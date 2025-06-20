import yastn.yastn as yastn
from yastn.yastn import YastnError

from yastn.yastn.backend import backend_torch as backend
from yastn.yastn.tensor._auxliary import _clear_axes, _struct
from yastn.yastn.tensor._tests import _test_axes_match
from yastn.yastn.tensor._merging import _mask_tensors_leg_intersection
from yastn.yastn.tensor._contractions import _apply_mask_axes
from yastn.yastn.tensor._contractions import _common_inds, ncon, einsum

import numpy as np
from collections import namedtuple
from copy import deepcopy
import opt_einsum as oe


Node = namedtuple("Node", "t_pos, ind")
class Contract_Graph:
    def __init__(self, num_nodes, directed=False):
        # For each node, store list of neighbors
        self.n = num_nodes
        self.directed = directed
        self.adj = [[[]]*self.n for _ in range(self.n)]

    def add_edge(self, u, v, e):
        """Add edge u–v. For undirected, add both ways."""
        self.adj[u][v].append(e)
        if not self.directed:
            self.adj[v][u].append(e)

    def __repr__ (self):
        return "\n".join(f"{u}: {self.adj[u]}" for u in range(self.n))


def _contracted_edges(subscripts):
    r"""
    Identify the contractions from einsum string.

    Args:
        subscripts (str): einsum string.

    Returns:
        contracted_egdes (list): list of edges representing the contractions.
        Each edge is a pair of nodes representing the pair-wise contracted tensors.
    """
    if not isinstance(subscripts, str):
        raise YastnError('The first argument should be a string.')

    subscripts = subscripts.replace(' ', '')

    tmp = subscripts.split('->')
    if len(tmp) == 1:
        sin, sout = tmp[0], ''
    elif len(tmp) == 2:
        sin, sout = tmp
    else:
        raise YastnError('Subscript should have at most one separator ->')

    alphabet1 = 'ABCDEFGHIJKLMNOPQRSTUWXYZabcdefghijklmnopqrstuvwxyz'
    alphabet2 = alphabet1 + ',*'
    if any(v not in alphabet1 for v in sout) or \
       any(v not in alphabet2 for v in sin):
        raise YastnError('Only alphabetic characters can be used to index legs.')

    if sout == '':
        for v in sin.replace(',', ''):
            if sin.count(v) == 1:
                sout += v
    elif len(sout) != len(set(sout)):
        raise YastnError('Repeated index after ->')

    contracted = {
        idx
        for idx in set(sin.replace(',', ''))
        if sin.count(idx) == 2 and idx not in sout
    }

    unpaired = {}
    contracted_edges = []
    for i, term in enumerate(sin.split(',')):
        for ind, s in enumerate(term):
            if s in contracted:
                if s in unpaired:
                    contracted_edges.append((unpaired[s], Node(i, ind)))
                    unpaired.pop(s)
                else:
                    unpaired[s] = Node(i, ind)

    return contracted_edges

def _filter_blocks(tensor_list, pos_a, pos_b, axes):
    r"""
    Identify the indices of the blocks used in the actual contraction between tensor A and B.

    Args:
        tensor_list (list): list of tensors.
        pos_a (int): position of tensor A.
        pos_B (int): position of tensor B.
        axes (Sequence[Sequence[int], Sequence[int]]): contracted axes of A and B.

    Returns:
        ind_a, ind_b (tuple[int, int]): indices of the blocks in A and B, respectively.
    """
    in_a, in_b = _clear_axes(*axes)  # contracted meta legs
    a, b = tensor_list[pos_a], tensor_list[pos_b]
    mask_needed, (nin_a, nin_b) = _test_axes_match(a, b, sgn=-1, axes=(in_a, in_b))

    if mask_needed:
        msk_a, msk_b, a_hfs, b_hfs = _mask_tensors_leg_intersection(a, b, nin_a, nin_b)
        a = _apply_mask_axes(a, nin_a, msk_a)
        b = _apply_mask_axes(b, nin_b, msk_b)
        a = a._replace(hfs=a_hfs)
        b = b._replace(hfs=b_hfs)

        # the same block structure as before masked, but block dimensions are reduced
        # use the reduced tensors instead
        tensor_list[pos_a], tensor_list[pos_b] = a, b


    ind_a, ind_b = _common_inds(a.struct.t, b.struct.t, nin_a, nin_b, a.ndim_n, b.ndim_n, a.config.sym.NSYM)

    ind_a = list(range(len(a.struct.D))) if ind_a is None else ind_a
    ind_b = list(range(len(b.struct.D))) if ind_b is None else ind_b
    return  ind_a, ind_b

def reduced_tensor(t, block_inds):
    r"""
    Construct the reduced tensor, given the indices of restricted blocks.

    Args:
        t (list): list of tensors.
        pos_a (int): position of tensor A.
        pos_B (int): position of tensor B.
        axes (Sequence[Sequence[int], Sequence[int]]): contracted axes of A and B.

    Returns:
        ind_a, ind_b (tuple[int, int]): indices of the blocks in A and B, respectively.
    """

    # Build new struct and slices referencing the same data
    new_t = tuple(t.struct.t[i] for i in block_inds)
    new_D = tuple(t.struct.D[i] for i in block_inds)
    new_size = sum([int(np.prod(D)) for D in new_D])
    new_slices = tuple(t.slices[i] for i in block_inds)


    new_struct = _struct(
        s=t.struct.s,
        n=t.struct.n,
        diag=t.struct.diag,
        t=new_t,
        D=new_D,
        size=new_size
    )

    # Prepare kwargs for new tensor (no data copy)
    new_kwargs = dict(
        data=t._data,      # share data array
        struct=new_struct,
        slices=new_slices,
        mfs=deepcopy(getattr(t, "mfs", None)),
        hfs=deepcopy(getattr(t, "hfs", None)),
    )

    return t._replace(
        config=t.config,
        s=t.struct.s,
        n=t.struct.n,
        isdiag=t.struct.diag,
        **new_kwargs
    )

def preprocess_contracted_dims(subscripts, *operands):
    r"""
    Simplify tensors by dropping unused blocks in the contraction.

    Args:
        subscripts (str): einsum string.
        operands: tensors.

    Returns:
        G (Contract_Graph): graph representing the contraction.
        reduced_operands (tuple): simplified tensors.
    """
    edges = _contracted_edges(subscripts)
    G = Contract_Graph(num_nodes=len(operands), directed=False)
    operands_list = list(operands)

    def iter_edges(edges):
        t_blocks = {}
        for edge in edges:
            node1, node2 = edge
            t1_pos, t2_pos = node1[0], node2[0]
            block_ind_a, block_ind_b = _filter_blocks(operands_list, t1_pos, t2_pos, axes=(node1[1], node2[1]))

            if t1_pos not in t_blocks:
                t_blocks[t1_pos] = set(block_ind_a)
            else:
                t_blocks[t1_pos] = t_blocks[t1_pos] & set(block_ind_a)

            if t2_pos not in t_blocks:
                t_blocks[t2_pos] = set(block_ind_b)
            else:
                t_blocks[t2_pos] = t_blocks[t2_pos] & set(block_ind_b)

            G.add_edge(t1_pos, t2_pos, edge)

        for pos, ind in t_blocks.items():
            # Modify meta data to get reduced tensors,
            # whose final contraction result is the same as that of the original tensors.
            operands_list[pos] = reduced_tensor(operands_list[pos], ind)

    prev_structs = [_struct()]*len(operands_list)

    converged = False
    while not converged:
        iter_edges(edges)
        converged = True
        for i in range(len(operands_list)):
            if operands_list[i].struct != prev_structs[i]:
                prev_structs[i] = operands_list[i].struct
                converged = False


    return G, tuple(operands_list)

def build_sizes_dict(subscripts, *operands):
    r"""
    Compute the effective shapes of YASTN tensors in operands.
    We simply add up dimensions for all symmetry sectors.

    Args:
        subscripts (str): einsum string.
        operands: tensors.

    Returns:
        sizes_dict (dict): {symbol: size}
    """
    sizes_dict = {}

    subscripts = subscripts.replace(' ', '')
    tmp = subscripts.split('->')
    sin = tmp[0]

    for i, term in enumerate(sin.split(",")):
        shape = operands[i].get_shape()
        for j, c in enumerate(term):
            if c not in sizes_dict:
                sizes_dict[c] = shape[j]
            else:
                assert sizes_dict[c] == shape[j], f"{sizes_dict[c]}, {shape[j]}"
    return sizes_dict

def convert_path_to_ncon(subscripts, path):
    """
    Convert an opt_einsum contraction path to ncon format.

    Args:
        subscripts (str): einsum string.
        path: list of tuples, e.g., [(1, 2), (0, 1)]

    Returns:
        connects (list): list of lists of ints, one per initial tensor, for ncon
        output_labels (list): list of ints (negative) for output indices
        order (list): list of ints, positive labels in contraction order for ncon
    """
    # Parse subscripts
    if '->' in subscripts:
        left, right = subscripts.split('->')
        implicit = False
    else:
        left = subscripts
        right = None
        implicit = True

    terms = left.split(',')
    # Build initial connects: list of lists of labels (strings)
    connects = [list(term) for term in terms]

    # Determine output labels if implicit
    if implicit:
        # Count occurrences of each label across all inputs
        from collections import Counter
        cnt = Counter(lbl for term in connects for lbl in term)
        # Labels with count == 1
        outs = [lbl for lbl, c in cnt.items() if c == 1]
        # Sort alphabetically for numpy/opt_einsum default
        right = ''.join(sorted(outs))

    # Map each label to an integer
    unique_labels = sorted({lbl for term in connects for lbl in term})
    label_to_int = {lbl: i+1 for i, lbl in enumerate(unique_labels)}
    # Determine output labels as negative integers
    for i, lbl in enumerate(right, start=1):
        label_to_int[lbl] = -i
    output_labels = [label_to_int[lbl] for lbl in list(right)]

    # Prepare current connects as lists of ints for simulating contractions
    current = [[label_to_int[lbl] for lbl in term] for term in connects]
    order = []

    # Simulate contractions according to the path
    for (i, j) in path:
        # Identify labels common to both tensors at positions i and j in current list
        labels_i = set(current[i])
        labels_j = set(current[j])
        common = sorted(labels_i & labels_j)
        # Append these positive labels (if not already) in the contraction order
        for lbl in common:
            if lbl not in order:
                order.append(lbl)
        # Form new tensor connects: union minus the contracted labels
        new_connect = sorted((labels_i | labels_j) - set(common))
        # Remove the two tensors and append the resulting new tensor
        for idx in sorted([i, j], reverse=True):
            current.pop(idx)
        current.append(new_connect)
    # Return the initial connects mapping, output labels, and contraction order
    return [[label_to_int[lbl] for lbl in term] for term in connects], output_labels, order


if __name__ == "__main__":
    config_U1 = yastn.make_config(sym='U1', backend=backend)

    # test_case 1
    print("============================Test-Case-1==========================")
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1, 2), D=(2, 3, 2, 4))
    leg2 = yastn.Leg(config_U1, s=1, t=(-3, 0, 1, 2), D=(2, 3, 2, 4))
    leg3 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(3, 2))
    leg4 = yastn.Leg(config_U1, s=1, t=(0, ), D=(3, ))

    a = yastn.rand(config=config_U1, legs=[leg1, leg1.conj()])
    b = yastn.rand(config=config_U1, legs=[leg2, leg2.conj()])
    c = yastn.rand(config=config_U1, legs=[leg3, leg3.conj()])
    d = yastn.rand(config=config_U1, legs=[leg4, leg4.conj()])

    ts = (a, b, c, d)
    einsum_string = "ab,bc,cd,de->ae"
    G, reduced_ts = preprocess_contracted_dims(einsum_string, *ts)

    sizes_dict = build_sizes_dict(einsum_string, *reduced_ts)
    views = oe.helpers.build_views(einsum_string, sizes_dict)
    path, path_info = oe.contract_path(einsum_string, *views)
    # print("ts:", ts)
    # print("reduced_ts:", reduced_ts)
    print(sizes_dict)
    print(path_info)
    input, output, order = convert_path_to_ncon(einsum_string, path)
    res = ncon(ts, input, order=order)
    assert yastn.allclose(einsum(einsum_string, *ts), res)


    # test_case 2
    print("============================Test-Case-2==========================")
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1, 2), D=(2, 3, 2, 4))
    leg2 = yastn.Leg(config_U1, s=1, t=(-3, 0, 3), D=(2, 3, 2))
    leg3 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(3, 2))

    a = yastn.rand(config=config_U1, legs=[leg1, leg1.conj(), leg1.conj(), leg1.conj()])
    b = yastn.rand(config=config_U1, legs=[leg2, leg2, leg2, leg2.conj()])
    c = yastn.rand(config=config_U1, legs=[leg3, leg3.conj()])

    a = a.fuse_legs(axes=(0, (1, 2), 3), mode='hard')
    a = a.fuse_legs(axes=(0, (1, 2)), mode='hard')
    b = b.fuse_legs(axes=((0, 1), 2, 3), mode='hard')
    b = b.fuse_legs(axes=((0, 1), 2), mode='hard')


    ts = (a, b, c)
    einsum_string = "ab,bc,cd->ad"
    G, reduced_ts = preprocess_contracted_dims(einsum_string, *ts)

    sizes_dict = build_sizes_dict(einsum_string, *reduced_ts)
    views = oe.helpers.build_views(einsum_string, sizes_dict)
    path, path_info = oe.contract_path(einsum_string, *views)
    # print("ts:", ts)
    # print("reduced_ts:", reduced_ts)
    print(sizes_dict)
    print(path_info)
    input, output, order = convert_path_to_ncon(einsum_string, path)
    res = ncon(ts, input, order=order)
    assert yastn.allclose(einsum(einsum_string, *ts), res)

    # test_case 3
    print("============================Test-Case-3==========================")
    ts = (a, b, c)
    einsum_string = "ab,bc,ca"
    G, reduced_ts = preprocess_contracted_dims(einsum_string, *ts)

    sizes_dict = build_sizes_dict(einsum_string, *reduced_ts)
    views = oe.helpers.build_views(einsum_string, sizes_dict)
    path, path_info = oe.contract_path(einsum_string, *views)
    # print("ts:", ts)
    # print("reduced_ts:", reduced_ts)
    print(sizes_dict)
    print(path_info)
    input, output, order = convert_path_to_ncon(einsum_string, path)
    res = ncon(ts, input, order=order)
    assert yastn.allclose(einsum(einsum_string, *ts), res)
