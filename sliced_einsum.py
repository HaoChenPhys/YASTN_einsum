import yastn.yastn as yastn
from yastn.yastn.tensor._auxliary import _struct
from yastn.yastn.initialize import decompress_from_1d
from yastn.yastn.backend import backend_torch as backend
from yastn.yastn.tensor._contractions import ncon, einsum
from preprocessing import *

import numpy as np
from torch.utils.checkpoint import checkpoint
from collections import defaultdict
from itertools import product



def _restrict_charges(tensor, axes, t):
    r"""
    Return a new Tensor that is a view on tensor, but with blocks restricted to those
    where the specified axes have the given charge(s).

    Args:
        axes  (int | tuple[int]): leg(s) to restrict.
        t (int | Sequence[int]): charge(s) to select for the axes.

    Returns:
        t (Yastn Tensor): new tensor with restricted block metadata (no data copy).
    """
    from copy import deepcopy
    # Normalize input
    if isinstance(axes, int):
        axes = (axes,)
    assert len(axes) == len(t), "axes and t must have the same length"

    # Identify blocks matching requested charges
    kept_indices = []
    for i, block_charge in enumerate(tensor.struct.t):
        match = True
        for ax, tq in zip(axes, t):
            if block_charge[ax] != tq:
                match = False
                break
        if match:
            kept_indices.append(i)

    if not kept_indices:
        raise YastnError("No blocks matched the requested charge sector(s).")

    # Build new struct and slices referencing the same data
    new_t = tuple(tensor.struct.t[i] for i in kept_indices)
    new_D = tuple(tensor.struct.D[i] for i in kept_indices)
    new_size = sum([int(np.prod(D)) for D in new_D])
    new_slices = tuple(tensor.slices[i] for i in kept_indices)


    new_struct = _struct(
        s=tensor.struct.s,
        n=tensor.struct.n,
        diag=tensor.struct.diag,
        t=new_t,
        D=new_D,
        size=new_size
    )

    # Prepare kwargs for new tensor (no data copy)
    new_kwargs = dict(
        data=tensor._data,      # share data array
        struct=new_struct,
        slices=new_slices,
        mfs=deepcopy(getattr(tensor, "mfs", None)),
        hfs=deepcopy(getattr(tensor, "hfs", None)),
    )

    return tensor._replace(
        config=tensor.config,
        s=tensor.struct.s,
        n=tensor.struct.n,
        isdiag=tensor.struct.diag,
        **new_kwargs
    )

def sliced_einsum(subscripts, *operands, sliced=None, checkpoint=False):
    r"""
    Perform einsum with optional slicing and checkpointing.

    Args:
        subscripts (str): einsum string.
        operands: YASTN tensors
        sliced (str): indices to be sliced.
        checkpoint (bool)

    Returns:
        res (YASTN tensor): contraction result.
    """
    G, reduced_ts = preprocess_contracted_dims(einsum_string, *operands)
    _, sliced_edges = contracted_edges(subscripts, sliced=sliced)

    sliced_charge_list = []
    for edge in sliced_edges:
        node = edge[0]
        sliced_charge_list.append(operands[node[0]].get_legs(axes=node[1]).tD.keys())

    res = None
    for charge_choice in product(*sliced_charge_list):
        try:
            # slice contracted legs
            sliced_ts = list(reduced_ts)
            for edge, charge in zip(sliced_edges, charge_choice):
                if len(edge) == 2:
                    node0, node1 = edge
                    sliced_ts[node0[0]] = _restrict_charges(sliced_ts[node0[0]], axes=node0[1], t=charge)
                    sliced_ts[node1[0]] = _restrict_charges(sliced_ts[node1[0]], axes=node1[1], t=charge)
                else: # len(edge) == 1 open leg
                    node = edge[0]
                    sliced_ts[node[0]] = _restrict_charges(sliced_ts[node[0]], axes=node[1], t=charge)
            _, reduced_tmp = preprocess_contracted_dims(einsum_string, *sliced_ts)

            sizes_dict = build_sizes_dict(subscripts, *reduced_tmp)
            views = oe.helpers.build_views(subscripts, sizes_dict)
            path, path_info = oe.contract_path(subscripts, *views)
            print(sizes_dict)
            print(path_info)
            input, order = convert_path_to_ncon(subscripts, path)
            if res is None:
                res = ncon(reduced_tmp, input, order=order)
            else:
                res += ncon(reduced_tmp, input, order=order)


        except YastnError as e: # no consistent blocks found
            continue

    return res


    # print(sliced_info)
    # for t_pos in sliced_info:



    # a_data, a_meta = a.compress_to_1d()
    # b_data, b_meta = b.compress_to_1d()

    # def _loop_fn(a_data, b_data):
    #     tmp = None
    #     a, b = decompress_from_1d(a_data, a_meta), decompress_from_1d(b_data, b_meta)
    #     slice_leg = a.get_legs(axes=2)
    #     for t, D in slice_leg.tD.items():
    #         a_res = _restrict_charges(a, axes=2, t=t)
    #         upper_half= b.flip_signature().tensordot(a_res, (1, 0))
    #         b_res = _restrict_charges(b, axes=1, t=t)
    #         lower_half = b_res.flip_signature().tensordot(a, (0, 2))
    #         if tmp is None:
    #             tmp = upper_half.tensordot(lower_half, ([0, 2], [1, 0]))
    #         else:
    #             tmp += upper_half.tensordot(lower_half, ([0, 2], [1, 0]))

    #     tmp_data, tmp_meta = tmp.compress_to_1d()
    #     return tmp_data, tmp_meta


    # # use_reentrant = True is important here
    # tmp_data, tmp_meta = checkpoint(_loop_fn, a_data, b_data, use_reentrant=True)
    # rho = decompress_from_1d(tmp_data, tmp_meta).unfuse_legs(axes=(0, 1))
    # return rho.trace(axes=((0, 2), (1, 3))).to_number()

if __name__ == "__main__":
    config_U1 = yastn.make_config(sym='U1', backend=backend)

    # test_case 1
    print("============================Test-Case-1==========================")
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1, 2), D=(2, 3, 2, 4))
    leg2 = yastn.Leg(config_U1, s=1, t=(-3, 0, 1, 2), D=(2, 3, 2, 4))
    leg3 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(3, 2))
    leg4 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(3, 2))

    a = yastn.rand(config=config_U1, legs=[leg1, leg1.conj()])
    b = yastn.rand(config=config_U1, legs=[leg2, leg2.conj()])
    c = yastn.rand(config=config_U1, legs=[leg3, leg3.conj()])
    d = yastn.rand(config=config_U1, legs=[leg4, leg4.conj()])

    ts = (a, b, c, d)
    einsum_string = "ab,bc,cd,de->ae"
    # G, reduced_ts = preprocess_contracted_dims(einsum_string, *ts)
    res = sliced_einsum(einsum_string, *ts, sliced=["a", "b", "d", "e"], checkpoint=False)
    assert yastn.allclose(einsum(einsum_string, *ts), res)

    # test_case 2
    print("============================Test-Case-2==========================")
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1, 2), D=(2, 3, 2, 4))
    leg2 = yastn.Leg(config_U1, s=1, t=(-3, 0, 1, 2), D=(2, 3, 2, 4))
    leg3 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(3, 2))
    leg4 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(3, 2))

    a = yastn.rand(config=config_U1, legs=[leg1, leg1, leg1.conj()])
    b = yastn.rand(config=config_U1, legs=[leg1.conj(), leg1, leg2.conj()])
    c = yastn.rand(config=config_U1, legs=[leg3, leg3.conj()])

    ts = (a, b, c)
    einsum_string = "abc,bcd,de->ae"
    # G, reduced_ts = preprocess_contracted_dims(einsum_string, *ts)
    res = sliced_einsum(einsum_string, *ts, sliced=["a", "b", "c", "e"], checkpoint=False)
    assert yastn.allclose(einsum(einsum_string, *ts), res)

    # test_case 3
    print("============================Test-Case-3==========================")
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1, 2), D=(2, 3, 2, 4))
    leg2 = yastn.Leg(config_U1, s=1, t=(-3, 0, 1, 2), D=(2, 3, 2, 4))
    leg3 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(3, 2))
    leg4 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(3, 2))

    a = yastn.rand(config=config_U1, legs=[leg1, leg1, leg1.conj()])
    b = yastn.rand(config=config_U1, legs=[leg1.conj(), leg1, leg2.conj()])
    c = yastn.rand(config=config_U1, legs=[leg3, leg3.conj()])

    ts = (a, b, c)
    einsum_string = "abb,ccd,de->ae"
    # G, reduced_ts = preprocess_contracted_dims(einsum_string, *ts)
    res = sliced_einsum(einsum_string, *ts, sliced=["a", "b", "c", "e"], checkpoint=False)
    assert yastn.allclose(einsum(einsum_string, *ts), res)