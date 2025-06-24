import yastn.yastn as yastn
from yastn.yastn.tensor._auxliary import _struct
from yastn.yastn.initialize import decompress_from_1d
from yastn.yastn.backend import backend_torch as backend
from yastn.yastn.tensor._contractions import ncon, einsum
from preprocessing import *

import numpy as np
import torch
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

def _compress_ts(*operands):
    data_t, meta_t= tuple(zip( *(t.compress_to_1d() for t in operands) ))
    return data_t, meta_t

def _decompress_ts(data_t, meta_t):
    return tuple(decompress_from_1d(data, meta_t[i]) for i, data in enumerate(data_t))

def sliced_einsum(subscripts, *operands, sliced=None, checkpoint_move=False, verbose=0):
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
    G, reduced_ts = preprocess_contracted_dims(subscripts, *operands)
    _, sliced_edges = contracted_edges(subscripts, sliced=sliced)

    sliced_charge_list = []
    for edge in sliced_edges:
        node = edge[0]
        sliced_charge_list.append(operands[node[0]].get_legs(axes=node[1]).tD.keys())

    tot = None
    def _loop_body(meta_t, input, order, *data_t):
        tmp_ts = _decompress_ts(data_t, meta_t)
        res = ncon(tmp_ts, input, order=order)
        res_data, res_meta = res.compress_to_1d()
        return res_data, res_meta
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

            _, reduced_tmp = preprocess_contracted_dims(subscripts, *sliced_ts)
            t_shapes = build_tensor_shapes(subscripts, *reduced_ts)
            path, path_info = oe.contract_path(subscripts, *t_shapes, shapes=True)
            input, order = convert_path_to_ncon(subscripts, path)
            if verbose > 2:
                print(path_info)

            if not checkpoint_move:
                res = ncon(reduced_tmp, input, order=order)
            else:
                data_t, meta_t = _compress_ts(*reduced_tmp)
                res_data, res_meta = checkpoint(_loop_body, meta_t, input, order, *data_t, use_reentrant=False)
                res = decompress_from_1d(res_data, res_meta)

            if tot is None:
                tot = res
            else:
                tot += res

        except YastnError as e: # no consistent blocks found
            continue
    return tot
    # data_t, meta_t = _compress_ts(*reduced_ts)
    # def _full_loop(meta_t, *data_t):
    #     reduced_ts = _decompress_ts(data_t, meta_t)
    #     tot = None
    #     for charge_choice in product(*sliced_charge_list):
    #         try:
    #             # slice contracted legs
    #             sliced_ts = list(reduced_ts)
    #             for edge, charge in zip(sliced_edges, charge_choice):
    #                 if len(edge) == 2:
    #                     node0, node1 = edge
    #                     sliced_ts[node0[0]] = _restrict_charges(sliced_ts[node0[0]], axes=node0[1], t=charge)
    #                     sliced_ts[node1[0]] = _restrict_charges(sliced_ts[node1[0]], axes=node1[1], t=charge)
    #                 else: # len(edge) == 1 open leg
    #                     node = edge[0]
    #                     sliced_ts[node[0]] = _restrict_charges(sliced_ts[node[0]], axes=node[1], t=charge)

    #             _, reduced_tmp = preprocess_contracted_dims(einsum_string, *sliced_ts)
    #             sizes_dict = build_sizes_dict(subscripts, *reduced_tmp)
    #             views = oe.helpers.build_views(subscripts, sizes_dict)
    #             path, path_info = oe.contract_path(subscripts, *views)
    #             input, order = convert_path_to_ncon(subscripts, path)
    #             if verbose > 2:
    #                 print(sizes_dict)
    #                 print(path_info)

    #             res = ncon(reduced_tmp, input, order=order)

    #             if tot is None:
    #                 tot = res
    #             else:
    #                 tot += res

    #         except YastnError as e: # no consistent blocks found
    #             continue
    #     res_data, res_meta = tot.compress_to_1d()
    #     return res_data, res_meta

    # if checkpoint_move:
    #     res_data, res_meta = checkpoint(_full_loop, meta_t, *data_t, use_reentrant=True)
    # else:
    #     res_data, res_meta = _full_loop(meta_t, *data_t)
    # res = decompress_from_1d(res_data, res_meta)
    # return res


if __name__ == "__main__":
    config_U1 = yastn.make_config(sym='U1', backend=backend)

    # test_case 1
    print("============================Test-Case-1==========================")
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1, 2), D=(2, 30, 20, 4))
    leg2 = yastn.Leg(config_U1, s=1, t=(-3, 0, 1, 2), D=(2, 30, 20, 4))
    leg3 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(30, 20))
    leg4 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(30, 20))

    a = yastn.rand(config=config_U1, legs=[leg1, leg1.conj()])
    b = yastn.rand(config=config_U1, legs=[leg2, leg2.conj()])
    c = yastn.rand(config=config_U1, legs=[leg3, leg3.conj()])
    d = yastn.rand(config=config_U1, legs=[leg4, leg4.conj()])

    ts = (a, b, c, d)
    for t in ts:
        t._data.requires_grad = True

    einsum_string = "ab,bc,cd,da->"
    # res = sliced_einsum(einsum_string, *ts, sliced=["a", "b", "d", "e"], checkpoint_move=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True) as prof:
        res = sliced_einsum(einsum_string, *ts, sliced="b", checkpoint_move=True)

    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Forward [checkpoint]: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True) as prof:
        res._data.sum().backward()
    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Backward [checkpoint]: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")
    assert yastn.allclose(einsum(einsum_string, *ts), res, atol=1e-9)

    ts = (a, b, c, d)
    for t in ts:
        t._data.grad.data.zero_()
        t._data.requires_grad = True
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True) as prof:
        res = sliced_einsum(einsum_string, *ts, sliced="b", checkpoint_move=False)

    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Forward YASTN_EINSUM: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True) as prof:
        res._data.sum().backward()

    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Backward YASTN_EINSUM: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")

    # test_case 2
    print("============================Test-Case-2==========================")
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1, 2), D=(20, 30, 20, 40))
    leg2 = yastn.Leg(config_U1, s=1, t=(-3, 0, 1, 2), D=(20, 30, 20, 40))
    leg3 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(30, 20))
    leg4 = yastn.Leg(config_U1, s=1, t=(0, 1), D=(30, 20))

    a = yastn.rand(config=config_U1, legs=[leg1, leg1, leg1.conj()])
    b = yastn.rand(config=config_U1, legs=[leg1.conj(), leg1, leg2.conj()])
    c = yastn.rand(config=config_U1, legs=[leg3, leg3.conj()])

    ts = (a, b, c)
    for t in ts:
        t._data.requires_grad = True
    einsum_string = "abc,bcd,da->"
    # G, reduced_ts = preprocess_contracted_dims(einsum_string, *ts)
    # res = sliced_einsum(einsum_string, *ts, sliced=["a", "b", "c", "e"], checkpoint_move=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True) as prof:
        res = sliced_einsum(einsum_string, *ts, sliced="b", checkpoint_move=True)

    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Forward [checkpoint]: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True) as prof:
        res.to_number().backward()

    print(b._data.grad.numpy().sum())
    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Backward [checkpoint]: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")
    assert yastn.allclose(einsum(einsum_string, *ts), res, atol=1e-9)

    for t in ts:
        t._data.grad.data.zero_()
        t._data.requires_grad = True

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True) as prof:
        res = sliced_einsum(einsum_string, *ts, sliced="b", checkpoint_move=False)
        # res = einsum(einsum_string, *ts)

    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Forward YASTN_EINSUM: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True) as prof:
        res._data.sum().backward()

    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Backward YASTN_EINSUM: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")


    # test_case 3
    print("============================Test-Case-3==========================")
    leg1 = yastn.Leg(config_U1, s=1, t=(-1, 0, 1), D=(5, 10, 5))
    leg2 = yastn.Leg(config_U1, s=1, t=(-4, -3, -2, -1, 0, 1, 2, 3, 4), D=(50, 50, 30, 30, 40, 30, 30, 50, 50))

    a = yastn.rand(config=config_U1, legs=[leg2, leg1, leg1.conj(), leg2])
    b = yastn.ones(config=config_U1, legs=[leg2, leg2])

    ts = (b.flip_signature(), a, b.flip_signature(), a)
    for t in ts:
        t._data.requires_grad = True

    einsum_string = "fg,gaad,ed,fbbe->"
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True) as prof:
        res = sliced_einsum(einsum_string, *ts, sliced="f", checkpoint_move=True)

    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Forward: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True) as prof:
        res._data.sum().backward()

    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Backward: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")

    assert yastn.allclose(einsum(einsum_string, *ts), res, atol=1e-9)

    for t in ts:
        t._data.grad.data.zero_()
        t._data.requires_grad = True

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True) as prof:
        res = sliced_einsum(einsum_string, *ts, sliced="f", checkpoint_move=False)
        # res = einsum(einsum_string, *ts)

    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Forward YASTN_EINSUM: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=True) as prof:
        res._data.sum().backward()

    peak_cpu = max(e.cpu_memory_usage for e in prof.events())
    print(f"Backward YASTN_EINSUM: Peak CPU memory usage: {peak_cpu / 1024:.2f} kB")