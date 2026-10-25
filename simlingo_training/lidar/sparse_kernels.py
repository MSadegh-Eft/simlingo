"""Checkpoint sparse convolution kernel choices, rather than re-tuning on resume.

spconv's default C++ tuner keeps its choices only in process memory. Its Python
dispatch uses the same kernels and exposes serializable tuning tables. Keep FP32
and split-K=1; preserve the fastest measured compatible kernel in checkpoints.
"""
import random

import numpy as np
import torch

TABLES=('kc_forward_cache','kc_dgrad_cache','kc_wgrad_cache')


def configure():
    import spconv.algo as algo
    import spconv.pytorch.ops as ops
    import spconv.constants as constants
    ops.SPCONV_CPP_GEMM=False
    constants.SPCONV_ALLOW_TF32=False
    constants.SPCONV_BWD_SPLITK=[1]
    algo.SPCONV_BWD_SPLITK=[1]


def dump(model):
    if not getattr(model,'lidar_numerics_policy',{}).get('sparse_kernel_checkpointing'): return None
    from spconv.algo import CONV
    return {'schema':1,'tables':{name:[{'key':list(key),'descriptor':str(v.algo_desp),'arch':list(v.arch),
        'splitk':v.splitk,'nvrtc':v.algo_desp.is_nvrtc} for key,v in sorted(getattr(CONV,name).items())] for name in TABLES}}


def restore(model,state):
    active=getattr(model,'lidar_numerics_policy',{}).get('sparse_kernel_checkpointing')
    if (state is not None)!=bool(active): raise ValueError('Sparse kernel checkpoint policy differs')
    if state is None: return
    if state['schema']!=1 or set(state['tables'])!=set(TABLES): raise ValueError('Sparse kernel state schema differs')
    from spconv.algo import CONV,ALL_CONV_ALGO_DESPS,BestConvAlgoByProfile
    choices={str(d):d for d in ALL_CONV_ALGO_DESPS}
    arch=(torch.cuda.get_device_properties(0).major,torch.cuda.get_device_properties(0).minor)
    for name,rows in state['tables'].items():
        table=getattr(CONV,name); table.clear()
        for row in rows:
            if row['descriptor'] not in choices or tuple(row['arch'])!=arch or row['splitk']!=1:
                raise ValueError('Sparse kernel identity/architecture/split-K differs')
            descriptor=choices[row['descriptor']]; descriptor.is_nvrtc=row['nvrtc']
            key=tuple(row['key'])
            if key in table: raise ValueError('Duplicate sparse kernel cache key')
            table[key]=BestConvAlgoByProfile(descriptor,arch,splitk=1)


def prime(model,cloud):
    """Populate all stage-specific kernels before checkpoint zero, without learning.

    Forward/backward initializes the tuner only. Restore buffers, modes and every
    training RNG; parameters and optimizer never receive an update.
    """
    buffers={k:v.detach().clone() for k,v in model.lidar.named_buffers()}
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prefix,geometry=model.lidar(cloud.to('cuda'))
            loss=prefix.float().square().mean()+sum(v.float().square().mean() for v in geometry.values())
        loss.backward(); torch.cuda.synchronize()
    finally:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name,value in model.lidar.named_buffers(): value.copy_(buffers[name])
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
