"""Explicit reproducibility and FP32 cumulative waypoint sums for LiDAR runs.

The pinned vision and Utonia code call FlashAttention directly, so the HF
environment flag alone does not configure their backwards. No installed source
or pretrained parameter is modified. Apply once, before training or validation.
"""
import functools
import os
import sys
import types

import torch


def prepare_environment():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    os.environ['FLASH_ATTENTION_DETERMINISTIC']='1'


def cumulative_waypoints(increments):
    """Small (10/20 point) triangular scan in FP32, with a stable adjoint.

    CUDA BF16 cumsum is not available under torch's strict deterministic policy.
    A triangular matrix gives the same prefix-sum map and reverse-sum adjoint,
    while avoiding BF16 accumulation/rounding in the driving output and gradient.
    """
    dtype=torch.float64 if increments.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=increments.device.type,enabled=False):
        length=increments.shape[-2]
        triangle=torch.ones(length,length,device=increments.device,dtype=dtype).tril()
        return triangle@increments.to(dtype)


def _deterministic(function):
    if getattr(function,'_simlingo_deterministic',False): return function
    @functools.wraps(function)
    def call(*args,**kwargs):
        kwargs['deterministic']=True
        return function(*args,**kwargs)
    call._simlingo_deterministic=True
    return call


def _predictions(adaptor,features,logits=None):
    offset=0; result={}
    for name in adaptor.order:
        size=adaptor.sizes[name]
        result[name]=cumulative_waypoints(adaptor.heads[name](features[:,offset:offset+size]))
        offset+=size
    return result


def native_grid_positions(embeddings,positions,height,width):
    side=embeddings.image_size//embeddings.patch_size
    if (height,width)!=(side,side):
        raise ValueError('A non-native vision grid needs a reviewed deterministic interpolation policy')
    # The pinned 448px tiles already have the pretrained 32x32 patch grid.
    # Bicubic interpolation to the identical grid is an identity operation,
    # but its generic CUDA backward is rejected by strict deterministic mode.
    return positions


def configure_model(model):
    prepare_environment()
    from simlingo_training.lidar.sparse_kernels import configure
    configure()
    import flash_attn
    # Utonia imports the package; InternVision imports the interface function
    # into its pinned remote module. Configure both references explicitly.
    flash_attn.flash_attn_varlen_qkvpacked_func=_deterministic(flash_attn.flash_attn_varlen_qkvpacked_func)
    vision_module=sys.modules[type(model.vision_model.image_encoder.model.vision_model).__module__]
    vision_module.flash_attn_varlen_qkvpacked_func=_deterministic(vision_module.flash_attn_varlen_qkvpacked_func)
    embeddings=model.vision_model.image_encoder.model.vision_model.embeddings
    embeddings._get_pos_embed=types.MethodType(native_grid_positions,embeddings)
    model.adaptors.driving.get_predictions=types.MethodType(_predictions,model.adaptors.driving)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    policy={'schema':2,'torch_deterministic':True,'flash_backward_deterministic':True,
            'cublas_workspace':os.environ['CUBLAS_WORKSPACE_CONFIG'],'tf32':False,
            'waypoint_accumulation':'FP32 triangular scan','cudnn_benchmark':False}
    policy['vision_position_grid']='Native grid identity; reject unreviewed resize'
    policy['sparse_kernel_checkpointing']='Python dispatch; FP32; split-K=1; checkpoint tuner tables; prime before update zero'
    model.lidar_numerics_policy=policy
    return policy


def make_training_model(**kwargs):
    """Create the pinned model with the required numerical policy installed."""
    prepare_environment()
    from simlingo_training.lidar.model import make_model
    model=make_model(**kwargs)
    configure_model(model)
    return model
