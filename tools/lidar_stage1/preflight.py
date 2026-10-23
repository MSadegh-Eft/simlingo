"""Bounded GPU-2 kernel checks and a real-scan Utonia forward pass; no training."""
import importlib
import importlib.metadata
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path('/ehsan/m.sadegh/simlingo_lidar')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise RuntimeError('This check requires CUDA_VISIBLE_DEVICES=2')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA device')
    expected_uuid = subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'], text=True).strip()
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(properties.uuid)
    if actual_uuid.removeprefix('GPU-') != expected_uuid.removeprefix('GPU-'):
        raise RuntimeError(f'CUDA device maps to {actual_uuid}, expected physical GPU 2 ({expected_uuid})')
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.20, 0)
    torch.cuda.reset_peak_memory_stats()
    assets = json.loads((ROOT / 'reports/utonia-assets.json').read_text())
    versions = {}
    for package, module in [('torch','torch'), ('torchvision','torchvision'),
                            ('transformers','transformers'), ('peft','peft'),
                            ('pytorch-lightning','pytorch_lightning'), ('timm','timm'),
                            ('spconv-cu124','spconv.pytorch'), ('torch-scatter','torch_scatter'),
                            ('flash-attn','flash_attn'), ('carla','carla'),
                            ('deepspeed','deepspeed'), ('laspy','laspy'), ('lazrs','lazrs')]:
        if module == 'carla':
            spec = importlib.util.find_spec(module)
            if spec is None or not Path(spec.origin).is_relative_to(Path(sys.prefix)):
                raise RuntimeError('Expected CARLA from the isolated environment; rerun bootstrap.sh and use run.sh')
        importlib.import_module(module)
        versions[package] = importlib.metadata.version(package)
    import flash_attn
    import spconv.pytorch as spconv
    import torch_scatter
    import utonia
    source_files = {}
    for source in (Path(assets['source_directory']) / 'utonia').rglob('*.py'):
        relative = source.relative_to(Path(assets['source_directory']) / 'utonia')
        installed = Path(utonia.__file__).parent / relative
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if not installed.exists() or hashlib.sha256(installed.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Installed Utonia differs from pinned source: {relative}')
        source_files[str(relative)] = digest
    # Kernel-level forward/backward checks, not optimization steps.
    q = torch.randn(1, 16, 2, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    flash_attn.flash_attn_func(q, q, q).float().square().mean().backward()
    assert torch.isfinite(q.grad).all(), 'FlashAttention BF16 backward failed'
    scattered = torch_scatter.scatter_add(torch.tensor([2., 3.], device='cuda'), torch.tensor([0, 0], device='cuda'))
    assert scattered.item() == 5., 'CUDA scatter failed'
    features = torch.randn(3, 4, device='cuda', requires_grad=True)
    sparse = spconv.SparseConvTensor(features, torch.tensor([[0,1,1,1],[0,1,1,2],[0,1,2,1]], device='cuda', dtype=torch.int32), [4,4,4], 1)
    conv = spconv.SubMConv3d(4, 8, 3, padding=1, bias=False).cuda()
    conv(sparse).features.square().mean().backward()
    assert torch.isfinite(features.grad).all(), 'Sparse convolution FP32 backward failed'
    del conv, sparse, features, q, scattered
    torch.cuda.empty_cache()
    model = utonia.load(assets['weight_path'], custom_config={'shuffle_orders': False}).float().eval().cuda()
    for module in model.modules():
        if hasattr(module, 'shuffle_orders'):
            module.shuffle_orders = False
        for name in ('shift_coords', 'jitter_coords', 'rescale_coords'):
            if hasattr(module, name):
                setattr(module, name, None)
    xyz = np.load(ROOT / 'audit/fixture/points.npy')
    coord = xyz * np.float32(0.2)
    grid = np.floor(coord / np.float32(0.01)).astype(np.int32)
    grid -= grid.min(axis=0)
    _, representatives = np.unique(grid, axis=0, return_index=True)
    feat = np.concatenate((coord[representatives], np.zeros((len(representatives), 6), dtype=np.float32)), axis=1)
    point = {'coord':torch.from_numpy(coord[representatives]).cuda(),
             'grid_coord':torch.from_numpy(grid[representatives]).cuda(),
             'feat':torch.from_numpy(feat).cuda(),
             'offset':torch.tensor([len(representatives)], device='cuda', dtype=torch.long)}
    with torch.inference_mode():
        output = model(dict(point))
        if not torch.isfinite(output.feat).all():
            raise RuntimeError('Non-finite Utonia features')
        reference = output.feat.clone()
        repeated = model(dict(point))
        max_diff = (reference - repeated.feat).abs().max().item()
        if not torch.allclose(reference, repeated.feat, atol=1e-4, rtol=1e-4):
            raise RuntimeError(f'Repeated Utonia output differs: {max_diff}')
    torch.cuda.synchronize()
    report = {'python': sys.version, 'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
              'physical_gpu_uuid':expected_uuid, 'torch_device_uuid':actual_uuid,
              'logical_device':0, 'device_name':torch.cuda.get_device_name(0),
              'torch_cuda':torch.version.cuda, 'versions':versions,
              'carla_module':importlib.import_module('carla').__file__,
              'utonia_module':utonia.__file__, 'verified_utonia_source_files':source_files,
              'flash_bf16_forward_backward':True, 'scatter_cuda':True,
              'sparse_fp32_forward_backward':True, 'strict_utonia_checkpoint_load':True,
              'utonia_parameter_count':sum(p.numel() for p in model.parameters()),
              'raw_points':len(xyz), 'voxel_points':len(representatives),
              'output_feature_shape':list(reference.shape),
              'deterministic_repeat_max_error':max_diff,
              'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
              'limits':'Inference and small kernel checks only; not an end-to-end training memory/throughput profile'}
    (ROOT / 'reports/preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
