"""Deterministic ego-frame preprocessing and pretrained 3D-to-language connector."""
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend

WEIGHT_SHA256 = 'a0a2e5e234ce943082ae9ba26bcff74e285f16ed8d04fed5c36ffe8dc30ea353'
SECTORS = ('front', 'right', 'rear', 'left')
PREPROCESS_VERSION = 'ego-xyz-scale0.2-voxel0.01-first-v1'


def sector_ids(xyz):
    """Half-open 90 degree sectors, X forward and Y right, front [-45,45)."""
    # Comparisons make exact diagonal boundaries agree across NumPy/Torch;
    # float32 atan2 can round the same 45 degree boundary in opposite directions.
    x,y = xyz[...,0],xyz[...,1]
    if isinstance(xyz,torch.Tensor):
        ids=torch.zeros_like(x,dtype=torch.long)
        for mask,value in (((y>=x)&(y>-x),1),((x<=-y)&(x<y),2),((y<=x)&(y<-x),3)):
            ids=torch.where(mask,value,ids)
        return ids
    ids=np.zeros_like(x,dtype=np.int64)
    for mask,value in (((y>=x)&(y>-x),1),((x<=-y)&(x<y),2),((y<=x)&(y<-x),3)):
        ids=np.where(mask,value,ids)
    return ids


@dataclass
class Cloud:
    coord: torch.Tensor
    grid: torch.Tensor
    feat: torch.Tensor
    metric: torch.Tensor
    timestamp: float

    def to(self, device):
        return replace(self, **{k: getattr(self, k).to(device) for k in ('coord', 'grid', 'feat', 'metric')})


def prepare_cloud(xyz, timestamp=0.):
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not 0 < len(xyz) <= 65536 or not np.isfinite(xyz).all():
        raise ValueError('Expected a finite, nonempty full sweep with at most 65536 XYZ points')
    coord = xyz * np.float32(.2)
    grid = np.floor(coord / np.float32(.01)).astype(np.int32)
    grid -= grid.min(0)
    _, representatives = np.unique(grid, axis=0, return_index=True)
    coord, grid, metric = coord[representatives], grid[representatives], xyz[representatives]
    feat = np.concatenate((coord, np.zeros((len(coord), 6), dtype=np.float32)), axis=1)
    return Cloud(*(torch.from_numpy(x.copy()) for x in (coord, grid, feat, metric)), float(timestamp))


def geometry_targets(cloud, boxes):
    """Training-only targets, scoped to observed vehicles within the 64m box radius."""
    xyz = cloud.metric.cpu().numpy()
    tree=cKDTree(xyz)
    presence = np.zeros(4, np.float32)
    valid = np.ones(4, bool)
    nearest = np.full(4, np.inf)
    for box in boxes:
        if box.get('class') != 'car':
            continue
        pos = np.asarray(box['position'])
        extent=np.asarray(box['extent'])
        if pos.shape!=(3,) or extent.shape!=(3,) or not np.isfinite(pos).all() or not np.isfinite(extent).all() or not (extent>0).all() or not np.isfinite(box['yaw']):
            raise ValueError('Invalid vehicle box geometry')
        distance = np.linalg.norm(pos[:2])
        if distance >= 64:
            continue
        sector = int(sector_ids(pos))
        c, s = np.cos(box['yaw']), np.sin(box['yaw'])
        # An enclosing sphere gives an exact candidate superset, then retain
        # the same oriented-box test. Avoid scanning the full sweep per vehicle.
        candidates=tree.query_ball_point(pos,float(np.linalg.norm(extent))+1e-6)
        local = (xyz[candidates] - pos) @ np.array([[c,-s,0], [s,c,0], [0,0,1]])
        hits = int((np.abs(local) < extent).all(1).sum())
        if hits >= 3:
            presence[sector] = 1
            nearest[sector] = min(nearest[sector], distance)
        elif hits > 0 or box.get('num_points', 0) > 0:
            valid[sector] = False
    # Presence of a supported object remains valid despite another ambiguous object;
    # nearest-distance supervision does not, as that ambiguous object may be closer.
    presence_valid = valid | presence.astype(bool)
    distance_valid = valid & presence.astype(bool)
    bins = np.digitize(nearest, [10.,20.,40.]).clip(0,3).astype(np.int64)
    return {'presence':torch.from_numpy(presence), 'presence_valid':torch.from_numpy(presence_valid),
            'distance':torch.from_numpy(bins), 'distance_valid':torch.from_numpy(distance_valid)}


class ResamplerBlock(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        self.q_norm = nn.LayerNorm(width)
        self.k_norm = nn.LayerNorm(width)
        self.cross = nn.MultiheadAttention(width, 8, batch_first=True)
        self.self_norm = nn.LayerNorm(width)
        self.self_attn = nn.MultiheadAttention(width, 8, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2048), nn.GELU(), nn.Linear(2048, width))

    def forward(self, queries, keys, mask):
        q, k = self.q_norm(queries), self.k_norm(keys)
        # cuDNN SDPA has no execution plan for these long, sector-masked keys
        # on the target GPU. Prefer fused attention with a portable math fallback.
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            queries = queries + self.cross(q, k, k, attn_mask=mask, need_weights=False)[0]
        q = self.self_norm(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]
        return queries + self.ff(queries)


class SpatialConnector(nn.Module):
    def __init__(self, channels, llm_width=896):
        super().__init__()
        self.project = nn.ModuleList(nn.Linear(c,512) for c in channels)
        self.position = nn.Sequential(nn.Linear(3,512), nn.GELU(), nn.Linear(512,512))
        self.levels = nn.Parameter(torch.randn(len(channels),512) * .02)
        self.queries = nn.Parameter(torch.randn(1,128,512) * .02)
        self.empty = nn.Parameter(torch.randn(4,512) * .02)
        self.blocks = nn.ModuleList(ResamplerBlock() for _ in range(2))
        self.out = nn.Sequential(nn.LayerNorm(512), nn.Linear(512,llm_width))
        self.presence = nn.Linear(512,1)
        self.distance = nn.Linear(512,4)

    def forward(self, hierarchy):
        keys, sectors = [], []
        for i, (features, metres) in enumerate(hierarchy):
            keys.append(self.project[i](features) + self.position(metres / 64.) + self.levels[i])
            sectors.append(sector_ids(metres))
        keys = torch.cat(keys)
        sectors = torch.cat(sectors)
        # Null keys are visible only when their sector contains no real keys.
        present = torch.bincount(sectors, minlength=4) > 0
        keys = torch.cat((keys, self.empty.to(keys.dtype)))
        sectors = torch.cat((sectors, torch.arange(4, device=keys.device)))
        query_sector = torch.arange(4, device=keys.device).repeat_interleave(32)
        mask = query_sector[:,None] != sectors[None,:]
        mask[:, -4:] |= present[None,:]
        queries = self.queries.to(keys.dtype)
        for block in self.blocks:
            queries = block(queries, keys[None], mask)
        pooled = queries.reshape(1,4,32,512).mean(2)
        return self.out(queries), {'presence':self.presence(pooled).squeeze(-1), 'distance':self.distance(pooled)}


class LidarPerception(nn.Module):
    def __init__(self, checkpoint, verify_hash=True):
        super().__init__()
        import utonia
        if verify_hash:
            digest = hashlib.sha256()
            with Path(checkpoint).open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2), b''): digest.update(block)
            if digest.hexdigest() != WEIGHT_SHA256:
                raise ValueError('Unexpected Utonia Base checkpoint hash')
        metadata = utonia.load(str(checkpoint), ckpt_only=True)
        channels = metadata['config']['enc_channels']
        self.level_indices = (0, len(channels)//2, len(channels)-1)
        self.encoder = utonia.load(str(checkpoint), custom_config={'shuffle_orders':False}).float()
        self.connector = SpatialConnector([channels[i] for i in self.level_indices])
        self.joint = False
        for module in self.encoder.modules():
            if hasattr(module, 'shuffle_orders'): module.shuffle_orders = False
            for key in ('shift_coords','jitter_coords','rescale_coords'):
                if hasattr(module,key): setattr(module,key,None)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if self.joint and mode:
            for stage in list(self.encoder.enc.children())[-2:]: stage.train()
        return self

    def forward(self, cloud):
        device_type = cloud.coord.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            point = self.encoder({'coord':cloud.coord.float(), 'origin_coord':cloud.metric.float(),
                                  'grid_coord':cloud.grid, 'feat':cloud.feat.float(),
                                  'offset':torch.tensor([len(cloud.coord)], device=cloud.coord.device)})
        hierarchy = []
        while True:
            hierarchy.append((point.feat, point.origin_coord))
            if 'pooling_parent' not in point: break
            point = point.pooling_parent
        hierarchy.reverse()
        return self.connector([hierarchy[i] for i in self.level_indices])
