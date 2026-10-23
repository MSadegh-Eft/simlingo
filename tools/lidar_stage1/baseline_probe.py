"""Strictly load SimLingo and compare fixed RGB/LLM/head outputs across environments.

Uses eager FP32 attention on CPU to isolate dependency changes. This is a component
compatibility fixture, not validation of the agent or the future LiDAR integration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoModel, AutoProcessor

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from simlingo_training.models.driving import DrivingModel
from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess

ROOT = Path('/ehsan/m.sadegh/simlingo_lidar')
SNAPSHOT = Path('/data/ghazaleh/.cache/huggingface/hub/models--OpenGVLab--InternVL2-1B/snapshots/0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187')
CHECKPOINT = REPO / 'models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)
    config = OmegaConf.load(REPO / 'models/simlingo/simlingo/.hydra/config.yaml')
    processor = AutoProcessor.from_pretrained(str(SNAPSHOT), trust_remote_code=True, local_files_only=True)
    original_load = AutoModel.from_pretrained

    def pinned_load(*unused_args, **unused_kwargs):
        return original_load(str(SNAPSHOT), trust_remote_code=True, local_files_only=True,
                             use_flash_attn=False, torch_dtype=torch.float32)

    cfg = OmegaConf.to_container(config.model, resolve=True)
    cfg.pop('_target_')
    with patch.object(AutoModel, 'from_pretrained', side_effect=pinned_load):
        model = DrivingModel(config.data_module, processor, None, **cfg)
    state = torch.load(str(CHECKPOINT), map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    image = Image.open(ROOT / 'audit/fixture/rgb.jpg').convert('RGB')
    image = image.crop((0, 0, image.width, int(image.height - (image.height * 4.8) // 16)))
    patches = dynamic_preprocess(image, image_size=448, use_thumbnail=False, max_num=2)
    pixels = torch.stack([build_transform(input_size=448)(p) for p in patches])
    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    ids = tokenizer('Current speed: 0.0 m/s. Predict the waypoints.', return_tensors='pt', add_special_tokens=False).input_ids
    with torch.inference_mode():
        vision = model.vision_model.image_encoder.model.extract_feature(pixels).reshape(1, -1, 896)
        text = model.adaptors.language.embed_tokens(ids)
        navigation = model.wp_encoder(torch.tensor([[[10., 0.], [20., 0.]]]))
        queries = torch.cat([model.adaptors.driving.queries[k] for k in model.adaptors.driving.order], dim=1)
        embeds = torch.cat((vision, text, navigation, queries), dim=1)
        hidden, logits = model.language_model(embeds, attention_mask=torch.ones(embeds.shape[:2], dtype=torch.bool))
        predictions = model.adaptors.driving.get_predictions(hidden[:, -queries.size(1):])
        arrays = {'vision': vision.numpy(), 'navigation': navigation.numpy(),
                  'last_logits':logits[:, -1].numpy(),
                  **{k: v.numpy() for k, v in predictions.items()}}
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise RuntimeError('Non-finite baseline output')
        # Repeated FP32 eager execution must agree with itself.
        repeated, _ = model.language_model(embeds, attention_mask=torch.ones(embeds.shape[:2], dtype=torch.bool))
        if not torch.allclose(hidden, repeated, atol=1e-6, rtol=1e-6):
            raise RuntimeError('Baseline fixture is not deterministic')
    comparisons = {}
    if args.reference:
        with np.load(args.reference) as reference:
            if set(reference.files) != set(arrays):
                raise RuntimeError('Reference component keys differ')
            for name, actual in arrays.items():
                expected = reference[name]
                comparisons[name] = {'max_absolute_error': float(np.max(np.abs(actual - expected))),
                                     'allclose': bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-4))}
            if not all(x['allclose'] for x in comparisons.values()):
                raise RuntimeError(f'Cross-environment parity failed: {comparisons}')
    np.savez(ROOT / 'reports' / f'baseline-{args.name}.npz', **arrays)
    digest = hashlib.sha256()
    with CHECKPOINT.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    report = {'name':args.name, 'python':sys.version, 'torch':torch.__version__,
              'checkpoint_sha256':digest.hexdigest(), 'hf_snapshot':SNAPSHOT.name,
              'strict_checkpoint_load':True, 'fixture_mode':'CPU eager FP32, model.eval(), original model components',
              'parameter_count':sum(p.numel() for p in model.parameters()),
              'output_shapes':{k:list(v.shape) for k,v in arrays.items()},
              'deterministic_repeat':True, 'comparisons':comparisons,
              'limits':'Does not exercise agent run_step, commentary decoding, or full training data assembly'}
    (ROOT / 'reports' / f'baseline-{args.name}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
