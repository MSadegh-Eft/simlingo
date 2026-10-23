"""Publish Stage 2 completion only when all persisted checks agree."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess

from simlingo_training.lidar.storage import digest_file, atomic_json

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/ehsan/m.sadegh/simlingo_lidar')
STAGE=ROOT/'stage2'


def read(path): return json.loads(Path(path).read_text())


def main():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    assert branch=='feat/simlingo-lidar',branch
    ready=read(STAGE/'pilot/ready.json'); audit=read(STAGE/'pilot/audit.json')
    for key,name in [('manifest_sha256','manifest.json'),('validation_sha256','validation.json'),('audit_sha256','audit.json')]:
        assert ready[key]==digest_file(STAGE/'pilot'/name),name
    assert ready['passed'] and 50000<=audit['samples']['train']<=100000
    assert audit['joint_cycle_replay']==192
    assert audit['joint_cycle_tasks']=={'dreamer':384,'driving':128,'qa':128,'commentary':128,'rear_driving':96,'geometry_qa':96}
    smoke=read(STAGE/'reports/smoke.json'); joint=read(STAGE/'reports/joint-step.json'); data=read(STAGE/'reports/data-checks.json')
    assert smoke['checks']['passed'] and joint['passed'] and data['passed']
    assert smoke['checks']['manifest']['sha256']==ready['manifest_sha256']
    assert data['manifest_sha256']==joint['manifest_sha256']==ready['manifest_sha256']
    metadata=read(STAGE/'reports/checkpoint-metadata.json')
    assert metadata['manifest_sha256']==ready['manifest_sha256']
    assert metadata['generation_cache'] is False and smoke['checks']['generation_cache_policy']['enabled'] is False
    for name,expected in metadata['code'].items(): assert digest_file(REPO/name)==expected,name
    unit=(STAGE/'logs/unit-tests.log').read_text()
    match=re.search(r'Ran (\d+) tests',unit)
    assert match and int(match[1])>=14 and '\nOK\n' in unit and 'FAILED' not in unit
    baseline=read(ROOT/'reports/baseline-stage2-regression.json')
    recheck=read(ROOT/'reports/baseline-stage1-recheck.json')
    for report in (baseline,recheck):
        assert report['strict_checkpoint_load'] and all(v['allclose'] for v in report['comparisons'].values())
        assert report['checkpoint_sha256']==metadata['released_sha256']
    preflight=read(ROOT/'reports/preflight.json')
    for key in ('flash_bf16_forward_backward','scatter_cuda','sparse_fp32_forward_backward','strict_utonia_checkpoint_load'): assert preflight[key]
    geometry=audit['geometry']; screen=audit['benchmark_screen']
    prepared=sum(p.stat().st_size for p in STAGE.rglob('*.tar*') if p.is_file())
    assert prepared<=100*1024**3 and shutil.disk_usage(STAGE).free>=150*1024**3
    assert not list((STAGE/'pilot').glob('*.part'))
    report={'stage':2,'status':'complete','recorded_at_utc':datetime.now(timezone.utc).isoformat(),'branch':branch,
        'stage1_commit':'93d2080','stage1_recheck':'passed','physical_gpu_uuid':smoke['physical_gpu_uuid'],
        'original_checkpoint_sha256':baseline['checkpoint_sha256'],'utonia_weight_sha256':metadata['utonia_sha256'],
        'unit_tests':int(match[1]),'worker_reproducibility':data,'pilot':audit,'prepared_shard_bytes':prepared,
        'pilot_ready':ready,'smoke':smoke,'joint_optimizer_step':joint,
        'long_training_started':False,'benchmark_started':False,'stage3_started':False,
        'next':'Review Stage 2, then profile the complete GPU-2 training/validation/checkpoint pipeline in Stage 3 before agreeing the training budget'}
    atomic_json(REPO/'tools/lidar_stage2/stage2-results.json',report)
    tiny=smoke['checks']['tiny_learning_and_frozen_weights']
    trainable=smoke['checks']['loaded_released_and_utonia_strictly']
    joint_count=sum(g['parameters'] for g in smoke['checks']['joint_optimizer_groups'].values())
    text=f'''# Stage 2 verification

Stage 1 passed its repeated pinned-source/environment, GPU-2 kernel and released
checkpoint checks. The corrected RGB components still match the saved baseline
within 1e-4. Work remains on `{branch}`; Stage 1 is commit `93d2080`.

Stage 2 is complete. The pretrained Utonia Base encoder supplies first/middle/final
features and metric positions to the two-block, width-512 spatial connector. Its
128 sector queries produce width-896 tokens for both language and driving. The
released RGB/projector, trained LoRA, navigation and driving weights are preserved.
There are {trainable['alignment_trainable']:,} trainable parameters in alignment
and {joint_count:,} in joint training. The last two Utonia stages account for
{smoke['checks']['joint_optimizer_groups']['lidar']['parameters']:,} of these.

The pilot contains **{audit['samples']['train']:,} training frames** across
{audit['groups']['train']} source groups and **{audit['samples']['val']:,} Town13
validation frames** across {audit['groups']['val']} groups. Prepared shards occupy
{prepared/1024**3:.2f} GiB including the small correctness dataset. Archives remain
intact. Every eligible RGB was decoded; LAZ scans were voxelized; boxes, navigation
and complete future measurements were checked. Geometry QA eligibility uses
retained-point support and ambiguous-label masks. Annotation memory is bounded,
and geometry masks use contiguous arrays for worker spawning.

The manifest excludes {len(screen['excluded_groups'])} source route group(s) identified
by the recorded benchmark containment screen. This screen is conservative and
does not establish independent scenario design or undo prior checkpoint exposure.
Its thresholds and matches are recorded in the audit. Fixed route-balanced
validation requests are persisted for driving, original QA, commentary, Dreamer,
rear/side driving and observed-geometry QA. Counts: `{json.dumps(audit['validation_counts'],sort_keys=True)}`.
The pilot has no `parkinglane` bucket; its missing-bucket exclusions and weight
renormalization are explicit. Parking coverage needs review before longer training.

Verification passed:

- {int(match[1])} contract tests: coordinates/sectors, visibility/range masks,
  empty sectors, zero-count gradients, accumulation normalization, task/replay
  quotas, bucket weighting, source groups, future boundaries, indexed payloads,
  photometric seeds, legacy inference masks and checkpoint recovery.
- Real samples match original waypoint/route construction. Ten presentations
  covering all tasks and RGB replay match with zero and two spawned loader workers.
- Combined loading is strict outside the new LiDAR modules. Prefix placement,
  no-target-leakage checks, direct-backbone parity and repeated uncached generation
  pass on the fixtures. A pilot geometry-QA example failed cached/uncached token parity, so KV
  caching remains disabled. The final suite uses uncached generation throughout.
- Language and driving gradients separately reach the connector through frozen
  modules. All intended joint groups receive finite nonzero gradients. A separate
  joint optimizer update changes each group, preserves frozen weights and keeps
  Adam moments FP32.
- The final disposable tiny alignment check ran {tiny['updates']} updates on
  {tiny['presentations']} presentations; loss changed from {tiny['losses'][0]:.3f}
  to {tiny['losses'][-1]:.3f}. Frozen parameters remained byte-identical. Full model,
  optimizer, sampler and CUDA random state recovered from an atomic checkpoint.

Development checks found and corrected an unsupported cuDNN attention path,
non-recursive imgaug seeding, unbounded annotation retention, the legacy RGB
projector freeze, zero-count loss division, inference padding/commentary-off
handling, agent evaluation mode and the overwritten overfit setting.

These are bounded correctness checks. They do not show improved driving, a
retention-passing checkpoint, full-mixture memory/throughput, or live-agent sensor
parity. Tiny-check weights were discarded. Longer training, Stage 3 profiling,
benchmark evaluation, temporal LiDAR and merging have not started.

Detailed artifacts: `{STAGE}/pilot/{{manifest,validation,audit,ready}}.json` and
`{STAGE}/reports/`. Reproduction commands are in
[`tools/lidar_stage2/README.md`](../tools/lidar_stage2/README.md). The machine-readable
summary is [`stage2-results.json`](../tools/lidar_stage2/stage2-results.json).
'''
    (REPO/'docs/lidar-stage2-report.md').write_text(text)
    print(json.dumps({'stage':2,'status':'complete','train':audit['samples']['train'],'val':audit['samples']['val'],'tests':int(match[1])},indent=2))


if __name__=='__main__': main()
