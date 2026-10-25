"""Publish readiness only after every concrete pre-training check has passed."""
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from simlingo_training.lidar.storage import digest_file,atomic_json
from simlingo_training.lidar.validation_run import identity

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/ehsan/m.sadegh/simlingo_lidar')
OUT=ROOT/'readiness'
PILOT=ROOT/'stage2/pilot-parking-v2'


def read(path): return json.loads(Path(path).read_text())


def main():
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!='feat/simlingo-lidar': raise ValueError('Unexpected branch')
    ready=read(PILOT/'ready.json'); audit=read(PILOT/'audit.json'); protocol=read(PILOT/'validation.json')
    for key,name in [('manifest_sha256','manifest.json'),('validation_sha256','validation.json'),('audit_sha256','audit.json')]:
        if digest_file(PILOT/name)!=ready[key]: raise ValueError('Dataset changed')
    if not ready['passed'] or any(audit['missing_buckets'].values()): raise ValueError('Coverage incomplete')
    artifacts={name:read(OUT/name) for name in ('assets.json','integrity.json','context-audit.json','metadata.json','baseline-resume-check.json','target-populations.json',
        'resume-align/result.json','resume-joint/result.json','resume-joint/global-stress.json','baseline/result.json')}
    for name,d in artifacts.items():
        if name!='baseline/result.json' and not d['passed']: raise ValueError('Incomplete check: '+name)
        if 'manifest_sha256' in d and d['manifest_sha256']!=ready['manifest_sha256']: raise ValueError('Manifest differs: '+name)
    baseline=artifacts['baseline/result.json']
    if not baseline['complete_protocol'] or not baseline['coverage_gate']['passed'] or baseline['samples']!=7680 or baseline['optimizer_updates']!=0:
        raise ValueError('Complete released baseline/coverage required')
    with (OUT/'baseline/records.jsonl').open() as f:
        binding=json.loads(next(f))['binding']; records=[json.loads(line) for line in f]
    expected={identity(r) for task,items in protocol['requests'].items() if task!='geometry_qa' for r in items}
    if len(records)!=len(expected) or {identity(r) for r in records}!=expected: raise ValueError('Full validation identities differ')
    if digest_file(OUT/'baseline/records.jsonl')!=baseline['journal_sha256']: raise ValueError('Baseline journal changed')
    for name,expected_population in artifacts['target-populations.json']['populations'].items():
        if any(baseline['metrics'][name][k]!=expected_population[k] for k in ('samples','groups','population_sha256')):
            raise ValueError('Full scorer differs from independent target audit: '+name)
    for name,sha in binding['code'].items():
        if digest_file(REPO/name)!=sha: raise ValueError('Current training source differs: '+name)
    shared=('manifest_sha256','validation_sha256','code','external_sources','versions')
    for name in ('resume-align/result.json','resume-joint/result.json'):
        d=artifacts[name]
        if not all(d['checks'].values()) or d['reference_pid']==d['resume_pid']: raise ValueError('Fresh-process proof incomplete')
        for key in shared:
            if d['metadata'][key]!=binding[key]: raise ValueError('Recovery/baseline provenance differs: '+key)
    sparse={stage:read(OUT/f'sparse-kernel-{stage}.json') for stage in ('alignment','joint')}
    for stage,snapshot in sparse.items():
        for rows in snapshot['sparse_kernels']['tables'].values():
            if any(row['splitk']!=1 or row['arch']!=[8,9] for row in rows): raise ValueError('Sparse kernel policy differs')
    if sparse['alignment']['source_checkpoint_sha256']!=artifacts['resume-joint/result.json']['metadata']['experiment_config']['run']['initial_weights_sha256']:
        raise ValueError('Joint smoke did not initialize from the checked aligned weights')
    if list(OUT.glob('resume-*/*.pt')) or list(OUT.glob('resume-*/*.part')): raise ValueError('Disposable recovery weights remain')
    unit=(OUT/'logs/unit-tests.log').read_text(); match=re.search(r'Ran (\d+) tests',unit)
    if not match or int(match[1])<37 or '\nOK\n' not in unit or 'FAILED' in unit: raise ValueError('Unit tests incomplete')
    stage3=read(REPO/'tools/lidar_stage3/stage3-results.json')
    for name,sha in stage3['artifacts_sha256'].items():
        if digest_file(Path(stage3['run'])/name)!=sha: raise ValueError('Historical Stage 3 evidence changed: '+name)
    for name in ('align-proposed.json','joint-one-epoch-proposed.json','joint-ceiling-proposed.json'):
        if read(REPO/'tools/lidar_readiness/configs'/name)['budget_reviewed'] is not False: raise ValueError('Training review was marked without review')
    if (ROOT/'stage4').exists():
        if list((ROOT/'stage4').rglob('*.pt')): raise ValueError('Main training weights unexpectedly exist')
    prepared=sum(p.stat().st_size for p in (ROOT/'stage2').rglob('*.tar*') if p.is_file())
    if prepared>100*1024**3 or shutil.disk_usage(OUT).free<150*1024**3:
        raise ValueError('Preparation cap or shared free-space reserve exceeded')
    # Adjust the Stage 3 training rates conservatively using two unique real
    # windows in the new pilot, including loading/observer overhead. This is a
    # short shared-load observation, not a replacement full-mixture profile.
    joint=artifacts['resume-joint/result.json']; align=artifacts['resume-align/result.json']
    nominal_joint=0.
    for r in joint['reference_initial_update']['requests']:
        mode='rgb' if r['replay'] else 'paired'
        nominal_joint+=stage3['task_rates'][r['task']+'/'+mode]['mean_compute_seconds']
    nominal_joint+=64*(stage3['joint_io_seconds_per_presentation']+stage3['joint_other_seconds_per_presentation'])
    joint_scale=max(1.,joint['reference_initial_update']['seconds']/nominal_joint)
    align_scale=max(1.,align['reference_initial_update']['seconds']/(64*stage3['alignment_seconds_per_presentation']))
    seconds_a=stage3['alignment_seconds_per_presentation']*align_scale
    seconds_b=stage3['joint_seconds_per_presentation']*joint_scale
    rgb_seconds=baseline['sum_example_seconds']
    val_scale=max(1.,rgb_seconds/stage3['validation_extrapolation']['rgb']['full_seconds'])
    paired_seconds=stage3['validation_extrapolation']['paired']['full_seconds']*val_scale
    monitor_rgb_keys={identity(r) for t,items in protocol['monitor'].items() if t!='geometry_qa' for r in items}
    monitor_rgb=sum(r['total_seconds'] for r in records if identity(r) in monitor_rgb_keys)
    monitor_pair=stage3['validation_extrapolation']['paired']['monitor_seconds']*val_scale
    epoch_updates=math.ceil(audit['samples']['train']/64)
    def budget(stage,updates,full,monitor,interval):
        rate=seconds_a if stage=='align' else seconds_b
        save=align['save_seconds'] if stage=='align' else joint['save_seconds']
        # Fixed intervals plus the initial checkpoint. Extra saves at full
        # validations are bounded explicitly rather than charged twice.
        saves=1+math.ceil(updates/interval)+full
        training=updates*64*rate
        overhead=saves*save+full*(paired_seconds+rgb_seconds)+monitor*(monitor_pair+monitor_rgb)+120
        hours=(training+overhead)/3600
        return {'updates':updates,'presentations':updates*64,'train_hours':training/3600,
            'full_validations_paired_and_rgb':full,'monitors_paired_and_rgb':monitor,'checkpoint_count_upper':saves,
            'checkpoint_interval_updates':interval,'checkpoint_save_seconds':save,'central_hours':hours,
            'planning_low_hours':.8*hours,'planning_high_hours':1.75*hours}
    budgets={'alignment':budget('align',750,1,2,15),
             'joint_one_sampled_epoch':budget('joint',epoch_updates,2,2,12),
             'joint_240k_ceiling':budget('joint',3750,4,4,12),
             'conditional_100k_epoch':budget('joint',math.ceil(100000/64),2,3,12)}
    report={'status':'technical_checks_complete_budget_review_pending','technical_checks_passed':True,
        'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'branch':branch,'manifest_sha256':ready['manifest_sha256'],
        'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'report_generator_sha256':digest_file(__file__),
        'dataset_ready':ready,'unit_tests':int(match[1]),'budget_reviewed':False,'main_training_started':False,
        'numerical_policy':binding['experiment_config']['numerics'],
        'physical_gpu_uuid':stage3['physical_gpu_uuid'],'dataset':audit,'baseline_metrics':baseline['metrics'],
        'hardware_at_report':subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=index,uuid,name,driver_version,memory.total',
            '--format=csv,noheader'],text=True).strip(),
        'baseline_structured_target_coverage':baseline['structured_target_coverage'],
        'independent_target_classes':artifacts['target-populations.json']['target_classes'],
        'baseline_parking_metrics':baseline['by_scenario_bucket']['parkinglane'],
        'baseline_full_seconds':rgb_seconds,'paired_full_forecast_seconds':paired_seconds,
        'recovery':{stage:artifacts['resume-'+stage+'/result.json'] for stage in ('align','joint')},
        'global_stress':artifacts['resume-joint/global-stress.json'],'context_audit':artifacts['context-audit.json'],
        'sparse_kernel_snapshots':sparse,
        'budgets':budgets,'time_calibration':{'alignment_scale':align_scale,'joint_scale':joint_scale,
            'validation_scale':val_scale,'training_rates_source':'Complete Stage 3 mixture, conservatively adjusted by a new unique accumulation-64 window per stage',
            'validation_source':'Complete new RGB baseline; paired forecast still extrapolates the Stage 3 timing subset'},
        'artifact_hashes':{name:digest_file(OUT/name) for name in artifacts},
        'unit_log_sha256':digest_file(OUT/'logs/unit-tests.log'),'free_shared_bytes':shutil.disk_usage(OUT).free,'prepared_stage2_bytes':prepared,
        'limits':['No alignment or main joint training launched; bounded disposable updates only',
            'Fresh interpreters simulate restart on the same server; no physical reboot was performed',
            'Paired full validation and RGB retention of a trained candidate run after alignment, not before it',
            'Timing range is a heuristic under shared GPU/storage load; future generation lengths can change costs',
            'CARLA sensor/runtime parity, closed-loop improvement, broader training and merging remain later reviewed stages'],
        'recommendation':'Review alignment 750 updates first. Review a single sampled joint epoch after its report; 240k is a ceiling, not the initial default.'}
    failed=OUT/'resume-align-before-sparse-fix/result.json'
    if failed.exists():
        original=read(failed)
        report['prior_fresh_process_failure']={'checks':original['checks'],'artifact_sha256':digest_file(failed),
            'fix':'Checkpoint spconv Python tuner tables, FP32/split-K=1 and prime kernels before checkpoint zero; initialize lazy imgaug RNG before seeding training',
            'accepted':False}
    atomic_json(OUT/'result.json',report)
    atomic_json(REPO/'tools/lidar_readiness/readiness-results.json',report)
    text=f'''# Pre-training readiness

Technical checks before Stage 4 are complete on `{branch}`. The actual training
budget still needs user review. No alignment or main joint run was launched.
Only disposable bounded checks used optimizer updates. The proposals remain
`budget_reviewed: false`; the training CLI rejects an unreviewed configuration.

The versioned pilot contains **{audit['samples']['train']:,} training frames** in
{audit['groups']['train']} source groups and **{audit['samples']['val']:,} Town13
validation frames** in {audit['groups']['val']} groups. All original pilot rows
remain. Parking adds Town12 training and Town13 validation source groups without
changing the old artifacts. The original grouping conservatively merges repeated
source IDs; 32 selected parking training groups add 31 previously unseen group IDs.
Parking bucket counts are {audit['bucket_counts']['driving']['parkinglane']:,} driving,
{audit['bucket_counts']['qa']['parkinglane']:,} QA and
{audit['bucket_counts']['commentary']['parkinglane']:,} commentary. No requested
sampling bucket is missing. Benchmark containment still excludes the same whole
Town04 source group; this does not establish complete benchmark independence or
undo exposure of the released checkpoint.

Verification passed:

- All prepared shard hashes and pinned source relationships; pretrained release,
  Utonia weights/source/installed code, and Hugging Face snapshot assets.
- {int(match[1])} tests, including failure before an optimizer boundary, update-zero
  checkpoint restoration, interrupted evaluation journals and budget guards.
- Independent uncached/cached metadata and complete sensor/target fingerprints on
  parking examples in both splits, covering every family and RGB replay.
- Exhaustive finite text-variant audit and runtime 2,048-token guards that reserve
  the 128-token prefix, up to 100 generated tokens and 30 driving queries.
- Full **7,680-request released RGB baseline** on the new immutable protocol,
  with generated text, route-group metrics, scenario buckets and parse coverage.
  No subset substitutes for the baseline. All 8,192 paired requests remain fixed
  for the later trained checkpoint validations.
- Fresh-process alignment and joint recovery using the production construction,
  update and boundary code: every model/Adam tensor, sampler, scheduler, input and
  target fingerprint, generated token, loss and Python/NumPy/Torch/CUDA RNG matches
  exactly. This simulates a restart without rebooting the shared server.
- Maximum training-cloud and all-split text stress with resident joint Adam state, retained teacher
  graph, exactly 100 uncached generated tokens and driving backward. This combined
  extreme is a resource test, not a training example or learning result.

The released RGB baseline, averaged within source groups and then across groups:

| Metric | Result | Scored requests |
|---|---:|---:|
| Route ADE | {baseline['metrics']['route_ade']['value']:.5f} m | {baseline['metrics']['route_ade']['samples']:,} |
| Speed-waypoint ADE | {baseline['metrics']['speed_ade']['value']:.5f} m | {baseline['metrics']['speed_ade']['samples']:,} |
| QA exact match | {baseline['metrics']['qa_exact_match']['value']:.2f}% | 2,048 |
| Structured QA accuracy | {baseline['metrics']['qa_structured_accuracy']['value']:.2f}% | 485 |
| Dreamer decision accuracy | {baseline['metrics']['dreamer_decision_accuracy']['value']:.2f}% | 501 |
| Commentary ROUGE-L | {baseline['metrics']['commentary_rouge_l']['value']:.2f} / 100 | 512 |

Both driving metrics measure waypoint position errors in metres. Structured QA
covers 485/2,048 targets (23.68%), and Dreamer covers 501/512 (97.85%); both span
all 59 groups. The independent annotation-only audit matches these exact scoring
populations. Unknown targets are excluded explicitly, while an unparseable
prediction for a known target is wrong. Parking-maneuver results include 155
driving requests across 16 groups; its commentary/Dreamer subgroups are small and
are descriptive only. Exact counts and generated examples remain in the artifacts.
These are engineering gates, not evidence of driving safety or improved rear
awareness. Complete paired evaluation and retention of a trained candidate are
performed after alignment.

The accepted GPU checks peak at {max(align['peak_reserved_bytes'],joint['peak_reserved_bytes'],artifacts['resume-joint/global-stress.json']['peak_reserved_bytes'])/1024**3:.2f} GiB reserved,
below the 40 GiB process limit. Exhaustive text maxima are 643 teacher tokens and
596 question tokens; reserving prefix, generation and driving queries needs at
most 854 tokens against the 2,048-token limit. The synthetic backward stress uses
the largest training cloud (31,361 voxels), resident joint Adam state and 100
uncached generated tokens. Validation's largest cloud is 31,516 voxels and uses
no-grad evaluation; all eligible scans were audited on CPU.

| Proposed next run | Presentations / updates | Central estimate | Planning range |
|---|---:|---:|---:|
'''
    for label,key in [('Alignment','alignment'),('One sampled joint epoch','joint_one_sampled_epoch'),('Older joint ceiling','joint_240k_ceiling'),('Conditional later 100k-frame epoch','conditional_100k_epoch')]:
        b=budgets[key]; text+=f"| {label} | {b['presentations']:,} / {b['updates']:,} | {b['central_hours']:.1f} h | {b['planning_low_hours']:.1f}–{b['planning_high_hours']:.1f} h |\n"
    text+=f'''
These include initial/periodic checkpoints, full paired+RGB validations and fixed
monitors. The initial released baseline is already complete and is not charged
again. Training uses the Stage 3 full-mixture rates, conservatively adjusted with
new-pilot bounded windows; actual full RGB validation took {rgb_seconds/3600:.2f} h.
Paired timing is still extrapolated. The range is a 0.8–1.75× planning band,
not a statistical interval or guaranteed completion time under shared load.
The 240k ceiling is {3750/epoch_updates:.2f} sampled epochs of this pilot.
The 100k-frame scenario requires preparing and auditing more eligible data first;
it is not the size of the entire original dataset, which remains unestablished.

Recommend reviewing the 750-update alignment run first, then its full report
before deciding the joint run. The concrete configurations are in
[`tools/lidar_readiness/configs`](../tools/lidar_readiness/configs/).
GPU 2 is shared; check current free memory before launch and notify the user.
KV caching stays disabled. Keep microbatch 1 / accumulation 64, preserve the
trained LoRA and image pipeline, and use the corrected numerical policy below.

The production CLI writes an initial checkpoint before any update, retains only
last/recovery/best plus an atomic write, and commits sampler position only after
successful optimizer windows. A failed window must reload its last completed
boundary. Resume binds data, code, pretrained assets, numerical policy, package
versions and the complete reviewed configuration. Validation journals bind their
exact checkpoint and request set. Joint initialization binds the selected aligned
weights and starts a fresh optimizer/schedule. tmux survives logout; checkpoints
and journals provide reboot recovery.

The first fresh-process attempt failed despite identical inputs and loaded
weights. Its logs and comparisons remain. spconv's process-local tuning choices
were not in the old checkpoint; lazy imgaug initialization also consumed NumPy
RNG only in the restarted process. The current policy exposes the same sparse
kernels through Python dispatch, primes stage-specific forward/backward kernels
before checkpoint zero, saves/restores their tuning tables, uses FP32 and
split-K=1, and initializes augmentation's global generator before training seeds
are set. Priming performs no optimizer update and restores buffers/RNG/gradients.
The accepted checks are fresh runs with this policy, without relaxed tolerances.

Earlier baseline startup/partial artifacts were superseded while finalizing the
training entry point; they remain for audit and are not used in these results.
All disposable recovery weights were removed after successful comparisons.
CARLA integration, sensor-fault behavior, temporal LiDAR, closed-loop benchmarking,
broader training and merging remain separate later stages.

Artifacts: `{OUT}`. Reproduction:
[`readiness README`](../tools/lidar_readiness/README.md). Exact metrics, coverage,
hashes, recovery comparisons, resource bounds and budget inputs:
[`readiness-results.json`](../tools/lidar_readiness/readiness-results.json).
'''
    (REPO/'docs/lidar-readiness-report.md').write_text(text)
    print(json.dumps({'technical_checks_passed':True,'budget_reviewed':False,'budgets':budgets},indent=2))


if __name__=='__main__': main()
