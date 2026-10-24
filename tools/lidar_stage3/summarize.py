"""Turn completed bounded measurements into explicit, provisional time budgets."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

from simlingo_training.lidar.data import TASKS, ALIGN_COUNTS, JOINT_COUNTS
from simlingo_training.lidar.storage import atomic_json, digest_file

REPO=Path(__file__).resolve().parents[2]


def read(path): return json.loads(Path(path).read_text())


def samples(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]


def hours(seconds):
    return {'central':seconds/3600,'planning_low':seconds*.8/3600,'planning_high':seconds*1.75/3600}


def summarize(run):
    run=Path(run); raw=read(run/'progress.json')
    assert raw['status']=='complete' and raw['long_training_started'] is False
    correction=read(run/'scoring-correction.json')
    assert correction['timed_scoring_source_sha256']==raw['provenance']['code']['simlingo_training/lidar/evaluation.py']
    assert correction['corrected_scoring_source_sha256']==digest_file(REPO/'simlingo_training/lidar/evaluation.py')
    assert all(not result['metrics_changed'] for result in correction['outputs'].values())
    for name,expected in raw['provenance']['code'].items():
        if name!='simlingo_training/lidar/evaluation.py': assert digest_file(REPO/name)==expected,name
    assert raw['alignment']['presentations']==64 and len(raw['joint_windows'])==3
    assert all(w['presentations']==64 for w in raw['joint_windows'])
    assert all(raw['checkpoint'][k] for k in ('loaded_model_exact','loaded_optimizer_exact','scheduler_equal','sampler_equal'))
    assert all(raw['checkpoint'][k]['within_tolerance'] for k in ('resumed_next_update_model','resumed_next_update_optimizer'))
    assert not list(run.glob('*.pt*'))
    unit_log=(run/'unit-tests.log').read_text()
    match=re.search(r'Ran (\d+) tests',unit_log)
    assert match and int(match[1])>=26 and '\nOK\n' in unit_log and 'FAILED' not in unit_log
    assert raw['numerics']['torch_deterministic'] and raw['numerics']['flash_backward_deterministic']
    align=samples(run/'align-samples.jsonl'); joint=samples(run/'joint-samples.jsonl'); resumed=samples(run/'resume-samples.jsonl')
    assert len(align)==64 and len(joint)==192 and len(resumed)==64
    peaks=[raw['alignment'],*raw['joint_windows'],raw['checkpoint']['resumed_window'],*raw['stress']]
    assert max(r['peak_reserved_bytes'] for r in peaks)<40*1024**3
    groups=defaultdict(list)
    for row in joint: groups[(row['task'],row['replay'])].append(row)
    task_rates={}; weighted_compute=0.
    for task,count in zip(TASKS,JOINT_COUNTS):
        for replay,fraction in ((False,.75),(True,.25)) if task in TASKS[:4] else ((False,1.),):
            rows=groups[(task,replay)]
            assert rows,f'Missing profile family/mode: {task}/{replay}'
            seconds=[r['forward_seconds']+r['backward_seconds'] for r in rows]
            mean=float(np.mean(seconds)); weight=count/sum(JOINT_COUNTS)*fraction
            weighted_compute+=weight*mean
            task_rates[task+('/rgb' if replay else '/paired')]={'samples':len(rows),'mixture_weight':weight,
                'mean_compute_seconds':mean,'median_compute_seconds':float(np.median(seconds)),
                'p95_compute_seconds':float(np.percentile(seconds,95)),
                'mean_generated_tokens':float(np.mean([r.get('generated_tokens',0) for r in rows])),
                'hit_100_token_cap':sum(r.get('generated_tokens',0)==100 for r in rows)}
    observed_compute=sum(w['seconds'] for w in raw['joint_windows'])
    sample_compute=sum(r['forward_seconds']+r['backward_seconds'] for r in joint)
    overhead=max(0.,observed_compute-sample_compute)/192
    joint_io=sum(w['io_seconds'] for w in raw['joint_windows'])/192
    joint_s=weighted_compute+overhead+joint_io
    align_s=(raw['alignment']['seconds']+raw['alignment']['io_seconds'])/64
    validation={}
    protocol=read('/ehsan/m.sadegh/simlingo_lidar/stage2/pilot/validation.json')
    for mode,measurement in raw['validation'].items():
        full=sum(len(protocol['requests'][task])*m['mean_seconds'] for task,m in measurement['per_task'].items())
        monitor=sum(min(64,len(protocol['requests'][task]))*m['mean_seconds'] for task,m in measurement['per_task'].items())
        validation[mode]={'full_seconds':full,'monitor_seconds':monitor,'full_hours':hours(full),'sample_count':measurement['samples']}
    paired_rgb_full=sum(v['full_seconds'] for v in validation.values())
    paired_rgb_monitor=sum(v['monitor_seconds'] for v in validation.values())
    checkpoint_s=raw['checkpoint']['save_seconds']
    startup=raw['model_load_seconds']+raw['train_dataset']['seconds']+raw['val_dataset']['seconds']
    def budget(presentations,seconds_per_sample,full_checks,monitor_checks):
        updates=math.ceil(presentations/64); presentations=updates*64
        train_seconds=presentations*seconds_per_sample
        checkpoint_interval=min(100,max(1,math.floor(1800/(seconds_per_sample*64))))
        checkpoint_count=math.ceil(updates/checkpoint_interval)
        total=train_seconds+checkpoint_count*checkpoint_s+full_checks*paired_rgb_full+monitor_checks*paired_rgb_monitor+startup
        return {'presentations':presentations,'updates':updates,'training_only_hours':hours(train_seconds),
                'total_hours':hours(total),'full_paired_and_rgb_validations':full_checks,'monitor_paired_and_rgb_validations':monitor_checks,
                'checkpoint_interval_updates':checkpoint_interval,'checkpoint_count':checkpoint_count,
                'checkpoint_seconds':checkpoint_count*checkpoint_s,'startup_seconds':startup}
    budgets={'alignment_48k':budget(48000,align_s,1,2),
             'joint_pilot_240k':budget(240000,joint_s,4,4),
             'one_pilot_sampled_epoch_61962_frames':budget(61962,joint_s,2,2),
             'larger_epoch_100k_eligible_frames':budget(100000,joint_s,2,3)}
    report={'stage':3,'status':'complete','branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
            'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),
            'physical_gpu_uuid':raw['hardware']['gpu'].split(',')[1].strip(),
            'gpu_name':raw['hardware']['gpu'].split(',')[0].strip(),'stage2_recheck':raw['stage2_recheck'],
            'executed':{'align_updates':1,'align_presentations':64,'joint_updates_executed':4,'joint_unique_presentations':192,
                        'joint_presentations_including_resume_repeat':256,'stress_backward_cases':len(raw['stress']),
                        'validation_examples':sum(m['samples'] for m in raw['validation'].values())},
            'trainability':'Released RGB/projector, LoRA, nav/driving weights preserved; base LLM frozen; fixed agreed Utonia policy',
            'generation_cache':False,'microbatch':1,'accumulation':64,'generation_cap':100,
            'numerical_policy':raw['numerics'],
            'peak_allocated_gib':max(r['peak_allocated_bytes'] for r in peaks)/1024**3,
            'peak_reserved_gib':max(r['peak_reserved_bytes'] for r in peaks)/1024**3,
            'task_rates':task_rates,'alignment_seconds_per_presentation':align_s,'joint_seconds_per_presentation':joint_s,
            'joint_weighted_compute_seconds':weighted_compute,'joint_io_seconds_per_presentation':joint_io,
            'joint_other_seconds_per_presentation':overhead,'realized_joint_counts':dict(Counter(r['task']+('/rgb' if r['replay'] else '/paired') for r in joint)),
            'loader':raw['loader'],'checkpoint':raw['checkpoint'],'validation_extrapolation':validation,'budgets':budgets,
            'initial_released_rgb_baseline':str(run/'validation-rgb-rescored.json'),
            'scoring_correction':correction,
            'baseline_limit':'Only four fixed requests per task, before any update. Full baseline and retention validation remain required before selecting a trained checkpoint.',
            'uncertainty':'Planning range 0.8x..1.75x is a heuristic, not a confidence interval. A small subset, shared GPU/SMB load and future generation lengths can change timing.',
            'validation_cadence':'Costed proposal only: A final full validation and two monitors; B full at 1000/2000/3000/3750, monitors at 500/1500/2500/3500; baseline RGB full is an additional one-time cost.',
            'data_limit':'61,962 eligible pilot training frames. 100k is a larger-data scenario, not a claim that a full-dataset eligibility census is complete. Parking-lane bucket absent.',
            'stress_limit':'Exact maximum training-cloud voxel count plus candidate-based long contexts; synthetic combined-cloud/text and forced 100-token diagnostics are resource checks only. No silent truncation or sample drops.',
            'weights_discarded':True,'long_training_started':False,'benchmark_started':False,
            'recovery_criterion':'Exact full model/Adam reload; next accumulation-64 update within predeclared model atol=1e-6/rtol=1e-5 and Adam atol=1e-8/rtol=1e-3; scheduler, sampler and augmented samples identical. No automatic tolerance changes.',
            'prior_profiles':'run-20261024a stopped on strict next-update bitwise inequality; run-20261024b failed the declared Adam tolerance. Both discarded weights. A paired-gradient diagnostic then isolated backward variability; deterministic attention, FP32 waypoint scan and native-grid identity removed it before this fresh full profile.',
            'next':'Review measured budget and parking coverage with the user. Notify before any main training on GPU 2; do not auto-launch Stage 4.',
            'artifacts_sha256':{str(p.relative_to(run)):digest_file(p) for p in sorted(run.iterdir()) if p.suffix in ('.json','.jsonl')},
            'profile_source_sha256':digest_file(REPO/'tools/lidar_stage3/profile_pipeline.py')}
    report['unit_tests']=int(match[1])
    report['unit_tests_log_sha256']=digest_file(run/'unit-tests.log')
    earlier=read('/ehsan/m.sadegh/simlingo_lidar/stage3/run-20261024b/progress.json')
    old_update=float(np.mean([w['seconds'] for w in earlier['joint_windows'][:2]]))
    fixed_update=float(np.mean([w['seconds'] for w in raw['joint_windows'][:2]]))
    report['observed_joint_runtime_change']={'earlier_mean_update_seconds':old_update,
        'corrected_mean_update_seconds':fixed_update,'relative_change':fixed_update/old_update-1,
        'scope':'First two joint windows; shared GPU/SMB load, not a controlled performance ablation'}
    report['maximum_stress_training_voxels']=max(r['voxel_points'] for r in raw['stress'])
    report['recovery_scope']='Full-state completed-boundary reload and next update in the same GPU process; server reboot/fresh-process trajectory parity was not tested.'
    diagnostic_path=Path('/ehsan/m.sadegh/simlingo_lidar/stage3/recovery-diagnostic-final.json')
    diagnostic=read(diagnostic_path)
    assert diagnostic['True']['rng_equal'] and diagnostic['True']['generation_equal']
    assert diagnostic['True']['runs'][1]['gradient_comparison']['bitwise']
    report['paired_gradient_diagnostic']={'path':str(diagnostic_path),'sha256':digest_file(diagnostic_path),
        'original_l2_difference':diagnostic['False']['runs'][1]['gradient_l2_difference'],
        'corrected_l2_difference':diagnostic['True']['runs'][1]['gradient_l2_difference'],
        'corrected_bitwise':True}
    resources=samples(run/'resources.jsonl')
    report['resource_monitor']={'samples':len(resources),
        'peak_process_and_children_rss_gib':max(r['process_rss_bytes']+r['children_rss_bytes'] for r in resources)/1024**3,
        'minimum_host_available_gib':min(r['host_available_bytes'] for r in resources)/1024**3,
        'peak_device_total_used_mib_including_other_jobs':max(float(r['gpu'].split(',')[1].strip().split()[0]) for r in resources)}
    report['maximum_observed_teacher_tokens']=max(r.get('teacher_tokens',0) for r in align+joint+raw['stress'])
    report['maximum_observed_question_tokens']=max(r['question_tokens'] for r in align+joint+raw['stress'])
    assert report['maximum_observed_question_tokens']+128+100+30<=2048
    atomic_json(REPO/'tools/lidar_stage3/stage3-results.json',report)
    a,b=budgets['alignment_48k'],budgets['joint_pilot_240k']
    text=f'''# Stage 3 single-GPU profile

Completed on `feat/simlingo-lidar`, using only physical GPU 2, an NVIDIA RTX 6000
Ada with 46,068 MiB reported memory. Stage 2's hashes and saved checks were rechecked;
the 15 original tests plus eleven evaluation/numerics/recovery tests pass. No main training, benchmark
evaluation or merge was started. Profiling weights and temporary checkpoints were
discarded. The run was bounded to 45 minutes in tmux.

One alignment update processed 64 paired presentations. Joint profiling executed
four accumulation-64 updates: 192 unique presentations and a repeated 64-sample
next-update recovery check. All six task families and original-family RGB replay
were measured with 100-token greedy caps and KV caching disabled. Joint rates
below are reweighted to the agreed 960-presentation task/replay cycle; the shorter
profile itself is not an exact 960-sample cycle.

| Measurement | Result |
|---|---:|
| Peak allocated GPU memory, including stress | {report['peak_allocated_gib']:.2f} GiB |
| Peak reserved GPU memory | {report['peak_reserved_gib']:.2f} GiB |
| Alignment, including measured I/O | {align_s:.3f} s/presentation |
| Joint, weighted mixture including I/O/step | {joint_s:.3f} s/presentation |
| Full model/Adam checkpoint size | {raw['checkpoint']['bytes']/1024**3:.2f} GiB |
| Atomic checkpoint save / load | {checkpoint_s:.1f} / {raw['checkpoint']['load_seconds']:.1f} s |

The full training/validation metadata took
{(raw['train_dataset']['seconds']+raw['val_dataset']['seconds'])/60:.1f} minutes
to read from shared storage. Loader measurements cover the same 64 requests:

| Spawn workers / prefetch | Wall time including startup | Median read wait | Summed process RSS peak |
|---|---:|---:|---:|
| 0 | {raw['loader'][0]['seconds']:.1f} s | {raw['loader'][0]['wait_median_seconds']:.3f} s | {raw['loader'][0]['combined_rss_peak_bytes']/1024**3:.1f} GiB |
| 2 / 1 per worker | {raw['loader'][1]['seconds']:.1f} s | {raw['loader'][1]['wait_median_seconds']:.3f} s | {raw['loader'][1]['combined_rss_peak_bytes']/1024**3:.1f} GiB |

RGB/prompt tensors matched across these requests. The full-update timings use
unoverlapped CPU reads with zero workers; benefits from two workers are not
credited in the budget. Host memory available during monitoring stayed above
{report['resource_monitor']['minimum_host_available_gib']:.1f} GiB, including the
temporary full-state recovery comparisons. Keep microbatch 1 / accumulation 64;
microbatch 2 has not been validated.

The profile used the original released RGB/projector, trained LoRA, navigation and
driving weights. The base LLM remained frozen. Alignment preserved every frozen
parameter. Joint Adam moments stayed FP32. Reloading a completed joint checkpoint
restored every model and Adam-state tensor exactly and reproduced all augmented
samples, scheduler and sampler. The full next accumulation-64 update passed the
predeclared numerical comparisons: model maximum absolute difference
{raw['checkpoint']['resumed_next_update_model']['max_absolute_error']:.3g}; Adam
maximum absolute difference
{raw['checkpoint']['resumed_next_update_optimizer']['max_absolute_error']:.3g}.
Tolerances were model atol=1e-6/rtol=1e-5 and Adam atol=1e-8/rtol=1e-3, checked
against every tensor. Bitwise next-update equality was
`{raw['checkpoint']['resumed_next_update_model_bitwise']}` for model tensors and
`{raw['checkpoint']['resumed_next_update_optimizer_bitwise']}` for Adam tensors.
This was a same-process reload. A fresh-process/reboot trajectory comparison was
not performed; the actual training entry point needs its own restart smoke check
before a long run.
Two earlier profiles stopped at failed recovery assertions; logs remain in
`run-20261024a` and `run-20261024b`. The second failed the declared Adam tolerance;
it was not accepted. A focused diagnostic found equal inputs, forward losses,
generated text and RNG states but differing backward gradients. The final policy
enables deterministic FlashAttention backward in Qwen, vision and Utonia, uses an
FP32 triangular scan for waypoint accumulation, and skips vision position
interpolation when the pretrained/input grids are identical. Other vision grid
sizes fail explicitly. The paired-gradient diagnostic then matched bit for bit,
and this fresh full run measures that corrected policy. Pretrained parameters
and architecture are unchanged; cumulative waypoint arithmetic now stays FP32.
Source/config/data hashes, numerical policy and RNG states are bound to
checkpoints. Checkpoint I/O was measured on `/ehsan`.

The first two corrected joint updates averaged {fixed_update:.1f} seconds versus
{old_update:.1f} seconds earlier (about {(fixed_update/old_update-1)*100:.0f}% slower).
This is an observed comparison under shared load, not a controlled ablation. The
budget below uses the corrected pipeline exclusively.

| Proposed budget (not launched) | Training only | Including proposed checks/I/O | Planning range, total |
|---|---:|---:|---:|
| Alignment: 48k / 750 updates | {a['training_only_hours']['central']:.1f} h | {a['total_hours']['central']:.1f} h | {a['total_hours']['planning_low']:.1f}–{a['total_hours']['planning_high']:.1f} h |
| Joint pilot: 240k / 3750 updates | {b['training_only_hours']['central']:.1f} h | {b['total_hours']['central']:.1f} h | {b['total_hours']['planning_low']:.1f}–{b['total_hours']['planning_high']:.1f} h |
| One sampled pilot epoch: 62,016 presentations | {budgets['one_pilot_sampled_epoch_61962_frames']['training_only_hours']['central']:.1f} h | {budgets['one_pilot_sampled_epoch_61962_frames']['total_hours']['central']:.1f} h | {budgets['one_pilot_sampled_epoch_61962_frames']['total_hours']['planning_low']:.1f}–{budgets['one_pilot_sampled_epoch_61962_frames']['total_hours']['planning_high']:.1f} h |
| Larger epoch scenario: 100,032 presentations | {budgets['larger_epoch_100k_eligible_frames']['training_only_hours']['central']:.1f} h | {budgets['larger_epoch_100k_eligible_frames']['total_hours']['central']:.1f} h | {budgets['larger_epoch_100k_eligible_frames']['total_hours']['planning_low']:.1f}–{budgets['larger_epoch_100k_eligible_frames']['total_hours']['planning_high']:.1f} h |

The total estimates include dataset/model startup, checkpoints at the earlier of
100 updates or about 30 minutes, and explicit validation counts. The proposal costs one final
full paired+RGB validation and two monitors for alignment; four full checks and
four monitors for the joint pilot. Full validation was extrapolated from four
fixed requests per task in each applicable mode ({report['executed']['validation_examples']}
examples total), not executed over all 8,192 paired and 7,680 RGB requests.
Estimated full paired validation is {validation['paired']['full_seconds']/60:.1f} minutes;
full RGB validation is {validation['rgb']['full_seconds']/60:.1f} minutes.
A complete released RGB baseline adds another {validation['rgb']['full_seconds']/60:.1f}
minutes once, before alignment. Generation lengths can change after training.
The 240k joint ceiling corresponds to approximately {240000/62016:.1f} sampled
epochs of this pilot; it remains a budget to review, not a required initial run.
The 0.8–1.75× planning band is a heuristic; another process was using GPU 2 and
SMB throughput varies. It is not a statistical confidence interval.

The evaluator computes generated-condition speed/route ADE, word-token ROUGE-L,
conservatively parsed QA/Dreamer decisions with explicit coverage, and per-sector
geometry accuracy. It averages within source route groups before averaging groups,
stores generated examples, and requires identical metric populations for retention
comparisons. The released RGB measurements here are a bounded baseline fixture;
they cannot establish retention or improved driving. Complete fixed validation
and sufficient parseable-target coverage remain required for checkpoint selection.
Real-sample checks exposed the decoded `assistant` role header in references.
Reference normalization was corrected before the final profile. Its saved
generations were re-scored as a CPU-only verification; the metrics were unchanged.
Original timing/scoring artifacts are preserved. `scoring-correction.json` records
the scoring-source hashes and outputs; earlier incorrect scoring from
`run-20261024a` is not used in this report. All learning/model/data source files
remained unchanged during the final profile.

Resource stress included the maximum audited training cloud
({report['maximum_stress_training_voxels']:,} voxels), candidate-based long
contexts, paired/replay tasks, and 100 uncached generated tokens with the teacher
graph retained before driving/backward. Synthetic combined-cloud/text cases are
only resource checks. Global maximum text length and a complete eligible count
for the original dataset have not been established; the 100k epoch is conditional.
The pilot still lacks parking-lane coverage, and KV caching remains disabled.
Do not silently truncate contexts, drop OOM samples, or treat these checks as
live-agent parity or safety evidence.

Measurements and generated examples: `{run}`.
Machine-readable task rates, resource readings, provenance and budget formula
inputs: [`stage3-results.json`](../tools/lidar_stage3/stage3-results.json).
Reproduction: [`Stage 3 README`](../tools/lidar_stage3/README.md).

**Next:** review the actual budget and parking coverage with the user before
Stage 4. Notify the user before any main training on GPU 2. Nothing auto-launches.
'''
    (REPO/'docs/lidar-stage3-report.md').write_text(text)
    print(json.dumps({'status':'complete','memory_gib':report['peak_reserved_gib'],'budgets':budgets},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path)
    summarize(parser.parse_args().run)
