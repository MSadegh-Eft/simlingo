"""Local artifacts explaining measured trends and proposed next experiments."""
import csv
import html
import json
from pathlib import Path

from simlingo_training.lidar.control import driving_score
from simlingo_training.lidar.storage import atomic_json


def read(path): return json.loads(Path(path).read_text())


def recommendations(stage,history,train_history,stop_reason,paired_vs_rgb,weak_buckets,val_language_history=None):
    proposals=[]
    if len(history)<3:
        return [{'reason':'Fewer than three monitoring observations','action':'Collect more validation evidence before changing parameters'}]
    first,last=history[-3],history[-1]
    improving=any(last['signals'][k]<first['signals'][k]*(1-(.005 if k=='score' else .01)) for k in first['signals'])
    if stop_reason=='budget_exhausted' and improving and last['retention_passed']:
        proposals.append({'reason':'Held-out metrics are still improving at the budget ceiling',
            'action':'Propose another 16,000 alignment presentations' if stage=='align' else 'Propose one additional sampled joint epoch',
            'requires':'New reviewed configuration, explicit continuation schedule and permission'})
    train_improved=(len(train_history)>=3 and train_history[-1] is not None and train_history[-3] is not None
                    and train_history[-1]<train_history[-3]*.99)
    val_worse=(val_language_history[-1]>val_language_history[-3]*1.01 if val_language_history is not None
               else last['signals']['language']>first['signals']['language']*1.01)
    if train_improved and val_worse:
        proposals.append({'reason':'Training language loss improves while validation language loss rises',
            'action':'Review the earlier checkpoint and overfitting; prioritize additional independent route groups'})
    if not last['retention_passed']:
        proposals.append({'reason':'Provisional RGB retention screen fails',
            'action':'Inspect full RGB retention; propose halving pretrained-component learning rates and increasing RGB replay from 20% to 30%'})
    if not improving and not train_improved:
        proposals.append({'reason':'Recent training/validation signals show limited progress',
            'action':'Inspect gradients, labels, task losses and LiDAR contribution before increasing training duration'})
    if paired_vs_rgb is not None and paired_vs_rgb<.005:
        proposals.append({'reason':'Paired driving benefit over same-checkpoint RGB-only is below 0.5% on this monitor',
            'action':'Inspect geometry learning and sector/scenario results; do not assume additional epochs fix modality use'})
    if weak_buckets:
        proposals.append({'reason':'Scenarios with relatively weak validation and limited observed training coverage',
            'scenarios':weak_buckets,'action':'Audit and prepare additional route groups, initially targeting a 100k-frame pool'})
    if not proposals: proposals.append({'reason':'No single change is supported by the current rules','action':'Review full metrics and generated examples before the next experiment'})
    return proposals


def svg_curves(history,train_history,val_language_history=None):
    """Standalone learning curves rendered with the installed plotting stack."""
    import io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    updates=[r['update'] for r in history]
    panels=[('Normalized driving error',[r['signals']['score'] for r in history]),
            ('Validation language CE (stopping signal)',[r['signals']['language'] for r in history]),
            ('Validation geometry loss',[r['signals']['geometry'] for r in history]),
            ('Matched task-average language CE',train_history)]
    figure,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
    try:
        for i,(axis,(name,values)) in enumerate(zip(axes.flat,panels)):
            axis.plot(updates,values,marker='o',markersize=3,label='Training diagnostic' if i==3 else 'Validation')
            if i==3 and val_language_history is not None:
                axis.plot(updates,val_language_history,marker='o',markersize=3,label='Validation diagnostic')
            axis.set_title(name,fontsize=10); axis.set_xlabel('Optimizer update')
            axis.grid(alpha=.2); axis.legend(fontsize=8)
        stream=io.StringIO(); figure.savefig(stream,format='svg')
        return stream.getvalue()
    finally: plt.close(figure)


def write_report(output,config,controller,tracking,selection,baseline,monitor):
    output=Path(output); history=controller['history']; train_history=[]; val_language=[]
    for row in history:
        train=read(output/f'monitor-{row["update"]}/train/result.json')['diagnostics']['by_task']
        val=read(output/f'monitor-{row["update"]}/paired/result.json')['diagnostics']['by_task']
        # Equal task weights on both sides avoid mistaking different task mixes
        # or answer lengths for a generalization gap.
        tasks=[t for t in train if train[t].get('language',{}).get('count') and val.get(t,{}).get('language',{}).get('count')]
        if not tasks: raise ValueError('No matched training/validation language diagnostic tasks')
        train_history.append(sum(train[t]['language']['value'] for t in tasks)/len(tasks))
        val_language.append(sum(val[t]['language']['value'] for t in tasks)/len(tasks))
    update=history[-1]['update']; paired=read(output/f'monitor-{update}/paired/result.json'); rgb=read(output/f'monitor-{update}/rgb/result.json')
    rgb_score=driving_score(rgb['metrics'],monitor['baseline_metrics'])
    gain=(rgb_score-history[-1]['signals']['score'])/max(1e-12,rgb_score)
    scenario_counts=tracking['bucket_counts']; candidates=[]
    for name,report in paired['by_scenario_bucket'].items():
        error=report['metrics'].get('route_ade',{})
        if error.get('samples',0)>=10 and name in scenario_counts:
            candidates.append((error['value'],name,scenario_counts[name]))
    # Descriptive ranking, with explicit support; this is not proof that more data helps.
    median=sorted(scenario_counts.values())[len(scenario_counts)//2] if scenario_counts else 0
    weak=[{'bucket':n,'route_ade':e,'training_presentations':c} for e,n,c in sorted(candidates,reverse=True)[:3] if c<=median]
    advice=recommendations(config['stage'],history,train_history,controller['stop_reason'],gain,weak,val_language)
    examples=[]
    journal=output/f'monitor-{update}/paired/records.jsonl'
    with journal.open() as handle:
        for line in handle:
            row=json.loads(line)
            if 'binding' not in row and row.get('generated'):
                examples.append({k:row[k] for k in ('uid','group','task','generated','reference','metrics')})
            if len(examples)>=12: break
    report={'status':'awaiting_review','stage':config['stage'],'stop_reason':controller['stop_reason'],
            'settings':config,'selection':selection,'history':history,'training_language_loss':train_history,
            'matched_validation_language_loss':val_language,'language_gap_rule':'Equal weights for the same eligible task families; token-normalized within each family',
            'unique_training_frames':len(tracking['seen_indices']),'presentations':update*64,
            'task_counts':tracking['task_counts'],'replay_count':tracking['replay_count'],
            'bucket_counts':scenario_counts,'latest_monitor_paired_vs_rgb_relative_gain':gain,
            'recommendations':advice,'examples':examples,
            'limits':'Engineering pilot and fixed held-out evaluation; thresholds are heuristics. No setting changes or further training are automatic. Bench2Drive is separate.'}
    atomic_json(output/'report.json',report)
    with (output/'learning-curves.csv').open('w',newline='') as handle:
        writer=csv.writer(handle); writer.writerow(['update','presentations','driving_score','val_language','val_geometry','train_language','rgb_screen_passed','patience'])
        for row,loss in zip(history,train_history): writer.writerow([row['update'],row['update']*64,row['signals']['score'],row['signals']['language'],row['signals']['geometry'],loss,row['retention_passed'],row['wait']])
    (output/'learning-curves.svg').write_text(svg_curves(history,train_history,val_language))
    text=['# Training review',f'Stage: {config["stage"]}; stopped because: {controller["stop_reason"]}.',
          f'{update*64:,} presentations; {len(tracking["seen_indices"]):,} unique training frames.',
          f'Full-validation retention-passing candidate: {selection["selected"] is not None}.',
          '## Proposed next actions']
    for item in advice: text.append(f'- {item["reason"]}: {item["action"]}.')
    text+=['','See learning-curves.svg, learning-curves.csv, report.json and per-check generated journals.',report['limits']]
    (output/'report.md').write_text('\n\n'.join(text)+'\n')
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>SimLingo training review</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:15px}pre{white-space:pre-wrap}img{width:100%}</style>'
        '<h1>SimLingo training review</h1><pre>'+html.escape('\n\n'.join(text))+'</pre>'
        '<img src="learning-curves.svg" alt="Learning curves"><h2>Counts, metrics, recommendations and examples</h2><pre>'
        +html.escape(json.dumps(report,indent=2))+'</pre>')
    return report
