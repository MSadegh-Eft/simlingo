"""Re-score persisted generated examples without running or changing the model."""
import argparse
import json
from pathlib import Path

from simlingo_training.lidar.evaluation import language_metrics, group_metrics
from simlingo_training.lidar.storage import atomic_json, digest_file

REPO=Path(__file__).resolve().parents[2]


def main(run):
    run=Path(run); progress=json.loads((run/'progress.json').read_text())
    if progress['status']!='complete': raise ValueError('Wait for the finite profile to finish before rescoring')
    report={'change':'Remove decoded assistant chat header from reference scoring; retain raw strings, predictions and timing measurements',
            'timed_scoring_source_sha256':progress['provenance']['code']['simlingo_training/lidar/evaluation.py'],
            'corrected_scoring_source_sha256':digest_file(REPO/'simlingo_training/lidar/evaluation.py'),'outputs':{}}
    for mode in ('rgb','paired'):
        original=run/f'validation-{mode}.json'; result=json.loads(original.read_text())
        original_metrics=result['metrics']
        for row in result['records']:
            # Every non-language field, including route/geometry scores, is kept.
            row['metrics'].update(language_metrics(row['task'],row['generated'],row['reference']))
        result['metrics']=group_metrics(result['records'])
        result['by_task']={task:group_metrics([r for r in result['records'] if r['task']==task]) for task in result['by_task']}
        result['scoring_correction']=report['change']; result['original_sha256']=digest_file(original)
        output=run/f'validation-{mode}-rescored.json'; atomic_json(output,result)
        report['outputs'][mode]={'path':str(output),'sha256':digest_file(output),'metrics':result['metrics'],
                                 'metrics_changed':result['metrics']!=original_metrics}
    atomic_json(run/'scoring-correction.json',report)
    print(json.dumps({'rescored':True,'examples':44,'model_executed':False},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path)
    main(parser.parse_args().run)
