"""Independent CPU audit of the fixed language-target scoring population."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from transformers import AutoProcessor

from simlingo_training.lidar.data import ManifestDataset,SNAPSHOT
from simlingo_training.lidar.evaluation import structured_answer,dreamer_decision,group_metrics
from simlingo_training.lidar.runtime import METADATA_CACHE
from simlingo_training.lidar.storage import atomic_json,digest_file


def main():
    root=Path('/ehsan/m.sadegh/simlingo_lidar/readiness')
    manifest=root.parent/'stage2/pilot-parking-v2/manifest.json'
    processor=AutoProcessor.from_pretrained(str(SNAPSHOT),trust_remote_code=True,local_files_only=True)
    dataset=ManifestDataset(manifest,getattr(processor,'tokenizer',processor),'val',metadata_cache=METADATA_CACHE)
    protocol=json.loads((manifest.parent/'validation.json').read_text())
    records=[]; classes={task:Counter() for task in ('qa','dreamer')}
    for task in classes:
        for request in protocol['requests'][task]:
            row=dataset.rows[request['index']]
            if row['id']!=request['uid']: raise ValueError('Target audit identity differs')
            rng=np.random.default_rng(request['seed'])
            if task=='qa':
                annotation=dataset.reader.json(row['annotations']['drivelm'])
                options=dataset._questions(annotation); option=options[int(rng.integers(len(options)))]
                target=option['A']; parsed=structured_answer(target); key='qa_structured_accuracy'
                category=str(parsed[0]) if parsed is not None else 'unparsed'
            else:
                options=dataset._options(dataset.reader.json(row['annotations']['dreamer']))
                option=options[int(rng.integers(len(options)))]; instructions=option['dreamer_instruction']
                if isinstance(instructions,list): rng.integers(len(instructions))
                safety=bool(rng.integers(2))
                target=option['dreamer_answer_safety'] if safety and not option['safe_to_execute'] else 'Following the given instruction. Waypoints:'
                parsed=dreamer_decision(target); key='dreamer_decision_accuracy'; category=parsed or 'unparsed'
            classes[task][category]+=1
            records.append({'uid':row['id'],'group':row['group'],'task':task,'seed':request['seed'],
                'metrics':{key:100. if parsed is not None else None}})
    populations=group_metrics(records)
    passed=all(v['samples']>=100 and v['groups']>=10 for v in populations.values())
    result={'passed':passed,'manifest_sha256':dataset.manifest_hash,'validation_sha256':digest_file(manifest.parent/'validation.json'),
        'populations':populations,'target_classes':{k:dict(v) for k,v in classes.items()},'audit_source_sha256':digest_file(__file__),
        'scope':'Annotation-only target selection with the fixed request seeds. Compare these population hashes with the full generated baseline; this is not model accuracy.'}
    atomic_json(root/'target-populations.json',result); print(json.dumps(result),flush=True)
    if not passed: raise ValueError('Insufficient fixed target scoring coverage')


if __name__=='__main__': main()
