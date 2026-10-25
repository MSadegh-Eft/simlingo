"""CPU enumeration of every finite pilot text variant, before long training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor
import torch

from simlingo_training.lidar.data import (ManifestDataset,SNAPSHOT,SPECIAL,navigation_prompt,
    check_context_lengths)
from simlingo_training.lidar.perception import SECTORS
from simlingo_training.lidar.storage import ShardReader,atomic_json,digest_file
from simlingo_training.utils.internvl2_utils import get_custom_chat_template


def variants(current,annotations):
    base=navigation_prompt(current)
    yield 'driving',base+'Predict the waypoints.','Waypoints:'
    for q in ManifestDataset._questions(annotations.get('drivelm')):
        yield 'qa',base+'Use only the front-camera view. Q: '+q['Q'],'A: '+q['A']
    comment=annotations.get('commentary',{})
    if comment.get('commentary') and comment.get('cause_object_visible_in_image') is True:
        yield 'commentary',base+'What should the ego do next?',comment['commentary']+' Waypoints:'
    for option in ManifestDataset._options(annotations.get('dreamer')):
        instructions=option['dreamer_instruction']
        for instruction in [instructions] if isinstance(instructions,str) else instructions:
            for safety in (False,True):
                prompt=('<SAFETY> ' if safety else '<INSTRUCTION_FOLLOWING> ')+base+instruction
                answer=option['dreamer_answer_safety'] if safety and not option['safe_to_execute'] else 'Following the given instruction. Waypoints:'
                yield 'dreamer',prompt,answer
    # Superset of geometry questions, including currently masked sector labels.
    for sector in SECTORS[1:]:
        prompt=base+f'Q: Are there LiDAR-observed vehicles in the {sector} sector within 64 metres?'
        for answer in ('A: Yes.','A: No observed vehicle; this does not establish that the lane is clear.'):
            yield 'geometry_qa',prompt,answer
        prompt=base+f'Q: What is the horizontal distance bin of the nearest observed vehicle in the {sector} sector within 64 metres?'
        for distance in ('0-10','10-20','20-40','40-64'):
            yield 'geometry_qa',prompt,'A: '+distance+' metres.'


def main(manifest,output):
    torch.set_num_threads(4)
    document=json.loads(manifest.read_text()); reader=ShardReader(manifest.parent,document['members'])
    processor=AutoProcessor.from_pretrained(str(SNAPSHOT),trust_remote_code=True,local_files_only=True)
    tokenizer=getattr(processor,'tokenizer',processor); tokenizer.add_special_tokens({'additional_special_tokens':SPECIAL}); tokenizer.padding_side='left'
    seen=set(); counts=Counter(); pending=[]; maximum={'teacher':0,'question':0,'reserved_context':0}; witness={}
    def flush():
        if not pending: return
        conversations=[[{'role':'user','content':[{'type':'text','text':p},{'type':'image'}]},
                        {'role':'assistant','content':[{'type':'text','text':a}]}] for _,p,a,uid in pending]
        teacher,question=get_custom_chat_template(conversations,tokenizer,'OpenGVLab/InternVL2-1B',512,conversation_path=str(SNAPSHOT/'conversation.py'))
        # Padded batch widths upper-bound every sample without treating EOS/pad
        # identity as a length mask. The maximum of all batch widths is exact.
        t,q=teacher['phrase_ids'].shape[1],question['phrase_ids'].shape[1]
        required=check_context_lengths(t,q,paired=True,driving=True)
        for key,length in [('teacher',t),('question',q),('reserved_context',required)]:
            if length>maximum[key]: maximum[key]=length; witness[key]=[{'uid':uid,'task':task} for task,_,_,uid in pending]
        pending.clear()
    for j,row in enumerate(document['samples']):
        annotations={k:reader.json(v) for k,v in row['annotations'].items()}
        current=reader.json(row['measurements'][0])
        for task,prompt,answer in variants(current,annotations):
            counts[task]+=1
            key=hashlib.sha256((task+'\0'+prompt+'\0'+answer).encode()).digest()
            if key in seen: continue
            seen.add(key); pending.append((task,prompt,answer,row['id']))
            if len(pending)==32: flush()
        if (j+1)%2000==0: print(f'Context audit {j+1}/{len(document["samples"])} rows; {len(seen)} distinct variants',flush=True)
    flush()
    report={'passed':True,'manifest_sha256':digest_file(manifest),'audit_source_sha256':digest_file(__file__),
        'data_source_sha256':digest_file(Path(__file__).resolve().parents[2]/'simlingo_training/lidar/data.py'),
        'rows':len(document['samples']),'enumerated_variants':dict(counts),'distinct_variants':len(seen),
        'maxima':maximum,'witness_batches':witness,'bound':2048,'generation_reserve':100,
        'lidar_prefix':128,'driving_queries':30,'image_tokens':512,
        'scope':'All original QA choices, visible commentary, Dreamer instructions and both safety modes; conservative superset of geometry texts. Both official splits. RGB-only lengths are smaller.'}
    atomic_json(output,report); print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); main(args.manifest,args.output)
