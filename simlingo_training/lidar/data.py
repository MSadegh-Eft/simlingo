"""Explicit task requests, immutable official splits, and sensor/target separation."""
from dataclasses import dataclass, fields, replace
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path

import cv2
import laspy
import numpy as np
import torch
from PIL import Image
from imgaug import augmenters as ia
from omegaconf import OmegaConf

from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess, get_custom_chat_template
from simlingo_training.lidar.perception import prepare_cloud, geometry_targets, sector_ids, SECTORS, PREPROCESS_VERSION
from simlingo_training.lidar.storage import ShardReader, SCHEMA

SNAPSHOT = Path('/data/ghazaleh/.cache/huggingface/hub/models--OpenGVLab--InternVL2-1B/snapshots/0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187')
SPECIAL = ['<WAYPOINTS>','<WAYPOINTS_DIFF>','<ORG_WAYPOINTS_DIFF>','<ORG_WAYPOINTS>','<WAYPOINT_LAST>','<ROUTE>','<ROUTE_DIFF>','<TARGET_POINT>']
TASKS = ('dreamer','driving','qa','commentary','rear_driving','geometry_qa')
JOINT_COUNTS = (384,128,128,128,96,96)  # 960 presentations; exact original-family/replay quotas.
ALIGN_COUNTS = (24,8,8,8,8,8)          # 64 paired presentations.


@dataclass(frozen=True)
class Request:
    index: int
    task: str
    replay: bool
    seed: int
    position: int


@dataclass
class Sample:
    pixels: torch.Tensor
    ids: torch.Tensor
    answer_mask: torch.Tensor
    question_ids: torch.Tensor
    navigation: torch.Tensor
    waypoints: torch.Tensor
    route: torch.Tensor
    cloud: object
    geometry: dict
    task: str
    uid: str
    position: int

    def to(self, device):
        values = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value,torch.Tensor): value = value.to(device)
            elif f.name=='geometry': value = {k:v.to(device) for k,v in value.items()}
            elif f.name=='cloud' and value is not None: value = value.to(device)
            values[f.name] = value
        return replace(self, **values)

    @property
    def driving_active(self): return self.task not in ('qa','geometry_qa')

    @property
    def language_active(self): return self.task in ('qa','geometry_qa','commentary','dreamer')


class TaskSampler:
    """Stateless sample RNG plus an explicit next-position cursor for exact resume."""
    def __init__(self, pools, stage='align', seed=42, position=0, manifest_hash='', bucket_pools=None, bucket_weights=None):
        if stage not in ('align','joint'): raise ValueError(stage)
        if any(not pools.get(t) for t in TASKS):
            raise ValueError('Every requested task needs an eligible pool; missing tasks must not be silently reassigned')
        self.pools, self.stage, self.seed, self.position, self.manifest_hash = pools,stage,seed,position,manifest_hash
        self.bucket_pools, self.bucket_weights = bucket_pools or {}, bucket_weights or {}
        self.sampling_hash = hashlib.sha256(json.dumps([pools,self.bucket_pools,self.bucket_weights],sort_keys=True).encode()).hexdigest()

    def request_at(self, position):
        counts = ALIGN_COUNTS if self.stage=='align' else JOINT_COUNTS
        length = sum(counts); cycle, offset = divmod(position,length)
        order = []
        for task,count in zip(TASKS,counts):
            order += [(task, self.stage=='joint' and task in TASKS[:4] and i < count//4) for i in range(count)]
        rng_cycle = np.random.default_rng(np.random.SeedSequence([self.seed,cycle,917]))
        task,replay = order[int(rng_cycle.permutation(length)[offset])]
        seed = int(np.random.SeedSequence([self.seed,position,31]).generate_state(1)[0])
        rng = np.random.default_rng(seed)
        candidates = self.pools[task]
        buckets = self.bucket_pools.get(task,{})
        available = [(k,v) for k,v in buckets.items() if v and self.bucket_weights.get(k,0)>0]
        if available:
            weights = np.asarray([self.bucket_weights[k] for k,_ in available],dtype=float)
            _,candidates = available[int(rng.choice(len(available),p=weights/weights.sum()))]
        index = candidates[int(rng.integers(len(candidates)))]
        return Request(index,task,replay,seed,position)

    def take(self, count):
        requests = [self.request_at(p) for p in range(self.position,self.position+count)]
        self.position += count
        return requests

    def state_dict(self):
        return {'stage':self.stage,'seed':self.seed,'position':self.position,'manifest_hash':self.manifest_hash,'sampling_hash':self.sampling_hash}

    def load_state_dict(self,state):
        for k in ('stage','seed','manifest_hash','sampling_hash'):
            if state[k] != getattr(self,k): raise ValueError(f'Sampler provenance differs: {k}')
        self.position = state['position']


def equal_spacing_route(points):
    points = np.asarray(points,dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points): raise ValueError('Invalid route target')
    route = np.concatenate((np.zeros_like(points[:1]),points))
    distances = np.r_[0., np.linalg.norm(np.diff(route,axis=0),axis=1)].cumsum()
    distances += np.arange(len(distances))*1e-4
    return np.stack([np.interp(np.arange(20),distances,route[:,i]) for i in range(2)],axis=1).astype(np.float32)


def scenario_buckets(measurements,boxes,route_name):
    """Reconstruct intended original bucket families from training-only labels."""
    m=measurements[0]; buckets={'all'}
    origin=np.asarray(m['ego_matrix'])
    wps=np.array([(origin[:3,:3].T@(np.asarray(x['ego_matrix'])[:3,3]-origin[:3,3]))[:2] for x in measurements[:11]])
    lateral=abs(wps[:,1].mean())
    if .1<=lateral<2: buckets.add('lateral_control_1_2')
    if lateral>=2: buckets.add('lateral_control_higher_5')
    # The original bucket generator used .2 here; keep that sampling definition.
    # Supervised waypoint times and the sensor clock remain .25 seconds.
    acceleration=np.diff([x['speed'] for x in measurements[:5]]).mean()/.2
    for bound,name in [(-20,None),(-5,'acceleration_negative_5'),(-1,'acceleration_negative_1'),(1,None),(5,'acceleration_positive_1'),(20,'acceleration_positive_5'),(np.inf,None)]:
        if acceleration<bound:
            if name: buckets.add(name)
            break
    if m['speed']<.5 and m['target_speed']>.8: buckets.add('start_from_stop')
    leader=m.get('speed_reduced_by_obj_type')
    if leader and m.get('speed_reduced_by_obj_distance') is not None and m['speed_reduced_by_obj_distance']<30:
        leader=leader.split('.')[0] if leader.startswith(('vehicle.','walker.')) else leader
        buckets.add('leading_object_'+leader)
    if m.get('changed_route'): buckets.add('changed_route')
    affecting=next((b for b in boxes if b.get('class')=='car' and b.get('id')==m.get('vehicle_affecting_id')),None)
    ego=next((b for b in boxes if b.get('class') in ('ego_car','ego_info')), {})
    junction_distance=ego.get('distance_to_junction')
    red=ego.get('traffic_light_state')=='Red' and junction_distance is not None and junction_distance<20
    if m.get('vehicle_hazard') and affecting and not red:
        yaw=abs(affecting['yaw'])
        if np.pi-.6<yaw<np.pi+.6: buckets.add('vehicle_front')
        elif yaw>.5: buckets.add('vehicle_side')
    if 'parking_lane' in route_name and lateral>.2: buckets.add('parkinglane')
    return buckets


def photometric(image,seed):
    probability=.5
    augmenter=ia.Sequential([
        ia.Sometimes(probability,ia.GaussianBlur((0,1.))),
        ia.Sometimes(probability,ia.AdditiveGaussianNoise(loc=0,scale=(0,.05*255),per_channel=.5)),
        ia.Sometimes(probability,ia.Dropout((.01,.1),per_channel=.5)),
        ia.Sometimes(probability,ia.Multiply((1/1.2,1.2),per_channel=.5)),
        ia.Sometimes(probability,ia.LinearContrast((1/1.2,1.2),per_channel=.5)),
        ia.Sometimes(probability,ia.Grayscale((0,.5)))],random_order=True,seed=seed)
    # Seeding Sequential alone leaves independently constructed children on
    # imgaug's global RNG. Seed the complete tree for exact worker/resume parity.
    augmenter.seed_(seed)
    return Image.fromarray(augmenter(image=np.array(image)))


class ManifestDataset:
    def __init__(self, manifest, tokenizer, split='train', augment=False, max_samples=None):
        self.path = Path(manifest)
        raw = self.path.read_bytes(); self.manifest_hash = hashlib.sha256(raw).hexdigest()
        document = json.loads(raw)
        if document.get('preparation_complete') is False: raise ValueError('Dataset preparation has not finished')
        if document['schema'] != SCHEMA or document['preprocess'] != PREPROCESS_VERSION: raise ValueError('Dataset schema/preprocessing differs')
        groups = document['group_splits']
        for row in document['samples']:
            if groups[row['group']] != row['split']: raise ValueError('Cross-task route split differs')
        self.rows = [r for r in document['samples'] if r['split']==split]
        if max_samples is not None:
            self.rows=sorted(self.rows,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:max_samples]
        self.reader = ShardReader(self.path.parent,document['members'])
        self.tokenizer, self.augment = tokenizer,augment
        self.tokenizer.add_special_tokens({'additional_special_tokens':SPECIAL})
        self.tokenizer.padding_side = 'left'
        self.pools = {t:[] for t in TASKS}
        self.row_buckets = {}
        # Four contiguous arrays avoid retaining all annotation objects or
        # creating hundreds of thousands of tiny shared Torch storages in workers.
        self.geometry={k:np.zeros((len(self.rows),4),dtype=bool if 'valid' in k else np.int64 if k=='distance' else np.float32)
                       for k in ('presence','presence_valid','distance','distance_valid')}
        for i,row in enumerate(self.rows):
            self.pools['driving'].append(i)
            annotations = {k:self.reader.json(p) for k,p in row['annotations'].items()}
            if self._questions(annotations.get('drivelm')): self.pools['qa'].append(i)
            c = annotations.get('commentary',{})
            # Retain visible causes; global/hidden-cause comments need a separate audit.
            if c.get('commentary') and c.get('cause_object_visible_in_image') is True: self.pools['commentary'].append(i)
            if self._options(annotations.get('dreamer')): self.pools['dreamer'].append(i)
            boxes = self.reader.json(row['boxes'])
            if 'geometry' in row:
                g={k:np.asarray(v) for k,v in row['geometry'].items()}
            else:
                laz=laspy.read(io.BytesIO(self.reader.read(row['lidar'])),laz_backend=laspy.LazBackend.Lazrs)
                g={k:v.numpy() for k,v in geometry_targets(prepare_cloud(np.column_stack((laz.x,laz.y,laz.z))),boxes).items()}
            for key,value in g.items(): self.geometry[key][i]=value
            if g['presence_valid'][1:].any(): self.pools['geometry_qa'].append(i)
            self.row_buckets[i] = scenario_buckets([self.reader.json(p) for p in row['measurements']],boxes,row['route'])
            if g['presence'][1:].any():
                self.pools['rear_driving'].append(i)
        config=OmegaConf.load(Path(__file__).resolve().parents[2]/'models/simlingo/simlingo/.hydra/config.yaml')
        self.bucket_weights=dict(config.data_module.train_partitions)
        self.bucket_pools={task:{bucket:[i for i in self.pools[task] if bucket in self.row_buckets[i]] for bucket in self.bucket_weights}
                           for task in ('driving','qa','commentary')}

    @staticmethod
    def _questions(annotation):
        if not annotation: return []
        return [q for values in annotation.get('QA',{}).values() for q in values if q.get('Q') and q.get('A')]

    @staticmethod
    def _options(annotation):
        if not annotation: return []
        options=[o for key,values in annotation.items() if 'factor' not in key and isinstance(values,list)
                for o in values if isinstance(o,dict) and all(k in o for k in ('waypoints','route','dreamer_instruction','safe_to_execute'))]
        for option in options:
            if not isinstance(option['safe_to_execute'],bool): raise ValueError('Dreamer safety target must be boolean')
            if not option['safe_to_execute'] and not isinstance(option.get('dreamer_answer_safety'),str): raise ValueError('Missing unsafe Dreamer answer')
            instructions=option['dreamer_instruction']
            instructions=[instructions] if isinstance(instructions,str) else instructions
            if not instructions or any(not isinstance(x,str) or not x for x in instructions): raise ValueError('Invalid Dreamer instructions')
            for key,shape in [('waypoints',(10,2)),('route',(20,2))]:
                value=option[key]
                if isinstance(value,str):
                    if value!='org': raise ValueError('Unrecognized original-trajectory placeholder')
                elif np.asarray(value).shape!=shape or not np.isfinite(np.asarray(value,dtype=float)).all():
                    raise ValueError(f'Invalid Dreamer {key} target')
        return options

    def __len__(self): return len(self.rows)

    def __getitem__(self, request):
        cv2.setNumThreads(0)
        if not isinstance(request,Request): request = Request(int(request),'driving',False,42,int(request))
        row = self.rows[request.index]
        rng = np.random.default_rng(request.seed)
        annotations={k:self.reader.json(p) for k,p in row['annotations'].items()}
        measurements = [self.reader.json(p) for p in row['measurements']]
        current = measurements[0]
        origin = np.asarray(current['ego_matrix'])
        targets = [(origin[:3,:3].T @ (np.asarray(m['ego_matrix'])[:3,3]-origin[:3,3]))[:2] for m in measurements[1:11]]
        waypoints = np.asarray(targets,dtype=np.float32)
        route = equal_spacing_route(current['route'])
        navigation = np.asarray([current['target_point'],current['target_point_next']],dtype=np.float32)
        # Inputs are selected current sensors and navigation only. Future measurements
        # and annotations are consumed exclusively below for supervised targets.
        image = cv2.imdecode(np.frombuffer(self.reader.read(row['rgb']),np.uint8),cv2.IMREAD_COLOR)
        if image is None: raise ValueError(f'Unreadable RGB: {row["id"]}')
        image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        if self.augment:
            image = photometric(image,request.seed)
        image = image.crop((0,0,image.width,int(image.height-(image.height*4.8)//16)))
        pixels = torch.stack([build_transform(448)(x) for x in dynamic_preprocess(image,image_size=448,use_thumbnail=False,max_num=2)])
        cloud = None
        geometry = {k:torch.zeros(4,dtype=torch.bool if 'valid' in k else torch.long if k=='distance' else torch.float32)
                    for k in ('presence','presence_valid','distance','distance_valid')}
        if not request.replay:
            laz = laspy.read(io.BytesIO(self.reader.read(row['lidar'])),laz_backend=laspy.LazBackend.Lazrs)
            cloud = prepare_cloud(np.column_stack((laz.x,laz.y,laz.z)), row['frame']*.25)
            geometry = {k:torch.from_numpy(values[request.index]) for k,values in self.geometry.items()}
        commands = {1:'go left at the next intersection',2:'go right at the next intersection',3:'go straight',4:'follow the road',5:'change lane left',6:'change lane right'}
        prompt = f"Current speed: {round(current['speed'],1)} m/s. Target waypoint: <TARGET_POINT><TARGET_POINT>. Command: {commands[current['command']]}. "
        answer = 'Waypoints:'
        if request.task=='qa':
            options = self._questions(annotations.get('drivelm'))
            if not options: raise ValueError('Requested QA has no labels')
            q = options[int(rng.integers(len(options)))]
            prompt += 'Use only the front-camera view. Q: '+q['Q']; answer = 'A: '+q['A']
        elif request.task=='commentary':
            c = annotations.get('commentary',{})
            if not c.get('commentary') or c.get('cause_object_visible_in_image') is not True: raise ValueError('Unsupported commentary target')
            prompt += 'What should the ego do next?'; answer = c['commentary']+' Waypoints:'
        elif request.task=='dreamer':
            options = self._options(annotations.get('dreamer'))
            if not options: raise ValueError('Requested Dreamer sample has no labels')
            option = options[int(rng.integers(len(options)))]
            instructions = option['dreamer_instruction']
            instruction = instructions[int(rng.integers(len(instructions)))] if isinstance(instructions,list) else instructions
            safety = bool(rng.integers(2))
            prompt = ('<SAFETY> ' if safety else '<INSTRUCTION_FOLLOWING> ')+prompt+instruction
            answer = 'Following the given instruction. Waypoints:'
            if safety and not option['safe_to_execute']:
                answer = option['dreamer_answer_safety']
            else:
                if not isinstance(option['waypoints'],str): waypoints = np.asarray(option['waypoints'],dtype=np.float32)
                if not isinstance(option['route'],str): route = np.asarray(option['route'],dtype=np.float32)
        elif request.task=='geometry_qa':
            if cloud is None: raise ValueError('Geometry QA requires LiDAR')
            sectors = [s for s in (1,2,3) if geometry['presence_valid'][s]]
            if not sectors: raise ValueError('All rear/side sectors are ambiguous')
            sector = sectors[int(rng.integers(len(sectors)))]
            prompt += f'Q: Are there LiDAR-observed vehicles in the {SECTORS[sector]} sector within 64 metres?'
            answer = 'A: Yes.' if geometry['presence'][sector] else 'A: No observed vehicle; this does not establish that the lane is clear.'
            if geometry['distance_valid'][sector] and rng.random()<.5:
                prompt = prompt.rsplit('Q:',1)[0]+f'Q: What is the horizontal distance bin of the nearest observed vehicle in the {SECTORS[sector]} sector within 64 metres?'
                answer = 'A: '+('0-10','10-20','20-40','40-64')[int(geometry['distance'][sector])]+' metres.'
        else:
            prompt += 'Predict the waypoints.'
        if waypoints.shape!=(10,2) or route.shape!=(20,2) or not np.isfinite(waypoints).all() or not np.isfinite(route).all():
            raise ValueError(f'Invalid driving targets: {row["id"]}')
        conv = [{'role':'user','content':[{'type':'text','text':prompt},{'type':'image'}]},
                {'role':'assistant','content':[{'type':'text','text':answer}]}]
        teacher, question = get_custom_chat_template([conv],self.tokenizer,'OpenGVLab/InternVL2-1B',len(pixels)*256,
                                                     conversation_path=str(SNAPSHOT/'conversation.py'))
        ids = teacher['phrase_ids'][0]; mask = teacher['loss_masking'][0] & teacher['phrase_valid'][0]
        if len(ids)+128+30>2048: raise ValueError('Context exceeds the unprofiled 2048-token bound; reprofile rather than truncate')
        return Sample(pixels,ids,mask,question['phrase_ids'][0],torch.from_numpy(navigation),torch.from_numpy(waypoints),
                      torch.from_numpy(route),cloud,geometry,request.task,row['id'],request.position)
