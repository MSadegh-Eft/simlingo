"""Pretrained SimLingo + LiDAR, with a shared causal context for language/driving."""
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from transformers import AutoProcessor

from simlingo_training.models.driving import DrivingModel
from simlingo_training.lidar.data import SNAPSHOT, Sample
from simlingo_training.lidar.perception import LidarPerception

ROOT = Path(__file__).resolve().parents[2]
RELEASED = ROOT/'models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt'


class LidarDrivingModel(DrivingModel):
    def __init__(self, *args, lidar_checkpoint, **kwargs):
        super().__init__(*args, **kwargs)
        self.lidar = LidarPerception(lidar_checkpoint)
        self.stage = 'align'
        self.use_cache = False
        self.set_stage('align')

    def load_released(self, checkpoint=RELEASED):
        from simlingo_training.lidar.storage import digest_file
        if digest_file(checkpoint)!='ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28':
            raise ValueError('Released SimLingo checkpoint hash differs')
        state = torch.load(str(checkpoint),map_location='cpu',weights_only=True,mmap=True)
        result = self.load_state_dict(state,strict=False)
        expected = {k for k in self.state_dict() if k.startswith('lidar.')}
        if result.unexpected_keys or set(result.missing_keys) != expected:
            raise ValueError(f'Checkpoint mismatch outside new LiDAR modules: {result}')
        return result

    def set_stage(self, stage):
        if stage not in ('align','joint'): raise ValueError(stage)
        self.stage = stage
        self.requires_grad_(False)
        self.lidar.connector.requires_grad_(True)
        self.lidar.joint = stage=='joint'
        self.vision_model.freeze = stage=='align'
        if stage=='joint':
            self.vision_model.requires_grad_(True)
            self.adaptors.driving.requires_grad_(True)
            self.wp_encoder.requires_grad_(True)
            for name,p in self.language_model.named_parameters(): p.requires_grad_('lora_' in name)
            for block in list(self.lidar.encoder.enc.children())[-2:]: block.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self,'lidar'):
            if self.stage=='align':
                self.vision_model.eval(); self.language_model.eval(); self.adaptors.driving.eval(); self.wp_encoder.eval()
            self.lidar.train(mode)
        return self

    @contextmanager
    def generation_mode(self):
        flags = {m:m.training for m in self.modules()}
        self.eval()
        try:
            with torch.no_grad(): yield
        finally:
            for module, flag in flags.items(): module.training = flag

    def backbone(self):
        # LoRA is installed directly in the base model's linear layers. Accessing
        # the backbone avoids materializing vocabulary logits for every image token.
        return self.language_model.model.get_base_model().model

    def hidden(self, embeds, past=None, use_cache=False):
        past_length = 0 if past is None else past[0][0].size(-2)
        mask = torch.ones((1,past_length+embeds.size(1)),device=embeds.device,dtype=torch.long)
        positions = torch.arange(past_length,past_length+embeds.size(1),device=embeds.device)[None]
        return self.backbone()(inputs_embeds=embeds,attention_mask=mask,position_ids=positions,
                               past_key_values=past,use_cache=use_cache,return_dict=True)

    def context(self, sample, ids=None):
        ids = sample.question_ids if ids is None else ids
        embeddings = self.adaptors.language.embed_tokens(ids.clamp(0,self.adaptors.language.embed_tokens.num_embeddings-1))[None]
        image_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        nav_id = self.tokenizer.convert_tokens_to_ids('<TARGET_POINT>')
        image_mask, nav_mask = ids==image_id, ids==nav_id
        unknown = (ids>=self.adaptors.language.embed_tokens.num_embeddings) & ~nav_mask
        if unknown.any(): raise ValueError('Unmapped added token in input')
        vision = self.vision_model.image_encoder.model.extract_feature(sample.pixels).reshape(-1,896)
        if image_mask.sum() != len(vision) or nav_mask.sum()!=2:
            raise ValueError('Image/navigation token count differs from sensor representation')
        embeddings = embeddings.clone()
        embeddings[0,image_mask] = vision.to(embeddings.dtype)
        embeddings[0,nav_mask] = self.wp_encoder(sample.navigation[None])[0].to(embeddings.dtype)
        geometry = None
        if sample.cloud is not None:
            prefix,geometry = self.lidar(sample.cloud)
            embeddings = torch.cat((prefix.to(embeddings.dtype),embeddings),dim=1)
        return embeddings,geometry

    def generate_ids(self, sample, max_new_tokens=100, use_cache=None):
        use_cache = self.use_cache if use_cache is None else use_cache
        with self.generation_mode():
            context,_ = self.context(sample)
            tokens, past, current = [],None,context
            for _ in range(max_new_tokens):
                result = self.hidden(current,past,use_cache)
                logits = self.adaptors.language.lm_head(result.last_hidden_state[:,-1])
                token = logits.argmax(-1)
                tokens.append(token)
                if int(token.item()) == self.tokenizer.eos_token_id: break
                embedding = self.adaptors.language.embed_tokens(token)[None]
                if use_cache:
                    past,current = result.past_key_values,embedding
                else:
                    current = torch.cat((current,embedding),dim=1)
            return torch.stack(tokens,dim=1) if tokens else torch.empty((1,0),device=context.device,dtype=torch.long)

    def driving(self, sample, generated=None):
        context,geometry = self.context(sample)  # Fresh differentiable recomputation.
        if generated is not None:
            context = torch.cat((context,self.adaptors.language.embed_tokens(generated)),dim=1)
        else:
            constant = self.tokenizer('Waypoints:',add_special_tokens=False,return_tensors='pt').input_ids.to(context.device)
            context = torch.cat((context,self.adaptors.language.embed_tokens(constant)),dim=1)
        queries = torch.cat([self.adaptors.driving.queries[k] for k in self.adaptors.driving.order],dim=1)
        result = self.hidden(torch.cat((context,queries.to(context.dtype)),dim=1))
        predictions = self.adaptors.driving.get_predictions(result.last_hidden_state[:,-queries.size(1):])
        return predictions,geometry

    def forward(self,sample):
        # The legacy agent's DrivingExample contains no LiDAR. Reject it rather
        # than silently taking DrivingModel.forward's RGB-only path.
        if not isinstance(sample,Sample):
            raise TypeError('LidarDrivingModel expects a lidar.data.Sample; legacy agent inputs require a separate sensor adapter')
        generated=self.generate_ids(sample) if sample.task in ('commentary','dreamer') else None
        return self.driving(sample,generated)

    def loss_counts(self, sample):
        paired = sample.cloud is not None
        return {'route':20 if sample.driving_active else 0, 'speed':10 if sample.driving_active else 0,
                'language':int(sample.answer_mask[1:].sum()) if sample.language_active else 0,
                'presence':int(sample.geometry['presence_valid'].sum()) if paired else 0,
                'distance':int(sample.geometry['distance_valid'].sum()) if paired else 0}

    def loss_terms(self, sample, generation_limit=100):
        terms, geometry = {},None
        if sample.language_active:
            context,geometry = self.context(sample,sample.ids)
            hidden = self.hidden(context).last_hidden_state
            prefix = 128 if sample.cloud is not None else 0
            mask = sample.answer_mask[1:]
            features = hidden[:,prefix:prefix+len(sample.ids)-1][0,mask]
            labels = sample.ids[1:][mask]
            logits = self.adaptors.language.lm_head(features)
            terms['language'] = F.cross_entropy(logits.float(),labels,reduction='sum') if len(labels) else hidden.sum()*0
        predictions = None
        if sample.driving_active:
            generated = self.generate_ids(sample,generation_limit) if sample.task in ('commentary','dreamer') else None
            predictions,geometry = self.driving(sample,generated)
            terms['route'] = F.smooth_l1_loss(predictions['route'].float(),sample.route[None],reduction='sum')
            terms['speed'] = F.smooth_l1_loss(predictions['speed_wps'].float(),sample.waypoints[None],reduction='sum')
        if geometry is not None:
            valid = sample.geometry['presence_valid']
            terms['presence'] = F.binary_cross_entropy_with_logits(geometry['presence'][0,valid].float(),sample.geometry['presence'][valid],reduction='sum')
            valid = sample.geometry['distance_valid']
            terms['distance'] = F.cross_entropy(geometry['distance'][0,valid].float(),sample.geometry['distance'][valid],reduction='sum') if valid.any() else geometry['distance'].sum()*0
        return terms,predictions

    def optimizer_groups(self):
        groups = []
        components = [('connector',self.lidar.connector,2e-4 if self.stage=='align' else 1e-4),
                      ('lidar',self.lidar.encoder,1e-5),('rgb',self.vision_model,1e-6),
                      ('lora',self.language_model,1e-5),('driving',self.adaptors.driving,1e-5),
                      ('navigation',self.wp_encoder,1e-5)]
        seen = set()
        for name,module,lr in components:
            parameters = [p for p in module.parameters() if p.requires_grad and id(p) not in seen]
            if not parameters: continue
            seen.update(map(id,parameters))
            groups.append({'params':parameters,'lr':lr,'name':name})
        expected = {id(p) for p in self.parameters() if p.requires_grad}
        if seen != expected: raise RuntimeError('Trainable parameters missing from optimizer')
        return groups


def make_model(lidar_checkpoint='/ehsan/m.sadegh/simlingo_lidar/cache/utonia.pth', use_flash_attn=True):
    config = OmegaConf.load(ROOT/'models/simlingo/simlingo/.hydra/config.yaml')
    for component in (config.model.vision_model,config.model.language_model):
        component.revision = SNAPSHOT.name
        component.local_files_only = True
        component.use_flash_attn = use_flash_attn
    processor = AutoProcessor.from_pretrained(str(SNAPSHOT),trust_remote_code=True,local_files_only=True)
    cfg = OmegaConf.to_container(config.model,resolve=True); cfg.pop('_target_')
    model = LidarDrivingModel(config.data_module,processor,None,lidar_checkpoint=lidar_checkpoint,**cfg)
    model.load_released()
    return model
