"""Independent judgment plus a zero-initialized correction; offline and equivariant."""
import math
import torch
from torch import nn
from baseline_models import CandidateScorer as OriginalScorer

MODEL_NAMES=('independent','attention','finetune','residual_self','residual_context','residual_gated','residual_joint')


class CandidateScorer(nn.Module):
    def __init__(self,name,hidden=128,dropout=0.1,question_dim=0):
        super().__init__()
        if name not in MODEL_NAMES or question_dim:
            raise ValueError('Unsupported scorer configuration')
        self.name=name
        self.has_question=False
        self.has_correction=name.startswith('residual_')
        self.frozen_base=name in ('residual_self','residual_context','residual_gated')
        self.gated=name in ('residual_gated','residual_joint')
        self.base=OriginalScorer('attention' if name=='attention' else 'independent',hidden,dropout)
        if self.has_correction:
            self.attention=nn.MultiheadAttention(hidden,4,dropout=dropout,batch_first=True)
            self.norm1=nn.LayerNorm(hidden)
            self.ff=nn.Sequential(nn.Linear(hidden,2*hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*hidden,hidden))
            self.norm2=nn.LayerNorm(hidden)
            self.residual_dropout=nn.Dropout(dropout)
            self.delta_head=nn.Sequential(nn.Linear(hidden,64),nn.GELU(),nn.Dropout(dropout),nn.Linear(64,1))
            nn.init.zeros_(self.delta_head[-1].weight)
            nn.init.zeros_(self.delta_head[-1].bias)
            if self.gated:
                self.gate=nn.Linear(2*hidden,1)
                nn.init.zeros_(self.gate.weight)
                nn.init.constant_(self.gate.bias,math.log(.1/.9))
        if self.frozen_base:
            self.base.requires_grad_(False)
            self.base.eval()

    def train(self,mode=True):
        super().train(mode)
        if self.frozen_base:
            self.base.eval()
        return self

    def components(self,x):
        if not self.has_correction:
            score=self.base(x)
            zero=torch.zeros_like(score)
            return {'base':score,'raw_delta':zero,'gate':zero,'applied_delta':zero,'final':score}
        h=self.base.encoder(x)
        base=self.base.head(h).squeeze(-1)
        mask=~torch.eye(4,dtype=torch.bool,device=x.device) if self.name=='residual_self' else None
        message,_=self.attention(h,h,h,attn_mask=mask,need_weights=False)
        c=self.norm1(h+self.residual_dropout(message))
        c=self.norm2(c+self.residual_dropout(self.ff(c)))
        delta=self.delta_head(c).squeeze(-1)
        gate=torch.sigmoid(self.gate(torch.cat((h,c),dim=-1))).squeeze(-1) if self.gated else torch.ones_like(delta)
        applied=gate*delta
        return {'base':base,'raw_delta':delta,'gate':gate,'applied_delta':applied,'final':base+applied}

    def forward(self,x,question=None):
        return self.components(x)['final']


def load_anchor(model,anchor_state):
    if not all(k.startswith('base.') for k in anchor_state):
        raise ValueError('Expected independent wrapper checkpoint')
    model.base.load_state_dict({k[len('base.'):]:v for k,v in anchor_state.items()},strict=True)
