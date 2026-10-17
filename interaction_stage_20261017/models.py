"""Same attention and FFN modules, explicit variations of forward order."""
import torch
from torch import nn
from reference_models import CandidateScorer as OriginalScorer

LAYOUTS=('s0_after','s1_before','s2_before_keep_norm_after',
         's3_before_before','s4_before_after','s5_after_after')


class CandidateScorer(OriginalScorer):
    def __init__(self,name='attention',hidden=128,dropout=.1,question_dim=0,layout='s0_after'):
        if name!='attention' or question_dim or layout not in LAYOUTS:
            raise ValueError('Unsupported stage configuration')
        # Exact original construction order preserves E/A/F1/H initialization.
        super().__init__(name,hidden,dropout,question_dim)
        self.layout=layout
        if layout in LAYOUTS[3:]:
            self.ff2=nn.Sequential(nn.Linear(hidden,2*hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*hidden,hidden))
            self.norm3=nn.LayerNorm(hidden)

    def attention_step(self,h):
        message,_=self.attention(h,h,h,need_weights=False)
        return self.norm1(h+self.residual_dropout(message))

    def p(self,h,index=1):
        ff=self.ff if index==1 else self.ff2
        return h+self.residual_dropout(ff(h))

    def f(self,h,index=1):
        norm=self.norm2 if index==1 else self.norm3
        return norm(self.p(h,index))

    def forward(self,x,question=None):
        h=self.encoder(x)
        if self.layout=='s0_after':
            h=self.f(self.attention_step(h))
        elif self.layout=='s1_before':
            h=self.attention_step(self.f(h))
        elif self.layout=='s2_before_keep_norm_after':
            h=self.norm2(self.attention_step(self.p(h)))
        elif self.layout=='s3_before_before':
            h=self.attention_step(self.f(self.f(h),2))
        elif self.layout=='s4_before_after':
            h=self.f(self.attention_step(self.f(h)),2)
        else:
            h=self.f(self.f(self.attention_step(h)),2)
        return self.head(h).squeeze(-1)
