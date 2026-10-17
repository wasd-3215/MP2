"""S2 with equal-size zero-initialized early/late additive feature injection."""
import torch
from torch import nn
from reference_models import CandidateScorer

class Scorer(nn.Module):
    def __init__(self,feature_mode,position):
        super().__init__()
        if feature_mode not in ('P','C') or position not in ('Late','Early'):raise ValueError((feature_mode,position))
        self.feature_mode=feature_mode;self.position=position
        self.base=CandidateScorer('attention',128,.1,0)
        self.feature_projection=nn.Linear(6,128,bias=False)
        nn.init.zeros_(self.feature_projection.weight)
        self.register_buffer('feature_mask',torch.tensor([1.]+[float(feature_mode=='C')]*5),persistent=False)
    def forward(self,x,features):
        b=self.base;h=b.encoder(x)
        u=self.feature_projection(features*self.feature_mask)
        if self.position=='Early':h=h+u
        h=h+b.residual_dropout(b.ff(h))
        message,_=b.attention(h,h,h,need_weights=False)
        h=b.norm2(b.norm1(h+b.residual_dropout(message)))
        if self.position=='Late':h=h+u
        return b.head(h).squeeze(-1)
