"""S2/S0/pool and explicit-feature head ablations, no data IO."""
import torch
from torch import nn
from reference_models import CandidateScorer
from explicit_features import FEATURES


def configuration(variant):
    if variant=='features_only':
        return {'architecture':'features_only','feature_mode':'all'}
    return {'architecture': 'pool' if variant=='pool' else ('s0' if variant=='s0' else 's2'),
            'feature_mode':variant.removeprefix('s2_') if variant.startswith('s2_') else 'none'}


class Scorer(nn.Module):
    def __init__(self,architecture='s2',feature_mode='none'):
        super().__init__()
        self.architecture=architecture
        self.feature_mode=feature_mode
        dim=len(FEATURES[feature_mode])
        if architecture=='features_only':
            self.feature_head=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,1))
            return
        self.base=CandidateScorer('pool' if architecture=='pool' else 'attention',128,.1,0)
        if dim:
            old=self.base.head[0]
            new=nn.Linear(old.in_features+dim,old.out_features)
            # Same initial embedding function as F0; new features start with zero weights.
            with torch.no_grad():
                new.weight.zero_()
                new.weight[:,:old.in_features].copy_(old.weight)
                new.bias.copy_(old.bias)
            self.base.head[0]=new

    def forward(self,x,features):
        if self.architecture=='features_only':
            return self.feature_head(features).squeeze(-1)
        b=self.base
        h=b.encoder(x)
        if self.architecture=='pool':
            h=torch.cat([h,h.mean(1,keepdim=True).expand_as(h)],-1)
        else:
            if self.architecture=='s2':
                h=h+b.residual_dropout(b.ff(h))
            m,_=b.attention(h,h,h,need_weights=False)
            h=b.norm1(h+b.residual_dropout(m))
            if self.architecture=='s0':
                h=h+b.residual_dropout(b.ff(h))
            h=b.norm2(h)
        if features.shape[-1]:
            h=torch.cat([h,features],-1)
        return b.head(h).squeeze(-1)
