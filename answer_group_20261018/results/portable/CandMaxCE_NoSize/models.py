"""Matched-parameter S2 backbone with candidate/global or answer-group mean/max readout."""
import torch
from torch import nn
from reference_models import CandidateScorer

class Scorer(nn.Module):
    def __init__(self,representation,size_mode):
        super().__init__()
        assert representation in ['CandBCE','CandMaxCE','GroupCE'] and size_mode in ['NoSize','Size']
        self.representation=representation;self.size_mode=size_mode
        self.base=CandidateScorer('attention',128,.1,0)
        old=self.base.head[0];new=nn.Linear(256+6,64)
        with torch.no_grad():
            new.weight.zero_();new.weight[:,:128].copy_(old.weight);new.bias.copy_(old.bias)
        self.base.head[0]=new
        self.register_buffer('feature_mask',torch.tensor([1.,float(size_mode=='Size'),0.,0.,0.,0.]),persistent=False)

    def forward(self,x,features,membership):
        b=self.base;h=b.encoder(x);h=h+b.residual_dropout(b.ff(h))
        message,_=b.attention(h,h,h,need_weights=False)
        h=b.norm1(h+b.residual_dropout(message));h=b.norm2(h)
        f=features*self.feature_mask
        mask=membership.bool();size=mask.sum(2,keepdim=True).clamp_min(1);weights=mask.to(h.dtype)
        if self.representation=='GroupCE':
            mean=torch.bmm(weights,h)/size
            maximum=h[:,None,:,:].expand(-1,4,-1,-1).masked_fill(~mask[:,:,:,None],-1e9).max(2).values
            maximum=maximum.masked_fill(~mask.any(2)[:,:,None],0.)
            pooled_f=torch.bmm(weights,f)/size
            logits=b.head(torch.cat([mean,maximum,pooled_f],dim=-1)).squeeze(-1)
            candidates=None
        else:
            context=h.mean(1,keepdim=True).expand_as(h)
            candidates=b.head(torch.cat([h,context,f],dim=-1)).squeeze(-1)
            logits=candidates[:,None,:].expand(-1,4,-1).masked_fill(~mask,-1e9).max(2).values
        return candidates,logits.masked_fill(~mask.any(2),-1e9)

def objective(model,outputs,y,membership,target):
    candidates,groups=outputs
    if model.representation=='CandBCE':
        valid=membership.any(1).to(y.dtype)
        losses=torch.nn.functional.binary_cross_entropy_with_logits(candidates,y,reduction='none')
        return ((losses*valid).sum(1)/valid.sum(1).clamp_min(1)).mean()
    return torch.nn.functional.cross_entropy(groups,target)
