"""Mask semantics, explicit attention reference, common initialization, baseline replay."""
from support import HERE, semantic_checks
import json
import math
import torch
import torch.nn.functional as F
from models import CandidateScorer
from reference_models import CandidateScorer as OriginalScorer


def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    torch.manual_seed(19)
    x=torch.randn(13,4,1024)
    y=torch.tensor([[1.,0,1,0]]*13)
    torch.manual_seed(17)
    reference=OriginalScorer('attention').eval()
    initial={k:v.clone() for k,v in reference.state_dict().items()}
    results=[]
    for variant,cfg in protocol['variants'].items():
        torch.manual_seed(17)
        model=CandidateScorer('attention',hidden=cfg['hidden'],dropout=cfg['dropout'],
                              num_heads=cfg['num_heads'],exclude_self=cfg['exclude_self']).eval()
        assert sum(p.numel() for p in model.parameters())==272001
        assert all(torch.equal(initial[k],v) for k,v in model.state_dict().items())
        with torch.no_grad():
            if variant=='include_h4':
                torch.testing.assert_close(model(x),reference(x),atol=0,rtol=0)
            h=model.encoder(x)
            mask=torch.eye(4,dtype=torch.bool) if model.exclude_self else None
            message,weights=model.attention(h,h,h,attn_mask=mask,need_weights=True,average_attn_weights=False)
            q,k,v=F.linear(h,model.attention.in_proj_weight,model.attention.in_proj_bias).chunk(3,-1)
            reshape=lambda t:t.reshape(len(x),4,cfg['num_heads'],-1).transpose(1,2)
            q,k,v=map(reshape,(q,k,v))
            logits=q@k.transpose(-2,-1)/math.sqrt(128/cfg['num_heads'])
            if mask is not None:
                logits=logits.masked_fill(mask,-torch.inf)
            expected=logits.softmax(-1)
            torch.testing.assert_close(weights,expected,atol=2e-6,rtol=1e-5)
            expected_message=F.linear((expected@v).transpose(1,2).reshape(len(x),4,128),
                                      model.attention.out_proj.weight,model.attention.out_proj.bias)
            torch.testing.assert_close(message,expected_message,atol=2e-6,rtol=1e-5)
            if mask is not None:
                assert weights.diagonal(dim1=-2,dim2=-1).abs().max()==0
            assert torch.allclose(weights.sum(-1),torch.ones_like(weights.sum(-1)),atol=1e-6)
        checks=semantic_checks(model,x,torch.zeros(len(x),0))
        assert checks['other_candidate_perturbation_max_abs_effect']>1e-7
        model.train()
        loss=F.binary_cross_entropy_with_logits(model(x),y)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert model.attention.in_proj_weight.grad.abs().sum()>0
        results.append({'variant':variant,'parameters':272001,'explicit_attention_reference':True,
                        'common_initialization':True,'finite_gradients':True,'checks':checks})
    print(json.dumps({'passed':True,'configurations':results},indent=2))


if __name__=='__main__':
    main()
