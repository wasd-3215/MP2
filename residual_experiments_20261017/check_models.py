"""Test exact initial preservation, freezing, live correction and context ablation."""
from support import semantic_checks
import json
import torch
from models import CandidateScorer,MODEL_NAMES,load_anchor


def main():
    torch.manual_seed(17)
    x=torch.randn(13,4,1024)
    y=torch.tensor([[1.,0,1,0]]*13)
    anchor=CandidateScorer('independent').eval()
    results=[]
    for name in MODEL_NAMES:
        torch.manual_seed(29)
        model=CandidateScorer(name)
        staged=name not in ('independent','attention')
        if staged:
            load_anchor(model,anchor.state_dict())
        model.eval()
        with torch.no_grad():
            parts=model.components(x)
            if staged:
                torch.testing.assert_close(parts['final'],anchor(x),rtol=0,atol=0)
            if model.has_correction:
                assert torch.equal(parts['raw_delta'],torch.zeros_like(parts['raw_delta']))
                if model.gated:
                    torch.testing.assert_close(parts['gate'],torch.full_like(parts['gate'],.1))
        before={k:v.clone() for k,v in model.base.state_dict().items()}
        model.train()
        if model.frozen_base:
            assert not model.base.training and not any(p.requires_grad for p in model.base.parameters())
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
        if model.frozen_base:
            assert all(torch.equal(before[k],v) for k,v in model.base.state_dict().items())
        model.eval()
        checks=semantic_checks(model,x,torch.zeros(13,0))
        with torch.no_grad():
            p=model.components(x)
            torch.testing.assert_close(p['final']-p['applied_delta'],p['base'])
            if model.has_correction:
                assert p['applied_delta'].abs().sum()>0
            if name in ('independent','finetune','residual_self'):
                assert checks['other_candidate_perturbation_max_abs_effect']<2e-5
            if name in ('residual_context','residual_gated','residual_joint'):
                assert checks['other_candidate_perturbation_max_abs_effect']>1e-7
            if model.frozen_base:
                torch.testing.assert_close(p['base'],anchor(x),rtol=0,atol=0)
        results.append({'model':name,'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'checks':checks})
    print(json.dumps({'passed':True,'configurations':results},indent=2))


if __name__=='__main__':
    main()
