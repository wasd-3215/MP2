"""Validate operation order, parameter matching and exact original baseline behavior."""
from support import semantic_checks
import json
import torch
import torch.nn.functional as F
from models import CandidateScorer,LAYOUTS
from reference_models import CandidateScorer as OriginalScorer


def main():
    torch.manual_seed(19)
    x=torch.randn(13,4,1024); y=torch.tensor([[1.,0,1,0]]*13)
    torch.manual_seed(17); reference=OriginalScorer('attention')
    initial={k:v.clone() for k,v in reference.state_dict().items()}
    expected=[['A','P1','N1'],['P1','N1','A'],['P1','A','N1'],
              ['P1','N1','P2','N2','A'],['P1','N1','A','P2','N2'],['A','P1','N1','P2','N2']]
    extra=None; results=[]
    for i,layout in enumerate(LAYOUTS):
        torch.manual_seed(17); model=CandidateScorer(layout=layout)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items())
        assert sum(p.numel() for p in model.parameters())==(272001 if i<3 else 338177)
        if i>=3:
            state={k:v for k,v in model.state_dict().items() if k.startswith(('ff2.','norm3.'))}
            if extra is None: extra=state
            assert all(torch.equal(extra[k],v) for k,v in state.items())
        trace=[]; hooks=[]
        modules={'A':model.attention,'P1':model.ff,'N1':model.norm2}
        if i>=3: modules.update({'P2':model.ff2,'N2':model.norm3})
        for label,module in modules.items():
            hooks.append(module.register_forward_hook(lambda m,inp,out,label=label:trace.append(label)))
        model.eval()
        with torch.no_grad(): model(x)
        for hook in hooks: hook.remove()
        assert trace==expected[i],(layout,trace)
        checks=semantic_checks(model,x,torch.zeros(13,0))
        assert checks['other_candidate_perturbation_max_abs_effect']>1e-7
        model.train(); F.binary_cross_entropy_with_logits(model(x),y).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if i==0:
            reference.eval(); model.eval()
            with torch.no_grad(): torch.testing.assert_close(model(x),reference(x),atol=0,rtol=0)
            model.zero_grad(set_to_none=True); reference.zero_grad(set_to_none=True)
            model.train(); reference.train(); state=torch.get_rng_state()
            ref_scores=reference(x); F.binary_cross_entropy_with_logits(ref_scores,y).backward()
            torch.set_rng_state(state)
            scores=model(x); F.binary_cross_entropy_with_logits(scores,y).backward()
            torch.testing.assert_close(scores,ref_scores,atol=0,rtol=0)
            for p,q in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
        results.append({'layout':layout,'parameters':sum(p.numel() for p in model.parameters()),
                        'operation_trace':trace,'common_initialization':True,'checks':checks})
    print(json.dumps({'passed':True,'baseline_train_eval_and_gradient_exact':True,'configurations':results},indent=2))


if __name__=='__main__': main()
