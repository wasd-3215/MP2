"""Initialization, feature placement, sampling and deterministic routing tests."""
import runtime
from support import OUT,write
from method import layout,decide,predict
from models import Scorer
from sampling import CyclicSampler
import numpy as np
import torch

def attention_input(model,x,f):
    captured=[]
    handle=model.base.attention.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    with torch.no_grad():z=model(x,f)
    handle.remove();return z,captured[0]
def main():
    torch.manual_seed(123);x=torch.randn(13,4,1024);f=torch.randn(13,4,6);f2=f.clone();f2[...,1:]+=torch.randn_like(f2[...,1:])*3
    models=[];rows=[]
    for feature in ('P','C'):
        for position in ('Late','Early'):
            torch.manual_seed(17);model=Scorer(feature,position).eval();models.append(model)
            with torch.no_grad():initial=model(x,f)
            assert sum(v.numel() for v in model.parameters())==272769
            if len(models)>1:
                assert all(torch.equal(v,models[0].state_dict()[k]) for k,v in model.state_dict().items())
                with torch.no_grad():assert torch.equal(initial,models[0](x,f))
            with torch.no_grad():model.feature_projection.weight.normal_(0,.05)
            a,q1=attention_input(model,x,f);b,q2=attention_input(model,x,f2)
            if feature=='P':assert torch.equal(a,b) and torch.equal(q1,q2)
            else:
                assert not torch.equal(a,b)
                if position=='Late':assert torch.equal(q1,q2)
                else:assert not torch.equal(q1,q2)
            perm=[2,0,3,1]
            with torch.no_grad():
                assert torch.max(torch.abs(a[:,perm]-model(x[:,perm],f[:,perm]))).item()<3e-5
                assert torch.max(torch.abs(a[:1]-model(x[:1],f[:1]))).item()<3e-5
            model.train();model.zero_grad(set_to_none=True);model(x,f).square().mean().backward()
            assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
            if feature=='P':assert torch.count_nonzero(model.feature_projection.weight.grad[:,1:]).item()==0
            rows.append({'feature':feature,'position':position,'attention_directly_feature_conditioned':feature=='C' and position=='Early'})
            # Restore zero projection for cross-configuration initialization comparisons.
            with torch.no_grad():model.feature_projection.weight.zero_()
            model.eval()
    s=CyclicSampler(np.arange(7),123);draw=torch.cat([s.draw(5),s.draw(8),s.draw(8)])
    for chunk in draw.reshape(3,7):assert sorted(chunk.tolist())==list(range(7))
    t=CyclicSampler(np.arange(7),123);assert torch.equal(draw,t.draw(21))
    bank=[{'candidates':[{'text':v} for v in texts]} for texts in [
        ['bad','#### 0.5','#### 1/2','bad'],['bad','#### 2','#### 3','bad'],['bad']*4]]
    _,valid,groups=layout(bank);assert groups.tolist()==[1,2,0]
    scores=np.array([[.9,.2,.8,.1],[.9,.4,.7,.1],[.2,.3,.1,.4]])
    assert decide(bank,scores).tolist()==[1,2,3]
    # No checkpoint is read and no neural model is loaded for a fully easy bank.
    actual=predict(bank[:1],{'embeddings':np.zeros((1,4,1024),np.float32)},'nonexistent-checkpoint')
    assert actual==[1] and type(actual[0]) is int
    OUT.mkdir(exist_ok=True)
    write(OUT/'implementation_checks.json',{'passed':True,'configurations':rows,'matched_parameters':272769,
        'zero_injection_identical_initial_state_and_function':True,'Late_attention_inputs_not_conditioned':True,
        'Early_attention_inputs_conditioned':True,'P_ignores_consensus':True,'finite_gradients':True,
        'permutation_and_single_batch':True,'cyclic_sampler_complete_passes_and_leftovers':True,
        'input_only_routing_parser_equivalence':True,'fully_easy_bank_without_model_loading':True,'all_invalid_neural_fallback':True})
    print('ALL MODEL, ROUTING AND SAMPLER CHECKS PASSED',flush=True)
if __name__=='__main__':main()
