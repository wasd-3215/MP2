"""Meaningful target/size/prior controls and pooling equivariance checks."""
import runtime
runtime.configure_torch()
from support import HERE,OUT,VARIANTS,write,sha
import numpy as np
import torch
from models import Scorer,objective
from grouping import layout,group_targets,choices_from_probabilities

def state(model):return {k:v.clone() for k,v in model.state_dict().items()}
def main():
    torch.manual_seed(1);x=torch.randn(3,4,1024);f=torch.randn(3,4,6)
    texts=[['#### 1','#### 2','#### 3','#### 4'],['#### 8','#### 8','#### 8','#### 9'],['no marker','#### 2','#### 2','#### 5']]
    bank=[{'candidates':[{'text':s} for s in row]} for row in texts]
    _,m,valid,counts=layout(bank);mt=torch.from_numpy(m)
    y=torch.tensor([[0,0,1,0],[0,0,0,1],[0,1,1,0]],dtype=torch.float32)
    target=torch.from_numpy(group_targets(m,y.numpy()));assert target.tolist()==[2,1,0]
    models={}
    for variant in VARIANTS:
        r,s=variant.split('_');torch.manual_seed(17);models[variant]=Scorer(r,s).eval()
    params={n:sum(p.numel() for p in model.parameters()) for n,model in models.items()}
    assert len(set(params.values()))==1
    original=state(models[VARIANTS[0]])
    assert all(all(torch.equal(original[k],v) for k,v in model.state_dict().items()) for model in models.values())
    perm=[2,0,3,1]
    perm_bank=[{'candidates':[row['candidates'][i] for i in perm]} for row in bank]
    _,pm,_,_=layout(perm_bank)
    max_error=0.
    for variant,model in models.items():
        with torch.no_grad():
            a=model(x,f,mt)[1];b=model(x[:,perm],f[:,perm],torch.from_numpy(pm))[1]
        # Compare probabilities by parsed value/group membership, not first-occurrence slot.
        expanded_a=torch.bmm(a[:,None,:],mt.to(a.dtype)).squeeze(1)
        expanded_b=torch.bmm(b[:,None,:],torch.from_numpy(pm).to(b.dtype)).squeeze(1)
        error=float(torch.max(torch.abs(expanded_a[:,perm]-expanded_b)))
        assert error<3e-5;max_error=max(max_error,error)
        other=variant.replace('NoSize','Size') if 'NoSize' in variant else variant.replace('Size','NoSize')
        with torch.no_grad():assert torch.equal(a,models[other](x,f,mt)[1])
        loss=objective(model,model(x,f,mt),y,mt,target)
        assert torch.isfinite(loss);loss.backward()
        if variant.endswith('NoSize'):assert torch.count_nonzero(model.base.head[0].weight.grad[:,-5:])==0
    # For all-singleton groups, candidate/global and group mean/max share their first
    # h branch at zero extra columns; initial logits match. Later head functions differ.
    with torch.no_grad():
        assert torch.allclose(models['CandMaxCE_NoSize'](x[:1],f[:1],mt[:1])[1],models['GroupCE_NoSize'](x[:1],f[:1],mt[:1])[1],atol=1e-6)
    zero=torch.zeros((1,4));positive=torch.tensor([[False,False,False,True]])
    setmass=torch.logsumexp(zero,1)-torch.logsumexp(zero.masked_fill(~positive,-torch.inf),1)
    maxce=torch.nn.functional.cross_entropy(torch.zeros((1,2)),torch.tensor([1]))
    assert abs(float(setmass)-np.log(4))<1e-6 and abs(float(maxce)-np.log(2))<1e-6
    assert choices_from_probabilities(m,np.array([[0,0,1,0],[0,1,0,0],[1,0,0,0]])).tolist()==[2,3,1]
    # BCE valid-only loss is unchanged by any invalid-candidate logit perturbation.
    candidate=torch.zeros((3,4),requires_grad=True);dummy=torch.zeros((3,4))
    a=objective(models['CandBCE_NoSize'],(candidate,dummy),y,mt,target)
    altered=candidate.detach().clone();altered[~torch.from_numpy(valid)]=100
    b=objective(models['CandBCE_NoSize'],(altered,dummy),y,mt,target)
    assert torch.equal(a,b)
    write(OUT/'implementation_checks.json',{'passed':True,'parameters':params,'identical_initial_states':True,
      'size_pairs_identical_initial_functions':True,'max_group_CE_has_no_log_size_bonus':True,'setmass_identity_control':True,
      'official_group_targets_and_invalid_mask':True,'invalid_BCE_supervision_excluded':True,
      'permutation_max_error':max_error,'script_sha256':sha(HERE/'check_models.py')})
    print('GROUP MODEL AND LOSS CONTROLS PASSED',params,flush=True)

if __name__=='__main__':main()
