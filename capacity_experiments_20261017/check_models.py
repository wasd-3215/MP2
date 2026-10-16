"""Width/dropout configurations and non-interfering clean diagnostics."""
from support import semantic_checks, HERE
import json
import torch
from models import CandidateScorer
from measurements import measure


def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    torch.manual_seed(19)
    x=torch.randn(13,4,1024)
    labels=torch.tensor([[1.,0,1,0]]*13)
    results=[]
    for name in protocol['models']:
        counts={}
        for variant,cfg in protocol['variants'].items():
            torch.manual_seed(17)
            model=CandidateScorer(name,hidden=cfg['hidden'],dropout=cfg['dropout'])
            count=sum(p.numel() for p in model.parameters())
            counts[variant]=count
            model.eval()
            before=torch.get_rng_state().clone()
            state={k:v.clone() for k,v in model.state_dict().items()}
            scores,metrics=measure(model,x,labels)
            repeat,again=measure(model,x,labels)
            assert torch.equal(scores,repeat) and metrics==again
            assert torch.equal(before,torch.get_rng_state())
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
            checks=semantic_checks(model,x,torch.zeros(13,0))
            model.train()
            try:
                measure(model,x,labels)
            except ValueError:
                pass
            else:
                raise AssertionError('Accepted dropout-active model for clean measurement')
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),labels)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            results.append({'model':name,'variant':variant,'parameters':count,'checks':checks})
        assert counts['width32']<counts['width64']<counts['baseline']
        assert counts['dropout30']==counts['decay01']==counts['baseline']
    print(json.dumps({'passed':True,'configurations':results},indent=2))


if __name__=='__main__':
    main()
