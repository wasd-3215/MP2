"""Analytical and gradient checks for ranking losses and budget sampler."""
import runtime
runtime.configure_torch()
import json
import math
import torch
import torch.nn.functional as F
from losses import objective_loss, QuestionSampler, OBJECTIVES


def main():
    s = torch.tensor([[2., -1., .5, 0.], [1., 3., -2., 0.]], dtype=torch.double, requires_grad=True)
    y = torch.tensor([[1., 0., 0., 0.], [1., 1., 0., 0.]], dtype=torch.double)
    expected = torch.stack([torch.stack([F.softplus(s[k,j]-s[k,i])
        for i in range(4) for j in range(4) if y[k,i] == 1 and y[k,j] == 0]).mean()
        for k in range(2)]).mean()
    torch.testing.assert_close(objective_loss(s,y,'pairwise'),expected)
    mass = -(torch.softmax(s,1) * y).sum(1).log().mean()
    torch.testing.assert_close(objective_loss(s,y,'set_mass'),mass)
    perm = [2,0,3,1]
    for obj in OBJECTIVES:
        torch.testing.assert_close(objective_loss(s,y,obj),objective_loss(s[:,perm],y[:,perm],obj))
        assert torch.autograd.gradcheck(lambda z: objective_loss(z,y,obj),(s,))
        gradient = torch.autograd.grad(objective_loss(s,y,obj),s)[0]
        assert (gradient[y == 1] < 0).all() and (gradient[y == 0] > 0).all()
        try:
            objective_loss(s, torch.zeros_like(y), obj)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted no-positive row')
    for obj in ['pairwise','set_mass']:
        torch.testing.assert_close(objective_loss(s,y,obj),objective_loss(s+83,y,obj))
        all_correct = objective_loss(s,torch.ones_like(y),obj)
        assert all_correct.item() == 0
        assert torch.autograd.grad(all_correct,s)[0].abs().sum() == 0
        extreme = s.detach().mul(10000).requires_grad_()
        loss = objective_loss(extreme,y,obj)
        assert torch.isfinite(loss) and torch.isfinite(torch.autograd.grad(loss,extreme)[0]).all()
    extra_s = torch.cat([s, s[:1]],0)
    extra_y = torch.cat([y, torch.ones_like(y[:1])],0)
    for obj in ['bce_mixed','pairwise','set_mass']:
        torch.testing.assert_close(objective_loss(extra_s,extra_y,obj),objective_loss(s,y,obj))
    zero = torch.zeros((2,4),dtype=torch.double)
    torch.testing.assert_close(objective_loss(zero,y,'pairwise'),torch.tensor(math.log(2),dtype=torch.double))
    sampler_labels = torch.tensor([[1,1,1,1],[1,0,0,0],[0,1,1,1],[1,1,1,1],[0,0,1,0.]])
    mixed = [QuestionSampler(sampler_labels,obj,17) for obj in ['bce_mixed','pairwise','set_mass']]
    full = QuestionSampler(sampler_labels,'bce_all',17)
    sampled = []
    for _ in range(10):
        draws = [sampler.draw(5) for sampler in mixed]
        assert all(torch.equal(draws[0],other) for other in draws[1:])
        assert (sampler_labels[draws[0]].sum(1)<4).all()
        assert torch.equal(full.draw(5).sort().values,torch.arange(5))
        sampled.extend(draws[0].tolist())
    counts = [sampled.count(i) for i in [1,2,4]]
    assert max(counts)-min(counts)<=1
    print(json.dumps({'passed':True,'checks':['manual pair averaging','set mass identity','permutation',
        'finite-difference gradients','gradient signs','no-positive rejection','shift invariance',
        'all-positive zero and exclusion','extreme finite scores','matched deterministic sampler']}))


if __name__ == '__main__':
    main()
