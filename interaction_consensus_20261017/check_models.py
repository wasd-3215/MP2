"""Information-path, initialization, gradient and parser constraint checks."""
import runtime
from support import OUT, ARCHITECTURES, write
from method import filter_choices
from models import Scorer
import numpy as np
import torch


def main():
    torch.manual_seed(123)
    x = torch.randn(13, 4, 1024)
    features = torch.randn(13, 4, 6)
    rows = []
    for architecture in ARCHITECTURES:
        torch.manual_seed(17); p = Scorer(architecture, 'P').eval()
        torch.manual_seed(17); c = Scorer(architecture, 'C').eval()
        assert all(torch.equal(v, c.state_dict()[k]) for k, v in p.state_dict().items())
        with torch.no_grad():
            assert torch.equal(p(x, features), c(x, features))
        # Give feature columns nonzero weights to check the feature mask itself.
        with torch.no_grad():
            p.base.head[0].weight[:, -6:].fill_(.05)
            c.base.head[0].weight[:, -6:].fill_(.05)
            f2 = features.clone(); f2[..., 1:] += 3
            assert torch.equal(p(x, features), p(x, f2))
            assert not torch.equal(c(x, features), c(x, f2))
            perm = [2, 0, 3, 1]
            for model in (p, c):
                a = model(x, features)
                b = model(x[:, perm], features[:, perm])
                assert torch.max(torch.abs(a[:, perm] - b)).item() < 3e-5
                assert torch.max(torch.abs(a[:1] - model(x[:1], features[:1]))).item() < 3e-5
            other = x.clone(); other[:, 1:] += 2
            difference = torch.max(torch.abs(p(x, features)[:, 0] - p(other, features)[:, 0])).item()
            if architecture == 'I':
                assert difference == 0
            else:
                assert difference > 1e-6
        for model in (p, c):
            model.train(); model.zero_grad(set_to_none=True)
            loss = model(x, features).square().mean()
            loss.backward()
            assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        assert torch.count_nonzero(p.base.head[0].weight.grad[:, -5:]).item() == 0
        assert torch.count_nonzero(c.base.head[0].weight.grad[:, -5:]).item() > 0
        rows.append({'architecture': architecture, 'parameters_per_PC_cell': sum(v.numel() for v in p.parameters()),
                     'PC_identical_initial_state_and_function': True, 'P_ignores_consensus_slots': True,
                     'C_can_use_consensus_slots': True, 'other_embedding_dependency_max': difference})
    bank = [{'candidates': [{'text': t} for t in texts]} for texts in [
        ['bad', '#### 1', '#### 2', 'bad'], ['bad'] * 4, ['#### 1'] * 4]]
    scores = np.array([[.9, .6, .6, .1], [.9, .1, .2, .3], [.8, .9, .1, .2]])
    assert filter_choices(bank, scores, [0, 2, 2]).tolist() == [1, 2, 2]
    OUT.mkdir(exist_ok=True)
    write(OUT / 'implementation_checks.json', {'passed': True, 'architectures': rows,
          'parser_filter_valid_choice_preserved': True, 'all_invalid_fallback': True,
          'ties_earliest': True, 'permutation_equivariance': True, 'single_batch_consistency': True,
          'finite_gradients': True, 'P_unused_feature_gradient_zero': True})
    print('MODEL CHECKS PASSED', rows, flush=True)


if __name__ == '__main__':
    main()
