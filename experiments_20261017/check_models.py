"""Behavior checks for gradient flow, symmetry, and train-only text features."""
import json

import run_suite as suite
import numpy as np
import torch

from features import fit_question_features, transform_questions
from models import MODEL_NAMES, CandidateScorer


def main():
    torch.manual_seed(5)
    x = torch.randn(17, 4, 1024)
    q = torch.randn(17, 1024)
    y = torch.tensor([[1, 0, 1, 0]] * 17, dtype=torch.float32)
    results = {}
    for name in MODEL_NAMES:
        model = CandidateScorer(name, question_dim=1024 if name.endswith("_question") else 0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x, q), y)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        result = suite.semantic_checks(model, x, q)
        if name in ("pool", "attention", "attention_question"):
            assert result["other_candidate_perturbation_max_abs_effect"] > 1e-6
        if model.has_question:
            model.eval()
            with torch.inference_mode():
                changed = model(x, q.roll(1, 0)) - model(x, q)
                # Question effect must change relative scores, not only a shared offset.
                assert (changed - changed.mean(1, keepdim=True)).abs().max() > 1e-6
        results[name] = {**result, "parameters": sum(p.numel() for p in model.parameters())}
    vocabulary, idf = fit_question_features([{"question": "A bus travels 60 miles"}] * 2)
    unknown = transform_questions([{"question": "zzzzqqqq"}], vocabulary, idf)
    assert np.isfinite(unknown).all() and not unknown.any()
    assert "zzzzqqqq" not in vocabulary
    print(json.dumps({"models": results, "unknown_question_check": "passed"}, indent=2))


if __name__ == "__main__":
    main()
