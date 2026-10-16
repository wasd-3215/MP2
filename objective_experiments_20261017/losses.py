"""Question-balanced objectives; all-positive rows have zero ranking loss."""
import torch
import torch.nn.functional as F

OBJECTIVES = ("bce_all", "bce_mixed", "pairwise", "set_mass")


def objective_loss(scores, labels, objective):
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective: {objective}")
    if scores.ndim != 2 or scores.shape != labels.shape or scores.shape[1] != 4:
        raise ValueError("Expected aligned (N, 4) scores and binary labels")
    if not torch.isfinite(scores).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Scores must be finite and labels binary")
    positive = labels.bool()
    if not positive.any(dim=1).all():
        raise ValueError("Every question needs at least one correct candidate")
    if objective == "bce_all":
        return F.binary_cross_entropy_with_logits(scores, labels) if len(scores) else scores.sum() * 0
    mixed = ~positive.all(dim=1)
    if not mixed.any():
        return scores.sum() * 0
    scores, labels, positive = scores[mixed], labels[mixed], positive[mixed]
    if objective == "bce_mixed":
        return F.binary_cross_entropy_with_logits(scores, labels)
    if objective == "set_mass":
        return (torch.logsumexp(scores, dim=1) -
                torch.logsumexp(scores.masked_fill(~positive, -torch.inf), dim=1)).mean()
    # Axis 1 is positive i; axis 2 is negative j. Normalize within each question.
    pairs = positive.unsqueeze(2) & ~positive.unsqueeze(1)
    differences = scores.unsqueeze(1) - scores.unsqueeze(2)
    per_question = (F.softplus(differences) * pairs).sum(dim=(1, 2)) / pairs.sum(dim=(1, 2))
    return per_question.mean()


class QuestionSampler:
    """Equal draw/step budgets, with shuffled cycling for the mixed-only pool."""
    def __init__(self, labels, objective, seed):
        if objective not in OBJECTIVES:
            raise ValueError(objective)
        self.full = objective == "bce_all"
        eligible = torch.ones(len(labels), dtype=torch.bool) if self.full else labels.sum(1) < 4
        self.pool = torch.where(eligible)[0]
        if not len(self.pool):
            raise ValueError("No eligible training questions")
        self.generator = torch.Generator().manual_seed(seed + 10000)
        self.pending = torch.empty(0, dtype=torch.long)

    def draw(self, count):
        if self.full:
            if count != len(self.pool):
                raise ValueError("Full BCE epoch must use each question once")
            return self.pool[torch.randperm(len(self.pool), generator=self.generator)]
        pieces, remaining = [], count
        while remaining:
            if not len(self.pending):
                self.pending = self.pool[torch.randperm(len(self.pool), generator=self.generator)]
            take = min(remaining, len(self.pending))
            pieces.append(self.pending[:take])
            self.pending = self.pending[take:]
            remaining -= take
        return torch.cat(pieces)
