"""Clean same-checkpoint metrics; diagnostics do not affect optimization."""
import torch
import torch.nn.functional as F


@torch.inference_mode()
def measure(model, x, labels):
    if model.training:
        raise ValueError('Clean measurements require model.eval()')
    scores = torch.cat([model(chunk) for chunk in x.split(256)])
    mixed = labels.sum(1)<4
    chosen = scores.argmax(1)
    hit = labels[torch.arange(len(labels)),chosen]
    return scores, {
        'bce': F.binary_cross_entropy_with_logits(scores,labels).item(),
        'mixed_bce': F.binary_cross_entropy_with_logits(scores[mixed],labels[mixed]).item(),
        'selection_accuracy_diagnostic': 100*hit.double().mean().item(),
        'mixed_selection_accuracy_diagnostic': 100*hit[mixed].double().mean().item()
    }
