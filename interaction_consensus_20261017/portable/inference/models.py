"""3 architectures x 2 information conditions, six equal feature slots."""
import torch
from torch import nn
from reference_models import CandidateScorer


class Scorer(nn.Module):
    def __init__(self, architecture, feature_mode):
        super().__init__()
        if architecture not in ('I', 'Pool', 'S2') or feature_mode not in ('P', 'C'):
            raise ValueError((architecture, feature_mode))
        self.architecture = architecture
        self.feature_mode = feature_mode
        name = {'I': 'independent', 'Pool': 'pool', 'S2': 'attention'}[architecture]
        self.base = CandidateScorer(name, 128, .1, 0)
        old = self.base.head[0]
        new = nn.Linear(old.in_features + 6, old.out_features)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_features].copy_(old.weight)
            new.bias.copy_(old.bias)
        self.base.head[0] = new
        self.register_buffer('feature_mask', torch.tensor([1.] + [float(feature_mode == 'C')] * 5), persistent=False)

    def forward(self, x, features):
        b = self.base
        h = b.encoder(x)
        if self.architecture == 'Pool':
            h = torch.cat([h, h.mean(1, keepdim=True).expand_as(h)], dim=-1)
        elif self.architecture == 'S2':
            h = h + b.residual_dropout(b.ff(h))
            message, _ = b.attention(h, h, h, need_weights=False)
            h = b.norm1(h + b.residual_dropout(message))
            h = b.norm2(h)
        # P's last five slots are identically zero AFTER standardization.
        features = features * self.feature_mask
        return b.head(torch.cat([h, features], dim=-1)).squeeze(-1)
