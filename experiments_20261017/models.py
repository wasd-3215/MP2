"""Shared candidate scorers. No bank identifiers, labels, or file IO."""
import torch
from torch import nn

MODEL_NAMES = ("linear", "independent", "pool", "attention", "independent_large",
               "attention_self", "independent_question", "attention_question")


class CandidateScorer(nn.Module):
    def __init__(self, name, hidden=128, dropout=0.1, question_dim=0):
        super().__init__()
        if name not in MODEL_NAMES:
            raise ValueError(name)
        self.name = name
        self.has_question = name.endswith("_question")
        self.has_attention = name.startswith("attention")
        self.encoder = nn.Sequential(nn.Linear(1024, hidden), nn.GELU(), nn.Dropout(dropout)) \
            if name != "linear" else nn.Identity()
        if name == "linear":
            self.head = nn.Linear(1024, 1)
            return
        if self.has_question:
            self.question_encoder = nn.Sequential(nn.Linear(question_dim, 32), nn.GELU())
            self.condition = nn.Sequential(nn.Linear(hidden + 32, hidden), nn.GELU(),
                                           nn.Dropout(dropout))
        if self.has_attention:
            self.attention = nn.MultiheadAttention(hidden, 4, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden)
            self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))
            self.norm2 = nn.LayerNorm(hidden)
            self.residual_dropout = nn.Dropout(dropout)
        if name == "independent_large":
            self.extra = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(4 * hidden, hidden))
            self.extra_norm = nn.LayerNorm(hidden)
        head_dim = 2 * hidden if name == "pool" else hidden
        self.head = nn.Sequential(nn.Linear(head_dim, 64), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(64, 1))

    def forward(self, x, question=None):
        h = self.encoder(x)
        if self.name == "linear":
            return self.head(h).squeeze(-1)
        if self.has_question:
            t = self.question_encoder(question).unsqueeze(1).expand(-1, h.shape[1], -1)
            h = self.condition(torch.cat((h, t), dim=-1))
        if self.name == "pool":
            c = h.mean(dim=1, keepdim=True).expand_as(h)
            h = torch.cat((h, c), dim=-1)
        elif self.has_attention:
            mask = None
            if self.name == "attention_self":
                mask = ~torch.eye(h.shape[1], dtype=torch.bool, device=h.device)
            message, _ = self.attention(h, h, h, attn_mask=mask, need_weights=False)
            h = self.norm1(h + self.residual_dropout(message))
            h = self.norm2(h + self.residual_dropout(self.ff(h)))
        elif self.name == "independent_large":
            h = self.extra_norm(h + self.extra(h))
        return self.head(h).squeeze(-1)

