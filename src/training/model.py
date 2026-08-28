"""The siamese architecture from Sentence-BERT, written out by hand.

Deliberately not sentence-transformers. Reproducing this is the point of the
project, and the whole architecture is about 60 lines once you see it.

The shape:

    sentence A -> encoder -> pool -> u  \\
                                         concat(u, v, |u-v|) -> Linear -> 3 classes
    sentence B -> encoder -> pool -> v  /

"encoder" is the SAME module both times, not a copy. That is what "siamese" means
and it is why `u` and `v` land in one comparable space. Two separate encoders would
drift into different coordinate systems and cosine similarity between them would be
meaningless.

After training the head is thrown away and only the encoder is kept.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

POOLINGS = ("mean", "max", "cls")

# The concatenation variants from the paper's Table 6, as data rather than seven
# near-identical classes — the ablation is then a config change, not a rewrite.
VARIANTS = {
    "u,v":           ("u", "v"),
    "|u-v|":         ("diff",),
    "u*v":           ("prod",),
    "|u-v|,u*v":     ("diff", "prod"),
    "u,v,u*v":       ("u", "v", "prod"),
    "u,v,|u-v|":     ("u", "v", "diff"),
    "u,v,|u-v|,u*v": ("u", "v", "diff", "prod"),
}


def pool(hidden, mask, how):
    """Collapse per-token vectors into one sentence vector."""
    if how == "cls":
        return hidden[:, 0]
    mask = mask.unsqueeze(-1)
    if how == "max":
        # Padding must not win a max. Fill it with a large negative instead of
        # -inf so fp16 autocast cannot produce NaN.
        return hidden.masked_fill(mask == 0, -1e4).max(dim=1).values
    # Mean, weighted by the mask. Averaging over padding is the silent bug that
    # cost points in Phase 2: it never raises, it just dilutes short sentences.
    mask = mask.to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class SiameseNLI(nn.Module):
    def __init__(self, model_name, pooling="mean", variant="u,v,|u-v|"):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.variant = variant
        self.parts = VARIANTS[variant]
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(len(self.parts) * hidden, 3)

    def embed(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return pool(out.last_hidden_state, attention_mask, self.pooling)

    def combine(self, u, v):
        pieces = {"u": u, "v": v, "diff": (u - v).abs(), "prod": u * v}
        return torch.cat([pieces[p] for p in self.parts], dim=-1)

    def forward(self, a_ids, a_mask, b_ids, b_mask):
        u = self.embed(a_ids, a_mask)   # same weights
        v = self.embed(b_ids, b_mask)   # same weights, same module
        return self.head(self.combine(u, v)), u, v
