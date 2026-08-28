"""BERT baselines — the two ways of squeezing a sentence vector out of raw BERT.

BERT outputs one vector per token, not per sentence, so something must collapse
them. The Sentence-BERT paper tests both obvious choices and finds both poor
without fine-tuning, which is the motivation for the whole paper.

  mean  average of the token vectors, weighted by the attention mask
  cls   the first token's vector, which BERT was pre-trained to use for
        classification and which is often wrongly assumed to be a sentence vector

sentence-transformers is deliberately not used here — Phase 3 reproduces it by hand.
"""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.device import get_device

MODEL = "bert-base-uncased"


def make_encoder(pooling: str = "mean", model_name: str = MODEL,
                 batch_size: int = 64, max_length: int = 128, quiet: bool = False):
    if pooling not in ("mean", "cls"):
        raise ValueError("pooling must be 'mean' or 'cls'")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def encode(sentences):
        chunks = []
        steps = range(0, len(sentences), batch_size)
        for start in tqdm(steps, desc=f"  bert-{pooling}", unit="batch",
                          disable=quiet, leave=False):
            batch = [s if s else " " for s in sentences[start:start + batch_size]]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**inputs).last_hidden_state

            if pooling == "cls":
                vectors = hidden[:, 0]
            else:
                # Weight by the attention mask: padding tokens must contribute
                # nothing. Averaging over padding is the classic silent bug here —
                # it does not crash, it just quietly lowers every score.
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

            chunks.append(vectors.float().cpu().numpy())
        return np.concatenate(chunks)

    return encode
