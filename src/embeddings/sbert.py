"""Load a trained siamese encoder and hand it to the Phase 2 evaluation harness.

The classification head is gone by this point — training threw it away. What
remains is an encoder whose vectors are meant to be comparable by cosine distance,
which is the only thing evaluate_sts() and evaluate_retrieval() ever ask for.

    from src.embeddings.sbert import make_encoder
    encode = make_encoder("models/sbert-debug/encoder")
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.training.model import pool
from src.utils.device import get_device


# Every trained encoder the evaluation harness knows about, newest last.
# name -> (encoder path, embedding-cache name). Add a model here and it appears
# in the pool, the retrieval table and the STS table automatically.
TRAINED = {
    "sbert-distilroberta-300k": ("models/sbert-distilroberta-300k/encoder",
                                 "reviews_sbert_300k"),
    "sbert-domain": ("models/sbert-domain/encoder", "reviews_sbert_domain"),
}


def available():
    """The subset of TRAINED whose encoder is actually on disk."""
    return {n: (Path(p), c) for n, (p, c) in TRAINED.items() if Path(p).exists()}


def make_encoder(path, batch_size: int = 64, max_length: int = 128,
                 quiet: bool = False):
    path = Path(path)
    pooling = "mean"
    meta = path / "pooling.json"
    if meta.exists():
        pooling = json.loads(meta.read_text())["pooling"]

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).to(device).eval()

    @torch.no_grad()
    def encode(sentences):
        chunks = []
        steps = range(0, len(sentences), batch_size)
        for start in tqdm(steps, desc=f"  sbert-{pooling}", unit="batch",
                          disable=quiet, leave=False):
            batch = [s if s else " " for s in sentences[start:start + batch_size]]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**inputs).last_hidden_state
            # Same pooling function the model was trained with — importing it
            # rather than reimplementing keeps train and inference identical.
            vectors = pool(hidden, inputs["attention_mask"], pooling)
            chunks.append(vectors.float().cpu().numpy())
        return np.concatenate(chunks)

    return encode
