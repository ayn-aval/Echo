"""Publish the trained encoder and its review vectors to the Hugging Face Hub.

    python -m src.embeddings.publish --user YOUR_HF_NAME --dry-run
    python -m src.embeddings.publish --user YOUR_HF_NAME

Why this exists: the deployed app cannot carry these files. GitHub rejects any
single file over 100 MB, and the encoder is 327 MB while the vectors are 141 MB.
The Hub is built to serve exactly this, is free, and gives the model a public
page of its own.

Two repositories, because they are two different kinds of thing:

    <user>/echo-sbert-domain     model    the encoder — config, weights, tokenizer
    <user>/echo-review-vectors   dataset  the 45,864 review vectors + their texts

Vectors are cast to float16 on the way up, which halves 141 MB to 70 MB. That was
verified rather than assumed: over 500 probe queries the top-1 result is identical
100% of the time and the same ten reviews come back every time, occasionally
reordered inside the ten.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENCODER = ROOT / "models/sbert-domain/encoder"
VECTORS = ROOT / "data/vectors/sbert-domain.npy"
CORPUS = ROOT / "data/vectors/corpus.parquet"

CARD = """---
license: apache-2.0
language: [en]
library_name: transformers
pipeline_tag: sentence-similarity
tags: [sentence-transformers, sentence-similarity, feature-extraction, sbert]
---

# echo-sbert-domain

A sentence embedding model for **short, messy, real-world app-store reviews**.

Built for [Echo](https://github.com/ayn-aval/Echo), which turns 100,000 Swiggy
Google Play reviews into tracked themes. Trained in two stages:

1. **Sentence-BERT reproduction.** `distilroberta-base` on 300,000 SNLI +
   MultiNLI pairs, siamese, mean pooling, classifier on `(u, v, |u-v|)` — written
   by hand in raw PyTorch, reproducing
   [Reimers & Gurevych (2019)](https://arxiv.org/abs/1908.10084).
2. **Domain adaptation.** Continued with `MultipleNegativesRankingLoss` on 53,061
   pairs mined from the reviews themselves, where TF-IDF and the stage-1 encoder
   independently agree, plus SimCSE dropout self-pairs.

## Results

| benchmark | score |
|---|---|
| STS average (7 datasets) after stage 1 | 72.17 |
| STS average after stage 2 | **74.54** |
| Review retrieval, Precision@10 | 61.15 |
| Review retrieval, + cross-encoder rerank | **75.77** |
| Theme assignment, blind hand-audit | 82.4% |

Stage 2 improved generic STS by +2.37 while adapting to the domain, which was
predicted to degrade and did not. Note that 74.54 is **not** "beating the paper's
74.21": that number comes from NLI training alone, this adds a second stage.

## Limitations, stated plainly

- **It does not bridge Hinglish.** "khana thanda tha" against "the food was cold"
  scores 0.066, versus 0.049 for a genuinely unrelated pair. Romanised Hindi
  reviews cluster by *language* rather than by subject.
- **On its own it loses to TF-IDF** for review retrieval (61.15 vs 65.00). It only
  wins in front of a cross-encoder reranker.
- **Part of the retrieval gain may be circular** — mined pairs required TF-IDF to
  agree, so the model may have partly learned to imitate it.
- Retrieval numbers rest on **26 hand-judged queries**, one judge.

## Use

```python
from transformers import AutoModel, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModel.from_pretrained("{repo}").eval()

def embed(sentences):
    x = tok(sentences, padding=True, truncation=True, max_length=128,
            return_tensors="pt")
    with torch.no_grad():
        h = model(**x).last_hidden_state
    mask = x["attention_mask"].unsqueeze(-1).float()      # mean pooling,
    v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # ignoring padding
    return torch.nn.functional.normalize(v, dim=1)
```

**Mean pooling, and it matters** — this model was trained with it. CLS pooling
scores 5.1 points lower in the ablation.
"""

DATASET_CARD = """---
license: apache-2.0
task_categories: [feature-extraction]
language: [en]
---

# echo-review-vectors

Precomputed embeddings for the 45,864 distinct review texts in
[Echo](https://github.com/ayn-aval/Echo), encoded with
[{model_repo}](https://huggingface.co/{model_repo}).

| file | what it is |
|---|---|
| `sbert-domain.fp16.npy` | `(45864, 768)` float16, L2-normalised, row *i* matches row *i* of the parquet |
| `corpus.parquet` | `row_index, content, n_rows, review_ids` |

45,864 vectors cover 64,280 review rows, because identical texts share one vector.
`corpus.parquet` is what joins a vector back to its reviews — without it the `.npy`
is an anonymous array.

Stored as float16 to halve the download. Verified over 500 probe queries: identical
top-1 result 100% of the time, and the same ten neighbours returned every time.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", required=True, help="your Hugging Face username")
    ap.add_argument("--dry-run", action="store_true",
                    help="check everything, upload nothing")
    args = ap.parse_args()

    from huggingface_hub import HfApi, whoami

    missing = [p for p in (ENCODER, VECTORS, CORPUS) if not p.exists()]
    if missing:
        raise SystemExit("Missing locally:\n" + "\n".join(f"  {p}" for p in missing))

    who = whoami()["name"]
    print(f"logged in as: {who}")
    if who != args.user:
        print(f"  note: uploading to '{args.user}', which is not your username. "
              f"That only works if it is an organisation you belong to.")

    model_repo = f"{args.user}/echo-sbert-domain"
    data_repo = f"{args.user}/echo-review-vectors"

    vectors = np.load(VECTORS).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    print(f"\nencoder     {sum(f.stat().st_size for f in ENCODER.iterdir())/1e6:6.1f} MB "
          f"-> {model_repo}")
    print(f"vectors     {vectors.astype(np.float16).nbytes/1e6:6.1f} MB (float16) "
          f"-> {data_repo}")
    print(f"corpus      {CORPUS.stat().st_size/1e6:6.1f} MB -> {data_repo}")

    if args.dry_run:
        print("\ndry run — nothing uploaded")
        return

    api = HfApi()
    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    api.create_repo(data_repo, repo_type="dataset", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "README.md").write_text(CARD.format(repo=model_repo))
        api.upload_folder(folder_path=str(ENCODER), repo_id=model_repo,
                          repo_type="model", commit_message="Trained encoder")
        api.upload_file(path_or_fileobj=str(tmp / "README.md"), path_in_repo="README.md",
                        repo_id=model_repo, repo_type="model",
                        commit_message="Model card")
        print(f"\n  https://huggingface.co/{model_repo}")

        np.save(tmp / "sbert-domain.fp16.npy", vectors.astype(np.float16))
        (tmp / "README.md").write_text(DATASET_CARD.format(model_repo=model_repo))
        api.upload_folder(folder_path=str(tmp), repo_id=data_repo,
                          repo_type="dataset", commit_message="Vectors and corpus")
        api.upload_file(path_or_fileobj=str(CORPUS), path_in_repo="corpus.parquet",
                        repo_id=data_repo, repo_type="dataset",
                        commit_message="Corpus mapping")
        print(f"  https://huggingface.co/datasets/{data_repo}")


if __name__ == "__main__":
    main()
