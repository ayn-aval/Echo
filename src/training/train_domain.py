"""Phase 4 — continue fine-tuning the Phase 3 encoder on Swiggy review pairs.

    python -m src.training.train_domain --pairs data/phase4_pairs.jsonl \
                                        --encoder models/sbert-distilroberta-300k/encoder \
                                        --out models/sbert-domain

Unlike Phase 3, `sentence-transformers` IS allowed here — the raw-PyTorch rule
existed so that reproducing the paper meant something, and that is done.

MultipleNegativesRankingLoss in one paragraph: each batch holds N pairs (a_i, b_i)
that we believe belong together. The loss pushes a_i's vector toward b_i and away
from every *other* b in the same batch, which it assumes are unrelated. So one
labelled positive silently buys N-1 negatives, which is why the batch size matters
more here than in ordinary training — a bigger batch is a harder, more useful
problem. The assumption is imperfect: if two pairs in a batch happen to be about
the same theme, one true positive gets pushed away as a negative. At batch 64 over
53k pairs that is rare enough to accept, and it is the standard tradeoff.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer import losses, modules
from torch.utils.data import DataLoader

from src.utils.device import get_device


def build_model(encoder_path: Path, max_seq_length: int):
    """Wrap the Phase 3 encoder, keeping the pooling it was trained with.

    Phase 3 saved a plain HuggingFace model plus a pooling.json, not a
    sentence-transformers bundle, so the two modules are assembled by hand.
    Reading pooling.json rather than hardcoding "mean" means an encoder trained
    with max or cls pooling would still be served correctly.
    """
    meta = encoder_path / "pooling.json"
    pooling = json.loads(meta.read_text())["pooling"] if meta.exists() else "mean"
    word = modules.Transformer(str(encoder_path), max_seq_length=max_seq_length)
    # renamed in sentence-transformers 6; the old name still works but warns.
    dim = (word.get_embedding_dimension() if hasattr(word, "get_embedding_dimension")
           else word.get_word_embedding_dimension())
    pool = modules.Pooling(dim, pooling_mode={"mean": "mean", "max": "max",
                                              "cls": "cls"}[pooling])
    print(f"encoder: {encoder_path}  pooling: {pooling}  dim: {dim}")
    return SentenceTransformer(modules=[word, pool]), pooling


def load_pairs(path: Path, mined_repeat: int, seed: int):
    """Read the JSONL written by src/training/mine_pairs.py."""
    rows = [json.loads(line) for line in path.open()]
    examples, counts = [], {}
    for r in rows:
        n = mined_repeat if r["source"] == "mined" else 1
        counts[r["source"]] = counts.get(r["source"], 0) + n
        examples.extend([InputExample(texts=[r["a"], r["b"]])] * n)
    random.Random(seed).shuffle(examples)
    print(f"pairs: {len(examples):,} total — " +
          " · ".join(f"{k} {v:,}" for k, v in sorted(counts.items())))
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", type=Path, default=Path("data/phase4_pairs.jsonl"))
    ap.add_argument("--encoder", type=Path,
                    default=Path("models/sbert-distilroberta-300k/encoder"))
    ap.add_argument("--out", type=Path, default=Path("models/sbert-domain"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq-length", type=int, default=128)
    ap.add_argument("--mined-repeat", type=int, default=1,
                    help="oversample the mined pairs; simcse outnumbers them 6:1")
    ap.add_argument("--steps-per-epoch", type=int, default=None,
                    help="cut the run short — for smoke tests only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    model, pooling = build_model(args.encoder, args.max_seq_length)
    examples = load_pairs(args.pairs, args.mined_repeat, args.seed)

    # drop_last matters: a short final batch gives MNRL fewer negatives, so the
    # last step would be measured against an easier problem than every other.
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size,
                        drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    steps = args.steps_per_epoch or len(loader)
    warmup = int(steps * args.epochs * 0.10)      # 10%, same as Phase 3
    # fp16 is a real speedup on Kaggle's T4 but unreliable on Apple Silicon,
    # so it follows the device rather than a flag someone has to remember.
    use_amp = device.type == "cuda"
    print(f"device: {device} · batch {args.batch_size} · lr {args.lr} · "
          f"{steps:,} steps/epoch · warmup {warmup:,} · amp {use_amp}")

    model.fit(train_objectives=[(loader, loss)],
              epochs=args.epochs,
              steps_per_epoch=args.steps_per_epoch,
              warmup_steps=warmup,
              optimizer_params={"lr": args.lr},
              use_amp=use_amp,
              show_progress_bar=True,
              output_path=None)

    args.out.mkdir(parents=True, exist_ok=True)
    # Saved in the same shape Phase 3 used, so src/embeddings/sbert.py and the
    # whole eval harness load it unchanged — no new code path to evaluate it.
    encoder = args.out / "encoder"
    model[0].auto_model.save_pretrained(encoder)
    model[0].tokenizer.save_pretrained(encoder)
    (encoder / "pooling.json").write_text(json.dumps({"pooling": pooling}))
    print(f"\nsaved encoder -> {encoder}")


if __name__ == "__main__":
    main()
