"""The siamese training loop, written by hand in raw PyTorch.

    python -m src.training.train --model nreimers/MiniLM-L6-H384-uncased --pairs 50000

Paper config (Reimers & Gurevych 2019, section 4): 1 epoch, batch 16, Adam at
2e-5, linear warmup over the first 10% of steps. Mean pooling, (u, v, |u-v|).

Checkpointing and resume exist before the first long run, not after one dies.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.training.model import SiameseNLI
from src.training.nli_data import load_nli
from src.utils.device import get_device

# The Phase 3 constraint, enforced rather than merely intended.
if "sentence_transformers" in sys.modules:
    raise RuntimeError("sentence-transformers is banned in Phase 3 — the point is "
                       "to write the siamese loop by hand.")


def collate(batch, tokenizer, max_len):
    def enc(key):
        return tokenizer([b[key] for b in batch], padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt")
    a, b = enc("premise"), enc("hypothesis")
    return (a["input_ids"], a["attention_mask"], b["input_ids"], b["attention_mask"],
            torch.tensor([x["label"] for x in batch]))


def lr_at(step, total, warmup_frac, base_lr):
    """Linear warmup then linear decay, written out rather than imported.

    Warmup matters: an untrained head sends large gradients back into pretrained
    encoder weights, and without a gentle start those first updates can wreck
    representations that took days to learn.
    """
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return base_lr * step / warmup
    return base_lr * max(0.0, (total - step) / max(1, total - warmup))


def save_checkpoint(path, model, optimizer, step, seen, cfg):
    """Atomic: write to a temp file, then rename. A disconnect mid-write cannot
    leave a half-written checkpoint that fails to load hours later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "seen": seen, "config": cfg,
                "torch_rng": torch.get_rng_state()}, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="nreimers/MiniLM-L6-H384-uncased")
    ap.add_argument("--pairs", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--variant", default="u,v,|u-v|")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/sbert-debug")
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None, help="cut the run short")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    out = Path(args.out)
    ckpt_path = out / "checkpoint.pt"
    cfg = {k: v for k, v in vars(args).items()}

    print(f"device      {device}")
    print(f"model       {args.model}")
    print(f"pooling     {args.pooling}   variant  {args.variant}")
    print(f"batch {args.batch_size}  lr {args.lr}  epochs {args.epochs}  "
          f"warmup {args.warmup:.0%}")

    data = load_nli(args.pairs, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = SiameseNLI(args.model, args.pooling, args.variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_step, seen = 0, 0
    if args.resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, seen = state["step"], state["seen"]
        torch.set_rng_state(state["torch_rng"].cpu())
        print(f"resumed at step {start_step:,} ({seen:,} pairs already seen)")

    # Resume by slicing the deterministically shuffled dataset rather than
    # spinning the DataLoader forward, which would waste minutes on a long run.
    if seen:
        data = data.select(range(seen % len(data), len(data)))

    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, tokenizer, args.max_len))
    total_steps = math.ceil(len(data) / args.batch_size) * args.epochs + start_step
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    use_amp = device.type == "cuda"   # CLAUDE.md rules fp16 out on MPS, not on T4
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    print(f"{len(data):,} pairs -> {total_steps:,} steps"
          f"{'  (fp16 autocast)' if use_amp else ''}\n")

    log_path = Path("results") / f"train_log_{out.name}.csv"
    log_path.parent.mkdir(exist_ok=True)
    if not args.resume or not log_path.exists():
        log_path.write_text("step,loss,accuracy,lr,seconds\n")

    step, started = start_step, time.time()
    model.train()
    bar = tqdm(total=total_steps, initial=start_step, unit="step")
    stop = False
    for _ in range(args.epochs):
        if stop:
            break
        for batch in loader:
            a_ids, a_mask, b_ids, b_mask, labels = (t.to(device) for t in batch)
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, args.warmup, args.lr)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits, _, _ = model(a_ids, a_mask, b_ids, b_mask)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            seen += labels.size(0)
            # Always log the very first step: the starting loss is the single
            # most diagnostic number in the run. It must be near ln(3) = 1.0986
            # for three balanced classes — anything else means a data bug.
            if step % args.log_every == 0 or step == start_step + 1:
                acc = (logits.argmax(-1) == labels).float().mean().item()
                bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}")
                with log_path.open("a") as fh:
                    fh.write(f"{step},{loss.item():.5f},{acc:.4f},"
                             f"{optimizer.param_groups[0]['lr']:.3e},"
                             f"{time.time() - started:.1f}\n")
            if step % args.checkpoint_every == 0:
                save_checkpoint(ckpt_path, model, optimizer, step, seen, cfg)
            bar.update(1)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

    bar.close()
    save_checkpoint(ckpt_path, model, optimizer, step, seen, cfg)

    # The head is discarded — only the encoder is ever used at inference.
    model.encoder.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "encoder")
    (out / "encoder" / "pooling.json").write_text(json.dumps({"pooling": args.pooling}))
    print(f"\nencoder -> {out / 'encoder'}   log -> {log_path}")


if __name__ == "__main__":
    main()
