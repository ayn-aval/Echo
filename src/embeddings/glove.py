"""Averaged GloVe sentence embeddings — the first baseline.

GloVe gives every word one fixed vector, learned from how often words co-occur.
A sentence embedding is just the average of its words' vectors. It has no idea
about word order — "dog bites man" and "man bites dog" get identical vectors —
which is exactly why it is a baseline rather than a solution.

The file holds 2.2 million words and is 5.3 GB; loading all of it costs about
2.6 GB of RAM. We only ever need the words that actually appear in the text being
encoded — roughly 30k for STS — so the file is streamed once and everything else
discarded. The filtered subset is cached, so this cost is paid once.
"""

import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

GLOVE_TXT = Path("data/glove/glove.840B.300d.txt")
CACHE_DIR = Path("data/glove")
DIM = 300
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")


def load_vectors(vocab, cache_key: str) -> dict:
    """Vectors for just these words. Cached under cache_key."""
    cache = CACHE_DIR / f"filtered_{cache_key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return dict(zip(z["words"].tolist(), z["vectors"]))

    # 840B is case-sensitive, so look for both forms and let the encoder pick.
    wanted = set(vocab) | {w.lower() for w in vocab}
    found = {}
    with GLOVE_TXT.open(encoding="utf-8") as fh:
        for line in tqdm(fh, total=2_196_018, desc="  scanning GloVe", unit="w"):
            word, _, rest = line.partition(" ")
            if word in wanted:
                found[word] = np.fromstring(rest, sep=" ", dtype=np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, words=np.array(list(found)),
                        vectors=np.stack(list(found.values())))
    print(f"  kept {len(found):,} of 2,196,018 vectors "
          f"({len(wanted):,} words wanted) -> {cache}")
    return found


def make_encoder(corpus, cache_key: str):
    """Build an encode(list[str]) -> np.ndarray over this corpus's vocabulary."""
    vocab = {t for text in corpus for t in tokenize(text)}
    vectors = load_vectors(vocab, cache_key)

    def encode(sentences):
        out = np.zeros((len(sentences), DIM), dtype=np.float32)
        for i, sentence in enumerate(sentences):
            hits = []
            for token in tokenize(sentence):
                # Lowercase first, cased form only as fallback. GloVe 840B holds
                # separate vectors for "The" and "the"; preferring the cased form
                # made sentence-initial words use a different vector from the same
                # word mid-sentence. Measured cost of getting this backwards:
                # 4.1 Spearman points on STS-B (44.11 -> 48.21).
                vec = vectors.get(token.lower())
                if vec is None:
                    vec = vectors.get(token)
                if vec is not None:
                    hits.append(vec)
            if hits:  # all-OOV sentences keep the zero vector
                out[i] = np.mean(hits, axis=0)
        return out

    return encode
