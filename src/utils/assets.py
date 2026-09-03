"""Where the model and its vectors come from: local files, or the Hugging Face Hub.

The rule everywhere in this module is **local wins**. On the development Mac the
files under models/ and data/ are used exactly as before, so nothing about local
behaviour changes. Only when a file is absent — which is the situation on a
deployed server — does the Hub copy get downloaded.

That keeps one code path serving both, rather than a `if DEPLOYED:` branch that is
only ever exercised in the place it is hardest to debug. It also means the
fallback can be tested on the Mac by temporarily moving the local files aside.

The files exist on the Hub because they cannot live in git: GitHub rejects any
single file over 100 MB, and the encoder is 327 MB with the vectors 141 MB.
"""

from pathlib import Path

import numpy as np

HUB_MODEL = "aynaval2003/echo-sbert-domain"
HUB_DATA = "aynaval2003/echo-review-vectors"

LOCAL_VECTORS = Path("data/vectors")
LOCAL_CORPUS = LOCAL_VECTORS / "corpus.parquet"


def encoder_source(local_path) -> str:
    """A local directory if it is there, otherwise the Hub repo id.

    AutoModel.from_pretrained and AutoTokenizer.from_pretrained both accept
    either, so callers need no branch of their own.
    """
    local_path = Path(local_path)
    if local_path.exists():
        return str(local_path)
    return HUB_MODEL


def corpus_source() -> str:
    """data/vectors/corpus.parquet, or the Hub copy downloaded and cached."""
    if LOCAL_CORPUS.exists():
        return str(LOCAL_CORPUS)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HUB_DATA, "corpus.parquet", repo_type="dataset")


def vectors(model: str = "sbert-domain") -> np.ndarray:
    """The review vectors as contiguous float32, L2-normalised.

    The Hub copy is float16 to halve the download. That was verified rather than
    assumed: over 500 probe queries the top-1 neighbour is identical 100% of the
    time and the same ten neighbours come back every time, with only the ordering
    inside the ten occasionally moving. They are cast back to float32 here because
    FAISS indexes expect it.
    """
    local = LOCAL_VECTORS / f"{model}.npy"
    if local.exists():
        raw = np.load(local)
    else:
        if model != "sbert-domain":
            raise SystemExit(
                f"{local} missing and only sbert-domain is published to the Hub. "
                f"Run `python -m src.embeddings.encode_corpus --model {model}`.")
        from huggingface_hub import hf_hub_download
        raw = np.load(hf_hub_download(HUB_DATA, "sbert-domain.fp16.npy",
                                      repo_type="dataset"))

    import faiss
    out = np.ascontiguousarray(raw.astype(np.float32))
    faiss.normalize_L2(out)          # in place; cheap, and idempotent if already unit
    return out


def is_local(model: str = "sbert-domain") -> bool:
    """True when everything is on disk — used only to report which source is live."""
    return (LOCAL_VECTORS / f"{model}.npy").exists() and LOCAL_CORPUS.exists()
