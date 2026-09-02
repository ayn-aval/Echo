"""UMAP + HDBSCAN over the review vectors.

Two stages, for two different reasons.

UMAP reduces 768 dimensions to a handful. This is not merely for speed: in high
dimensions almost every pair of points sits at a similar distance from every
other ("the curse of dimensionality"), so density-based clustering has no density
to find. UMAP keeps the neighbourhood structure and throws away the rest.

HDBSCAN then finds dense regions and, crucially, is allowed to say "this review
belongs to nothing" — it labels those -1, noise. K-means cannot do that; it would
force every generic "good" review into some theme and quietly poison it.

UMAP is the slow part (minutes) and HDBSCAN the fast part (seconds), so the
reduction is cached per model and parameter sweeps only re-run HDBSCAN.
"""

from pathlib import Path

import numpy as np

VECTORS = Path("data/vectors")
SEED = 42


def load_vectors(model: str) -> np.ndarray:
    path = VECTORS / f"{model}.npy"
    if not path.exists():
        raise SystemExit(f"{path} missing — run `python -m src.embeddings.encode_corpus`")
    return np.load(path).astype(np.float32)


def reduce(model: str, n_components: int = 5, n_neighbors: int = 15,
           min_dist: float = 0.0, force: bool = False) -> np.ndarray:
    """UMAP to n_components dimensions, cached to disk.

    metric="cosine" because these are sentence embeddings: direction carries the
    meaning and length does not. min_dist=0.0 because we are reducing *for
    clustering*, not for a picture — packing points tightly is what gives HDBSCAN
    clean density to work with.

    random_state is set so a rerun reproduces the same themes. UMAP warns that
    this disables its parallelism and costs speed; a clustering nobody can
    reproduce is not a result, so the trade is worth it.
    """
    cache = VECTORS / f"{model}_umap{n_components}_nn{n_neighbors}.npy"
    if cache.exists() and not force:
        print(f"  umap: cached {cache.name}")
        return np.load(cache)

    import umap
    vectors = load_vectors(model)
    print(f"  umap: reducing {vectors.shape} -> {n_components}d "
          f"(n_neighbors={n_neighbors}, min_dist={min_dist}) — a few minutes")
    reduced = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, metric="cosine",
                        random_state=SEED).fit_transform(vectors)
    reduced = np.asarray(reduced, dtype=np.float32)
    np.save(cache, reduced)
    print(f"  umap: {reduced.shape} -> {cache}")
    return reduced


def cluster(reduced: np.ndarray, min_cluster_size: int, min_samples=None):
    """HDBSCAN on the reduced space. Returns (labels, probabilities).

    Label -1 means noise. probabilities[i] is how firmly point i belongs to its
    cluster, which Phase 7 can use to show the most typical reviews first.
    """
    import hdbscan
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            metric="euclidean",           # UMAP output, not cosine
                            cluster_selection_method="eom",
                            core_dist_n_jobs=-1)
    labels = model.fit_predict(reduced)
    return labels, model.probabilities_


def summarise(labels: np.ndarray) -> dict:
    real = labels[labels >= 0]
    sizes = np.bincount(real) if real.size else np.array([0])
    return {"n_clusters": int(len(np.unique(real))) if real.size else 0,
            "noise_pct": round(float((labels < 0).mean()) * 100, 2),
            "largest": int(sizes.max()),
            # The share of the whole corpus sitting in one cluster. Without this,
            # a model that dumps everything into a single bucket looks *good* —
            # it posts the lowest noise percentage in the comparison.
            "largest_pct": round(float(sizes.max()) / len(labels) * 100, 2),
            "median_size": int(np.median(sizes))}


def silhouette(vectors: np.ndarray, labels: np.ndarray, sample: int = 5000):
    """Silhouette in the ORIGINAL embedding space, noise excluded.

    Computed on the original vectors rather than the UMAP output: scoring the
    UMAP space would mostly measure how well HDBSCAN carved up UMAP's own
    picture, which is circular. Sampled because the exact figure is O(n^2) —
    2.1 billion pairs at this corpus size.

    Read it as a within-model diagnostic only. Different embedding models produce
    spaces with different geometry, so silhouette is NOT strictly comparable
    across them; the hand-audit is what settles that question.
    """
    from sklearn.metrics import silhouette_score
    keep = labels >= 0
    if keep.sum() < 100 or len(set(labels[keep])) < 2:
        return None
    rng = np.random.default_rng(SEED)
    idx = np.flatnonzero(keep)
    if idx.size > sample:
        idx = rng.choice(idx, sample, replace=False)
    return round(float(silhouette_score(vectors[idx], labels[idx],
                                        metric="cosine")), 4)
