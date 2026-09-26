"""Smart sorting: group images (or part crops) by look, spot near-duplicates, suggest classes from a few
labeled examples and find labels that look wrong. Training-free: DINOv3 features (engine/embed.py) plus
clustering and nearest-neighbour arithmetic that runs in milliseconds.

Choices (research 2026-09-26, spikes/REPORT.md S10):
- Group: L2-normalise, PCA to 128 dims, Ward clustering; the number of groups is picked by the simplified
  silhouette (distance to the own group's centre vs the nearest other centre) and can be changed with a slider
  by re-cutting the same tree. Items with a negative silhouette go to "Unsure". HDBSCAN was rejected: it tends to
  call half of the items noise. Above MAX_TREE items the tree is built over k-means micro-groups.
- Near-duplicates: nearest-neighbour distance below 0.13 x the collection's median (Cleanlab's rule).
- Suggest: class-mean prototypes while a class has fewer than 5 examples, then logistic regression. An item is only
  suggested a class when it is as close to one of that class's examples as the closest 90% of neighbour pairs in
  the collection are to each other: images of kinds nobody has named yet stay unclassed (S10: 1-5% of those got a
  suggestion, against 100% without this rule, while 46-72% of the named kinds were suggested correctly).
- Odd ones: a labeled item whose nearest labeled neighbours (itself left out) mostly carry another class.
"""
import numpy as np

MAX_TREE = 5000
PCA_DIMS = 128


def normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def reduce(v: np.ndarray, dims: int = PCA_DIMS) -> np.ndarray:
    v = normalise(np.asarray(v, np.float32))
    if len(v) > dims and v.shape[1] > dims:
        from sklearn.decomposition import PCA
        v = PCA(dims, random_state=0).fit_transform(v)
    return normalise(v.astype(np.float32))


def _centres(v, labels, k):
    return normalise(np.stack([v[labels == g].mean(0) for g in range(k)]))


def silhouette(v, labels, k) -> np.ndarray:
    """Simplified silhouette per item: (b - a) / max(a, b), a = distance to its group's centre, b = to the nearest
    other centre. O(n k), so it runs for every candidate k."""
    if k < 2:
        return np.zeros(len(v))
    d = 1 - v @ _centres(v, labels, k).T
    a = d[np.arange(len(v)), labels]
    d[np.arange(len(v)), labels] = np.inf
    b = d.min(1)
    return (b - a) / np.maximum(np.maximum(a, b), 1e-12)


class Groups:
    """A clustering tree over the items; `cut(k)` gives k groups instantly (the page's slider)."""

    def __init__(self, vectors: np.ndarray, max_k: int = 50):
        from scipy.cluster.hierarchy import linkage
        self.v = reduce(vectors)
        n = len(self.v)
        self.leaf = np.arange(n)                          # item -> tree leaf
        pts = self.v
        if n > MAX_TREE:                                  # ponytail: approximate tree over micro-groups past 5000 items
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(min(1000, n // 5), random_state=0, n_init=3).fit(self.v)
            self.leaf, pts = km.labels_, normalise(km.cluster_centers_)
        self.tree = linkage(pts, method="ward") if len(pts) > 1 else None
        self.max_k = max(1, min(max_k, n // 5, len(pts)))
        self.scores = {k: float(silhouette(self.v, self.labels(k), k).mean()) for k in range(2, self.max_k + 1)}
        self.best_k = max(self.scores, key=self.scores.get) if self.scores else 1

    def labels(self, k: int) -> np.ndarray:
        from scipy.cluster.hierarchy import fcluster
        if self.tree is None or k <= 1:
            return np.zeros(len(self.v), int)
        return (fcluster(self.tree, k, "maxclust") - 1)[self.leaf]

    def cut(self, k: int | None = None) -> dict:
        """{"k", "groups": [[item, ...] closest to the centre first, biggest group first], "unsure": [item, ...]}.
        Unsure items (negative silhouette) are listed in their group too, and again under "unsure"."""
        k = max(1, min(k or self.best_k, self.max_k))
        labels = self.labels(k)
        k = int(labels.max()) + 1
        sil = silhouette(self.v, labels, k)
        centre = _centres(self.v, labels, k)
        near = (self.v * centre[labels]).sum(1)
        groups = [np.flatnonzero(labels == g) for g in range(k)]
        groups = [g[np.argsort(-near[g])].tolist() for g in sorted(groups, key=len, reverse=True)]
        return {"k": k, "best_k": self.best_k, "max_k": self.max_k, "groups": groups,
                "unsure": np.flatnonzero(sil < 0)[np.argsort(sil[sil < 0])].tolist()}


def near_duplicates(v: np.ndarray, ratio: float = 0.13, chunk: int = 2048) -> list[list[int]]:
    """Stacks of near-identical items (each stack sorted), nearest-neighbour cosine distance below `ratio` x the
    median nearest-neighbour distance."""
    v = normalise(np.asarray(v, np.float32))
    n = len(v)
    if n < 2:
        return []
    nn_d, nn_i = np.empty(n), np.empty(n, int)
    for s in range(0, n, chunk):
        sim = v[s:s + chunk] @ v.T
        sim[np.arange(len(sim)), np.arange(s, s + len(sim))] = -np.inf
        nn_i[s:s + chunk] = sim.argmax(1)
        nn_d[s:s + chunk] = 1 - sim.max(1)
    limit = ratio * float(np.median(nn_d))
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in np.flatnonzero(nn_d < limit):
        parent[root(i)] = root(int(nn_i[i]))
    stacks = {}
    for i in range(n):
        stacks.setdefault(root(i), []).append(i)
    return sorted((s for s in stacks.values() if len(s) > 1), key=lambda s: s[0])


def nn_similarity(v: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Each item's similarity to its nearest other item."""
    out = np.empty(len(v))
    for s in range(0, len(v), chunk):
        sim = v[s:s + chunk] @ v.T
        sim[np.arange(len(sim)), np.arange(s, s + len(sim))] = -np.inf
        out[s:s + chunk] = sim.max(1)
    return out


def predict(v: np.ndarray, labeled: dict[int, int], fit_quantile: float = 0.1):
    """(class, score, fits) per item from the labeled ones {item: class}: prototypes (the mean of each class's
    examples) while any class has fewer than 5 examples, else logistic regression. `fits` is False where the item
    is not close enough to any example of its class to suggest it (see the module notes). None without 2 classes."""
    classes = sorted(set(labeled.values()))
    if len(classes) < 2:
        return None
    v = normalise(np.asarray(v, np.float32))
    idx = np.array(list(labeled))
    y = np.array([labeled[i] for i in idx])
    if min(np.bincount(y)[classes]) < 5:
        proto = normalise(np.stack([v[idx[y == c]].mean(0) for c in classes]))
        logits = v @ proto.T / 0.05                           # temperature: cosine gaps of ~0.1 are decisive
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(v[idx], y)
        p, classes = clf.predict_proba(v), list(clf.classes_)
    best = p.argmax(1)
    cls = np.array(classes)[best]
    closest = np.full(len(v), -np.inf)                        # similarity to the nearest example of the predicted class
    for c in classes:
        ex = v[idx[y == c]]
        on = cls == c
        for s in range(0, int(on.sum()), 4096):
            rows = np.flatnonzero(on)[s:s + 4096]
            closest[rows] = (v[rows] @ ex.T).max(1)
    fits = closest >= np.quantile(nn_similarity(v), fit_quantile) if len(v) > 1 else np.ones(len(v), bool)
    fits[idx] = True
    return cls, p[np.arange(len(v)), best], fits


def odd_ones(v: np.ndarray, labeled: dict[int, int], k: int = 5, agree: float = 0.6):
    """[(item, likely class, share of neighbours saying so)] for labeled items whose k nearest labeled neighbours
    (itself left out, weighted by similarity) mostly carry another class; strongest first."""
    if len(set(labeled.values())) < 2 or len(labeled) <= k:
        return []
    v = normalise(np.asarray(v, np.float32))
    idx = np.array(list(labeled))
    y = np.array([labeled[i] for i in idx])
    x = v[idx]
    out = []
    for s in range(0, len(idx), 2048):
        sim = x[s:s + 2048] @ x.T
        sim[np.arange(len(sim)), np.arange(s, s + len(sim))] = -np.inf
        top = np.argsort(-sim, 1)[:, :k]
        for row, nb in enumerate(top):
            w = np.maximum(sim[row, nb], 0) + 1e-6
            votes = {}
            for c, wi in zip(y[nb], w):
                votes[c] = votes.get(c, 0) + wi
            c, share = max(votes.items(), key=lambda cw: cw[1])
            share /= w.sum()
            if c != y[s + row] and share >= agree:
                out.append((int(idx[s + row]), int(c), float(share)))
    return sorted(out, key=lambda o: -o[2])
