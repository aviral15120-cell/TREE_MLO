"""Size minimisation that provably preserves 100% training accuracy.

Locality lemma: the whole tree is correct on the training set iff every node's corrected decision
is correct on its own domain (samples outside a child's domain are exactly its don't-cares).
Hence each node can be pruned *independently*, checking only its own domain:
  1. structural pruning - drop the least important hidden units (|w2_j| * ||W1[:,j]||),
  2. magnitude pruning  - zero the smallest remaining first-layer weights,
binary-searching the largest reduction that leaves the node's corrected decisions unchanged.
The shared multiclass root is checked globally (final arg-max on the full training set).
"""
import numpy as np
import jax
from .node import MLPNode, SharedRoot, mlp_apply
from .tree import subtree_corrected


def _np(p):
    return {k: np.array(v) for k, v in p.items()}


def eff_params(m):
    if m.kind not in ("mlp", "shared"):
        return m.n_params()
    p = m.p
    alive = np.abs(np.asarray(p["W2"])).sum(1) > 0
    return int((np.asarray(p["W1"])[:, alive] != 0).sum() + alive.sum() + (np.asarray(p["W2"])[alive] != 0).sum()
               + np.asarray(p["b2"]).size)


def _largest_ok(lo, hi, ok):
    """Largest integer k in [lo, hi] with ok(k) (assumes ok is monotone; ok(lo) must hold)."""
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _prune_params(p0, order, k_units, frac):
    p = _np(p0)
    dead = order[:k_units]
    p["W1"][:, dead] = 0; p["b1"][dead] = 0; p["W2"][dead] = 0
    if frac > 0:
        alive = np.setdiff1d(np.arange(p["W1"].shape[1]), dead)
        w = np.abs(p["W1"][:, alive])
        thr = np.quantile(w, frac)
        sub = p["W1"][:, alive]; sub[np.abs(sub) < thr] = 0; p["W1"][:, alive] = sub
    return p


def prune_model(m, ok_fn):
    """Generic: m has .p (MLP params); ok_fn(params)->bool checks correctness."""
    p0 = _np(m.p)
    imp = np.abs(p0["W2"]).sum(1) * np.linalg.norm(p0["W1"], axis=0)
    order = np.argsort(imp)
    H = p0["W1"].shape[1]
    k = _largest_ok(0, H - 1, lambda k: ok_fn(_prune_params(p0, order, k, 0.0)))
    fr = _largest_ok(0, 98, lambda f: ok_fn(_prune_params(p0, order, k, f / 100)))
    m.p = _prune_params(p0, order, k, fr / 100)
    assert ok_fn(m.p)
    return dict(units_removed=int(k), units=int(H), weight_frac_zeroed=fr / 100)


def prune_tree_classifier(model, Xtr, ytr):
    log = []
    for tree in model.trees:
        for node in sorted(tree.nodes, key=lambda n: -n.depth):
            if node.model.kind != "mlp":
                continue
            Xd, td = Xtr[node.dom], node.t
            a = (subtree_corrected(node.A, Xd) >= 0) if node.A is not None else 0
            b = (subtree_corrected(node.B, Xd) >= 0) if node.B is not None else 0
            corr = node.wA * a - node.wB * b
            th = node.model.theta

            def ok(p, Xd=Xd, td=td, corr=corr, th=th):
                r = np.asarray(mlp_apply(p, Xd))[:, 0] - th
                return bool((((r + corr) >= 0).astype(int) == td).all())
            before = eff_params(node.model)
            info = prune_model(node.model, ok)
            log.append(dict(depth=node.depth, before=before, after=eff_params(node.model), **info))
    if model.shared is not None:
        # global check for the shared root: children fixed, final arg-max must stay perfect
        corr = np.zeros((len(Xtr), model.C))
        for c, tree in enumerate(model.trees):
            r = tree.root
            if r.A is not None:
                corr[:, c] += r.wA * (subtree_corrected(r.A, Xtr) >= 0)
            if r.B is not None:
                corr[:, c] -= r.wB * (subtree_corrected(r.B, Xtr) >= 0)

        def ok(p):
            z = np.asarray(mlp_apply(p, Xtr))
            mg = np.stack([z[:, c] - np.delete(z, c, 1).max(1) for c in range(model.C)], 1)
            return bool(((mg + corr).argmax(1) == ytr).all())
        before = eff_params(model.shared)
        info = prune_model(model.shared, ok)
        log.append(dict(depth=0, before=before, after=eff_params(model.shared), shared=True, **info))
    return log


def effective_params(model):
    p = sum(eff_params(n.model) + (n.A is not None) + (n.B is not None) for t in model.trees for n in t.nodes)
    if model.shared is not None:
        p += eff_params(model.shared)
    return int(p)
