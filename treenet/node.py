"""Nodes of the tree: small neural networks.

* MLPNode   : d -> h (ReLU) -> 1 logit.  Trained with (weighted) logistic loss + Adam, then the
              output bias is calibrated to minimise the *number* of errors on the node's domain
              (the zero-norm objective of eq. (1) in the lecture notes, solved exactly in 1-D).
* SharedRoot: d -> h (ReLU) -> C logits trained with softmax cross-entropy.  For class c it
              exposes the margin  z_c - max_{c' != c} z_c'  as that class's root net input.
* BallNode  : a single linear neuron on the lifted input (x, ||x||^2).  Lifting onto the
              paraboloid puts every point in convex position, so any single point can be cut
              off by a hyperplane.  Used only as a fallback that guarantees the growth
              procedure terminates (every node removes at least one error).
"""
from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp


def mlp_apply(p, x):
    return jax.nn.relu(x @ p["W1"] + p["b1"]) @ p["W2"] + p["b2"]


def init_mlp(seed, d, h, out=1):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    return dict(W1=jax.random.normal(k1, (d, h)) * np.sqrt(2.0 / d), b1=jnp.zeros(h),
                W2=jax.random.normal(k2, (h, out)) * np.sqrt(1.0 / h), b2=jnp.zeros(out))


def _bce(p, x, t, w):
    z = mlp_apply(p, x)[:, 0]
    return (w * (jnp.logaddexp(0.0, z) - t * z)).sum() / w.sum()


def _ce(p, x, y, w):
    z = jax.nn.log_softmax(mlp_apply(p, x))
    return -(w * jnp.take_along_axis(z, y[:, None], 1)[:, 0]).sum() / w.sum()


@lru_cache(None)
def _make_step(kind):
    loss = _bce if kind == "bce" else _ce

    @jax.jit
    def step(p, m, v, k, x, t, w, lr):
        g = jax.grad(loss)(p, x, t, w)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        mh = jax.tree_util.tree_map(lambda a: a / (1 - 0.9 ** k), m)
        vh = jax.tree_util.tree_map(lambda a: a / (1 - 0.999 ** k), v)
        p = jax.tree_util.tree_map(lambda a, b, c: a - lr * b / (jnp.sqrt(c) + 1e-8), p, mh, vh)
        return p, m, v
    return step


_apply = jax.jit(mlp_apply)


def train_mlp(p, X, T, W, kind, epochs=40, lr=3e-3, batch=256, min_steps=300, max_steps=6000,
              check_every=200, err_fn=None, seed=0):
    """Mini-batch Adam.  Stops early when err_fn(params) == 0 (interpolation reached)."""
    n = len(X)
    # power-of-two batch buckets: avoids one XLA recompilation per distinct small domain size
    B = int(min(batch, 2 ** int(np.ceil(np.log2(max(n, 2))))))
    replace = n < B
    steps = int(np.clip(epochs * int(np.ceil(n / B)), min_steps, max_steps))
    step = _make_step(kind)
    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    rng = np.random.default_rng(seed)
    Xj, Tj, Wj = jnp.asarray(X), jnp.asarray(T), jnp.asarray(W)
    perm, pos = rng.permutation(n), 0
    for k in range(1, steps + 1):
        if replace:
            ib = rng.integers(0, n, B)
        else:
            if pos + B > n:
                perm, pos = rng.permutation(n), 0
            ib = perm[pos:pos + B]; pos += B
        p, m, v = step(p, m, v, k, Xj[ib], Tj[ib], Wj[ib], lr)
        if err_fn is not None and k % check_every == 0 and err_fn(p) == 0:
            break
    return p, k


def n_params_mlp(d, h, out=1):
    return d * h + h + h * out + out


class MLPNode:
    kind = "mlp"

    def __init__(self, d, h, seed=0):
        self.d, self.h, self.seed = d, h, seed
        self.p = init_mlp(seed, d, h, 1)
        self.theta = 0.0

    def raw(self, X):
        return np.asarray(_apply(self.p, jnp.asarray(X))[:, 0]) - self.theta

    def fit(self, X, t, weighting="sqrt", **kw):
        npos = max(t.sum(), 1); nneg = max(len(t) - t.sum(), 1)
        wpos = {"none": 1.0, "sqrt": np.sqrt(nneg / npos), "balanced": nneg / npos}[weighting]
        W = np.where(t == 1, wpos, 1.0).astype(np.float32)

        def err(p):
            s = np.asarray(_apply(p, jnp.asarray(X))[:, 0])
            return int(((s >= 0) != (t == 1)).sum())
        self.p, self.steps = train_mlp(self.p, X, t.astype(np.float32), W, "bce", err_fn=err,
                                       seed=self.seed, **kw)
        self.calibrate(X, t)
        return self

    def calibrate(self, X, t):
        """Choose the threshold that minimises #errors on the domain (exact 1-D zero-norm fit),
        subject to at least one true positive so that the A-child task strictly shrinks."""
        self.theta = 0.0
        s = self.raw(X)
        order = np.argsort(-s)
        ss, tt = s[order], t[order]
        # predicting the top-j samples as positive:
        tp = np.concatenate([[0], np.cumsum(tt)])
        fp = np.concatenate([[0], np.cumsum(1 - tt)])
        errs = (tt.sum() - tp) + fp
        first_pos = int(np.argmax(tt == 1)) if tt.any() else 0
        errs[:first_pos + 1] = 10 ** 9  # forbid thresholds with zero true positives
        j = int(np.argmin(errs))
        hi = ss[j - 1]; lo = ss[j] if j < len(ss) else ss[j - 1] - 1.0
        self.theta = float(0.5 * (hi + lo))
        if not np.isfinite(self.theta):
            self.theta = 0.0

    def n_params(self):
        return n_params_mlp(self.d, self.h, 1)

    def first_layer(self):
        return np.asarray(self.p["W1"])


class BallNode:
    """net(x) = r^2 - ||x - c||^2  ==  linear neuron on (x, ||x||^2).  Fires inside a ball."""
    kind = "ball"

    def __init__(self, d):
        self.d = d

    def fit(self, X, t, n_candidates=64, seed=0):
        pos = np.where(t == 1)[0]; neg = np.where(t == 0)[0]
        rng = np.random.default_rng(seed)
        cand = pos if len(pos) <= n_candidates else rng.choice(pos, n_candidates, replace=False)
        best = None
        cand = list(cand) + [c for c in pos if c not in set(cand)]  # widen search if needed
        for c in cand:
            d2 = ((X - X[c]) ** 2).sum(1)
            r2 = 0.5 * d2[neg].min() if len(neg) else d2.max() + 1.0
            covered = int((d2[pos] < r2).sum())
            if r2 > 0 and (best is None or covered > best[0]):
                best = (covered, c, r2)
            if best is not None and c not in pos[:0] and len(cand) > n_candidates and cand.index(c) >= n_candidates:
                break
        if best is None:
            raise RuntimeError("duplicate points with conflicting labels: cannot separate")
        _, c, r2 = best
        self.c, self.r2 = X[c].copy(), float(r2)
        return self

    def raw(self, X):
        return self.r2 - ((X - self.c) ** 2).sum(1)

    def n_params(self):
        return self.d + 2  # weights on (x, ||x||^2) + bias

    def first_layer(self):
        return self.c[:, None]


class SharedRoot:
    """One multiclass MLP shared by all C class-trees (efficient multiclass extension)."""
    kind = "shared"

    def __init__(self, d, h, C, seed=0):
        self.d, self.h, self.C, self.seed = d, h, C, seed
        self.p = init_mlp(seed, d, h, C)

    def fit(self, X, y, **kw):
        def err(p):
            z = np.asarray(_apply(p, jnp.asarray(X)))
            return int((z.argmax(1) != y).sum())
        W = np.ones(len(y), np.float32)
        self.p, self.steps = train_mlp(self.p, X, y.astype(np.int32), W, "ce", err_fn=err,
                                       seed=self.seed, **kw)
        return self

    def logits(self, X):
        return np.asarray(_apply(self.p, jnp.asarray(X)))

    def margins(self, X):
        z = self.logits(X)
        out = np.empty_like(z)
        for c in range(self.C):
            other = np.delete(z, c, axis=1).max(1)
            out[:, c] = z[:, c] - other
        return out

    def n_params(self):
        return n_params_mlp(self.d, self.h, self.C)

    def first_layer(self):
        return np.asarray(self.p["W1"])


class MarginView:
    """Root node of class-tree c when the root is shared: raw net = margin of class c."""
    kind = "view"

    def __init__(self, shared, c):
        self.shared, self.c, self.d = shared, c, shared.d

    def raw(self, X):
        return self.shared.margins(X)[:, self.c]

    def n_params(self):
        return 0  # counted once on the SharedRoot

    def first_layer(self):
        return self.shared.first_layer()
