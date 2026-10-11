"""Feature trunk for the tree-type network.

A single-layer convolutional trunk in the style of Coates, Lee & Ng (2011):
  image -> 6x6 patches (stride 1) -> per-patch contrast normalisation -> ZCA whitening
        -> "triangle" k-means encoding  f_k = max(0, mean_j z_j - z_k),  z_k = ||p - c_k||
        -> sum pooling over a GxG grid  -> G*G*K features.

The trunk is trained *unsupervised* (k-means on whitened patches), separately per dataset,
so the label information enters the model only through the tree nodes.  A "random" trunk
with Gaussian centroids is provided as an ablation.  Everything is written in jax.numpy so
the full map image -> tree-node output can be differentiated for receptive-field analysis.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

PATCH = 6


def extract_patches(imgs):
    """imgs (B,32,32,3) float -> (B, H', W', 3*6*6) patches (channel-major ordering)."""
    x = jnp.transpose(imgs, (0, 3, 1, 2))  # NCHW
    p = lax.conv_general_dilated_patches(x, (PATCH, PATCH), (1, 1), "VALID")
    return jnp.transpose(p, (0, 2, 3, 1))  # (B, 27, 27, 108)


def normalise(p):
    m = p.mean(-1, keepdims=True)
    v = p.var(-1, keepdims=True)
    return (p - m) / jnp.sqrt(v + 10.0)  # pixels are on a 0..255 scale


class KMeansTrunk:
    def __init__(self, K=256, grid=3, seed=0, kind="kmeans"):
        self.K, self.grid, self.seed, self.kind = K, grid, seed, kind

    # ---------------------------------------------------------------- training
    def fit(self, imgs_uint8, n_patches=150_000, iters=15, eps_zca=0.1):
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(imgs_uint8), n_patches)
        ys = rng.integers(0, 32 - PATCH + 1, n_patches)
        xs = rng.integers(0, 32 - PATCH + 1, n_patches)
        P = np.stack([imgs_uint8[i, y:y + PATCH, x:x + PATCH, :] for i, y, x in zip(idx, ys, xs)])
        P = np.transpose(P.astype(np.float32), (0, 3, 1, 2)).reshape(n_patches, -1)  # channel-major
        P = np.asarray(normalise(jnp.asarray(P)))
        self.M = P.mean(0)
        C = np.cov(P - self.M, rowvar=False)
        d, V = np.linalg.eigh(C)
        self.W = (V @ np.diag(1.0 / np.sqrt(d + eps_zca)) @ V.T).astype(np.float32)  # ZCA
        self.Winv = (V @ np.diag(np.sqrt(d + eps_zca)) @ V.T).astype(np.float32)
        Pw = (P - self.M) @ self.W
        if self.kind == "random":
            c = rng.standard_normal((self.K, Pw.shape[1])).astype(np.float32)
            self.C = c / np.linalg.norm(c, axis=1, keepdims=True) * np.median(np.linalg.norm(Pw, axis=1))
            return self
        # k-means (Lloyd), initialised from random whitened patches
        c = Pw[rng.choice(len(Pw), self.K, replace=False)].copy()
        for it in range(iters):
            d2 = (Pw ** 2).sum(1)[:, None] - 2 * Pw @ c.T + (c ** 2).sum(1)[None]
            a = d2.argmin(1)
            counts = np.bincount(a, minlength=self.K)
            sums = np.zeros_like(c)
            np.add.at(sums, a, Pw)
            empty = counts == 0
            c[~empty] = sums[~empty] / counts[~empty, None]
            c[empty] = Pw[rng.choice(len(Pw), empty.sum(), replace=False)]  # re-seed dead clusters
        self.C = c.astype(np.float32)
        return self

    # --------------------------------------------------------------- encoding
    def params(self):
        return dict(M=jnp.asarray(self.M), W=jnp.asarray(self.W), C=jnp.asarray(self.C))

    @staticmethod
    def forward(params, imgs, grid):
        """Differentiable trunk: imgs (B,32,32,3) float (0..255) -> (B, grid*grid*K)."""
        p = normalise(extract_patches(imgs))
        p = (p - params["M"]) @ params["W"]
        C = params["C"]
        d2 = (p ** 2).sum(-1, keepdims=True) - 2 * p @ C.T + (C ** 2).sum(-1)
        z = jnp.sqrt(jnp.maximum(d2, 0.0) + 1e-6)
        f = jax.nn.relu(z.mean(-1, keepdims=True) - z)  # triangle activation
        B, H, Wd, K = f.shape
        s = H // grid  # 27 -> 9 for grid=3
        f = f[:, :s * grid, :s * grid].reshape(B, grid, s, grid, s, K).sum((2, 4))
        return f.reshape(B, -1)

    def transform(self, imgs_uint8, batch=200):
        params = self.params()
        fn = jax.jit(lambda x: self.forward(params, x, self.grid))
        out = []
        for i in range(0, len(imgs_uint8), batch):
            out.append(np.asarray(fn(jnp.asarray(imgs_uint8[i:i + batch], dtype=jnp.float32))))
        return np.concatenate(out)

    def filters_rgb(self):
        """Centroids mapped back to (de-whitened) pixel space for display: (K,6,6,3) in [0,1]."""
        c = self.C @ self.Winv
        c = c.reshape(-1, 3, PATCH, PATCH).transpose(0, 2, 3, 1)
        lo = c.min((1, 2, 3), keepdims=True); hi = c.max((1, 2, 3), keepdims=True)
        return (c - lo) / (hi - lo + 1e-8)


class FeatureMap:
    """Standardise trunk features (sqrt-compressed) and project to D principal components."""

    def fit(self, F, D=256):
        F = np.sqrt(F + 1e-6)
        self.mu = F.mean(0)
        self.sd = F.std(0) + 1e-3
        Z = (F - self.mu) / self.sd
        U, S, Vt = np.linalg.svd(Z[:: max(1, len(Z) // 20000)], full_matrices=False)
        self.P = Vt[:D].T.astype(np.float32)
        proj = Z @ self.P
        self.scale = proj.std(0).mean()
        self.explained = float((S[:D] ** 2).sum() / (S ** 2).sum())
        return self

    def params(self):
        return dict(mu=jnp.asarray(self.mu), sd=jnp.asarray(self.sd), P=jnp.asarray(self.P),
                    scale=float(self.scale))

    @staticmethod
    def forward(params, F):
        Z = (jnp.sqrt(F + 1e-6) - params["mu"]) / params["sd"]
        return (Z @ params["P"]) / params["scale"]

    def transform(self, F):
        Z = (np.sqrt(F + 1e-6) - self.mu) / self.sd
        return ((Z @ self.P) / self.scale).astype(np.float32)
