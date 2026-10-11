"""Receptive fields of tree nodes.

For every node n we differentiate its net input through the *whole* pipeline
    image (32x32x3) -> k-means trunk -> sqrt/standardise/PCA -> node MLP
with JAX, giving pixel-space receptive fields:
  * gradient RF      : mean over the node's positive samples of d net / d image
  * |gradient| map   : where in the image the node looks
  * preferred stimulus: gradient-ascent image that maximises the node's net input
and three per-node statistics, plotted against depth:
  * coherence    : mean cosine between each positive sample's gradient and the node's mean gradient
                   (high = one shared pattern, low = a different template for every image)
  * concentration: fraction of |gradient| energy in the top 10% of pixels (spatial localisation)
  * self-match   : mean cosine between a positive sample's gradient and that same (centred) image
                   (high = the node responds to "this particular picture")
"""
import sys, json, pickle
sys.path.insert(0, ".")
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from treenet.trunk import KMeansTrunk, FeatureMap
from treenet.node import mlp_apply
from treenet.utils import load

rng = np.random.default_rng(0)
CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_trunk(ds, kind="kmeans"):
    with open(f"data/trunk_{ds}_{kind}.pkl", "rb") as f:
        trunk, fm = pickle.load(f)
    return trunk, fm


def feat_fn(trunk, fm):
    tp, fp = trunk.params(), fm.params()
    return lambda img: FeatureMap.forward(fp, KMeansTrunk.forward(tp, img, trunk.grid))


def node_net(model, c=None):
    k = model.kind
    if k == "mlp":
        p = jax.tree_util.tree_map(jnp.asarray, model.p); th = model.theta
        return lambda f: mlp_apply(p, f)[:, 0] - th
    if k == "ball":
        cc, r2 = jnp.asarray(model.c), model.r2
        return lambda f: r2 - ((f - cc) ** 2).sum(1)
    if k == "view":
        p = jax.tree_util.tree_map(jnp.asarray, model.shared.p); c = model.c
        def g(f):
            z = mlp_apply(p, f)
            return z[:, c] - jnp.max(jnp.where(jnp.arange(z.shape[1]) == c, -1e9, z), 1)
        return g
    raise ValueError(k)


def make_grad(ff, net):
    fn = jax.jit(jax.grad(lambda img: net(ff(img)).sum()))
    val = jax.jit(lambda img: net(ff(img)))
    return fn, val


def grads(gfn, imgs, bs=32):
    return np.concatenate([np.asarray(gfn(jnp.asarray(imgs[i:i + bs], jnp.float32))) for i in range(0, len(imgs), bs)])


def preferred_stimulus(gfn, steps=120, lr=6.0):
    x = 128 + 8 * rng.standard_normal((1, 32, 32, 3)).astype(np.float32)
    for s in range(steps):
        g = np.asarray(gfn(jnp.asarray(x)))
        x = x + lr * g / (g.std() + 1e-8)
        x = 0.98 * (x - 128) + 128  # L2 decay toward grey
        if s % 10 == 9:  # light 3x3 box blur as a smoothness prior
            xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
            x = sum(xp[:, i:i + 32, j:j + 32] for i in range(3) for j in range(3)) / 9
        x = np.clip(x, 0, 255)
    return x[0]


def show_signed(g):
    g = g / (np.abs(g).max() + 1e-12)
    return np.clip(0.5 + 0.5 * g, 0, 1)


def stats(G, imgs):
    gm = G.mean(0).ravel()
    Gf = G.reshape(len(G), -1)
    coh = float(np.mean(Gf @ gm / (np.linalg.norm(Gf, axis=1) * np.linalg.norm(gm) + 1e-12)))
    e = np.abs(G).sum(-1).reshape(len(G), -1)
    e_sorted = -np.sort(-e, 1)
    conc = float(np.mean(e_sorted[:, : e.shape[1] // 10].sum(1) / e.sum(1)))
    Xc = (imgs - imgs.mean((1, 2, 3), keepdims=True)).reshape(len(imgs), -1)
    selfm = float(np.mean((Gf * Xc).sum(1) / (np.linalg.norm(Gf, axis=1) * np.linalg.norm(Xc, axis=1) + 1e-12)))
    return coh, conc, selfm


def analyse(ds, model_path, label_names):
    raw = np.load(f"data/{ds}_train.npz")["X"]
    trunk, fm = load_trunk(ds)
    ff = feat_fn(trunk, fm)
    model = load(model_path)
    rows, cache = [], {}
    for ti, tree in enumerate(model.trees):
        for ni, node in enumerate(tree.nodes):
            pos = node.idx_pos
            if len(pos) == 0:
                continue
            sel = pos if len(pos) <= 32 else rng.choice(pos, 32, replace=False)
            gfn, _ = make_grad(ff, node_net(node.model))
            G = grads(gfn, raw[sel].astype(np.float32))
            coh, conc, selfm = stats(G, raw[sel].astype(np.float32))
            r = dict(tree=ti, node=ni, depth=node.depth, role=node.role, kind=node.model.kind,
                     npos=int(len(pos)), domain=int(len(node.dom)), coherence=coh, concentration=conc,
                     self_match=selfm)
            rows.append(r)
            cache[(ti, ni)] = (G.mean(0), sel, gfn)
    return model, rows, cache, raw


def node_gallery(fname, items, raw, cache, titlefn, n_pos=4):
    fig, axs = plt.subplots(len(items), 3 + n_pos, figsize=(1.45 * (3 + n_pos), 1.5 * len(items)))
    axs = np.atleast_2d(axs)
    for r, (key, title) in enumerate(items):
        gmean, sel, gfn = cache[key]
        ps = preferred_stimulus(gfn)
        heat = np.abs(gmean).sum(-1)
        panels = [show_signed(gmean), heat, ps / 255.0] + [raw[i] / 255.0 for i in sel[:n_pos]]
        for c, im in enumerate(panels):
            ax = axs[r, c]
            ax.imshow(im, cmap="magma" if im.ndim == 2 else None, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
        axs[r, 0].set_ylabel(title, fontsize=6.5, rotation=0, ha="right", va="center")
    for c, t in enumerate(["gradient RF", "|grad| map", "preferred\nstimulus"] + ["positive"] * n_pos):
        axs[0, c].set_title(t, fontsize=7)
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


def hidden_units(fname, ds, Wsrc, title, n=16):
    """RFs of first-layer hidden units of a root network (gradient of unit pre-activation)."""
    raw = np.load(f"data/{ds}_train.npz")["X"]
    trunk, fm = load_trunk(ds); ff = feat_fn(trunk, fm)
    p = jax.tree_util.tree_map(jnp.asarray, Wsrc)
    H = p["W1"].shape[1]
    sample = raw[rng.choice(len(raw), 2000, replace=False)].astype(np.float32)
    F = np.concatenate([np.asarray(jax.jit(ff)(jnp.asarray(sample[i:i + 200]))) for i in range(0, 2000, 200)])
    A = F @ np.asarray(p["W1"]) + np.asarray(p["b1"])
    units = np.argsort(-np.abs(np.asarray(p["W2"])).sum(1))[: min(n, H)]
    fig, axs = plt.subplots(2, len(units), figsize=(1.1 * len(units), 2.4))
    for k, j in enumerate(units):
        top = np.argsort(-A[:, j])[:24]
        g = jax.jit(jax.grad(lambda img: (ff(img) @ p["W1"][:, j]).sum()))
        G = grads(g, sample[top])
        axs[0, k].imshow(show_signed(G.mean(0))); axs[1, k].imshow(sample[top[0]] / 255)
        for a in axs[:, k]:
            a.set_xticks([]); a.set_yticks([])
        axs[0, k].set_title(f"u{j}", fontsize=6)
    axs[0, 0].set_ylabel("RF", fontsize=7); axs[1, 0].set_ylabel("top img", fontsize=7)
    fig.suptitle(title, fontsize=8); plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


if __name__ == "__main__":
    out = {}
    # ---- trunk filters
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, (ds, kind) in zip(axs, [("cifar10", "kmeans"), ("catsdogs", "kmeans"), ("cifar10", "random")]):
        trunk, _ = load_trunk(ds, kind)
        F = trunk.filters_rgb()[:100]
        tile = np.ones((10 * 7, 10 * 7, 3))
        for i, f in enumerate(F):
            r, c = divmod(i, 10); tile[r * 7:r * 7 + 6, c * 7:c * 7 + 6] = f
        ax.imshow(tile, interpolation="nearest"); ax.set_axis_off()
        ax.set_title(f"{ds} trunk, {kind} filters (100 of 256)", fontsize=9)
    plt.tight_layout(); plt.savefig("results/figs/rf_trunk_filters.png", dpi=150); plt.close()

    # ---- Cats vs Dogs tree
    m, rows, cache, raw = analyse("catsdogs", "results/main_catsdogs_kmeans_shared_r16_h16_margin.pkl", ["cat", "dog"])
    out["catsdogs"] = rows
    t = m.trees[0]
    items = [((0, i), f"{n.role} d{n.depth}\n{n.model.kind} pos={len(n.idx_pos)}\ndom={len(n.dom)}")
             for i, n in enumerate(t.nodes) if len(n.idx_pos)]
    node_gallery("results/figs/rf_catsdogs_nodes.png", items, raw, cache, None)
    hidden_units("results/figs/rf_catsdogs_root_units.png", "catsdogs", t.nodes[0].model.p,
                 "Cats vs Dogs root: hidden-unit receptive fields (16 units)")

    # ---- CIFAR-10 tree
    m, rows, cache, raw = analyse("cifar10", "results/main_cifar10_kmeans_shared_r16_h16_margin.pkl", CIFAR)
    out["cifar10"] = rows
    items = [((c, 0), f"root/{CIFAR[c]}\nmargin view") for c in range(10)]
    node_gallery("results/figs/rf_cifar_class_roots.png", items, raw, cache, None)
    deep = sorted([(r["depth"], r["tree"], r["node"]) for r in rows if r["depth"] >= 1], key=lambda z: (z[0], z[1]))
    pick = []
    for d in sorted(set(z[0] for z in deep)):
        pick += [z for z in deep if z[0] == d][:3]
    items = []
    for d, ti, ni in pick:
        n = m.trees[ti].nodes[ni]
        items.append(((ti, ni), f"{CIFAR[ti]} {n.role} d{d}\n{n.model.kind} pos={len(n.idx_pos)}\ndom={len(n.dom)}"))
    node_gallery("results/figs/rf_cifar_child_nodes.png", items, raw, cache, None)
    hidden_units("results/figs/rf_cifar_root_units.png", "cifar10", m.shared.p,
                 "CIFAR-10 shared root: hidden-unit receptive fields (16 units)")

    # ---- quantitative: RF statistics vs depth
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.2))
    for ds, mk in [("catsdogs", "o"), ("cifar10", "s")]:
        R = out[ds]
        d = np.array([r["depth"] for r in R]) + (0.08 if ds == "cifar10" else -0.08)
        for ax, key in zip(axs, ["coherence", "concentration", "self_match", "npos"]):
            ax.scatter(d, [r[key] for r in R], marker=mk, alpha=0.7, label=ds)
            ax.set_xlabel("node depth"); ax.set_title(key, fontsize=9)
    axs[3].set_yscale("log"); axs[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("results/figs/rf_stats_vs_depth.png", dpi=150); plt.close()
    summ = {}
    for ds in out:
        for d in sorted(set(r["depth"] for r in out[ds])):
            R = [r for r in out[ds] if r["depth"] == d]
            summ[f"{ds}_depth{d}"] = {k: float(np.mean([r[k] for r in R])) for k in ["coherence", "concentration", "self_match", "npos"]} | {"n_nodes": len(R)}
    json.dump(dict(nodes=out, by_depth=summ), open("results/receptive_fields.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
