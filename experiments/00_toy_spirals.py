"""Sanity check on 2-D two-spirals: tiny nodes (h=2) must still reach 100% training accuracy."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from treenet.tree import BinaryTree

rng = np.random.default_rng(0)
n = 300
th = np.sqrt(rng.random(n)) * 3 * np.pi
r = th / (3 * np.pi)
X0 = np.c_[r * np.cos(th), r * np.sin(th)]
X1 = -X0
X = np.r_[X0, X1].astype(np.float32) + 0.02 * rng.standard_normal((2 * n, 2)).astype(np.float32)
t = np.r_[np.zeros(n), np.ones(n)].astype(np.int32)

res = {}
fig, axs = plt.subplots(1, 4, figsize=(16, 4))
for ax, (h, dc) in zip(axs, [(2, "paper"), (2, "margin"), (8, "margin"), (64, "margin")]):
    tree = BinaryTree(h=h, dontcare=dc, train_kw=dict(epochs=200, lr=1e-2, max_steps=3000)).fit(X, t)
    acc = float((tree.predict(X) == t).mean())
    res[f"h={h},{dc}"] = dict(train_acc=acc, **tree.summary())
    g = np.linspace(-1.1, 1.1, 300).astype(np.float32)
    G = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    Z = tree.predict(G).reshape(300, 300)
    ax.contourf(g, g, Z, levels=[-.5, .5, 1.5], colors=["#cfe0f3", "#f6d5c8"])
    ax.scatter(*X0.T, s=4, c="#1f5fa8"); ax.scatter(*X1.T, s=4, c="#c0492b")
    s = tree.summary()
    ax.set_title(f"h={h}, don't-care={dc}\ntrain acc={acc:.3f}, nodes={s['nodes']}, depth={s['depth']}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig("results/figs/toy_spirals.png", dpi=130)
json.dump(res, open("results/toy_spirals.json", "w"), indent=1)
print(json.dumps(res, indent=1))
