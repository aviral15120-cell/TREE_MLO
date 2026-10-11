"""Capacity sweep: node width h (all nodes, root included) from tiny to huge, with and without
label noise.  For every width we record the root-only network (depth-0 truncation, i.e. an ordinary
2-layer MLP) and the full interpolating tree.  Results are appended to results/dd.jsonl."""
import sys, json, time, argparse
sys.path.insert(0, ".")
import numpy as np
from treenet.tree import TreeClassifier

ap = argparse.ArgumentParser()
ap.add_argument("--ds", default="catsdogs")
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--noise", type=float, default=0.0)
ap.add_argument("--widths", default="1,2,4,8,16,32,64,128,256,512")
ap.add_argument("--dontcare", default="margin")
ap.add_argument("--trunk", default="kmeans")
ap.add_argument("--max_nodes", type=int, default=400)
ap.add_argument("--out", default="results/dd.jsonl")
a = ap.parse_args()

f = np.load(f"data/feat_{a.ds}_{a.trunk}.npz")
C = int(f["Ytr"].max() + 1)
rng = np.random.default_rng(0)
# class-balanced subset
idx = np.concatenate([rng.permutation(np.where(f["Ytr"] == c)[0])[: a.n // C] for c in range(C)])
Xtr, ytr_clean = f["Xtr"][idx], f["Ytr"][idx]
ytr = ytr_clean.copy()
flip = rng.random(len(ytr)) < a.noise
ytr[flip] = (ytr[flip] + rng.integers(1, C, flip.sum())) % C   # always a *different* class
Xte, yte = f["Xte"], f["Yte"]

for h in [int(w) for w in a.widths.split(",")]:
    t0 = time.time()
    m = TreeClassifier(C, h=h, root_h=h, mode="shared", dontcare=a.dontcare, max_nodes=a.max_nodes,
                       train_kw=dict(epochs=60, lr=3e-3, max_steps=3000), seed=0).fit(Xtr, ytr)
    s = m.summary()
    root_params = m.shared.n_params() if m.shared is not None else m.trees[0].nodes[0].model.n_params()
    row = dict(ds=a.ds, n=len(ytr), noise=a.noise, h=h, dontcare=a.dontcare, trunk=a.trunk,
               train_acc=m.accuracy(Xtr, ytr), test_acc=m.accuracy(Xte, yte),
               root_train_acc=m.accuracy(Xtr, ytr, 0), root_test_acc=m.accuracy(Xte, yte, 0),
               root_params=int(root_params), params=s["params"], nodes=s["nodes"], depth=s["depth"],
               kinds=s["kinds"], truncated=any(t["truncated"] for t in s["per_tree"]),
               noisy_memorised=float((m.predict(Xtr[flip]) == ytr[flip]).mean()) if flip.any() else None,
               noisy_test_like_clean=float((m.predict(Xtr[flip]) == ytr_clean[flip]).mean()) if flip.any() else None,
               seconds=time.time() - t0)
    with open(a.out, "a") as fo:
        fo.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
