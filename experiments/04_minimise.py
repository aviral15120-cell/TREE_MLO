"""Minimise network size: prune trained interpolating trees while keeping 100% train accuracy."""
import sys, json, glob
sys.path.insert(0, ".")
import numpy as np
from treenet.utils import load, save
from treenet.prune import prune_tree_classifier, effective_params

out = {}
for path in sorted(glob.glob("results/main_*_kmeans_*.pkl")):
    if "pruned" in path:
        continue
    name = path.split("/")[-1][:-4]
    ds = name.split("_")[1]
    f = np.load(f"data/feat_{ds}_kmeans.npz")
    Xtr, ytr, Xte, yte = f["Xtr"], f["Ytr"], f["Xte"], f["Yte"]
    m = load(path)
    before = dict(params=effective_params(m), train=m.accuracy(Xtr, ytr), test=m.accuracy(Xte, yte))
    log = prune_tree_classifier(m, Xtr, ytr)
    after = dict(params=effective_params(m), train=m.accuracy(Xtr, ytr), test=m.accuracy(Xte, yte))
    out[name] = dict(before=before, after=after, per_node=log)
    save(m, path.replace(".pkl", "_pruned.pkl"))
    print(name, before, after, flush=True)
json.dump(out, open("results/minimise.json", "w"), indent=1)
