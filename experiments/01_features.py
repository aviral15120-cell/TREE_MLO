"""Train one unsupervised trunk per dataset (plus a random-filter control) and cache features."""
import sys, time, json, pickle
sys.path.insert(0, ".")
import numpy as np
from treenet.trunk import KMeansTrunk, FeatureMap

D = 256
out = {}
for ds in ["cifar10", "catsdogs"]:
    tr = np.load(f"data/{ds}_train.npz"); te = np.load(f"data/{ds}_test.npz")
    for kind in ["kmeans", "random"]:
        t0 = time.time()
        trunk = KMeansTrunk(K=256, grid=3, seed=0, kind=kind).fit(tr["X"])
        Ftr = trunk.transform(tr["X"]); Fte = trunk.transform(te["X"])
        fm = FeatureMap().fit(Ftr, D=D)
        Xtr, Xte = fm.transform(Ftr), fm.transform(Fte)
        np.savez(f"data/feat_{ds}_{kind}.npz", Xtr=Xtr, Xte=Xte, Ytr=tr["Y"], Yte=te["Y"])
        with open(f"data/trunk_{ds}_{kind}.pkl", "wb") as f:
            pickle.dump((trunk, fm), f)
        out[f"{ds}_{kind}"] = dict(raw_dim=int(Ftr.shape[1]), D=D, pca_explained=fm.explained,
                                  seconds=time.time() - t0)
        print(ds, kind, out[f"{ds}_{kind}"], flush=True)
json.dump(out, open("results/features.json", "w"), indent=1)
