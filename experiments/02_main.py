"""Main experiment: grow the tree on the full training set until 100% training accuracy."""
import sys, json, time, argparse
sys.path.insert(0, ".")
import numpy as np
from treenet.tree import TreeClassifier
from treenet.utils import save, depth_profile

ap = argparse.ArgumentParser()
ap.add_argument("--ds", default="catsdogs")
ap.add_argument("--trunk", default="kmeans")
ap.add_argument("--mode", default="shared")
ap.add_argument("--root_h", type=int, default=512)
ap.add_argument("--h", type=int, default=16)
ap.add_argument("--dontcare", default="margin")
ap.add_argument("--max_steps", type=int, default=6000)
ap.add_argument("--tag", default="")
a = ap.parse_args()

f = np.load(f"data/feat_{a.ds}_{a.trunk}.npz")
Xtr, Xte, ytr, yte = f["Xtr"], f["Xte"], f["Ytr"], f["Yte"]
C = int(ytr.max() + 1)
t0 = time.time()
model = TreeClassifier(C, h=a.h, root_h=a.root_h, mode=a.mode, dontcare=a.dontcare,
                       train_kw=dict(epochs=40, lr=3e-3, max_steps=a.max_steps), seed=0).fit(Xtr, ytr, verbose=True)
res = dict(args=vars(a), seconds=time.time() - t0, train_acc=model.accuracy(Xtr, ytr),
           test_acc=model.accuracy(Xte, yte), summary=model.summary(),
           depth_profile=depth_profile(model, Xtr, ytr, Xte, yte))
name = f"main_{a.ds}_{a.trunk}_{a.mode}_r{a.root_h}_h{a.h}_{a.dontcare}{a.tag}"
json.dump(res, open(f"results/{name}.json", "w"), indent=1)
save(model, f"results/{name}.pkl")
print(json.dumps({k: v for k, v in res.items() if k != "summary"}, indent=1))
print("nodes", res["summary"]["nodes"], "params", res["summary"]["params"], "kinds", res["summary"]["kinds"])
