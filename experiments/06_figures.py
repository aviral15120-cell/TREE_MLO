"""Produce summary figures/tables from results/*.json(l)."""
import sys, json, glob
sys.path.insert(0, ".")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open("results/dd.jsonl")] if glob.glob("results/dd.jsonl") else []


def sel(**kw):
    out = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    return sorted(out, key=lambda r: r["h"])


# ---------------------------------------------------------------- double descent
fig, axs = plt.subplots(2, 2, figsize=(11, 7.5))
for i, ds in enumerate(["catsdogs", "cifar10"]):
    for j, noise in enumerate([0.0, 0.2]):
        ax = axs[i, j]
        R = sel(ds=ds, noise=noise, dontcare="margin", trunk="kmeans")
        if not R:
            ax.set_title(f"{ds}, label noise {noise:.0%}: not run (compute budget)", fontsize=10); ax.set_axis_off(); continue
        rp = np.array([r["root_params"] for r in R]); tp = np.array([r["params"] for r in R])
        ax.plot(rp, [1 - r["root_test_acc"] for r in R], "o-", c="#1f5fa8", label="root only: test error")
        ax.plot(rp, [1 - r["root_train_acc"] for r in R], "o--", c="#1f5fa8", alpha=.45, label="root only: train error")
        ax.plot(tp, [1 - r["test_acc"] for r in R], "s-", c="#c0492b", label="interpolating tree: test error")
        for r, x in zip(R, tp):
            ax.annotate(f"h={r['h']}", (x, 1 - r["test_acc"]), fontsize=6, xytext=(2, 3), textcoords="offset points", color="#c0492b")
        interp = [r["root_params"] for r in R if r["root_train_acc"] >= 0.999]
        if interp:
            ax.axvline(min(interp), c="k", ls=":", lw=1)
            ax.text(min(interp), ax.get_ylim()[1] * 0.98, " root interpolates", fontsize=7, va="top")
        ax.axvline(R[0]["n"], c="grey", ls="-.", lw=0.8)
        ax.text(R[0]["n"], 0.01, " #params = n", fontsize=7, color="grey")
        ax.set_xscale("log"); ax.set_xlabel("number of parameters"); ax.set_ylabel("error")
        ax.set_title(f"{ds}  (n={R[0]['n']}, label noise {noise:.0%})", fontsize=10)
        ax.set_ylim(bottom=0)
        if i == 0 and j == 0:
            ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/figs/double_descent.png", dpi=150); plt.close()

# ---------------------------------------------------------------- size vs width
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
for ds, noise, mk in [("catsdogs", 0.0, "o-"), ("catsdogs", 0.2, "o--"), ("cifar10", 0.0, "s-"), ("cifar10", 0.2, "s--")]:
    R = sel(ds=ds, noise=noise, dontcare="margin", trunk="kmeans")
    if not R:
        continue
    h = [r["h"] for r in R]; lab = f"{ds}, noise {noise:.0%}"
    axs[0].plot(h, [r["nodes"] for r in R], mk, label=lab)
    axs[1].plot(h, [r["params"] for r in R], mk, label=lab)
    axs[2].plot(h, [r["test_acc"] - r["root_test_acc"] for r in R], mk, label=lab)
for ax, t in zip(axs, ["nodes needed to interpolate", "total parameters at interpolation",
                       "test acc: tree minus root-only"]):
    ax.set_xscale("log", base=2); ax.set_xlabel("node width h"); ax.set_title(t, fontsize=10)
axs[0].set_yscale("log"); axs[1].set_yscale("log"); axs[2].axhline(0, c="k", lw=.6)
axs[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/figs/size_vs_width.png", dpi=150); plt.close()

# ---------------------------------------------------------------- depth profiles (main runs)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, ds in zip(axs, ["catsdogs", "cifar10"]):
    f = glob.glob(f"results/main_{ds}_kmeans_shared_r16_h16_margin.json")
    if not f:
        continue
    P = json.load(open(f[0]))["depth_profile"]
    d = [p["depth"] for p in P]
    ax.plot(d, [p["train_acc"] for p in P], "o-", label="train acc")
    ax.plot(d, [p["test_acc"] for p in P], "s-", label="test acc")
    for p in P:
        ax.annotate(f"{p['nodes']} nodes\n{p['params']/1e3:.0f}k par.", (p["depth"], p["train_acc"]), fontsize=6,
                    xytext=(3, -22), textcoords="offset points")
    ax.set_xlabel("tree truncated at depth"); ax.set_title(f"{ds}: all nodes h=16", fontsize=10)
    ax.set_xticks(d); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/figs/depth_profile.png", dpi=150); plt.close()

# ---------------------------------------------------------------- tables for the report
tab = {}
for ds in ["catsdogs", "cifar10"]:
    for dc in ["paper", "margin"]:
        for r in sel(ds=ds, noise=0.0, dontcare=dc, trunk="kmeans"):
            tab.setdefault(f"dontcare|{ds}|{r['h']}", {})[dc] = dict(nodes=r["nodes"], params=r["params"], test=r["test_acc"], depth=r["depth"])
    for tr in ["random", "kmeans"]:
        for r in sel(ds=ds, noise=0.0, dontcare="margin", trunk=tr):
            tab.setdefault(f"trunk|{ds}|{r['h']}", {})[tr] = dict(nodes=r["nodes"], params=r["params"], test=r["test_acc"],
                                                                 root_test=r["root_test_acc"], root_train=r["root_train_acc"])
json.dump(tab, open("results/ablation_tables.json", "w"), indent=1)
print("figures done;", len(rows), "sweep rows")

# ---------------------------------------------------------------- what do corrections do on test data?
from treenet.utils import load
flip = {}
for ds in ["catsdogs", "cifar10"]:
    path = f"results/main_{ds}_kmeans_shared_r16_h16_margin.pkl"
    if not glob.glob(path):
        continue
    m = load(path)
    f = np.load(f"data/feat_{ds}_kmeans.npz")
    Xtr, ytr, Xte, yte = f["Xtr"], f["Ytr"], f["Xte"], f["Yte"]
    rows_f = []
    prev = m.predict(Xte, 0)
    for d in range(1, m.depth() + 1):
        cur = m.predict(Xte, d)
        ch = cur != prev
        rows_f.append(dict(depth=d, changed=int(ch.sum()), fixed=int((ch & (cur == yte)).sum()),
                           broken=int((ch & (prev == yte)).sum()), wrong_to_wrong=int((ch & (cur != yte) & (prev != yte)).sum())))
        prev = cur
    # firing rates of corrector nodes: on their training positives vs on the test set
    fire = []
    for t in m.trees:
        for n in t.nodes:
            if n.depth == 0 or n.model.kind == "view":
                continue
            fire.append(dict(depth=n.depth, role=n.role, npos=int(len(n.idx_pos)),
                             train_rate=float(len(n.idx_pos) / len(Xtr)),
                             train_fire_rate_all=float((n.model.raw(Xtr) >= 0).mean()),
                             test_fire_rate=float((n.model.raw(Xte) >= 0).mean())))
    flip[ds] = dict(flips=rows_f, firing=fire)
json.dump(flip, open("results/correction_effects.json", "w"), indent=1)

# RF statistics summary robust to NaN (zero-gradient nodes)
rf = json.load(open("results/receptive_fields.json")) if glob.glob("results/receptive_fields.json") else None
if rf:
    summ = {}
    for ds, R in rf["nodes"].items():
        for d in sorted(set(r["depth"] for r in R)):
            S = [r for r in R if r["depth"] == d]
            summ[f"{ds}|{d}"] = {k: float(np.nanmean([r[k] for r in S])) for k in ["coherence", "concentration", "self_match", "npos"]}
            summ[f"{ds}|{d}"]["n_nodes"] = len(S)
    json.dump(summ, open("results/rf_summary.json", "w"), indent=1)
print("analysis done")
