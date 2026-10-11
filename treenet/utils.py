import pickle
import numpy as np
import jax


def to_numpy(obj):
    """Recursively convert jax arrays inside model objects to numpy (for pickling)."""
    for n in getattr(obj, "trees", []):
        for node in n.nodes:
            m = node.model
            if hasattr(m, "p"):
                m.p = jax.tree_util.tree_map(np.asarray, m.p)
    if getattr(obj, "shared", None) is not None:
        obj.shared.p = jax.tree_util.tree_map(np.asarray, obj.shared.p)
    return obj


def save(obj, path):
    with open(path, "wb") as f:
        pickle.dump(to_numpy(obj), f)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def depth_profile(model, Xtr, ytr, Xte, yte):
    """Train/test accuracy, #nodes and #params of the tree truncated at each depth."""
    D = model.depth()
    out = []
    for d in range(D + 1):
        nodes = sum(sum(n.depth <= d for n in t.nodes) for t in model.trees)
        if model.shared is not None:
            nodes -= model.C - 1
        params = sum(sum(n.model.n_params() + (n.A is not None and n.A.depth <= d) + (n.B is not None and n.B.depth <= d)
                         for n in t.nodes if n.depth <= d) for t in model.trees)
        if model.shared is not None:
            params += model.shared.n_params()
        out.append(dict(depth=d, nodes=int(nodes), params=int(params),
                        train_acc=model.accuracy(Xtr, ytr, d), test_acc=model.accuracy(Xte, yte, d)))
    return out
