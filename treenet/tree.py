"""Tree-type network whose nodes are small neural networks.

Growth rule (lecture notes, Tables I & II), applied recursively to every node:
    net   = raw output of the node;  decision = [net >= 0]
    C3    = target 1, decision 0  (false negatives)  -> child A must fire (weight w_A > 0)
    C4    = target 0, decision 1  (false positives)  -> child B must fire (weight w_B < 0)
    corrected net = net + w_A * step(A) - |w_B| * step(B)
with  w_A = max_{C3}(-net) + margin,  |w_B| = max_{C4}(net) + margin  (the "by inspection"
choice of Fig. 4).  Child A must output 1 on C3 and 0 on every target-0 sample whose decision
*would* flip if A fired; child B symmetrically.  The lecture marks all of C2 (for A) and all
of C1 (for B) as don't-care.  We add a sharper, still exact, rule:

  margin-aware don't-cares: a target-0 sample with net < -w_A cannot be flipped by A, and a
  target-1 sample with net >= |w_B| cannot be flipped by B, so they are don't-cares too.

This keeps every child's training set small.  Children are grown with the same rule until
each node is error-free on its own domain; by induction the root is then error-free on the
whole training set (100% training accuracy).
"""
import time
import numpy as np
from .node import MLPNode, BallNode, SharedRoot, MarginView


class TreeNode:
    def __init__(self, model, depth, role, parent):
        self.model, self.depth, self.role, self.parent = model, depth, role, parent
        self.A = self.B = None
        self.wA = self.wB = 0.0
        self.info = {}


class BinaryTree:
    """One error-correcting tree for a binary target (1 = positive)."""

    def __init__(self, h=16, dontcare="margin", margin=0.05, weighting="sqrt", max_nodes=3000,
                 train_kw=None, seed=0, verbose=False, root_h=None):
        self.root_h = root_h or h
        self.h, self.dontcare, self.margin, self.weighting = h, dontcare, margin, weighting
        self.max_nodes, self.train_kw, self.seed, self.verbose = max_nodes, train_kw or {}, seed, verbose
        self.nodes = []

    # ------------------------------------------------------------------ growth
    def _new_mlp(self, X, t, depth, role, parent, idx):
        h = self.root_h if role == "root" else self.h
        if h == 0:  # pure "ball" network (nearest-prototype style), used as an extreme baseline
            return self._new_ball(X, t, depth, role, parent, idx)
        m = MLPNode(X.shape[1], h, seed=self.seed + len(self.nodes)).fit(
            X, t, weighting=self.weighting, **self.train_kw)
        return self._register(m, depth, role, parent, idx, t)

    def _new_ball(self, X, t, depth, role, parent, idx):
        m = BallNode(X.shape[1]).fit(X, t, seed=self.seed + len(self.nodes))
        return self._register(m, depth, role, parent, idx, t)

    def _register(self, m, depth, role, parent, idx, t):
        n = TreeNode(m, depth, role, parent)
        n.info = dict(domain=len(idx), npos=int(t.sum()), kind=m.kind)
        n.idx_pos = idx[t == 1]          # global training indices this node must fire on
        n.dom, n.t = idx, t              # the node's domain and its binary targets there
        self.nodes.append(n)
        return n

    def fit(self, X, t, root_model=None):
        """X (n,d) features, t (n,) in {0,1}.  root_model: optional pre-trained root (MarginView)."""
        t0 = time.time()
        idx = np.arange(len(X))
        if root_model is None:
            root = self._new_mlp(X, t, 0, "root", None, idx)
        else:
            root = self._register(root_model, 0, "root", None, idx, t)
        stack = [(root, idx, t)]
        while stack:
            node, dom, td = stack.pop()
            net = node.model.raw(X[dom])
            pred = net >= 0
            c3 = (td == 1) & ~pred
            c4 = (td == 0) & pred
            node.info.update(fn=int(c3.sum()), fp=int(c4.sum()))
            if len(self.nodes) >= self.max_nodes and (c3.any() or c4.any()):
                node.info["truncated"] = True
                continue
            for side, err in (("A", c3), ("B", c4)):
                if not err.any():
                    continue
                if side == "A":
                    w = float((-net[c3]).max() + self.margin)
                    if self.dontcare == "margin":
                        keep = c3 | ((td == 0) & (net + w >= 0))
                    else:  # lecture rule: C2 is don't-care
                        keep = ~((td == 1) & pred)
                    tc = c3[keep].astype(np.int32)
                else:
                    w = float(net[c4].max() + self.margin)
                    if self.dontcare == "margin":
                        keep = c4 | ((td == 1) & (net - w < 0))
                    else:  # lecture rule: C1 is don't-care
                        keep = ~((td == 0) & ~pred)
                    tc = c4[keep].astype(np.int32)
                cdom = dom[keep]
                Xc = X[cdom]
                child = self._new_mlp(Xc, tc, node.depth + 1, side, node, cdom)
                # termination guard: the child must make progress (fire on >=1 of its positives
                # and leave >=1 sample of its domain correctly silent); otherwise use a ball node.
                s = child.model.raw(Xc) >= 0
                no_progress = not (s & (tc == 1)).any() or len(cdom) >= len(dom)
                if no_progress and child.model.kind == "mlp":
                    self.nodes.pop()
                    try:
                        child = self._new_ball(Xc, tc, node.depth + 1, side, node, cdom)
                    except RuntimeError:
                        # identical inputs with conflicting targets (e.g. duplicate images whose
                        # labels disagree): no network can separate them -> irreducible error
                        node.info["irreducible"] = int(tc.sum())
                        continue
                if side == "A":
                    node.A, node.wA = child, w
                else:
                    node.B, node.wB = child, w
                stack.append((child, cdom, tc))
            if self.verbose and len(self.nodes) % 25 == 0:
                print(f"  nodes={len(self.nodes)} depth={node.depth} dom={len(dom)}", flush=True)
        self.root = root
        self.fit_seconds = time.time() - t0
        return self

    # -------------------------------------------------------------- inference
    def corrected(self, X, max_depth=None, raw_cache=None):
        """Corrected net input of the root (vectorised, bottom-up, no recursion)."""
        order = sorted(range(len(self.nodes)), key=lambda i: -self.nodes[i].depth)
        pos = {id(n): i for i, n in enumerate(self.nodes)}
        val = [None] * len(self.nodes)
        for i in order:
            n = self.nodes[i]
            if max_depth is not None and n.depth > max_depth:
                continue
            r = n.model.raw(X) if raw_cache is None or i not in raw_cache else raw_cache[i]
            r = r.copy()
            if n.A is not None and val[pos[id(n.A)]] is not None:
                r += n.wA * (val[pos[id(n.A)]] >= 0)
            if n.B is not None and val[pos[id(n.B)]] is not None:
                r -= n.wB * (val[pos[id(n.B)]] >= 0)
            val[i] = r
        return val[0]

    def predict(self, X, max_depth=None):
        return (self.corrected(X, max_depth) >= 0).astype(int)

    # ------------------------------------------------------------- bookkeeping
    def n_params(self):
        corr = sum((n.A is not None) + (n.B is not None) for n in self.nodes)  # w_A, w_B
        return int(sum(n.model.n_params() for n in self.nodes) + corr)

    def depth(self):
        return max(n.depth for n in self.nodes)

    def summary(self):
        kinds = {}
        for n in self.nodes:
            kinds[n.model.kind] = kinds.get(n.model.kind, 0) + 1
        return dict(nodes=len(self.nodes), depth=self.depth(), params=self.n_params(), kinds=kinds,
                    irreducible=int(sum(n.info.get("irreducible", 0) for n in self.nodes)),
                    truncated=any(n.info.get("truncated") for n in self.nodes))


class TreeClassifier:
    """Binary: one tree.  Multiclass: one error-correcting tree per class (one-vs-rest), with
    either independent roots ('ovr') or a single shared multiclass root MLP ('shared')."""

    def __init__(self, n_classes, h=16, mode="shared", root_h=None, **tree_kw):
        self.C, self.h, self.mode = n_classes, h, mode
        self.root_h = root_h or h
        self.tree_kw = tree_kw

    def fit(self, X, y, verbose=False):
        t0 = time.time()
        self.shared = None
        if self.C == 2:
            self.trees = [BinaryTree(h=self.h, root_h=self.root_h, **self.tree_kw).fit(
                X, (y == 1).astype(np.int32))]
        else:
            if self.mode == "shared":
                self.shared = SharedRoot(X.shape[1], self.root_h, self.C, seed=self.tree_kw.get("seed", 0)).fit(
                    X, y, **self.tree_kw.get("train_kw", {}))
            self.trees = []
            for c in range(self.C):
                bt = BinaryTree(h=self.h, root_h=self.root_h,
                                **{**self.tree_kw, "seed": self.tree_kw.get("seed", 0) + 1000 * c})
                rm = MarginView(self.shared, c) if self.shared is not None else None
                bt.fit(X, (y == c).astype(np.int32), root_model=rm)
                self.trees.append(bt)
                if verbose:
                    print(f" class {c}: {bt.summary()}", flush=True)
        self.fit_seconds = time.time() - t0
        return self

    def scores(self, X, max_depth=None):
        if self.C == 2:
            return self.trees[0].corrected(X, max_depth)
        return np.stack([t.corrected(X, max_depth) for t in self.trees], 1)

    def predict(self, X, max_depth=None):
        s = self.scores(X, max_depth)
        return (s >= 0).astype(int) if self.C == 2 else s.argmax(1)

    def accuracy(self, X, y, max_depth=None):
        return float((self.predict(X, max_depth) == y).mean())

    def n_params(self):
        p = sum(t.n_params() for t in self.trees)
        if self.C > 2 and self.shared is not None:
            p += self.shared.n_params()
        return int(p)

    def n_nodes(self):
        # a shared root is one network but appears as a (parameter-free) view in each class tree
        return int(sum(len(t.nodes) for t in self.trees)) - (self.C - 1 if self.shared is not None else 0)

    def depth(self):
        return int(max(t.depth() for t in self.trees))

    def summary(self):
        s = dict(nodes=self.n_nodes(), params=self.n_params(), depth=self.depth(),
                 per_tree=[t.summary() for t in self.trees])
        s["kinds"] = {}
        for t in self.trees:
            for k, v in t.summary()["kinds"].items():
                s["kinds"][k] = s["kinds"].get(k, 0) + v
        return s


def subtree_nodes(node):
    out, st = [], [node]
    while st:
        n = st.pop(); out.append(n)
        st += [c for c in (n.A, n.B) if c is not None]
    return out


def subtree_corrected(node, X, raw_override=None):
    """Corrected net of `node` on X (children evaluated bottom-up)."""
    nodes = sorted(subtree_nodes(node), key=lambda n: -n.depth)
    val = {}
    for n in nodes:
        r = (raw_override[id(n)] if raw_override and id(n) in raw_override else n.model.raw(X)).copy()
        if n.A is not None:
            r += n.wA * (val[id(n.A)] >= 0)
        if n.B is not None:
            r -= n.wB * (val[id(n.B)] >= 0)
        val[id(n)] = r
    return val[id(node)]
