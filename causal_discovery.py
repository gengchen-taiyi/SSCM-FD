import argparse
import csv
import json
from collections import Counter
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import cdist, pdist
from threadpoolctl import threadpool_limits


BASE = Path(__file__).resolve().parent


def load_data(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        names = next(reader, [])
        if len(names) < 2 or len(set(names)) != len(names) or any(not v.strip() for v in names):
            raise ValueError("Expected at least two uniquely named variable columns.")
        if any(v.strip().lower() in {"label", "index", "id"} for v in names):
            raise ValueError("Input must contain only normal sensor measurements, without labels or IDs.")
        data = np.asarray(list(reader), dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(names) or len(data) < 20 or not np.isfinite(data).all():
        raise ValueError("Expected at least 20 complete, finite numeric observations.")
    scale = data.std(axis=0)
    if (scale == 0).any():
        raise ValueError("Constant variables: " + ", ".join(n for n, s in zip(names, scale) if s == 0))
    return names, (data - data.mean(axis=0)) / scale


class KernelEvidence:
    def __init__(self, data, args):
        self.data = data
        self.n = len(data)
        self.alpha = args.alpha
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(self.n)
        self.landmarks = order[:min(args.rank, self.n)]
        self.bandwidth_rows = order[:min(160, self.n)]
        signs = rng.choice([-1.0, 1.0], size=(args.resamples, self.n))
        self.weights = signs - signs.mean(axis=1, keepdims=True)

    @lru_cache(maxsize=128)
    def features(self, group):
        x = self.data[:, group]
        distances = pdist(x[self.bandwidth_rows], metric="sqeuclidean")
        positive = distances[distances > 0]
        width = np.median(positive) if len(positive) else 1.0
        kernel = np.exp(-cdist(x, x[self.landmarks], metric="sqeuclidean") / (2 * width))
        values, vectors = eigh(kernel[self.landmarks])
        keep = values > max(1e-12, values[-1] * 1e-10)
        phi = kernel @ (vectors[:, keep] / np.sqrt(values[keep]))
        return phi - phi.mean(axis=0)

    @lru_cache(maxsize=128)
    def ridge(self, group):
        z = self.features(group)
        return cho_factor(z.T @ z / self.n + self.alpha * np.eye(z.shape[1]))

    def residual(self, features, group):
        if not group:
            return features
        z = self.features(group)
        return features - z @ cho_solve(self.ridge(group), z.T @ features / self.n)

    @lru_cache(maxsize=None)
    def ci_pvalue(self, i, j, group):
        x = self.features(tuple(sorted((i,) + group)))
        y = self.features((j,))
        x = self.residual(x, group)
        y = self.residual(y, group)
        products = (x[:, :, None] * y[:, None, :]).reshape(self.n, -1)
        observed = np.square(products.sum(axis=0)).sum()
        null = np.square(self.weights @ products).sum(axis=1)
        return (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)

    @lru_cache(maxsize=None)
    def trace(self, target, group):
        x = self.features((target,))
        return np.sum(x * self.residual(x, group)) / self.n

    @lru_cache(maxsize=None)
    def conflict(self, i, j, group):
        cross = self.features((i,)).T @ self.residual(self.features((j,)), group)
        return np.square(cross / self.n).sum()


def partition(evidence, count, args):
    graph = nx.complete_graph(count)
    for depth in range(args.ci_order + 1):
        remove = []
        for i, j in graph.edges:
            groups = set()
            for a, b in ((i, j), (j, i)):
                groups.update(combinations(sorted(set(graph[a]) - {b}), depth))
            if any(evidence.ci_pvalue(i, j, group) > args.ci_level for group in sorted(groups)):
                remove.append((i, j))
        graph.remove_edges_from(remove)
    best = None
    for size in range(min(args.max_cut, count - 2) + 1):
        for cut in combinations(range(count), size):
            parts = sorted(tuple(sorted(c)) for c in nx.connected_components(
                graph.subgraph(set(graph) - set(cut))))
            if len(parts) < 2:
                continue
            key = (max(map(len, parts)), tuple(cut))
            if best is None or key < best[0]:
                best = key, cut, parts
        if best is not None:
            return best[1], best[2], graph
    return (), [tuple(range(count))], graph


def boundary(evidence, target, region, args):
    chosen = ()
    remaining = set(region) - {target}
    base = evidence.trace(target, ())
    tolerance = args.mb_tol * base
    while remaining and len(chosen) < args.mb_max:
        cost, v = min((evidence.trace(target, tuple(sorted((*chosen, u)))), u) for u in remaining)
        if evidence.trace(target, chosen) - cost <= tolerance:
            break
        chosen = tuple(sorted((*chosen, v)))
        remaining.remove(v)
    while chosen:
        cost, subset = min((evidence.trace(target, chosen[:i] + chosen[i+1:]), chosen[:i] + chosen[i+1:])
                           for i in range(len(chosen)))
        if cost > evidence.trace(target, chosen) + tolerance:
            break
        chosen = subset
    return chosen


def bits(mask):
    while mask:
        one = mask & -mask
        yield one.bit_length() - 1
        mask ^= one


def transitive_parents(parents):
    closure = list(parents)
    for k, ancestors in enumerate(closure):
        for i, mask in enumerate(closure):
            if mask & (1 << k):
                closure[i] |= ancestors
    return closure


def d_separated(i, j, given, parents, closure):
    family = given | (1 << i) | (1 << j)
    ancestral = family
    for v in bits(family):
        ancestral |= closure[v]
    neighbors = [0] * len(parents)
    for v in bits(ancestral):
        parents_v = parents[v]
        neighbors[v] |= parents_v
        for p in bits(parents_v):
            neighbors[p] |= parents_v | (1 << v)
    seen = given
    frontier = 1 << i
    while frontier:
        if frontier & (1 << j):
            return False
        seen |= frontier
        adjacent = 0
        for v in bits(frontier):
            adjacent |= neighbors[v]
        frontier = adjacent & ~seen
    return True


def local_dag(evidence, nodes, beta):
    size = len(nodes)
    statements = []
    for i, j in combinations(range(size), 2):
        others = sorted(set(range(size)) - {i, j})
        for k in range(len(others) + 1):
            for given in combinations(others, k):
                weight = evidence.conflict(nodes[i], nodes[j], tuple(nodes[v] for v in given))
                statements.append((i, j, sum(1 << v for v in given), weight))

    @lru_cache(maxsize=None)
    def score(parents):
        closure = transitive_parents(parents)
        if any(a & (1 << v) for v, a in enumerate(closure)):
            return float("inf")
        return (beta * sum(len(tuple(bits(p))) for p in parents)
                + sum(w for i, j, g, w in statements if d_separated(i, j, g, parents, closure)))

    state = (0,) * size
    while True:
        best, cost = state, score(state)
        for i in range(size):
            for j in range(size):
                if i == j or state[i] & (1 << j):
                    continue
                changed = list(state)
                changed[j] ^= 1 << i
                moves = [tuple(changed)]
                if state[j] & (1 << i):
                    changed[i] |= 1 << j
                    moves.append(tuple(changed))
                for move in moves:
                    value = score(move)
                    if value < cost - 1e-12:
                        best, cost = move, value
        if best == state:
            break
        state = best
    return [(nodes[i], nodes[j]) for j, mask in enumerate(state) for i in bits(mask)]


def discover(names, data, args):
    evidence = KernelEvidence(data, args)
    cut, parts, skeleton = partition(evidence, len(names), args)
    votes = Counter()
    learned = {}
    records = []
    for part in parts:
        region = tuple(sorted((*part, *cut)))
        for t in region:
            mb = boundary(evidence, t, region, args)
            nodes = tuple(sorted((*mb, t)))
            if nodes not in learned:
                learned[nodes] = local_dag(evidence, nodes, args.beta)
            dag = learned[nodes]
            votes.update(dag)
            records.append(dict(region=[names[v] for v in region], target=names[t],
                                boundary=[names[v] for v in mb], edges=[[names[i], names[j]] for i, j in dag]))
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(names)))
    ranked = sorted(votes, key=lambda e: (-votes[e], e))
    for i, j in ranked[:args.max_edges]:
        if votes[i, j] > args.nu and not nx.has_path(graph, j, i):
            graph.add_edge(i, j)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("The global graph contains a directed cycle.")
    report = dict(observations=len(data), variables=names, cut_set=[names[i] for i in cut],
                  clusters=[[names[i] for i in p] for p in parts],
                  partition_skeleton=[[names[i], names[j]] for i, j in skeleton.edges],
                  local_graphs=records, edges=[[names[i], names[j]] for i, j in graph.edges],
                  parameters={k: v for k, v in vars(args).items() if k not in {"csv", "out_dir"}})
    return graph, votes, ranked, report


def main():
    parser = argparse.ArgumentParser(description="Kernelized local-to-global causal discovery from normal observations.")
    parser.add_argument("--csv", type=Path, default=BASE / "Sample_data.csv")
    parser.add_argument("--out-dir", type=Path, default=BASE / "causal_discovery_results")
    parser.add_argument("--alpha", type=float, default=0.002, help="Kernel operator regularization")
    parser.add_argument("--max-edges", "-M", type=int, default=32, help="Directed candidates retained before cycle checks")
    parser.add_argument("--beta", type=float, default=0.001, help="Local penalty per edge")
    parser.add_argument("--nu", type=float, default=0.5, help="Global penalty per edge in vote units")
    parser.add_argument("--rank", type=int, default=16, help="Maximum Nystrom rank")
    parser.add_argument("--ci-level", type=float, default=0.05)
    parser.add_argument("--ci-order", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--resamples", type=int, default=199)
    parser.add_argument("--max-cut", type=int, default=3, choices=range(5))
    parser.add_argument("--mb-max", type=int, default=4, choices=range(1, 7))
    parser.add_argument("--mb-tol", type=float, default=0.005, help="Fraction of unconditional target trace")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not (args.alpha > 0 and args.beta > 0 and args.nu >= 0 and args.max_edges > 0
            and args.rank >= 2 and args.resamples >= 1 and 0 < args.ci_level < 1
            and 1 / (args.resamples + 1) <= args.ci_level and 0 <= args.mb_tol < 1 and args.seed >= 0):
        parser.error("Invalid penalty, sample count, rank, level, edge limit, tolerance, or seed.")
    names, data = load_data(args.csv)
    with threadpool_limits(limits=1):
        graph, votes, ranked, report = discover(names, data, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "adjacency.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", *names])
        for i, name in enumerate(names):
            writer.writerow([name, *(int(graph.has_edge(i, j)) for j in range(len(names)))])
    with (args.out_dir / "edge_votes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "votes", "in_top_m", "selected"])
        for k, (i, j) in enumerate(ranked):
            writer.writerow([names[i], names[j], votes[i, j], int(k < args.max_edges), int(graph.has_edge(i, j))])
    (args.out_dir / "discovery.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(data)} observations, {len(names)} variables, {len(report['clusters'])} clusters, "
          f"{graph.number_of_edges()} directed edges. Results: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
