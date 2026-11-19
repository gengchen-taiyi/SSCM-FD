import argparse
import ast
import json
import operator
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


BASE = Path(__file__).resolve().parent


def graph_from(path):
    a = pd.read_csv(path, index_col=0)
    if not a.index.is_unique or set(a.index) != set(a.columns) or not a.isin([0, 1]).all().all():
        raise ValueError("Expected a binary adjacency matrix with matching row and column names.")
    names = list(a.columns)
    if any(not n.isidentifier() for n in names):
        raise ValueError("Use sensor identifiers such as X1, X2, ...")
    graph = nx.from_numpy_array(a.loc[names, names].to_numpy(), create_using=nx.DiGraph)
    if graph.number_of_edges() == 0 or not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The graph must be a DAG with at least one edge.")
    return names, graph


def read_data(path, names, normal=False):
    frame = pd.read_csv(path)
    missing = set(names) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Empty data or missing variables in {path}: {sorted(missing)}")
    x = frame[names].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite sensor measurements in {path}.")
    labels = None
    if "label" in frame:
        if not frame.label.isin([0, 1]).all() or (normal and frame.label.any()):
            raise ValueError(f"Expected binary monitoring labels or exclusively normal reference rows: {path}")
        labels = frame.label.to_numpy(dtype=int)
    return x, labels


def expression(text, n):
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}
    funcs = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh, "log": np.log,
             "exp": np.exp, "sqrt": np.sqrt, "Abs": np.abs, "abs": np.abs, "atan": np.arctan,
             "asin": np.arcsin, "acos": np.arccos, "sinh": np.sinh, "cosh": np.cosh, "sign": np.sign}
    variables = {f"x_{i+1}": i for i in range(n)}

    def parse(node):
        if isinstance(node, ast.Constant) and type(node.value) in (float, int):
            return lambda x: float(node.value)
        if isinstance(node, ast.Name) and node.id in variables:
            return lambda x: x[:, variables[node.id]]
        if isinstance(node, ast.Name) and node.id in ("pi", "E"):
            return lambda x: np.pi if node.id == "pi" else np.e
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right, op = parse(node.left), parse(node.right), ops[type(node.op)]
            return lambda x: op(left(x), right(x))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            f = parse(node.operand)
            return f if isinstance(node.op, ast.UAdd) else lambda x: -f(x)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in funcs
                and len(node.args) == 1 and not node.keywords):
            f, func = parse(node.args[0]), funcs[node.func.id]
            return lambda x: func(f(x))
        raise ValueError(f"Unsupported symbolic expression component: {ast.dump(node)}")

    return parse(ast.parse(text, mode="eval").body)


def mechanism(path, mode, device, target, parents, args):
    if mode == "numeric":
        import torch
        from KAN_1 import build
        saved = torch.load(path, map_location=device, weights_only=True)
        config = saved["config"]
        if not args.calibration and (config["seed"] != args.seed or config["test_ratio"] != args.cal_fraction):
            raise ValueError(f"{target}: checkpoint split differs; match --seed/--cal-fraction or supply --calibration.")
    else:
        saved = json.loads(path.read_text(encoding="utf-8"))
        config = saved["metrics"]
    order = config["parents"]
    if config["target"] != target or set(order) != set(parents) or len(order) != len(parents):
        raise ValueError(f"{path}: mechanism target or parents differ from the supplied graph.")
    if mode == "numeric":
        model = build(config, device)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        model.save_act = False
        lower, span = np.array(config["x_min"]), np.array(config["x_span"])

        def predict(x):
            with torch.inference_mode():
                z = torch.as_tensor(2 * (x - lower) / span - 1, dtype=torch.float32, device=device)
                parts = [model(block).flatten().cpu().numpy().astype(float) for block in z.split(512)]
            return np.concatenate(parts) * config["y_std"] + config["y_mean"]
    else:
        formula = expression(saved["formula_raw"], len(order))
        scaler = config["scaler"]
        if (scaler["x_scale"] not in ("minmax_m11", "none") or scaler["y_scale"] not in ("standard", "none")
                or scaler["y_link"] not in ("logit", "none")):
            raise ValueError(f"{path}: unsupported transformation.")

        def predict(x):
            if scaler["x_scale"] == "minmax_m11":
                x = 2 * (x - scaler["x_min"]) / scaler["x_span"] - 1
            y = np.broadcast_to(formula(x), (len(x),))
            if scaler["y_scale"] == "standard":
                y = y * scaler["y_std"] + scaler["y_mean"]
            return expit(y) if scaler["y_link"] == "logit" else y
    return order, predict


class Monitor:
    def __init__(self, names, graph, models, reference, calibration, args):
        self.names, self.graph, self.args = names, graph, args
        self.reference = reference
        self.models = models
        self.edges = list(graph.edges)
        self.topo = list(nx.topological_sort(graph))
        self.roots = [v for v in graph if graph.in_degree(v) == 0]
        descendants = {v: nx.descendants(graph, v) for v in graph}
        self.desc = {v: [d for d in self.topo if d in members] for v, members in descendants.items()}
        self.delta = np.maximum(args.kappa * np.subtract(*np.quantile(reference, [.75, .25], axis=0)), 1e-6)
        self.scale = reference.std(axis=0)
        self.scale[self.scale < 1e-12] = 1
        self.km = KMeans(n_clusters=args.K, random_state=args.seed, n_init=10).fit(reference[:, self.roots])
        self.regimes = self.context(reference)
        if len(np.unique(self.regimes)) != args.K:
            raise ValueError("Not enough distinct reference operating conditions for K clusters.")
        effects = self.effects(reference)
        bands = [np.quantile(effects[self.regimes == k], [args.xi, 1 - args.xi], axis=0) for k in range(args.K)]
        self.lo, self.hi = np.stack(bands, axis=1)
        self.ce_threshold = float(np.quantile(self.excursions(calibration, self.context(calibration)).sum(1), args.theta))
        self.res_threshold = None
        if args.version == 2:
            r = self.residual(reference)
            self.center = np.median(r, axis=0)
            self.mad = np.maximum(np.median(np.abs(r - self.center), axis=0), 1e-6)
            self.res_threshold = float(np.quantile(self.res_score(calibration), args.res_quantile))

    def context(self, x):
        return self.km.predict(x[:, self.roots])

    def predict(self, v, inputs):
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                result = np.asarray(self.models[v][1](inputs), dtype=float)
        except (FloatingPointError, ValueError, OverflowError) as err:
            raise ValueError(f"Invalid mechanism evaluation for {self.names[v]}") from err
        if result.shape != (len(inputs),) or not np.isfinite(result).all():
            raise ValueError(f"Non-finite or invalid prediction for {self.names[v]}; check model domain and scaling.")
        return result

    def effects(self, x):
        eta = np.empty((len(x), len(self.edges)))
        for e, (u, v) in enumerate(self.edges):
            pa = self.models[v][0]
            plus, minus = x[:, pa].copy(), x[:, pa].copy()
            j = pa.index(u)
            plus[:, j] += self.delta[u]
            minus[:, j] -= self.delta[u]
            eta[:, e] = (self.predict(v, plus) - self.predict(v, minus)) / (2 * self.delta[u])
        return eta

    def excursions(self, x, ctx):
        eta = self.effects(x)
        lo, hi = self.lo[ctx], self.hi[ctx]
        return np.maximum(0, np.maximum(lo - eta, eta - hi)) / np.maximum(hi - lo, 1e-6)

    def residual(self, x):
        return np.column_stack([x[:, v] - self.predict(v, x[:, pa]) for v, (pa, _) in self.models.items()])

    def res_score(self, x):
        return (np.abs(self.residual(x) - self.center) / self.mad).max(axis=1)

    def repair(self, x, ctx, u):
        out = x.copy()
        if u in self.models:
            out[:, u] = self.predict(u, x[:, self.models[u][0]])
        else:
            other = [v for v in self.roots if v != u]
            for k in np.unique(ctx):
                ref = self.reference[self.regimes == k]
                for t in np.flatnonzero(ctx == k):
                    if other:
                        dist = np.sum(((ref[:, other] - x[t, other]) / self.scale[other]) ** 2, axis=1)
                        chosen = np.argsort(dist, kind="stable")[:self.args.neighbors]
                        out[t, u] = np.median(ref[chosen, u])
                    else:
                        out[t, u] = np.median(ref[:, u])
        for v in self.desc[u]:
            out[:, v] = self.predict(v, out[:, self.models[v][0]])
        return out

    def diagnose(self, x):
        ctx = self.context(x)
        strength = self.excursions(x, ctx)
        before = strength.sum(axis=1)
        ce_alarm = before > self.ce_threshold
        alarm = ce_alarm.copy()
        scores = pd.DataFrame(dict(sample_index=np.arange(len(x)), regime=ctx, score_before=before,
                                   ce_threshold=self.ce_threshold, ce_alarm=ce_alarm.astype(int)))
        if self.res_threshold is not None:
            r = self.res_score(x)
            res_alarm = r >= self.res_threshold
            alarm &= res_alarm
            scores = scores.assign(residual_max=r, residual_threshold=self.res_threshold, residual_alarm=res_alarm.astype(int))
        scores["pred"] = alarm.astype(int)
        priority = [{} for _ in x]
        top_edges = [[] for _ in x]
        for i in np.flatnonzero(alarm):
            for e in np.argsort(-strength[i], kind="stable")[:self.args.top_edges]:
                if strength[i, e] <= 0:
                    break
                u, v = self.edges[e]
                top_edges[i].append([self.names[u], self.names[v]])
                for node in (u, v):
                    priority[i][node] = priority[i].get(node, 0.0) + strength[i, e]
        by_sample = [[] for _ in x]
        for u in self.graph:
            idx = np.array([i for i, p in enumerate(priority) if u in p], dtype=int)
            if len(idx):
                repaired = self.repair(x[idx], ctx[idx], u)
                after = self.excursions(repaired, ctx[idx]).sum(axis=1)
                for i, a in zip(idx, after):
                    by_sample[i].append((u, float(a), float(before[i] - a)))
        rows, ranks, best_after, best_gain = [], [], [], []
        for i, candidates in enumerate(by_sample):
            candidates.sort(key=lambda c: (-c[2], -priority[i][c[0]], c[0]))
            ranks.append([self.names[c[0]] for c in candidates])
            best_after.append(candidates[0][1] if candidates else np.nan)
            best_gain.append(candidates[0][2] if candidates else np.nan)
            for j, (u, a, gain) in enumerate(candidates, 1):
                rows.append((i, ctx[i], j, self.names[u], priority[i][u], before[i], a, gain))
        scores["top_anomalous_edges"] = [json.dumps(es) for es in top_edges]
        scores["root_top1"] = [r[0] if r else "" for r in ranks]
        scores["root_top3"] = [json.dumps(r[:3]) for r in ranks]
        scores["score_after"] = best_after
        scores["score_reduction"] = best_gain
        details = pd.DataFrame(rows, columns=["sample_index","regime","rank","candidate","priority",
                                              "score_before","score_after","score_reduction"])
        return scores, details


def run(version):
    p = argparse.ArgumentParser(description="Regime-calibrated KAN fault detection and virtual-repair root localization.")
    p.add_argument("--normal", type=Path, default=BASE / "Sample_data.csv")
    p.add_argument("--csv", type=Path, required=True, help="Monitoring observations; label is optional")
    p.add_argument("--adj", type=Path, default=BASE / "causal_discovery_results" / "adjacency.csv")
    p.add_argument("--model", choices=["numeric", "symbolic"], default="numeric")
    p.add_argument("--models", type=Path)
    p.add_argument("--fit-missing", action="store_true", help="Train missing numeric KANs from the normal CSV")
    p.add_argument("--fit-steps", type=int, default=120)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--calibration", type=Path, help="Independent normal calibration CSV instead of a normal holdout")
    p.add_argument("--cal-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K", type=int, default=7)
    p.add_argument("--xi", type=float, default=0.005)
    p.add_argument("--kappa", type=float, default=0.1)
    p.add_argument("--theta", type=float, default=0.996)
    p.add_argument("--top-edges", type=int, default=3)
    p.add_argument("--neighbors", type=int, default=10)
    if version == 2:
        p.add_argument("--res-quantile", type=float, default=0.99)
    p.add_argument("--true-roots", help="Optional shared injected root names, comma-separated; evaluation only")
    p.add_argument("--out-dir", type=Path, default=BASE / f"fault_diagnosis_results_{version}")
    args = p.parse_args()
    args.version = version
    if not (0 < args.xi < .5 and 0 < args.theta < 1 and args.kappa > 0 and 0 < args.cal_fraction < 1
            and min(args.K, args.top_edges, args.neighbors, args.fit_steps) >= 1
            and (version == 1 or 0 < args.res_quantile < 1)):
        p.error("Invalid quantile, perturbation, holdout fraction, or count.")
    if args.fit_missing and args.model != "numeric":
        p.error("--fit-missing is only for numeric models; generate symbolic artifacts with KAN_2.py.")
    if not args.adj.is_file():
        p.error(f"Missing {args.adj}. Run causal_discovery.py first.")
    names, graph = graph_from(args.adj)
    x, labels = read_data(args.csv, names)
    normal, _ = read_data(args.normal, names, normal=True)
    if args.calibration:
        reference, calibration = normal, read_data(args.calibration, names, normal=True)[0]
    else:
        reference, calibration = train_test_split(normal, test_size=args.cal_fraction, random_state=args.seed)
    if len(reference) < args.K or len(calibration) < 2:
        p.error("Insufficient normal reference or calibration rows.")
    truth = {v.strip() for v in args.true_roots.split(",")} if args.true_roots else set()
    if truth and (not truth.issubset(names) or labels is None):
        p.error("--true-roots requires known sensor names and a monitoring label column.")
    directory = args.models or BASE / ("KAN_1_results" if args.model == "numeric" else "KAN_2_results")
    paths = {v: directory / (names[v] + ("_model.pt" if args.model == "numeric" else "_result.json"))
             for v in graph if graph.in_degree(v)}
    missing = [names[v] for v, path in paths.items() if not path.is_file()]
    if missing and not args.fit_missing:
        p.error(f"Missing {args.model} mechanisms in {directory}: {', '.join(missing)}. "
                "Use --fit-missing for numeric models or run KAN_2.py for each missing symbolic target.")
    models = {}
    for v, path in paths.items():
        if not path.is_file():
            subprocess.run([sys.executable, str(BASE / "KAN_1.py"), "--data_csv", str(args.normal),
                            "--adj_csv", str(args.adj), "--target", names[v], "--out_dir", str(directory),
                            "--steps", str(args.fit_steps), "--seed", str(args.seed), "--test_ratio",
                            str(args.cal_fraction), "--device", args.device], check=True)
        parents = [names[u] for u in graph.predecessors(v)]
        order, predict = mechanism(path, args.model, args.device, names[v], parents, args)
        models[v] = [names.index(n) for n in order], predict
    with threadpool_limits(limits=2):
        monitor = Monitor(names, graph, models, reference, calibration, args)
        scores, details = monitor.diagnose(x)
    summary = {"version": version, "model": args.model, "reference_rows": len(reference),
               "calibration_rows": len(calibration), "monitoring_rows": len(x), "alerts": int(scores.pred.sum()),
               "ce_threshold": monitor.ce_threshold, "residual_threshold": monitor.res_threshold,
               "parameters": {k: str(val) if isinstance(val, Path) else val for k, val in vars(args).items()}}
    if labels is not None:
        scores["label"] = labels
        tn, fp, fn, tp = confusion_matrix(labels, scores.pred, labels=[0, 1]).ravel()
        summary.update(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                       precision=float(precision_score(labels, scores.pred, zero_division=0)),
                       recall=float(recall_score(labels, scores.pred, zero_division=0)),
                       f1=float(f1_score(labels, scores.pred, zero_division=0)))
        if len(np.unique(labels)) == 2:
            for key, col in [("ce", "score_before"), ("residual", "residual_max")]:
                if col in scores:
                    summary[key + "_auc"] = float(roc_auc_score(labels, scores[col]))
                    summary[key + "_ap"] = float(average_precision_score(labels, scores[col]))
        if truth:
            for k in (1, 3):
                hits = sum(y == 1 and bool(truth.intersection(json.loads(r)[:k])) for y, r in zip(labels, scores.root_top3))
                summary[f"hit_at_{k}"] = dict(hits=int(hits), detected=float(hits/tp) if tp else None,
                                               all_faults=float(hits/(tp+fn)) if tp+fn else None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.out_dir / "scores.csv", index=False)
    details.to_csv(args.out_dir / "repairs.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Version {version}: {len(x)} rows, {summary['alerts']} alarms. Results: {args.out_dir.resolve()}")
