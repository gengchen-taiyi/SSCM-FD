import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import sympy as sp
from sympy import lambdify


BASE = Path(__file__).resolve().parent


def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def reorder_x_cols(df: pd.DataFrame) -> pd.DataFrame:
    x_cols = [c for c in df.columns if re.fullmatch(r"[Xx]\d+", str(c).strip())]
    if not x_cols:
        return df
    x_cols = sorted(x_cols, key=lambda c: int(re.findall(r"\d+", c)[0]))
    others = [c for c in df.columns if c not in x_cols]
    return df[x_cols + others].copy()


def load_adj(path: str) -> pd.DataFrame:
    adj = pd.read_csv(path, index_col=0)
    adj = adj.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return adj


def align_by_names(df: pd.DataFrame, adj: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vars_in_both = [c for c in df.columns if c in adj.index and c in adj.columns]
    if not vars_in_both:
        raise ValueError("数据列名与邻接矩阵行列名无交集，请检查命名。")
    return df[vars_in_both].copy(), adj.loc[vars_in_both, vars_in_both].copy()


def parents_of(target: str, adj: pd.DataFrame, orientation: str = "row_to_col") -> List[str]:
    if orientation == "row_to_col":
        pa = adj.index[(adj[target].values != 0)].tolist()
    elif orientation == "col_to_row":
        pa = adj.columns[(adj.loc[target].values != 0)].tolist()
    else:
        raise ValueError("orientation must be row_to_col or col_to_row")
    return [x for x in pa if x != target]


def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def build_width(in_dim: int, hidden: int, depth: int) -> List[int]:
    return [in_dim] + [hidden] * depth + [1]


def fit_minmax_m11(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    span = np.where((xmax - xmin) < 1e-8, 1.0, (xmax - xmin))
    return xmin.astype(np.float32), span.astype(np.float32)


def transform_minmax_m11(X: np.ndarray, xmin: np.ndarray, span: np.ndarray) -> np.ndarray:
    Z = 2.0 * (X - xmin[None, :]) / span[None, :] - 1.0
    return Z.astype(np.float32)


def fit_y_standard(y: np.ndarray) -> Tuple[float, float]:
    mu = float(y.mean())
    sd = float(y.std())
    if sd < 1e-8:
        sd = 1.0
    return mu, sd


def transform_y_standard(y: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return ((y - mu) / sd).astype(np.float32)


def invert_y_standard(y_scaled: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (y_scaled * sd + mu).astype(np.float32)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logit(y: np.ndarray, eps: float) -> np.ndarray:
    y = np.clip(y, eps, 1.0 - eps)
    return np.log(y / (1.0 - y))


def call_fit_or_train(model, dataset: Dict[str, torch.Tensor], *, steps: int, opt: str,
                      lamb: float, lamb_entropy: float, device: str, verbose: int = 1):
    fn = model.fit if hasattr(model, "fit") else model.train
    tried = []
    for kwargs in [
        dict(opt=opt, steps=steps, lamb=lamb, lamb_entropy=lamb_entropy, device=device),
        dict(opt=opt, steps=steps, lamb=lamb, lamb_entropy=lamb_entropy),
        dict(opt=opt, steps=steps, device=device),
        dict(opt=opt, steps=steps),
        dict(steps=steps),
    ]:
        try:
            if verbose:
                print(f"  -> {fn.__name__} kwargs={kwargs}")
            return fn(dataset, **kwargs)
        except TypeError as e:
            tried.append((kwargs, str(e)))
            continue
    raise TypeError("fit/train 参数不兼容，尝试记录：\n" + "\n".join([f"{k} -> {e}" for k, e in tried]))


def try_symbolic(model, lib: List[str], verbose: int = 1) -> str:
    if hasattr(model, "auto_symbolic"):
        try:
            if verbose:
                print(f"  -> auto_symbolic(lib={lib})")
            model.auto_symbolic(lib=lib)
        except Exception as e:
            if verbose:
                print(f"  -> auto_symbolic 失败：{e}")
            return ""

    if hasattr(model, "symbolic_formula"):
        try:
            if verbose:
                print("  -> symbolic_formula()")
            out = model.symbolic_formula()
            expr = out[0][0] if isinstance(out, (list, tuple)) else out
            return str(expr)
        except Exception as e:
            if verbose:
                print(f"  -> symbolic_formula 失败：{e}")
            return ""
    return ""


def rename_formula_vars(formula: str, var_names: List[str]) -> str:
    if not formula:
        return formula
    out = formula
    for i, name in enumerate(var_names, start=1):
        out = re.sub(rf"\bx_{i}\b", str(name), out)
    return out


def eval_symbolic_scaled(formula_raw: str, X_scaled: np.ndarray) -> np.ndarray:
    d = X_scaled.shape[1]
    xs = [sp.Symbol(f"x_{i+1}") for i in range(d)]
    expr = sp.sympify(formula_raw)
    f = lambdify(xs, expr, "numpy")
    result = np.asarray(f(*[X_scaled[:, i].astype(np.float64) for i in range(d)]))
    yhat = np.broadcast_to(result, (len(X_scaled),)).astype(np.float64)
    return yhat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=Path, default=BASE / "Sample_data.csv")
    ap.add_argument("--adj_csv", type=Path, default=BASE / "causal_discovery_results" / "adjacency.csv")
    ap.add_argument("--target", type=str, default="X9")
    ap.add_argument("--out_dir", type=Path, default=BASE / "KAN_2_results")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--orientation", type=str, default="row_to_col", choices=["row_to_col", "col_to_row"])

    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--opt", type=str, default="Adam", choices=["LBFGS", "Adam"])
    ap.add_argument("--grid", type=int, default=15)
    ap.add_argument("--k", type=int, default=3)

    ap.add_argument("--hidden_md", type=int, default=6)
    ap.add_argument("--depth_md", type=int, default=2)

    ap.add_argument("--lamb", type=float, default=1e-4)
    ap.add_argument("--lamb_entropy", type=float, default=0.0)

    ap.add_argument("--lib", type=str, default="x,tanh,log,exp")
    ap.add_argument("--x_scale", type=str, default="minmax_m11", choices=["none", "minmax_m11"])
    ap.add_argument("--y_scale", type=str, default="standard", choices=["none", "standard"])
    ap.add_argument("--y_link", type=str, default="none", choices=["none", "logit"])
    ap.add_argument("--logit_eps", type=float, default=1e-4)

    ap.add_argument("--symbolic_after_fit", action="store_true", default=True, help=argparse.SUPPRESS)
    ap.add_argument("--verbose", type=int, default=1)

    args = ap.parse_args()
    set_seed(args.seed)
    safe_mkdir(args.out_dir)

    from kan import KAN

    if not args.adj_csv.is_file():
        ap.error(f"Missing adjacency: {args.adj_csv}. Run causal_discovery.py first or pass --adj_csv.")
    df = pd.read_csv(args.data_csv)
    df = reorder_x_cols(df)

    adj = load_adj(args.adj_csv)
    df, adj = align_by_names(df, adj)

    if args.target not in df:
        ap.error(f"Target not present in both input tables: {args.target}")
    pa = parents_of(args.target, adj, orientation=args.orientation)
    print(f"Target={args.target} | #Parents={len(pa)} | Parents={pa}")
    if len(pa) == 0:
        print("该变量无父节点，跳过。")
        return

    X = df[pa].to_numpy(dtype=np.float32)
    y = df[[args.target]].to_numpy(dtype=np.float32).reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_ratio, random_state=args.seed
    )

    scaler = {"x_scale": args.x_scale, "y_scale": args.y_scale, "y_link": args.y_link, "logit_eps": args.logit_eps}
    if args.x_scale == "minmax_m11":
        x_min, x_span = fit_minmax_m11(X_train)
        Xtr_s = transform_minmax_m11(X_train, x_min, x_span)
        Xte_s = transform_minmax_m11(X_test, x_min, x_span)
        scaler["x_min"] = x_min.tolist()
        scaler["x_span"] = x_span.tolist()
    else:
        Xtr_s, Xte_s = X_train.astype(np.float32), X_test.astype(np.float32)

    if args.y_link == "logit":
        ytr_link = logit(y_train, args.logit_eps).astype(np.float32)
        yte_link = logit(y_test, args.logit_eps).astype(np.float32)
    else:
        ytr_link = y_train.astype(np.float32)
        yte_link = y_test.astype(np.float32)

    if args.y_scale == "standard":
        y_mu, y_sd = fit_y_standard(ytr_link)
        ytr_s = transform_y_standard(ytr_link, y_mu, y_sd)
        yte_s = transform_y_standard(yte_link, y_mu, y_sd)
        scaler["y_mean"] = y_mu
        scaler["y_std"] = y_sd
    else:
        y_mu, y_sd = 0.0, 1.0
        ytr_s, yte_s = ytr_link.astype(np.float32), yte_link.astype(np.float32)

    dataset = {
        "train_input": to_tensor(Xtr_s, args.device),
        "train_label": to_tensor(ytr_s.reshape(-1, 1), args.device),
        "test_input": to_tensor(Xte_s, args.device),
        "test_label": to_tensor(yte_s.reshape(-1, 1), args.device),
    }

    width = build_width(len(pa), args.hidden_md, args.depth_md)
    print(f"Use KAN width={width} | x_scale={args.x_scale} | y_link={args.y_link} | y_scale={args.y_scale}")

    model = KAN(width=width.copy(), grid=args.grid, k=args.k, seed=args.seed, device=args.device, auto_save=False)

    call_fit_or_train(model, dataset, steps=args.steps, opt=args.opt,
                      lamb=args.lamb, lamb_entropy=args.lamb_entropy,
                      device=args.device, verbose=args.verbose)

    model.eval()
    with torch.no_grad():
        pred_tr_s = model(dataset["train_input"]).detach().cpu().numpy().reshape(-1).astype(np.float32)
        pred_te_s = model(dataset["test_input"]).detach().cpu().numpy().reshape(-1).astype(np.float32)

    pred_tr_link = invert_y_standard(pred_tr_s, y_mu, y_sd) if args.y_scale == "standard" else pred_tr_s
    pred_te_link = invert_y_standard(pred_te_s, y_mu, y_sd) if args.y_scale == "standard" else pred_te_s

    if args.y_link == "logit":
        pred_tr = sigmoid(pred_tr_link)
        pred_te = sigmoid(pred_te_link)
    else:
        pred_tr = pred_tr_link
        pred_te = pred_te_link

    metrics = {
        "target": args.target,
        "parents": pa,
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "r2_train": float(r2_score(y_train, pred_tr)),
        "r2_test": float(r2_score(y_test, pred_te)),
        "mse_test": float(mean_squared_error(y_test, pred_te)),
        "mae_test": float(mean_absolute_error(y_test, pred_te)),
        "width": width,
        "grid": args.grid,
        "k": args.k,
        "steps": args.steps,
        "opt": args.opt,
        "lamb": args.lamb,
        "lib": args.lib,
        "scaler": scaler,
    }
    print("Metrics:", metrics)

    out = {"metrics": metrics}
    formula_raw = ""
    formula_named = ""
    wrapper = ""

    if args.symbolic_after_fit:
        lib = [s.strip() for s in args.lib.split(",") if s.strip()]
        with torch.no_grad():
            model(dataset["train_input"])
        formula_raw = try_symbolic(model, lib=lib, verbose=args.verbose)
        formula_named = rename_formula_vars(formula_raw, pa) if formula_raw else ""
        if formula_raw:
            steps_txt = []
            if args.x_scale == "minmax_m11":
                for i, name in enumerate(pa):
                    steps_txt.append(f"u{i+1}=2*({name}-{scaler['x_min'][i]})/{scaler['x_span'][i]}-1")
            else:
                for i, name in enumerate(pa):
                    steps_txt.append(f"u{i+1}={name}")

            inv_std = f"z = {scaler.get('y_mean',0.0)} + {scaler.get('y_std',1.0)}*f(u1..u{len(pa)})" if args.y_scale=="standard" else "z=f(u1..)"
            if args.y_link == "logit":
                inv_link = "y = sigmoid(z)"
            else:
                inv_link = "y = z"
            wrapper = " ; ".join(steps_txt) + " ; " + inv_std + " ; " + inv_link

            yhat_te_scaled = eval_symbolic_scaled(formula_raw, Xte_s)
            z_te = invert_y_standard(yhat_te_scaled, y_mu, y_sd) if args.y_scale=="standard" else yhat_te_scaled
            yhat_te = sigmoid(z_te) if args.y_link=="logit" else z_te
            metrics["r2_symbolic_test"] = float(r2_score(y_test, yhat_te))
            metrics["mse_symbolic_test"] = float(mean_squared_error(y_test, yhat_te))
            print("Symbolic test R2 (orig y):", metrics["r2_symbolic_test"])
        else:
            raise RuntimeError("Symbolic formula extraction failed.")

    out.update({"formula_raw": formula_raw, "formula_named": formula_named, "wrapper": wrapper, "metrics": metrics})
    out_path = os.path.join(args.out_dir, f"{args.target}_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
