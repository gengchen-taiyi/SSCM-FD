import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kan import KAN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE = Path(__file__).resolve().parent


def inputs(frame, parents, lower, span, device):
    values = frame.loc[:, parents].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Parent measurements must be finite.")
    values = 2.0 * (values - np.asarray(lower)) / np.asarray(span) - 1.0
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def build(config, device):
    return KAN(width=list(config["width"]), grid=config["grid"], k=config["k"],
               seed=config["seed"], symbolic_enabled=False, auto_save=False, device=device)


def numeric_prediction(model, x, mean, scale):
    model.eval()
    with torch.no_grad():
        values = model(x).flatten().cpu().numpy().astype(np.float64)
    return values * scale + mean


def predict(checkpoint, frame, device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    config = saved["config"]
    model = build(config, device)
    model.load_state_dict(saved["state_dict"])
    x = inputs(frame, config["parents"], config["x_min"], config["x_span"], device)
    return numeric_prediction(model, x, config["y_mean"], config["y_std"])


def main():
    parser = argparse.ArgumentParser(description="Fit a numerical spline KAN to a target's parents.")
    parser.add_argument("--data_csv", type=Path, default=BASE / "Sample_data.csv")
    parser.add_argument("--adj_csv", type=Path, default=BASE / "causal_discovery_results" / "adjacency.csv")
    parser.add_argument("--target", default="X9")
    parser.add_argument("--out_dir", type=Path, default=BASE / "KAN_1_results")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--opt", choices=("LBFGS", "Adam"), default="LBFGS")
    parser.add_argument("--grid", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--lamb", type=float, default=0.001)
    args = parser.parse_args()
    if not args.adj_csv.is_file():
        parser.error(f"Missing adjacency: {args.adj_csv}. Run causal_discovery.py first or pass --adj_csv.")
    if not (0 < args.test_ratio < 1 and args.steps > 0 and args.grid > 0 and args.k > 0
            and args.depth > 0 and (args.hidden is None or args.hidden > 0) and args.lamb >= 0):
        parser.error("Invalid split, network dimensions, step count, or regularization.")
    frame = pd.read_csv(args.data_csv)
    adj = pd.read_csv(args.adj_csv, index_col=0)
    if not adj.index.is_unique or set(adj.index) != set(adj.columns) or not adj.isin([0, 1]).all().all():
        parser.error("Expected a named binary adjacency matrix (row = source, column = target).")
    if args.target not in adj or args.target not in frame:
        parser.error(f"Target not present in both input tables: {args.target}")
    parents = adj.index[adj[args.target] == 1].tolist()
    if args.target in parents or any(p not in frame for p in parents):
        parser.error("The parent list contains the target or a missing sensor.")
    if not parents:
        print(f"{args.target} has no parents in the supplied graph; there is no conditional model to fit.")
        return
    train, test = train_test_split(np.arange(len(frame)), test_size=args.test_ratio, random_state=args.seed)
    if min(len(train), len(test)) < 2:
        parser.error("At least two rows are required in both splits.")
    x_train = frame.loc[train, parents].to_numpy(dtype=np.float64)
    y = frame[args.target].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        parser.error("Target measurements must be finite.")
    lower, span = x_train.min(axis=0), np.ptp(x_train, axis=0)
    span[span < 1e-8] = 1.0
    mean, scale = float(y[train].mean()), float(y[train].std())
    if scale < 1e-8:
        scale = 1.0
    hidden = args.hidden or (6 if len(parents) == 1 else 2 * len(parents) + 1)
    config = dict(target=args.target, parents=parents, x_min=lower.tolist(), x_span=span.tolist(),
                  y_mean=mean, y_std=scale, width=[len(parents)] + [hidden] * args.depth + [1],
                  grid=args.grid, k=args.k, seed=args.seed, steps=args.steps, opt=args.opt,
                  lamb=args.lamb, test_ratio=args.test_ratio)
    dataset = {}
    for name, rows in [("train", train), ("test", test)]:
        dataset[name + "_input"] = inputs(frame.iloc[rows], parents, lower, span, args.device)
        dataset[name + "_label"] = torch.as_tensor(
            (y[rows, None] - mean) / scale, dtype=torch.float32, device=args.device)
    model = build(config, args.device)
    model.fit(dataset, opt=args.opt, steps=args.steps, lamb=args.lamb, lamb_entropy=0.0,
              lr=1.0 if args.opt == "LBFGS" else 0.001)
    pred = numeric_prediction(model, dataset["test_input"], mean, scale)
    metrics = dict(r2_test=float(r2_score(y[test], pred)), mse_test=float(mean_squared_error(y[test], pred)),
                   mae_test=float(mean_absolute_error(y[test], pred)), n_train=len(train), n_test=len(test))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / f"{args.target}_model.pt"
    torch.save({"config": config, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
               checkpoint)
    pd.DataFrame(dict(row_index=test, observed=y[test], predicted=pred)).sort_values("row_index").to_csv(
        args.out_dir / f"{args.target}_predictions.csv", index=False)
    (args.out_dir / f"{args.target}_result.json").write_text(
        json.dumps(dict(config=config, metrics=metrics), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.target} <- {', '.join(parents)} | Test R2 = {metrics['r2_test']:.5f} | {checkpoint}")


if __name__ == "__main__":
    main()
