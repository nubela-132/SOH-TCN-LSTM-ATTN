import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 全局划分比例（运行时由参数覆盖）
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1


def extract_cycle_features(
    cycle_df: pd.DataFrame,
    v_cut: float = 4.2,
    high_frac: float = 0.8,
    low_frac: float = 0.2,
) -> Optional[Tuple[float, float, float, float]]:
    """Return (CCCT, CVCT, CCAV, SOH) for one cycle."""
    g = cycle_df.sort_values("time")
    med = g["charge_current"].median()
    sign = 1.0 if med == 0 else np.sign(med)
    curr = g["charge_current"] * sign
    chg = g[curr > 0]
    if chg.empty:
        return None
    curr = curr.loc[chg.index]
    volt = chg["terminal_voltage"]
    time = chg["time"]
    curr_max = curr.max()
    if curr_max <= 0:
        return None
    cc_mask = (curr >= high_frac * curr_max) & (volt <= v_cut + 1e-3)
    if cc_mask.sum() < 2:
        cc_mask = curr >= curr.quantile(0.7)
    cv_mask = (curr <= low_frac * curr_max) & (volt >= v_cut - 0.05)
    if cv_mask.sum() < 2:
        cv_mask = (curr <= curr.quantile(0.3)) & (volt >= volt.quantile(0.7))
    if cc_mask.sum() < 2 or cv_mask.sum() < 2:
        return None
    cc_time = time[cc_mask]
    cv_time = time[cv_mask]
    ccct = float(cc_time.max() - cc_time.min())
    cvct = float(cv_time.max() - cv_time.min())
    ccav = float(volt[cc_mask].mean())
    soh = float(g["SOH"].iloc[0])
    return ccct, cvct, ccav, soh


def build_feature_table_for_file(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows: List[dict] = []
    for cyc, g in df.groupby("cycle"):
        vals = extract_cycle_features(g)
        if vals is None:
            continue
        ccct, cvct, ccav, soh = vals
        rows.append(
            {
                "source": csv_path.stem,
                "cycle": cyc,
                "CCCT": ccct,
                "CVCT": cvct,
                "CCAV": ccav,
                "SOH": soh,
            }
        )
    feat_df = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)
    return feat_df


def build_feature_table(data_path: Path) -> pd.DataFrame:
    """Build feature table from a single CSV or all CSVs in a directory."""
    data_path = Path(data_path)
    if data_path.is_dir():
        parts: List[pd.DataFrame] = []
        for csv_file in sorted(data_path.glob("*.csv")):
            part = build_feature_table_for_file(csv_file)
            if not part.empty:
                parts.append(part)
        if not parts:
            return pd.DataFrame(columns=["source", "cycle", "CCCT", "CVCT", "CCAV", "SOH"])
        return pd.concat(parts, ignore_index=True)
    return build_feature_table_for_file(data_path)


class SampleDataset(Dataset):
    def __init__(self, samples: np.ndarray, labels: np.ndarray):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx], self.labels[idx]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.chomp1(self.conv1(x))
        out = self.relu1(out)
        out = self.chomp2(self.conv2(out))
        out = self.relu2(out)
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class TCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: List[int], kernel_size: int = 3):
        super().__init__()
        layers = []
        for i, c in enumerate(hidden_channels):
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    n_inputs=in_channels if i == 0 else hidden_channels[i - 1],
                    n_outputs=c,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2))
        return y.transpose(1, 2)


class CrossAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = float(dim) ** 0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        att = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(att, v)


class TcnLstmModel(nn.Module):
    def __init__(self, feat_dim: int = 3, hidden_dim: int = 64, lstm_layers: int = 1):
        super().__init__()
        self.tcn = TCN(in_channels=feat_dim, hidden_channels=[hidden_dim, hidden_dim])
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.attn = CrossAttention(dim=hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tcn_out = self.tcn(x)
        lstm_out, _ = self.lstm(x)
        fused = self.attn(tcn_out, lstm_out, lstm_out)
        out = self.fc(fused[:, -1, :])
        return out.squeeze(-1)


def make_loaders(feat_df: pd.DataFrame, window: int, batch_size: int):
    """Create loaders without跨电芯滑窗，按每个 source 独立切分后汇总。"""
    train_feat_chunks: List[np.ndarray] = []
    train_label_chunks: List[np.ndarray] = []
    val_feat_chunks: List[np.ndarray] = []
    val_label_chunks: List[np.ndarray] = []
    test_feat_chunks: List[np.ndarray] = []
    test_label_chunks: List[np.ndarray] = []
    for _, g in feat_df.groupby("source"):
        feats = g[["CCCT", "CVCT", "CCAV"]].to_numpy()
        labels = g["SOH"].to_numpy()
        if len(feats) < window:
            continue
        n_train = int(len(feats) * TRAIN_RATIO)
        n_val = int(len(feats) * VAL_RATIO)
        n_train = max(window, min(n_train, len(feats)))
        remaining = max(0, len(feats) - n_train)
        n_val = min(n_val, remaining)
        train_feat_chunks.append(feats[:n_train])
        train_label_chunks.append(labels[:n_train])
        start_val = n_train
        end_val = n_train + n_val
        if n_val > 0:
            val_feat_chunks.append(feats[start_val:end_val])
            val_label_chunks.append(labels[start_val:end_val])
        if end_val < len(feats):
            test_feat_chunks.append(feats[end_val:])
            test_label_chunks.append(labels[end_val:])

    if not train_feat_chunks:
        return None, None, None, {}

    train_feats_all = np.concatenate(train_feat_chunks, axis=0)
    mean = train_feats_all.mean(axis=0)
    std = train_feats_all.std(axis=0) + 1e-8

    def build_samples(feats: np.ndarray, labels: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        feats = (feats - mean) / std
        xs: List[np.ndarray] = []
        ys: List[float] = []
        if len(feats) < window:
            return xs, ys
        for i in range(len(feats) - window + 1):
            xs.append(feats[i : i + window])
            ys.append(labels[i + window - 1])
        return xs, ys

    train_samples: List[np.ndarray] = []
    train_labels_list: List[float] = []
    val_samples: List[np.ndarray] = []
    val_labels_list: List[float] = []
    test_samples: List[np.ndarray] = []
    test_labels_list: List[float] = []

    for feats, labels in zip(train_feat_chunks, train_label_chunks):
        xs, ys = build_samples(feats, labels)
        train_samples.extend(xs)
        train_labels_list.extend(ys)

    for feats, labels in zip(val_feat_chunks, val_label_chunks):
        xs, ys = build_samples(feats, labels)
        val_samples.extend(xs)
        val_labels_list.extend(ys)

    for feats, labels in zip(test_feat_chunks, test_label_chunks):
        xs, ys = build_samples(feats, labels)
        test_samples.extend(xs)
        test_labels_list.extend(ys)

    train_ds = SampleDataset(np.stack(train_samples), np.array(train_labels_list)) if train_samples else None
    val_ds = SampleDataset(np.stack(val_samples), np.array(val_labels_list)) if val_samples else None
    test_ds = SampleDataset(np.stack(test_samples), np.array(test_labels_list)) if test_samples else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) if train_ds else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None
    norm = {"mean": mean.tolist(), "std": std.tolist()}
    return train_loader, val_loader, test_loader, norm


def predict_all(feat_df: pd.DataFrame, model: TcnLstmModel, window: int, norm: dict) -> pd.DataFrame:
    """Run sliding-window prediction per source, return DataFrame with true/pred."""
    mean = np.array(norm.get("mean", [0, 0, 0]), dtype=float)
    std = np.array(norm.get("std", [1, 1, 1]), dtype=float) + 1e-8
    device = next(model.parameters()).device
    rows = []
    for src, g in feat_df.groupby("source"):
        feats = g[["CCCT", "CVCT", "CCAV"]].to_numpy()
        labels = g["SOH"].to_numpy()
        cycles = g["cycle"].to_numpy()
        feats_norm = (feats - mean) / std
        if len(feats_norm) < window:
            continue
        for i in range(len(feats_norm) - window + 1):
            seq = feats_norm[i : i + window]
            target = float(labels[i + window - 1])
            cycle_end = int(cycles[i + window - 1])
            with torch.no_grad():
                pred = (
                    model(torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0))
                    .cpu()
                    .item()
                )
            rows.append({"source": src, "cycle": cycle_end, "true": target, "pred": pred})
    return pd.DataFrame(rows)


def save_per_source_outputs(output_dir: Path, feat_df: pd.DataFrame, pred_df: pd.DataFrame):
    """Save features/predictions per source into numbered subfolders."""
    for idx, src in enumerate(sorted(feat_df["source"].unique()), start=1):
        subdir = output_dir / f"{idx:03d}_{src}"
        subdir.mkdir(parents=True, exist_ok=True)
        feat_df[feat_df["source"] == src].to_csv(subdir / "features.csv", index=False)
        if not pred_df.empty:
            pred_df[pred_df["source"] == src].to_csv(subdir / "predictions.csv", index=False)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8)
    r2 = 1 - ss_res / ss_tot
    return {"RMSE": rmse, "MAE": float(mae), "MAPE%": mape, "R2": float(r2)}


def train_and_eval(
    data_path: Path,
    output_dir: Path,
    window: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    progress_cb: Optional[Callable[[int, int, float, Optional[float]], None]] = None,
):
    global TRAIN_RATIO, VAL_RATIO
    TRAIN_RATIO = train_ratio
    VAL_RATIO = val_ratio
    feat_df = build_feature_table(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_dir / "features_per_cycle.csv", index=False)
    train_loader, val_loader, test_loader, norm = make_loaders(feat_df, window, batch_size)
    if train_loader is None:
        return {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TcnLstmModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loss_hist: List[float] = []
    val_loss_hist: List[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_loss_hist.append(train_loss)

        # 验证集 loss
        val_loss = None
        if val_loader is not None:
            model.eval()
            v_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    v_pred = model(xb)
                    v_loss = loss_fn(v_pred, yb)
                    v_losses.append(v_loss.item())
            if v_losses:
                val_loss = float(np.mean(v_losses))
                val_loss_hist.append(val_loss)
            else:
                val_loss_hist.append(float("nan"))

        msg = f"Epoch {epoch+1}/{epochs} train_loss={train_loss:.6f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.6f}"
        print(msg)
        if progress_cb:
            progress_cb(epoch + 1, epochs, train_loss, val_loss)

    model.eval()
    metrics: dict = {}
    with torch.no_grad():
        def run_eval(loader):
            preds, trues = [], []
            if loader is None:
                return np.array([]), np.array([])
            for xb, yb in loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())
            return (np.concatenate(preds), np.concatenate(trues)) if preds else (np.array([]), np.array([]))

        val_pred, val_true = run_eval(val_loader)
        test_pred, test_true = run_eval(test_loader)

    if len(test_true):
        metrics.update(eval_metrics(test_true, test_pred))
    if len(val_true):
        val_metrics = eval_metrics(val_true, val_pred)
        for k, v in val_metrics.items():
            metrics[f"val_{k}"] = v
    metrics["train_loss_history"] = train_loss_hist
    if val_loss_hist:
        metrics["val_loss_history"] = val_loss_hist

    torch.save({"model_state": model.state_dict(), "norm": norm, "window": window}, output_dir / "tcn_lstm_model.pt")

    # 保存预测（总表 + 按 source 分类编号）
    pred_df = predict_all(feat_df, model, window, norm)
    if not pred_df.empty:
        pred_df.to_csv(output_dir / "predictions.csv", index=False)
    save_per_source_outputs(output_dir, feat_df, pred_df)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 写入日志记录（包含参数和时间戳），并保存快照便于历史查看
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{run_id}.json"
    run_dir = log_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_info = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "params": {
            "data_path": str(data_path),
            "output_dir": str(output_dir),
            "window": window,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
        "metrics": metrics,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    # 保存快照文件
    for name in ["features_per_cycle.csv", "predictions.csv", "metrics.json", "tcn_lstm_model.pt"]:
        src = output_dir / name
        if src.exists():
            shutil.copyfile(src, run_dir / name)
    # 保存按 source 分类的子目录
    for sub in output_dir.glob("0*_*"):
        if sub.is_dir():
            dest = run_dir / sub.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(sub, dest)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train TCN-LSTM SOH predictor.")
    parser.add_argument(
        "--data-path",
        "--csv",
        dest="data_path",
        type=Path,
        default=Path("dataset"),
        help="Path to a CSV file or directory containing multiple CSVs.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Where to save model and metrics.")
    parser.add_argument("--window", type=int, default=8, help="Sequence length (number of cycles).")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio per source.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio per source.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = train_and_eval(
        data_path=args.data_path,
        output_dir=args.output_dir,
        window=args.window,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )
    print("Finished. Metrics:", metrics)


if __name__ == "__main__":
    main()
