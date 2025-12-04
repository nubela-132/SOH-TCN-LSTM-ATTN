import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from soh_model import TcnLstmModel, train_and_eval


@st.cache_data(show_spinner=False)
def load_features(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    model = TcnLstmModel()
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    window = ckpt.get("window", 8)
    norm = ckpt.get("norm", {"mean": [0, 0, 0], "std": [1, 1, 1]})
    return model, window, norm


@st.cache_data(show_spinner=False)
def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_history(log_dir: Path) -> pd.DataFrame:
    rows = []
    if not log_dir.exists():
        return pd.DataFrame()
    for p in sorted(log_dir.glob("run_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                info = json.load(f)
            row = {
                "run_id": info.get("run_id", p.stem),
                "timestamp": info.get("timestamp", ""),
            }
            params = info.get("params", {})
            metrics = info.get("metrics", {})
            row.update(params)
            row.update({f"test_{k}": v for k, v in metrics.items() if not k.startswith("val_")})
            row.update({k: v for k, v in metrics.items() if k.startswith("val_")})
            rows.append(row)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp", ascending=False)
    return df


def build_predictions(feat_df: pd.DataFrame, model: TcnLstmModel, window: int, norm: dict) -> pd.DataFrame:
    mean = np.array(norm.get("mean", [0, 0, 0]), dtype=float)
    std = np.array(norm.get("std", [1, 1, 1]), dtype=float) + 1e-8
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
                pred = model(torch.tensor(seq, dtype=torch.float32).unsqueeze(0)).item()
            rows.append({"source": src, "cycle": cycle_end, "true": target, "pred": pred})
    return pd.DataFrame(rows)


def plot_features(feat_df: pd.DataFrame, sources: list):
    data = feat_df[feat_df["source"].isin(sources)]
    for feature in ["CCCT", "CVCT", "CCAV"]:
        fig = px.line(
            data,
            x="cycle",
            y=feature,
            color="source" if len(sources) > 1 else None,
            markers=True,
            title=f"{feature} 随循环变化",
        )
        st.plotly_chart(fig, width="stretch")


def plot_prediction_curve(pred_df: pd.DataFrame, sources: list):
    data = pred_df[pred_df["source"].isin(sources)]
    if data.empty:
        st.info("所选电芯数据不足，无法生成预测曲线。")
        return
    melted = data.melt(id_vars=["source", "cycle"], value_vars=["true", "pred"], var_name="type", value_name="SOH")
    fig = px.line(
        melted,
        x="cycle",
        y="SOH",
        color="type",
        facet_row="source" if len(sources) > 1 else None,
        markers=True,
        title="SOH 真实值 vs 预测值",
    )
    st.plotly_chart(fig, width="stretch")


def plot_error_scatter(pred_df: pd.DataFrame, sources: list):
    data = pred_df[pred_df["source"].isin(sources)]
    if data.empty:
        st.info("所选电芯数据不足，无法生成误差散点。")
        return
    fig = px.scatter(
        data,
        x="true",
        y="pred",
        color="source",
        title="误差散点（理想为对角线）",
        labels={"true": "真实 SOH", "pred": "预测 SOH"},
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
    st.plotly_chart(fig, width="stretch")


def show_metrics_sidebar(metrics: dict):
    if not metrics:
        st.sidebar.info("未找到指标文件或为空。")
        return
    st.sidebar.subheader("测试指标")
    for key in ["RMSE", "MAE", "MAPE%", "R2"]:
        if key in metrics:
            st.sidebar.metric(f"test {key}", f"{metrics[key]:.4f}")
    for key in ["val_RMSE", "val_MAE", "val_MAPE%", "val_R2"]:
        if key in metrics:
            st.sidebar.metric(f"val {key.split('_',1)[1]}", f"{metrics[key]:.4f}")


def plot_loss_curve(metrics: dict):
    train_hist = metrics.get("train_loss_history", [])
    val_hist = metrics.get("val_loss_history", [])
    if not train_hist:
        return
    epochs = list(range(1, len(train_hist) + 1))
    rows = [{"epoch": e, "loss": l, "type": "train"} for e, l in zip(epochs, train_hist)]
    if val_hist:
        rows.extend({"epoch": e, "loss": l, "type": "val"} for e, l in zip(epochs, val_hist))
    df = pd.DataFrame(rows)
    fig = px.line(
        df,
        x="epoch",
        y="loss",
        color="type",
        markers=True,
        title="Loss 随 Epoch 变化（对数轴）",
        log_y=True,
    )
    st.plotly_chart(fig, width="stretch")


def show_history(log_dir: Path):
    hist_df = load_history(log_dir)
    if hist_df.empty:
        st.info("暂无历史记录。")
        return
    st.subheader("训练历史记录")
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
    return hist_df


def run_training_form(default_data: Path, default_out: Path):
    with st.sidebar.form("train_form"):
        st.markdown("### 训练配置")
        data_path = st.text_input("数据路径", str(default_data))
        output_dir = st.text_input("输出路径", str(default_out))
        window = st.number_input("窗口长度", min_value=1, max_value=256, value=8, step=1)
        train_ratio = st.number_input("训练占比", min_value=0.1, max_value=0.9, value=0.7, step=0.05, format="%.2f")
        val_ratio = st.number_input("验证占比", min_value=0.0, max_value=0.5, value=0.1, step=0.05, format="%.2f")
        epochs = st.number_input("训练轮数", min_value=1, max_value=5000, value=30, step=1)
        lr = st.number_input("学习率", min_value=1e-6, max_value=1.0, value=1e-3, format="%.6f")
        weight_decay = st.number_input("权重衰减", min_value=0.0, max_value=1.0, value=1e-4, format="%.6f")
        batch_size = st.number_input("批大小", min_value=1, max_value=1024, value=32, step=1)
        submitted = st.form_submit_button("运行训练")
    return submitted, data_path, output_dir, window, train_ratio, val_ratio, epochs, lr, weight_decay, batch_size


def main():
    st.set_page_config(page_title="SOH Dashboard", layout="wide")
    st.title("SOH 预测可视化面板")

    default_dir = Path("outputs")
    default_data = Path("dataset")

    submitted, data_path, output_dir, window_cfg, train_ratio, val_ratio, epochs, lr, weight_decay, batch_size = run_training_form(
        default_data, default_dir
    )

    if submitted:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        def progress_cb(current, total, train_loss, val_loss):
            ratio = current / total
            progress_bar.progress(ratio)
            status_placeholder.text(f"训练进度: {current}/{total} epoch, train_loss={train_loss:.6f}" + (f", val_loss={val_loss:.6f}" if val_loss is not None else ""))

        with st.spinner("训练中，请稍候..."):
            metrics = train_and_eval(
                data_path=Path(data_path),
                output_dir=Path(output_dir),
                window=int(window_cfg),
                train_ratio=float(train_ratio),
                val_ratio=float(val_ratio),
                epochs=int(epochs),
                lr=float(lr),
                weight_decay=float(weight_decay),
                batch_size=int(batch_size),
                progress_cb=progress_cb,
            )
        # 清理缓存，确保读取最新输出
        load_features.clear()
        load_model.clear()
        load_metrics.clear()
        status_placeholder.success("训练完成")
        progress_bar.progress(1.0)

    # 文件路径输入（默认指向 output_dir）
    feature_path = Path(st.sidebar.text_input("特征文件路径", str(Path(output_dir) / "features_per_cycle.csv")))
    model_path = Path(st.sidebar.text_input("模型文件路径", str(Path(output_dir) / "tcn_lstm_model.pt")))
    metrics_path = Path(st.sidebar.text_input("指标文件路径", str(Path(output_dir) / "metrics.json")))

    # 选择历史记录以加载对应快照
    hist_df = load_history(Path(output_dir) / "logs")
    if not hist_df.empty:
        run_options = hist_df["run_id"].tolist()
        selected_run = st.sidebar.selectbox("选择历史 run", run_options, index=0)
        run_dir = Path(output_dir) / "logs" / f"run_{selected_run}"
        if (run_dir / "features_per_cycle.csv").exists():
            feature_path = run_dir / "features_per_cycle.csv"
        if (run_dir / "tcn_lstm_model.pt").exists():
            model_path = run_dir / "tcn_lstm_model.pt"
        if (run_dir / "metrics.json").exists():
            metrics_path = run_dir / "metrics.json"

    if not feature_path.exists() or not model_path.exists():
        st.error("请确认特征文件和模型文件路径是否正确。")
        return

    feat_df = load_features(feature_path)
    model, window_loaded, norm = load_model(model_path)
    metrics = load_metrics(metrics_path)
    pred_df = build_predictions(feat_df, model, window_loaded, norm)

    st.sidebar.write(f"窗口长度: {window_loaded}")
    st.sidebar.write(f"可用样本数: {len(pred_df)}")
    show_metrics_sidebar(metrics)

    all_sources = sorted(pred_df["source"].unique().tolist()) if not pred_df.empty else []
    selected = st.sidebar.multiselect("选择电芯", all_sources, default=all_sources[:1] if all_sources else [])

    if not selected:
        st.info("请选择至少一个电芯。")
        return

    plot_features(feat_df, selected)

    col1, col2 = st.columns(2)
    with col1:
        plot_prediction_curve(pred_df, selected)
    with col2:
        plot_error_scatter(pred_df, selected)

    plot_loss_curve(metrics)
    show_history(Path(output_dir) / "logs")


if __name__ == "__main__":
    main()
