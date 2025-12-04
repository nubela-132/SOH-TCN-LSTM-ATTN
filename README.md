# SOH Prediction (TCN + LSTM + Cross-Attention)

利用 `soh_model.py` 对目录中的多份电芯 CSV 进行特征提取和 SOH 预测。每个 CSV 需包含列：
`terminal_voltage, terminal_current, temperature, charge_current, charge_voltage, time, capacity, cycle, SOH`。

## 环境准备
```bash
python -m pip install -r requirements.txt
```
建议用干净的 conda 环境（例如 `conda create -n soh python=3.10`）。

## 训练/评估
```bash
python soh_model.py --data-path dataset --output-dir outputs --window 8 --epochs 30 \
  --lr 1e-3 --weight-decay 1e-4 --batch-size 32 --train-ratio 0.7 --val-ratio 0.1
```
参数说明：
- `--data-path`：单个 CSV 或包含多个 CSV 的目录（默认 `dataset`）。目录时遍历所有 `*.csv`，`source` 列标记文件名。
- `--output-dir`：输出目录（默认 `outputs`）。
- `--window`：滑动窗口长度（多少个循环作为输入序列）。
- `--train-ratio` / `--val-ratio`：对每个电芯内部按时间顺序切分的训练/验证占比，剩余自动作为测试。
- 其余为训练超参。

## 可视化仪表盘
运行 Streamlit 仪表盘查看特征、预测曲线、误差散点、测试与验证指标、loss 曲线：
```bash
streamlit run dashboard.py
```
侧栏可输入特征/模型/指标文件路径（默认指向 `outputs`），选择电芯，并可一键提交训练表单（无需绝对路径）。

## 代码流程
1. **特征提取**：每个 CSV、每个循环，从充电段提取 CCCT、CVCT、CCAV 和 SOH；根据充电电流符号自动判定充电段，启发式区分 CC/CV。
2. **按电芯划分训练/验证/测试**：每个 `source` 内按时间顺序 70% 训练、10% 验证、剩余测试（比例可调）；滑窗只在同一电芯内构造。
3. **标准化**：用全部训练样本的均值/标准差归一化特征。
4. **模型**：并行 TCN 与 LSTM，TCN 输出作 Query，LSTM 输出作 Key/Value 做交叉注意力，融合后回归 SOH。
5. **评估/保存**：输出特征表、模型权重、归一化信息和指标；终端打印每个 epoch 的 loss 及 `Finished. Metrics: {...}`，指标文件包含 test_*、val_*（若有）和 loss 历史。


## 技术路线简述
- **数据侧**：仅用充电过程曲线，提取三类健康指标：恒流时长 CCCT、恒压时长 CVCT、恒流平均电压 CCAV，作为 SOH 的输入特征。
- **建模侧**：
  - TCN 分支：因果空洞卷积 + 残差结构，抓取局部时序模式和较长感受野。
  - LSTM 分支：捕获长期依赖。
  - 交叉注意力：TCN 输出为 Query，LSTM 输出为 Key/Value，融合两路特征以提升预测精度。
- **训练侧**：按电芯时间顺序切分训练/验证/测试，滑窗构造序列样本，MSE 损失回归 SOH，可调窗口、学习率、轮数、正则等超参。
- **工程侧**：训练输出包含特征汇总、预测、模型、指标，按电芯分目录保存，并生成按时间戳的日志快照便于回溯；仪表盘支持参数输入、训练进度条、历史 run 选择和结果可视化。


## 输出文件
位于 `--output-dir`（默认 `outputs`），且按输入文件分目录保存：
- `features_per_cycle.csv`：全量特征汇总。
- `predictions.csv`：全量预测汇总（如有）。
- `tcn_lstm_model.pt`：模型参数、归一化均值/方差、窗口长度。
- `metrics.json`：指标（RMSE、MAE、MAPE%、R2），若有验证集则附带 `val_*`，并含 `train_loss_history` / `val_loss_history`。
- `XXX/` 子目录：对每个输入 CSV 建立按序号命名的子目录（如 `001_B05_discharge_soh`），内含该电芯的 `features.csv`、`predictions.csv`。
- `logs/`：每次训练生成 `run_*.json`，记录时间戳、参数与指标，并保存当前快照，便于历史回溯。



## 目录结构（关键文件）
```
dataset\         # 多个 *_discharge_soh.csv
outputs\         # 运行后生成的输出
soh_model.py     # 训练/评估脚本
dashboard.py     # 可视化仪表盘
requirements.txt # 依赖
README.md        # 本说明
```
