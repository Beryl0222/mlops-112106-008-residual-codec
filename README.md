# 残差量化码流

samples/scene.json 是本地合成的码本、片段和消费级别。可检查资料的形状、来源关系或交换文件边界。时间序列条目表示调用方送达或发起读取的顺序，数值为输入。

完整交换约定见 [交换格式](exchange.md)。运行环境为 Python 3.11 和 NumPy 2.2.6，使用本地文件即可运行资料工具。

```sh
python3 -m pip install -r requirements.txt
python3 inspect_exchange.py samples/scene.json
python3 run_scene.py samples/scene.json
python3 -m unittest discover -s tests -v
```

## 量化与码流实现

`rvq/` 交付可直接复用的 Python 量化、码流读写与训练反馈路径，统一入口为
`rvq.QuantizationArchive`：

- `encode_segment(片段, 通道组, 精度层数, 码本)`：分块多级残差量化，平方
  欧氏距离选码、平局取小编号；缺帧不参与距离与更新，零向量仍是正常候选。
- `read_layers` / `read_frame_layer`：按层重建或随机访问单层；读第一层不
  触碰高层字节。声明无高层 (`LayerNotDeclared`)、层内缺字节
  (`TruncatedLayer`)、码本版本查不到 (`UnknownCodebookVersion`) 分别报错。
- `commit` / `publish`：只把已提交层冻结的残差与分配送入训练，重放同一
  提交不重复学习；EMA 更新受 `min_assigned` 约束，冷码按最大量化误差复活
  并以片段 ID、帧序裁决平局，一次发布为整版原子可见。
- `release` 与版本留存：旧码本供历史读者查旧文件；容量不足时
  `RetentionExceeded` 报告仍引用旧版的未消费片段，由上游判断继续时机。
- 长片段按 `budget_bytes` 推导块行数，距离矩阵最大只到 块行数 × 码字数，
  冻结状态超预算抛 `MemoryBudgetExceeded`。

`run_scene.py` 按样本 `timeline` 驱动 encode/read_layers/commit/publish/
release，并把读取消费结果、发布版本与预算占用打印为 JSON。

