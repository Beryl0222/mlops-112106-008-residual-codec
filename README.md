# 残差量化码流

samples/scene.json 是本地合成的码本、片段和消费级别。可检查资料的形状、来源关系或交换文件边界。时间序列条目表示调用方送达或发起读取的顺序，数值为输入。

完整交换约定见 [交换格式](exchange.md)。运行环境为 Python 3.11 和 NumPy 2.2.6，使用本地文件即可运行资料工具。

```sh
python3 -m pip install -r requirements.txt
python3 inspect_exchange.py samples/scene.json
python3 -m unittest discover -s tests -v
```
