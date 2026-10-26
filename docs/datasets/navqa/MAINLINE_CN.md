# NAVQA 当前主线

原始完整 CG6；7 序列，每个序列有独立真实时间索引。

139 道 non-position 问题；官方风格正确数 A/B/C = 73/87/101。

资产入口由 `registry/datasets/navqa.yaml` 声明，不从旧实验目录猜测版本。

代码：`src/vlkg/datasets/navqa/{graph,retrieval,qa,runtime,audit}`。

数据：`datasets/navqa/{source,graphs,indexes,ir,retrieval,runs,reports,cache,scratch}`。

统一入口：`.venv/Scripts/python.exe scripts/datasets/pipeline.py --dataset navqa --stage validate`。

新计算任务传入 `--run-id`，默认仅打印计划；显式 `--execute` 运行。当前主线的原始记录和历史资料保存在 archive；活动运行不得读取封存目录。

迁移只重定位路径和必要的默认引用，不据此声称在线 QA 或全量模型索引已经重跑。
