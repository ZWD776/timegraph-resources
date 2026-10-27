# OpenEQA 当前主线

统一 time-fact-core；HM3D 和 ScanNet 是执行分片，V5 主线。

1636 题；项目归一化 LLM-Judge 分数 A/B/C = 48.915/49.3888/53.2549。

以上 C 为保留的 routed reference。外部 QA 综合资料采用论文冻结最终 C：53.4994（0–100 归一化 Judge 分），不覆盖旧结果；所选历史 Judge 记录和来源见 [QA 资料说明](../../github_release_20261027/QA_RESOURCES_CN.md)。

资产入口由 `registry/datasets/openeqa.yaml` 声明，不从旧实验目录猜测版本。

代码：`src/vlkg/datasets/openeqa/{graph,retrieval,qa,runtime,audit}`。

数据：`datasets/openeqa/{source,graphs,indexes,ir,retrieval,runs,reports,cache,scratch}`。

统一入口：`.venv/Scripts/python.exe scripts/datasets/pipeline.py --dataset openeqa --stage validate`。

新计算任务传入 `--run-id`，默认仅打印计划；显式 `--execute` 运行。当前主线的原始记录和历史资料保存在 archive；活动运行不得读取封存目录。

迁移只重定位路径和必要的默认引用，不据此声称在线 QA 或全量模型索引已经重跑。
