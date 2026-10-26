# NExT-QA 当前主线

human v2 time-fact-core；570 视频，IRv3 与 strict-pairs v2 索引。

4996 道五选一题；exact-choice 正确数 A/B/C = 3024/3257/3334。

资产入口由 `registry/datasets/nextqa.yaml` 声明，不从旧实验目录猜测版本。

代码：`src/vlkg/datasets/nextqa/{graph,retrieval,qa,runtime,audit}`。

数据：`datasets/nextqa/{source,graphs,indexes,ir,retrieval,runs,reports,cache,scratch}`。

统一入口：`.venv/Scripts/python.exe scripts/datasets/pipeline.py --dataset nextqa --stage validate`。

新计算任务传入 `--run-id`，默认仅打印计划；显式 `--execute` 运行。当前主线的原始记录和历史资料保存在 archive；活动运行不得读取封存目录。

迁移只重定位路径和必要的默认引用，不据此声称在线 QA 或全量模型索引已经重跑。

## 已解压源输入（2026-10-08）

当前 source manifest 声明 `input_storage: normalized_directory`。570 个视频使用
`datasets/nextqa/source/nextqa-val-v1-repaired29/payload/items/<official_video_id>/`，
每个目录包含 objects、attributes、relations、hierarchy_mapping 四张观察表，
timeline、source_trace，以及 frames 中的原采样 PNG。共 23,381 帧，其中
23,044 帧有 HSGG，337 帧为 skipped 占位。保留原有 29 个修复视频的选择、帧和时间。

全量建图入口 `scripts/datasets/nextqa/build/build_graph_full.py` 直接读取这些目录，
不扫描压缩包，也不向每次 run 复制观察表或图片。历史 image_path 保持原始字节，
运行时通过 frames_dir 解析实际图片位置。显式传入旧 archive manifest 时仍支持导入；
该模式按 HSGG 压缩包分组，避免基础包和修复包混批。

源目录的压缩副本在全量内容与图片哈希核验后撤销。benchmark_data 是受保护备份，
完整保留。唯一的 special_splice 修复包成员以原始 JSON 字节完整解压保存在
`payload/provenance/history/special_splice/original_records/`；历史清单和迁移凭据完整保留。
当前清单中的 archive_provenance 仅记录历史来源，不要求原包在线。

核验已解压输入（只读，不调用模型）：

```powershell
& .venv/Scripts/python.exe scripts/datasets/nextqa/audit/verify_materialized_source.py --verify-image-sha256 --output <新的报告.json>
```

可用 `--baseline-report <已有报告.json>` 检查六类 JSON 与先前报告的字节指纹一致。
本次迁移与删除凭据位于 `registry/retirements/nextqa_source_archives_20261008/`。

2026-10-09 完成该数据集内部整理。570 个视频各有一个选定版本：原前半批次
263 个、原后半批次 278 个、奇数高度修复 21 个、低 FPS 修复 8 个。
这些是来源分类，不是需要再拼接的四份活动输入。目录按官方 video ID 命名；
objects 与 source_trace 的运行图片引用为 item 相对路径 `frames/<frame>.png`。
更改前的两份原始 JSON 已保存在 provenance/history/item_metadata。
问题与评分标签位于 evaluation；官方注释位于 official；修复子集清单位于
manifests/repaired29_manifest.json。旧批次和旧清单仅存 provenance/history。
图像、描述、关系、层级、帧状态和时间内容保持，历史图与索引的旧图片引用
由工作区内路径别名只读解析；SQLite 特征查询键未改写。
