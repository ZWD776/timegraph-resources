# 各数据集独立整理（2026-10-09）

本次仅整理当前项目内各数据集的源目录，不合并三个数据集，不读取或修改 benchmark_data。图像采用同盘目录移动；原始观察、时间、QA 与历史记录保留。

## NExT-QA

活动输入：datasets/nextqa/source/nextqa-val-v1-repaired29/payload/items/<official_video_id>/。
每个视频保持 objects、attributes、relations、hierarchy_mapping、timeline、source_trace 和 frames。全 570 视频仅有一个选定版本；官方 ID 是目录键，原 local ID 仍作元数据。

来源统计：前半批次 263、后半批次 278、奇数高度补跑 21、低 FPS 补跑 8。原两个 50% 批次的未选定旧版本不再作为活动输入入口。

问题与评分标签在 evaluation，官方注释在 official，29 视频修复子集清单在 manifests/repaired29_manifest.json，旧批次清单、原修复 JSON 与修改前路径记录在 provenance/history。

objects 与 source_trace 的运行图像路径为 item 相对 frames/<frame>.png；修改前的原文件保存在 history/item_metadata。23,381 帧、23,044 完成 HSGG、337 skipped 均保留，时间及观察内容不变。

## NAVQA

保留原 seq/cam/every10 结构和图像、时间、QA 的路径。每个采样目录的 hsgg 是唯一活动输入，原 runtime_inputs 不再与旧 local/upstream 三份表示混在一起。

原 upstream、旧 local、旧 aggregate 与源目录内旧图产物归入本数据集 provenance/history；已有处理属性缓存移动到本数据集 cache/processed_attributes，代码通过声明的缓存位置读取，避免写入原始 HSGG。

以 9,759 个完成 sidecar 为有效观察。130 张没有 HSGG 的图像包括 105 skipped 和 25 error，仍完整保存，不补造观察。seq22 原聚合表仅 1,528 帧，而完成 sidecar 为 1,530；新聚合表按完整逐帧内容派生，原 1,528 帧版本完整保留。这是数据表示完整性修复，未改变 HSGG 观察或 GT。

## OpenEQA

HM3D 和 ScanNet 仍是 OpenEQA 自身的两个独立分片，各自目录为 shard/hsgg、shard/frames、shard/provenance。

HM3D 保持 63 episodes、4,956 完成 HSGG、189 skipped；ScanNet 保持 89 episodes、26,078 完成 HSGG、519 skipped。QA episode 集合与活动源一致，无多余、缺少或重复 episode。

旧 generation/preparation 日志和 aggregate 保存在对应分片 provenance。HM3D 六个 episode 的旧 aggregate 比完成 sidecar 少 37 帧；新建图优先完整 sidecar，只在新运行输出目录生成标准表，不覆写原始 HSGG。时间保持原 ordinal 或明确外部索引，不从 stride 推造 FPS。

## 代码与兼容

各数据集仍使用自己的 source_manifest 和适配器。Next loader/visual index 使用选定 item；NAV evidence、visual-index、grounding 尊重显式 HSGG/图像位置；Open builder 默认读自身 manifest，保留显式旧 root 导入模式，默认 config 使用 OpenEQA 自有配置。

registry/source_path_aliases.json 仅提供同一数据集所有权内的旧路径兼容，不创建共享数据集。历史图、索引图片引用在内存中重定位，SQLite 特征原查询键不改写。活动 registry/pointer/lock 已更新。

旧冻结论文图和预测没有重新构建。后续新版建图会读到原聚合表漏掉的 NAVQA 2 帧和 HM3D 37 帧；与旧冻结结果比较时应明确输入表示修订，或在同一新版输入上重建基线。

## 验收

迁移日志、原控制文件与全部源文件身份清单位于 registry/organization/data_sources_20261009。
运行 `.venv/Scripts/python.exe scripts/datasets/organize_source_layouts.py --verify` 可只读复核；禁止把原始备份目录作为操作范围。

本轮广泛代码回归 857 项通过，2 项失败与原项目 registry/audits/final_regression_summary.json 中已记载的 NAVQA semantic_event_candidates 历史语义断言相同，未为目录整理改动算法行为。
