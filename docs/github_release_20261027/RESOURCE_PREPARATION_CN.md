# 数据与运行资源准备说明

本文说明 GitHub 代码仓库与网盘资源如何对应。三个数据集保持各自独立的清单和目录，不把它们拼成一个数据集。

## 发布边界与资源入口

- GitHub 准备代码、配置、测试、使用说明、资源清单和必要的版本记录。
- 图片、HSGG、主线图记忆与选定视觉特征由维护者通过同一受控网盘分享提供，不随 GitHub 主仓库上传；访问方式单独提供，本仓库不保存分享链接或提取凭据。
- 用户已确认图和视觉特征上传完成。新增的 30 组、60 个图／特征包在本地逐文件 SHA256 与完整 gzip EOF/CRC 校验通过；这不等于已经从云端重新下载核验。
- 上传的视觉附件为选定 SigLIP FAISS 及配套 metadata；MiniLM 文本向量、历史混合 embedding SQLite、证据层数据库和其他可再生缓存不在这 60 包内。
- 环境目录、模型权重、密钥及本地包缓存不属于上述数据资源。环境与模型的准备以最终发布的环境说明为准。

资源入口以 `registry/datasets/<dataset>.yaml`、对应 `asset_pointer.json` 和 `source_manifest.json` 为准。不要从目录修改日期、旧实验名称或压缩包名称猜测有效版本。

下文 `<DATA_ROOT>` 表示资源目录的根，目录结构相对于该根展开。现有项目入口主要按仓库根解析 `datasets/...`；直接运行这些入口时，资源需还原到对应的仓库相对位置。本文不宣称所有脚本已经支持任意外置数据根。

| 数据集 | 选中源清单，相对于 `<DATA_ROOT>` | 选中源版本 |
| --- | --- | --- |
| NAVQA | `datasets/navqa/source/navqa-coda-every10-v1/payload/source_manifest.json` | `navqa-coda-every10-v1` |
| NExT-QA | `datasets/nextqa/source/nextqa-val-v1-repaired29/payload/source_manifest.json` | `nextqa-val-v1-repaired29` |
| OpenEQA | `datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/source_manifest.json` | `openeqa-em-eqa-hm3d-scannet-v1` |

网盘对齐包以 `catalog/aligned_packages.json` 和 `catalog/aligned_units.json` 登记文件名、大小、SHA256、组别及完整单元列表。`catalog/*_frames.csv` 与 `unit_catalog.json` 中的 `datasets/...` 是旧项目规范目录；下文原目录树用于解释来源，不是所有网盘压缩包直接解压后的结构。不要仅把旧项目盘符替换成新盘符。

## 1. 图片、视频与 HSGG 的对应关系

### NAVQA

七个序列为 `seq0、seq3、seq4、seq6、seq16、seq21、seq22`，保留 `cam0/every10` 结构。

```text
datasets/navqa/source/navqa-coda-every10-v1/payload/
  navqa/seq<id>/cam0/every10/
    images/
    hsgg/frame_scene_graphs/
    frame_time_index.json
    frame_time_index.csv
  navqa/seq<id>/qa_unfilled.json
  official/qa.csv
  source_manifest.json
```

每条清单明确提供 `images_dir`、`hsgg_input_dir` 和 `frame_time_index_json`。权威 HSGG 为 `hsgg/frame_scene_graphs/*.json`，完成记录共 9,759 份；图片共 9,889 张。其余 130 张对应生成记录中的 105 次 skipped 与 25 次 error，不能补造空观察来冒充完成 HSGG。

图片匹配应读取清单、单帧记录的图片名称及时间索引；不要仅凭文件排序重命名后配对。目录中的聚合表是兼容表示，不应再与逐帧记录叠加计数。

### NExT-QA

```text
datasets/nextqa/source/nextqa-val-v1-repaired29/payload/
  items/<official_video_id>/
    frames/
    objects.json
    attributes.json
    relations.json
    hierarchy_mapping.json
    timeline.json
    source_trace.json
  evaluation/
  official/val.csv
  manifests/repaired29_manifest.json
  source_manifest.json
```

当前输入为 `normalized_directory`，570 个视频各有一个选中目录。四表以相同帧键连接，`timeline` 提供顺序、帧号、时间和完成状态，`source_trace` 保留图片及原 HSGG 来源。运行图片引用为 item 相对路径 `frames/<frame>.png`。

共 23,381 个采样帧，其中 23,044 个完成 HSGG、337 个 skipped 占位。该数据集有意保留 skipped 时间槽。来源分类为前半批次 263、后半批次 278、奇数高度修复 21、低 FPS 修复 8；这些已经组成选定的 570 视频，不是四套需要重新叠加的输入。

当前活动输入是选中四表与时间、来源记录。原始压缩包路径、重试结果和修复前文件仅属于追溯信息，不应代替当前 `items`，也不要求旧压缩包在线。

### OpenEQA

```text
datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/
  hm3d/
    frames/hm3d-v0/<episode>/
    hsgg/hm3d-v0/<episode>/frame_scene_graphs/
    provenance/
  scannet/
    frames/scannet-v0/<episode>/
    hsgg/scannet-v0/<episode>/frame_scene_graphs/
    provenance/
  official/open-eqa-v0.json
  source_manifest.json
```

`shards.hm3d`、`shards.scannet` 分别声明图片、HSGG 和 provenance 根。题目的 `episode_history` 与路径中的 `hm3d-v0/<episode>` 或 `scannet-v0/<episode>` 对应。

| 分片 | episodes | 图片 | 完成 HSGG | skipped |
| --- | ---: | ---: | ---: | ---: |
| HM3D | 63 | 5,145 | 4,956 | 189 |
| ScanNet | 89 | 26,597 | 26,078 | 519 |

权威输入为逐帧 HSGG。已核查所有完成 HSGG 均有同 episode、同帧名的 RGB，episode 集合与 QA 一致。当前源清单直接提供采样帧，不提供独立的 `video_path` 入口。

## 2. 采样与时间必须分别记录

| 数据集 | 已确认的时间语义 | 使用要求 |
| --- | --- | --- |
| NAVQA | `frame_time_index` 提供 Unix 时间及序列内相对时间 | 按索引连接原帧号、采样帧和时间；`every10` 不表示每十秒。 |
| NExT-QA | 帧文件名中的 FPS 推导名义视频秒数；不是容器 PTS | 保留 `timeline.timestamp_sec`、原帧号与 skipped 槽；不得统一假设固定采样秒数。 |
| OpenEQA | 已保存图使用 ordinal 顺序值；若有明确外部时间索引才使用真实时间 | `timestamp_is_real_time=false` 时，即使字段名为 `timestamp_sec` 也不能解释为秒。 |

OpenEQA 图片数字帧号为 `0、2、4……`，并有 stride=2、offset=0 的生成记录及存档测试证据。这仅确认抽帧间隔，不提供真实 FPS。当前活动代码中未找到原 `make_frame_subset.py` 采样实现，不能把现有记录写成“完整采样脚本已附带”。

OpenEQA 建图脚本 `scripts/datasets/openeqa/build/build_hm3d_graphs_from_hsgg.py` 支持显式 `--frame-timestamp-index`；没有该索引时使用 ordinal。现有两分片构建清单均记录 `timestamp_is_real_time=false`，不应从 stride2 推造时间戳。

NExT-QA 的两批运行、补帧、21 个奇数高度修复与 8 个低 FPS 修复，以及 NaVQA 的原帧时间映射示例和已保留的 HSGG 模型记录，集中见 [采样与 HSGG 生成说明](SAMPLING_AND_HSGG_CN.md)。

## 3. QA 版本、标签与可见历史窗口

| 数据集 | 题目与标签入口 | 本项目选中范围 |
| --- | --- | --- |
| NAVQA | `official/qa.csv`，配合各序列 `qa_unfilled.json` 与选中配置 | CSV 原有 210 题；本项目评测为七序列的 139 道 non-position 题，不能把整个 CSV 都称为当前评测集。 |
| NExT-QA | `evaluation/questions.runtime.json`、`evaluation/labels.evaluator_only.json`；官方对照为 `official/val.csv` | 570 视频、4,996 道五选一题。 |
| OpenEQA | `official/open-eqa-v0.json` | 1,636 道开放式问答：HM3D 557、ScanNet 1,079。263 题含 `extra_answers`。 |

QA 标准答案属于评分侧数据，不可写入记忆图或用于身份关联。NExT-QA 的 `labels.evaluator_only.json` 尤其应保持与建图输入分离。

时间窗口也属于评测协议，不能用一句“整段视频”替代所有数据集的约定：

- NAVQA 的现有 QA 入口支持题目时间元数据及检索窗口；应保存所用 `qa_unfilled`、Question IR、route 配置、窗口开关和参数。官方 CSV 的答案时间字段不能自动视为每题可见历史的截止时间。
- NExT-QA 当前题目关联视频级观察，输入为该视频选中采样帧及 timeline。若另设在线前缀或截止时间，必须另行记录协议，不应把它算作既有评测。
- OpenEQA 的 `episode_history` 本身是一段已提供的观察历史。当前实现使用对应 episode 的完整选中图；官方 QA 没有逐题 timestamp、history_start/end 或 cutoff 字段。因此不能声称已有逐题不同截止时间的协议，也不能说它“没有历史”。

评分时保留相应脚本、提示词、模型设置和结果版本。OpenEQA 原协议是开放回答及 LLM Judge，不能与另行提供的选择题变体混报。

## 4. 图记忆、特征与来源配套

QA 配套现已按论文冻结 C 制作完整主包：各 `QA_<DATASET>/qa_<dataset>_paper_frozen_c_v1.tar.gz` 包含原始问题、参考答案、模型回答、历史逐题评分、汇总及 HSGG 分包绑定。官方题目共 6,842 条，论文 C 回答共 6,771 条（NaVQA 的 71 道 position 题仅保留参考题目）。论文 C 为 NaVQA 102/139、NExT-QA 3,334/4,996、OpenEQA 归一化 Judge 分 53.4994；不同于旧运行入口的部分 C 快照。包已在本地完成校验，尚未确认云端上传。目录和读取说明见 [QA 资料说明](QA_RESOURCES_CN.md)，实际文件名、SHA256 与状态以 `catalog/qa_resources.json` 为准。

已经上传的图包中，主线图文件是 `temporal_graph.json` 与 `canonical_temporal_memory_graph.json`。当前登记版本如下：

| 数据集 | graph asset | 对应 index asset |
| --- | --- | --- |
| NAVQA | `navqa-cg6-time-fact-core-v1` | `navqa-cg6-retrieval-v1` |
| NExT-QA | `nextqa-cg6-human-v2-time-fact-core-v1` | `nextqa-visual-index-v2-strict-pairs-full570` |
| OpenEQA | `openeqa-time-fact-core-v1` | `openeqa-v5-full-visual-indexes` |

资源包应提供对应 graph manifest。`original` 是迁移前原字节图，`runtime` 是路径适配版本，不能把它们计为两个新方法。`time_fact_core_report`、`video_graph_audit`、运行配置和迁移记录是报告或来源证据，不是第三类主线图。

需要视觉检索时，FAISS 必须与同一版本的 metadata、候选单位及图中实体映射配套，不能只下载 `.faiss`。现有视觉索引为 SigLIP 768 维；旧 canonical record 文本缓存为 MiniLM 384 维，二者不是可互换特征。

已有特征应区分三种用途：

1. 原方法复现：按选中 index manifest 还原配套文件，并保留图版本、模型及预处理约定。
2. 新方法纯 HSGG 输入：直接读取权威观察，自行构建记忆图；旧实体、tracklet、事件索引及旧图文本向量不自动成为新方法输入。
3. 低层特征复用：只有源图片或文本、模型版本、裁剪/文本处理、归一化及向量语义一致时才可考虑；旧 SQLite 的路径和框查询键也必须匹配。仅模型名称相同不足以证明可复用。

现有 30 组图与特征包分别对应图片／HSGG 旧分包：NaVQA 7 个序列组、NExT-QA 6 个视频组、HM3D 7 个 episode 组、ScanNet 10 个 episode 组。图包共保留 729 份 canonical 与 729 份 temporal；视觉包共保留 1,611,961 个 768 维 float32 向量。没有重新建图、编码或归一化。

| 数据集／分片 | 分组方式 | 图包所在目录 | 特征包所在目录 |
| --- | --- | --- | --- |
| NaVQA | 每个 seq 一组，共 7 组 | `TIMEGRAPH_NAVQA/` | `VISUAL_FEATURES_NAVQA/` |
| NExT-QA | 100×5＋70，共 6 组 | `TIMEGRAPH_NEXTQA/` | `VISUAL_FEATURES_NEXTQA/` |
| HM3D | 10×6＋3，共 7 组 | `TIMEGRAPH_OPENEQA/HM3D/` | `VISUAL_FEATURES_OPENEQA/HM3D/` |
| ScanNet | 9×9＋8，共 10 组 | `TIMEGRAPH_OPENEQA/SCANNET/` | `VISUAL_FEATURES_OPENEQA/SCANNET/` |

OpenEQA 文件名中的 episode 范围是排序位置，不是 episode 的实际编号；准确成员以 `group_id` 与 `unit_ids` 为准。四类资源同组对应，不跨数据集合并。

图包内的两个 JSON 是同一单元 `timegraph/` 下的并列文件：

```text
<unit-root>/timegraph/
  canonical_temporal_memory_graph.json
  temporal_graph.json
  unit_binding.json
  ...时间索引、构建报告或来源记录
```

`<unit-root>` 为 `navqa/seq<id>/cam0/every10`、`nextqa-val/<video_id>`、`hm3d-v0/<episode>` 或 `scannet-v0/<episode>`。canonical 由 temporal 转换得到；canonical 本身也包含带时间的事实，并非纯静态图。

原图 JSON 以原字节保留，内部历史路径需要依据 `unit_binding.json` 映射到资源根；绑定表不是自动改写器。NaVQA 旧 JPEG 包为平铺目录，旧 HSGG 根为 `hsgg_navqa_seq<id>_every10_full/`，不能假设几个包解压到同一目录就已经统一为 `<unit-root>/frames/`。NaVQA 七个旧图内部的 `video_id` 同名，必须以外部 `dataset + seq` 区分。

NaVQA／HM3D／ScanNet 视觉包按组保存 `__group_features__/<group_id>/object.faiss`、`object.metadata.jsonl` 和导出报告；metadata 同时保留新局部行号与原全局行号。NExT-QA 在各单元 `visual_features/` 下保留六种索引及原 metadata，另提供便携 metadata；10 个空事件／前后对分片只有空 metadata，属于已有索引的有效状态。

旧 NaVQA／OpenEQA runtime 读取的是全局索引与 SQLite `object_metadata` 表，新分组 JSONL 包不能直接当作原 SQLite 使用。便携读取和显式路径绑定见 [读取示例与校验](ALIGNED_READER_CN.md)；完整旧主线复现仍需恢复其注册目录和其他声明的依赖，不能把单个向量读取通过写成完整 QA 已复现。

## 当前权威输入与旧冻结结果的差异

现有目录整理保留了权威逐帧观察，也保留旧聚合表作来源证据。需要特别记录：

- NAVQA seq22 的旧聚合表比完成 sidecar 少 2 帧；当前权威 sidecar 包含完整 1,530 帧。
- OpenEQA HM3D 六个 episode 的旧聚合表合计比完成 sidecar 少 37 帧；当前建图优先完成 sidecar。
- 旧论文冻结图和预测没有因目录整理而重新计算。用新版完整观察重新建图时，应在同一输入版本重建基线，或明确报告该输入差异。

`provenance/history` 中的旧聚合表、重试输出和历史派生图不应与主线权威输入叠加。来源文档见 [各数据集独立整理](../datasets/SOURCE_ORGANIZATION_20261009_CN.md)。

## 资源还原后的检查顺序

1. 对照 `aligned_packages.json` 的文件名、SHA256 和 `unit_ids`，确认下载了同组图片、HSGG、图和特征；保持原帧名及单元 ID。
2. 根据绑定信息显式指定实际解压位置，按读取示例核查图片、HSGG、图文件和向量行的对应；不要将采样序号或 OpenEQA ordinal 当成秒。
3. 选择复现或新建图路线。新项目可独立读取这些资源；完整旧主线需要进一步恢复其 `datasets/...` 布局、全局索引接口及其他依赖，新建图使用独立 run ID。
4. 若已经恢复为旧项目规范布局，可从仓库根运行下列检查。此命令不是新分包的通用导入器，也不替代资源下载后的 SHA256 校验：

```powershell
python scripts/datasets/manage.py validate --dataset all
```

NExT-QA 另有 `scripts/datasets/nextqa/audit/verify_materialized_source.py`，可逐项核验解压输入与图片。详细用法见 [NExT-QA 当前主线](../datasets/nextqa/MAINLINE_CN.md)。

发布时还需明确外部数据、标注及模型各自的来源和再分发条件；本说明不代替上游许可，也不表示已取得所有第三方材料的公开授权。

## 本次交付边界

全量帧映射、采样和时间说明、旧图／视觉向量配套以及 QA ID／版本指纹已经提供。按用户安排，本次不整理第一轮人工回看的视频／帧短名单。原采样脚本缺失处和已有图／特征覆盖缺项保留原状，不以推断补造。网盘上传状态由用户确认；云端字节复核、完整旧算法重建与在线 QA 不是本次读取示例的验收结论。
