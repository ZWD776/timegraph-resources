# 数据与运行资源准备说明

本文说明 GitHub 代码仓库与网盘资源如何对应。三个数据集保持各自独立的清单和目录，不把它们拼成一个数据集。

## 发布边界与资源入口

- GitHub 准备代码、配置、测试、使用说明、资源清单和必要的版本记录。
- 图记忆文件、HSGG 输出、图片及视频由网盘提供，不随 GitHub 主仓库上传。
- 网盘链接：**待用户提供，尚未录入**。本文不提供占位网址，也不假定网盘已经公开。
- FAISS、SQLite 特征库及文本向量缓存单独登记，发布位置待确定；不因它们不是图片就自动复制进代码仓库。
- 环境目录、模型权重、密钥及本地包缓存不属于上述数据资源。环境与模型的准备以最终发布的环境说明为准。

资源入口以 `registry/datasets/<dataset>.yaml`、对应 `asset_pointer.json` 和 `source_manifest.json` 为准。不要从目录修改日期、旧实验名称或压缩包名称猜测有效版本。

下文 `<DATA_ROOT>` 表示资源目录的根，目录结构相对于该根展开。现有项目入口主要按仓库根解析 `datasets/...`；直接运行这些入口时，资源需还原到对应的仓库相对位置。本文不宣称所有脚本已经支持任意外置数据根。

| 数据集 | 选中源清单，相对于 `<DATA_ROOT>` | 选中源版本 |
| --- | --- | --- |
| NAVQA | `datasets/navqa/source/navqa-coda-every10-v1/payload/source_manifest.json` | `navqa-coda-every10-v1` |
| NExT-QA | `datasets/nextqa/source/nextqa-val-v1-repaired29/payload/source_manifest.json` | `nextqa-val-v1-repaired29` |
| OpenEQA | `datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/source_manifest.json` | `openeqa-em-eqa-hm3d-scannet-v1` |

网盘资源清单至少应标明资源版本、目标目录、文件数、字节数和校验值。链接尚未提供时，不能把本地已有资源误写成其他人可以下载的资源。

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

图记忆放网盘时，主线图文件是 `temporal_graph.json` 与 `canonical_temporal_memory_graph.json`。当前登记版本如下：

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

特征附件的存放位置和下载链接尚未确定。本准备步骤不复制现有 GB 级 FAISS、SQLite 或向量缓存，也不把未验证覆盖范围的 cache 宣称为完整特征集。

## 当前权威输入与旧冻结结果的差异

现有目录整理保留了权威逐帧观察，也保留旧聚合表作来源证据。需要特别记录：

- NAVQA seq22 的旧聚合表比完成 sidecar 少 2 帧；当前权威 sidecar 包含完整 1,530 帧。
- OpenEQA HM3D 六个 episode 的旧聚合表合计比完成 sidecar 少 37 帧；当前建图优先完成 sidecar。
- 旧论文冻结图和预测没有因目录整理而重新计算。用新版完整观察重新建图时，应在同一输入版本重建基线，或明确报告该输入差异。

`provenance/history` 中的旧聚合表、重试输出和历史派生图不应与主线权威输入叠加。来源文档见 [各数据集独立整理](../datasets/SOURCE_ORGANIZATION_20261009_CN.md)。

## 资源还原后的检查顺序

1. 对照资源版本及校验清单，将网盘包还原到所声明的相对目录；保持题目 ID、video ID、episode_history 和原帧名。
2. 确认三个 source_manifest 能解析到权威 HSGG、时间信息及对应图片；检查 skipped/error 与完成记录的区别。
3. 选择复现或新建图路线。复现时使用配套旧图、索引与 QA 版本；新建图使用独立 run ID，避免覆盖冻结资产。
4. 环境准备完成后，可从仓库根运行下列检查。它需要本地资源，单独 clone 代码不能替代资源下载：

```powershell
python scripts/datasets/manage.py validate --dataset all
```

NExT-QA 另有 `scripts/datasets/nextqa/audit/verify_materialized_source.py`，可逐项核验解压输入与图片。详细用法见 [NExT-QA 当前主线](../datasets/nextqa/MAINLINE_CN.md)。

发布时还需明确外部数据、标注及模型各自的来源和再分发条件；本说明不代替上游许可，也不表示已取得所有第三方材料的公开授权。
