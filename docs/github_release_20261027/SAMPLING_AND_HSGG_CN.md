# 视频采样、时间与 HSGG 生成记录

本文根据选中源清单、逐帧来源记录和保留的运行 manifest 汇总已有数据的生成方式。它说明当前受控网盘资源，不执行重新采样或生成 HSGG，也不整理人工回看短名单。

## 处理链与权威输入

已有处理链为：视频或原始序列图像 → 选中采样画面 → HSGG 单帧检测、属性与关系 → 按单元保存逐帧 HSGG → 旧 TimeGraph 关联及 canonical 转换。HSGG 单帧对象下标只是该帧的局部索引，不是跨帧稳定身份；旧轨迹、实体和事件由旧算法生成，不是人工身份真值。

图片和 HSGG 保留已有网盘分包。NExT-QA／OpenEQA 对应单元内为 `frames/` 和 `hsgg/`；逐帧 `hsgg/frame_scene_graphs/*.json` 是权威观察。NExT-QA 本机 `items/<video_id>/` 的 objects、attributes、relations、hierarchy_mapping 四表是规范化读取视图，不能据此声称旧 HSGG 网盘包已经具有相同四表布局。旧聚合表可用于来源核对，不与逐帧记录叠加；重新规范化时应从完成的逐帧 sidecar 构造。

## NaVQA：每 10 个原帧取一帧

范围为 `seq0、seq3、seq4、seq6、seq16、seq21、seq22` 的 cam0，共 9,889 张图片、9,759 个完成 HSGG；其余 130 张对应 105 次 skipped 和 25 次 error。`every10` 表示原始帧号步长 10，不是每 10 秒。

时间应依次读取：

1. `sampled_index`：在已提供采样图片中的顺序。
2. `graph_frame_index`：旧图使用的帧标识，不能代替原帧号。
3. `source_frame_index`／源索引中的 `source_frame`：原始序列帧号。
4. `timestamp_unix`：通过原始 timestamps 数组按原帧号取得的时间。
5. `elapsed_seconds`：Unix 时间减该序列起始时间。

例如 seq0 的第二张采样图，`sampled_index=1`、`graph_frame_index=2`、`source_frame=10`，相对序列开始约 1.000084 秒。这三个编号不同，不能把前缀 `1` 或图帧号 `2` 直接当秒。

GitHub 的 `catalog/navqa_frames.csv` 已提供全量映射。网盘图包中每个单元有 `timegraph/frame_time_index.json`；原始时间索引构造入口为 `scripts/build_navqa_frame_time_index.py`。旧图片包是平铺的 JPEG 名称，HSGG 旧根为 `hsgg_navqa_seq<id>_every10_full/`。新包内的 `<unit-root>/frames/` 是绑定约定，需要显式指向实际图片目录。

NaVQA QA 的 `start_time/end_time` 是题目窗口元数据；CSV 的答案事件时间不能自动当作可见历史截止。旧检索加载完整序列图后使用窗口检索，并非所有路线都统一执行严格前缀截断。

## NExT-QA：两批运行与 29 个选中修复

当前为 validation 的 570 个视频，共 23,381 张采样图片，其中 23,044 个完成 HSGG、337 个 skipped 时间槽。视频键始终使用 `official_video_id`，而非服务器任务序号。

最初两个 GPU 分别处理前后两半。选中源清单记录的流程为：前半先按 every60 运行，再补 every30 所需画面；后半直接按 every30 运行。最终普通视频的规则是每 30 个解码帧取一帧，不是所有视频每秒一帧，也不是 uniform32。

| 当前选中来源 | 视频数 | 采样／处理约定 |
| --- | ---: | --- |
| `primary_first` | 263 | 前半 every60 初跑＋every30 补充后的选中版本 |
| `primary_second` | 278 | 后半 direct every30 版本 |
| `repair_oddheight` | 21 | 奇数高度相关解码修复；仍为 `every_30_decoded_frames` |
| `repair_lowfps` | 8 | 实际 10／15 FPS 视频，改为 `one_frame_per_second_at_actual_source_fps` |
| 合计 | 570 | 每视频仅保留一个选中版本 |

21＋8 个修复按视频 ID 替换原版本，不与旧版本叠加，也不把重跑文件当新视频。12 个未采用的正常 FPS 重跑不在当前选择范围；这不表示删除这 12 个视频的原始选中版本。

八个实际低 FPS 视频为 `2442307875、4882414082、5047408177、6188449000、6244675392、8188651630、9304738817、9528990839`。其中第一个为约 10 FPS，其余约 15 FPS。这份列表说明特殊采样规则的适用范围，不是人工回看样本选择。

文件名 `frame_<source-frame>_<seconds>s.png` 中的 seconds 采用原帧号与源 FPS 推导的名义秒数，保留到三位小数；不应宣称来自容器 PTS。例：

- 普通视频 `10001787725`：原帧 0／30／60 对应 0／1.001／2.002 秒，约 29.97 FPS。
- 低 FPS 修复 `2442307875`：原帧 0／10／20 对应 0／1／2 秒，约 10 FPS。

因此 ordinary every30 的真实间隔可能约 1.001、1.2、1.25 或 1.251 秒。读取时使用每视频 `timeline` 或 `catalog/nextqa_frames.csv` 中保留的帧号与秒数，不把第 n 张图片直接解释为第 n 秒。修复清单旧字段曾描述 source PTS／actual FPS 的可能来源，当前选中主清单明确采用 FPS-derived nominal seconds；该较严格定义作为使用约定。

HSGG 的 skipped 包括检测框少于两个对象后不生成场景图的情况。采样图片及时间槽仍保留，不能把没有逐帧 JSON 的 skipped 槽补造为完成观察。旧聚合表曾有保存和重试后的历史缺帧，当前规范输入使用完成 sidecar，不从聚合表决定真实完成帧数。

网盘六组按既有项目视频顺序划分为 100×5＋70；文件名的 `videos001-100` 是排序位置。实际视频集合由组清单完整 `unit_ids` 决定，不能按 `official_video_id` 字典序重新分组。

## OpenEQA：stride2 是顺序采样，不是真实秒数

当前 HM3D 为 63 个 episode、5,145 张图片、4,956 个完成 HSGG；ScanNet 为 89 个 episode、26,597 张图片、26,078 个完成 HSGG。其余分别为 189 和 519 个 skipped。

图片数字帧号通常为 0、2、4……；保留的生成记录确认 stride=2、offset=0。当前活动源码未找到原 `make_frame_subset.py` 实现，真实 FPS 也未得到可靠记录。因此这里交付的是采样规则及实际图片／HSGG 映射说明，不能声称原采样脚本和真实视频时间均已恢复。

旧图与构建 manifest 使用 ordinal 时间，`timestamp_is_real_time=false`；图中名为 `timestamp_sec` 的值在这种配置下仍是排序值，不是真实秒。`catalog/openeqa_frames.csv` 将真实秒字段留空，未找到旧图排序映射的 745 张图片也保留空值；其中包含 skipped 及比旧冻结图新增完成的帧。

建图入口 `scripts/datasets/openeqa/build/build_hm3d_graphs_from_hsgg.py` 可显式接收 `--frame-timestamp-index`，但这不代表现有数据已经具有该真实时间索引。没有来源依据时不根据 stride2 推造 FPS 或秒数。

HM3D 分为 10×6＋3 共七组，ScanNet 分为 9×9＋8 共十组。episode 文件名范围是既有排序位置，不是原 episode 数字前缀连续区间。

## 保留的 HSGG 模型记录

保留的 NaVQA 上游 `run_manifest.json.model_config` 记录以下模型；OpenEQA HM3D 运行命令也保存了同名检测、深度模型及本地 LLaVA 权重目录名：

| 用途 | 记录中的模型 |
| --- | --- |
| 视觉语言属性／关系生成 | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` |
| 对象检测 | `iSEE-Laboratory/llmdet_large` |
| 深度估计 | `depth-anything/da3-base` |

8B 是模型参数规模，不是权重磁盘体积。这里保存的是历史运行模型名称／目录证据，未核实到所有批次权重文件的精确 revision 或二进制 SHA；不据此宣称可以逐位复现。当前 HSGG 源码的默认回退模型可能与历史实际运行配置不同，应以对应运行 manifest／命令为准。读取已生成 HSGG、图与向量不要求重新下载这些模型。

NaVQA seq6 记录温度为 0.3，HM3D 记录命令温度为 0.1；不能把一个数据集的生成配置推广到所有批次。原始模型权重不在 GitHub，也不在图／视觉特征这 60 个包中。

## 来源证据与读取边界

上述说明来自已有文件，路径相对于原项目数据根；这些保留证据本体未因此自动包含于代码仓库：

- NExT-QA：`datasets/nextqa/source/nextqa-val-v1-repaired29/payload/source_manifest.json` 的 `hsgg_generation_provenance`、`source_layout.generation_counts` 与 `observed_timing_summary`；`manifests/repaired29_manifest.json` 中逐视频 `special_patch_sample_policy`；各 item 的 `timeline.json` 和 `source_trace.json`。
- NaVQA：`datasets/navqa/source/navqa-coda-every10-v1/payload/navqa/seq<id>/cam0/every10/frame_time_index.json`；`provenance/history/seq6/hsgg_upstream/run_manifest.json`。
- OpenEQA：`datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/hm3d/provenance/generation/run_manifest.json`；两分片的生成／准备记录及图内时间索引。
- 网盘旧 HSGG 包的单元布局由既有配对打包规则固定；新增图／特征包的文件与 SHA256 见 `catalog/aligned_packages.json`，单元入口见 `catalog/aligned_units.json`。

图包的 `temporal_graph.json` 和 `canonical_temporal_memory_graph.json` 在各单元 `timegraph/` 下并列保存，保留原图字节及其历史路径。视觉包保存已有向量和 metadata。具体路径绑定与单行向量读取见 [便携读取示例](ALIGNED_READER_CN.md)；完整旧主线的全局索引／SQLite 导入和在线 QA 另行验证。
