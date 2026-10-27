# 图与视觉特征对齐包：解包和最小读取

本文对应已制作的 30 组图记忆/视觉特征配套包。资源访问方式由维护者单独提供，本文不包含网盘地址或提取码。三个数据集各自独立；OpenEQA 分为 HM3D、ScanNet。

## 目录与文件身份

同组图片、HSGG、timegraph、visual_features 以 `group_id` 和完整 `unit_ids` 对齐。文件名中的范围是排序位置，不是 episode 名称前面的实际编号。组数为 NaVQA 7、NExT-QA 6、HM3D 7、ScanNet 10。新图/特征包总计 60 个，文件大小、SHA256 和配对名称以 `catalog/aligned_packages.json` 为准。

图包和特征包解压到自行选择的同一目录 `<UNPACK_ROOT>` 后，主要目录为：

```text
<UNPACK_ROOT>/
  navqa/seq0/cam0/every10/
    timegraph/
      temporal_graph.json
      canonical_temporal_memory_graph.json
      unit_binding.json
    visual_features/unit_binding.json
  nextqa-val/10001787725/
    timegraph/{temporal_graph.json,canonical_temporal_memory_graph.json,unit_binding.json}
    visual_features/
      object.faiss
      object.metadata.json
      object.metadata.original.json
      ...其余候选类型...
      unit_binding.json
  hm3d-v0/<episode>/{timegraph/,visual_features/unit_binding.json}
  scannet-v0/<episode>/{timegraph/,visual_features/unit_binding.json}
  __group_features__/<group_id>/
    object.faiss
    object.metadata.jsonl
    export_report.json
```

`canonical` 与 `temporal` 是同一单元下并列的文件；生成过程中先有 temporal，再转换为 canonical。旧 canonical 包含带时间的实体与事实，不是纯静态拓扑。每个单元一份两类主线图，未加入历史重复图。

NaVQA 旧图片包内 JPEG **平铺**，旧 HSGG 根为 `hsgg_navqa_seq<id>_every10_full/`，与上面的新单元目录不同。不要把不同序列的平铺图片混进同一个目录；读取时分别传入该序列实际的图片和 HSGG 目录。`portable_images_root` 只是绑定约定，不表示旧图片包本身已有这个目录。NaVQA 七份图内部的 `video_id` 相同，单元身份必须使用外部 `dataset + seq`。

NExT-QA、HM3D、ScanNet 原图片/HSGG分包以单元为目录，通常可与新包在同一 `<UNPACK_ROOT>` 下合并。仍应对照具体文件名和 `unit_binding.json`；NExT-QA HSGG 逐帧 sidecar 与旧项目规范化四表是两种表示，不能重复计数。

## 最小读取命令

`scripts/release/read_aligned_unit.py` 只需 Python 标准库，不安装 FAISS、Torch 或模型。它读取一条 metadata，随机定位已有 768D float32 向量，检查相应图片、HSGG 记录及两份图文件存在。它不编码图片、不下载、不写文件，也不自动搜索原电脑路径。

已合并解包 NExT-QA 图片、HSGG、图和特征后：

```text
python scripts/release/read_aligned_unit.py --root /data/unpacked --unit-root nextqa-val/10001787725
```

NaVQA 显式绑定旧包解压后的实际位置：

```text
python scripts/release/read_aligned_unit.py --root /data/unpacked --unit-root navqa/seq0/cam0/every10 --images-root /data/navqa_seq0_images --hsgg-root /data/hsgg_navqa_seq0_every10_full
```

HM3D / ScanNet 的 `--unit-root` 分别填写 `hm3d-v0/<episode>`、`scannet-v0/<episode>`。必要时也可用 `--images-root` 和 `--hsgg-root` 指定该单元的实际位置。HSGG 根可直接包含 sidecar，或包含 `frame_scene_graphs/`；NExT-QA 还支持包含 `objects.json`、`attributes.json`、`relations.json`、`hierarchy_mapping.json` 的规范化目录，并校验该帧键同时存在于四张表。

可加 `--visual-archive /downloads/<visual_features>.tar.gz` 直接从本地特征包读取，省去解压该特征包。图片、HSGG 和图仍须已存在；tar.gz 内定位需要解码所经的压缩数据，速度慢于已解压索引，但不会整载大索引或图正文。

默认选择该单元已有的第一条对象向量。`--local-faiss-id N` 选择该单元对应的索引局部行号；`--include-vector` 输出全部 768 个值。常规输出仅给向量 SHA256、L2 范数和前八维。函数 `read_index_vector()` 可供 Python 代码直接取得数值及原始 float32 字节。

NExT-QA 还提供 frame、tracklet、temporal_window、event_boundary、before_after_pair 类型索引；`--kind` 可以选类型，但这份最小观察绑定示例需要候选有单帧路径及帧键，建议使用 `object`。10 个空事件/前后对 shard 只有空 metadata，没有 FAISS，属于正常情况。

## 特征行号与使用边界

NaVQA/HM3D/ScanNet 是分组索引：JSONL 的 `local_faiss_id`（亦即 `faiss_id`）才是新包中的向量行号；`original_faiss_id` 是原全局索引行号，用于追溯，不可直接索引新包。原始 metadata 字符串和数据库字段作为来源保留，实际读取使用新相对路径。NExT-QA 每视频六类索引保留原字节，便携 `.metadata.json` 与 `.metadata.original.json` 并列。

所有已打包向量为既有 SigLIP 768D float32，不重新编码、归一化或降精度。旧 tracklet/window/event/pair 特征依赖原图分段，不能直接用作新方法身份真值。MiniLM 文本向量、混合 embedding SQLite 和证据层数据库不在这 60 个包内。

两类图正文保留原始字节，其中的历史绝对路径没有逐项改写。`unit_binding.json` 是对应关系，不是自动导入器。本读取器检查的是选定观察的文件/帧键和向量行，不验证整图身份、拓扑或所有路径，也不把旧图当作人工标注。

旧 NaVQA/OpenEQA 运行时仍使用全局 FAISS 和 `metadata.sqlite`；新分组 JSONL 不能直接替换它。本文提供的读取路线用于复用资源与新项目输入核验；完整恢复旧 QA 主线还需按旧注册目录导入相应索引和其他派生资源。

覆盖缺项保持原状：NaVQA 旧图比当前 HSGG 少 2 帧、HM3D 少 37 帧，NaVQA 对象特征也不是当前所有对象的完整覆盖；缺失记录和缺失特征没有补造。OpenEQA 的旧排序时间不是已知真实秒数。

## 验证范围

维护端可以显式加 `--legacy-data-root /path/to/original-root`，将新包的相对来源绑定到原始项目，只读核验图片、HSGG和图。该选项不会默认启用，也不是普通下载用户需要准备的额外项目。对这种本机检查，应报告为“本地新包与原始数据绑定验证”；它不等于从云端重新下载后的解包验收。

网盘上传状态由维护者确认；压缩包本地 SHA256、成员 SHA256 和 gzip 完整性已检查。云端下载字节的复核需下载后对照 `catalog/aligned_packages.json` 中的哈希。脚本不访问网盘。
