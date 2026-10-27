# TimeGraph 代码与资源清单

本私有仓库整理已有 TimeGraph 项目的代码、配置和资源对应关系，供实验复现及后续研究参考。
三个数据集分别管理：NaVQA、NExT-QA、OpenEQA；OpenEQA 内保留 HM3D 与 ScanNet 分片。
这里的旧轨迹与图是算法产物，不是人工身份真值，也不代表新图记忆方法已经实现。

## 数据获取

图片、HSGG、图记忆和视觉特征已由维护者上传至同一受控分享。访问方式由维护者单独提供，仓库不记录分享链接、提取码或其他访问凭据。

- 云端使用原分包：NaVQA 7 组、NExT-QA 6 组、HM3D 7 组、ScanNet 10 组；四类资源按相同单元范围对应。
- 新增图与特征共 60 个压缩包，覆盖 729 个单元，约 12.98 GiB；单包小于 4 GB。文件名、大小、SHA256 和配对关系见 [分包清单](catalog/aligned_packages.json)，单元入口见 [单元清单](catalog/aligned_units.json)。
- 图包保留每单元的 `canonical_temporal_memory_graph.json` 与 `temporal_graph.json`；视觉包保留已有 SigLIP 向量及行号映射。上传由维护者确认，本地包已完整校验，云端字节尚未重新下载复核。

本仓库不含图片、视频、HSGG 正文、图正文、embedding、FAISS、数据库或模型权重。
官方 QA、参考答案、模型回答和逐题评分正文不进入 GitHub；外部 QA 综合包已包含这些资料，`catalog/source_versions.json` 提供原始标注版本与 SHA256，获取和使用遵循各上游条款。

## 先检查资源清单：只需要 Python 标准库

在本仓库根目录运行：

```text
python scripts/release/verify_resource_catalog.py
```

该命令校验文件 SHA256、65,012 帧映射、6,842 道官方 QA 的 ID 映射、729 个单元、时间类型及跨表对应。
它不下载数据、不加载模型、不调用 API，也不自动查找原电脑的数据目录。
如果已有按 `datasets/...` 布局放置的外部数据，可显式指定根目录，附加只读文件存在性检查：

```text
python scripts/release/verify_resource_catalog.py --data-root /path/to/resource-root
```

`--data-root` 是包含 `datasets/` 的旧项目资源根，仅检查图片/HSGG文件存在性。它不导入新分包，也不验证图或索引的读取。网盘压缩包的实际单元目录见 [读取与绑定示例](docs/github_release_20261027/ALIGNED_READER_CN.md)，不能只更换盘符。

新包的 `unit_binding.json` 用于明确绑定图、画面、HSGG 与特征。NaVQA 旧图片包为平铺结构；NaVQA/HM3D/ScanNet 新视觉包为分组 FAISS + JSONL，与旧 runtime 的全局索引 + SQLite 不同。读取示例不需要模型，不会自动改写旧图或原数据。完整旧主线复现仍需对应索引/目录适配及其余检索资源。

| 数据集 | 图片映射 | 当前完成 HSGG | 官方 QA ID | 单元 |
| --- | ---: | ---: | ---: | ---: |
| NaVQA | 9,889 | 9,759 | 210（主线 non-position 为 139） | 7 |
| NExT-QA | 23,381 | 23,044 | 4,996 | 570 |
| OpenEQA | 31,742 | 31,034 | 1,636 | 152 |

`image_available`、`hsgg_available` 表示导出时的原始资源状态，不表示新克隆已下载这些文件。
所有文件引用相对于明确选定的数据根目录；清单内不含本机绝对路径。

## 时间、身份和版本说明

- NaVQA 保留采样序号、图帧号、原帧号、Unix 时间和经过秒数；不能将文件名前缀当秒数。
- NextQA 保留最终选中的 570 个视频版本；秒数来自帧号与 FPS，不是容器 PTS。网盘 HSGG 使用逐帧 `hsgg/frame_scene_graphs/`；旧项目规范化四表视图需按 `hsgg_record_key` 一起读取。
- OpenEQA 的 `ordinal_time` 是旧图排序值，不是真实秒数；缺失旧图映射的 745 帧保持空值。
- NaVQA QA 窗口使用 `start_time/end_time`，与答案事件时间分开；旧检索不保证统一硬截止。NextQA/OpenEQA 使用整段提供的视频或 episode 历史，没有逐题截止字段。
- 当前 HSGG 与冻结旧图的覆盖不同：NaVQA 9,759 对 9,757；OpenEQA 31,034 对 30,997。NextQA 图的 23,381 帧计数包含 skipped 时间槽。

详见 [采样与 HSGG 生成说明](docs/github_release_20261027/SAMPLING_AND_HSGG_CN.md)、[资源准备说明](docs/github_release_20261027/RESOURCE_PREPARATION_CN.md) 和 [发布范围](docs/github_release_20261027/RELEASE_SCOPE_CN.md)。首轮人工回看短名单暂不制作。

QA 配套已按 **2026-09-04 论文冻结 C** 整理成三个完整主包：`qa_navqa_paper_frozen_c_v1.tar.gz`、`qa_nextqa_paper_frozen_c_v1.tar.gz`、`qa_openeqa_paper_frozen_c_v1.tar.gz`，分别位于资源目录的 `QA_NAVQA/`、`QA_NEXTQA/`、`QA_OPENEQA/`。三包共 6,494,001 字节，包含原始问题、参考答案、模型回答、逐题历史评分、汇总和 HSGG 分包绑定；旧参考 QA 包保留备份，无需同时解压。`CURRENT_PACKAGE.json` 指向推荐主包。

| 论文冻结 C | 评测范围 | 保存结果 |
| --- | --- | --- |
| NaVQA | 139 道 non-position | 102/139，73.3813% |
| NExT-QA | 4,996 道五选一 | 3,334/4,996，66.7334% |
| OpenEQA | 1,636 道开放问答 | 53.4994（0–100 归一化 LLM Judge 分数） |

这些是已有冻结回答及历史评判，未重新调用模型或 Judge。当前旧运行入口的 NaVQA C=101/139、OpenEQA C=53.2549 是不同结果快照，不能替代论文冻结 C。包内目录、读取方法、版本及时间标签限制见 [QA 资料说明](docs/github_release_20261027/QA_RESOURCES_CN.md)，文件大小与 SHA256 见 [QA 包清单](catalog/qa_resources.json)。QA 综合包已在本地完成并校验，尚未确认上传到外部分享。

## 运行已有代码

只阅读和校验 `catalog/` 不需要安装模型依赖。运行旧图构建或 QA 代码时，先创建独立环境，再安装：

```text
python -m venv .venv
# 激活该环境后：
python -m pip install -e ".[test]"
python scripts/datasets/manage.py validate --metadata-only
```

`--metadata-only` 检查三个数据集的注册表、指针和配置，不要求大型 payload 或历史清理台账存在。
完整 `status/validate/build/index/evaluate` 等阶段仍要求对应外部资源；没有数据时会明确报缺项，不会回退到原项目。
旧算法的注册入口使用本仓库下的 `datasets/...` 布局，路径均以当前仓库定位；`verify_resource_catalog.py --data-root` 的外部根参数仅用于清单检查，不会偷偷重定向算法输出。
阶段脚本仍接受其显式输入参数，模型/API 需求以 `configs/datasets/<dataset>/` 为准。

已适配入口使用当前 Python 解释器，不固定 Windows 虚拟环境路径。在线 API 需要自行设置 `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY` 等配置中指定的环境变量；仓库不会携带凭据。
`pyproject.toml` 是依赖声明，不是完整环境锁；此次未安装新包、下载模型或重跑完整数据集实验。

最小离线发布回归：

```text
python -m pytest tests/test_release_resource_catalog.py tests/test_release_metadata_registry.py tests/datasets/nextqa -q
```

缺少外部数据的集成测试会明确跳过。完整旧项目测试还包含历史迁移或全数据依赖，不能把这些测试当作未下载数据的新克隆的验收条件。

## 维护与发布记录

- `scripts/release/export_resource_catalog.py` 是维护导出工具，必须显式传入完整 `--data-root` 和新的 `--output`，不覆盖非空目录。
- `publication_inventory.json` 记录当前发布文件校验值；适配仅发生在独立发布副本。
- `release_validation.json` 保留首次发布的历史验收，`resource_delivery_validation.json` 记录本次维护验收；均不代表云端下载和在线模型实验已验证。
- 旧删除、迁移和封存工具只用于相应历史工作流，不是常规数据准备入口；本仓库没有携带其依赖的历史 archive 台账。

本仓库为私有协作资源。自有代码许可证由作者确定；第三方来源、许可和引用应保留，不能将所有数据统一视为同一代码许可证。
