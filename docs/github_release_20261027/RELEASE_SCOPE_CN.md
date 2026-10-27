# GitHub 发布范围与验收说明

日期：2026-10-27。本文记录私有 GitHub 代码与受控网盘资源的交付范围。用户已确认图片、HSGG、图与视觉特征上传至同一分享；本地包完整性已校验，云端字节尚未重新下载核验。

## 1. 主仓库范围

GitHub 主仓库以复现代码和资源说明为主，采用明确的文件白名单导出：

- 本项目代码、运行与构建脚本、必要测试、配置模板、环境与依赖说明。
- 方法说明、数据适配说明、QA 与评测入口、选定主线的运行命令及结果摘要。
- 经检查并脱敏的 registry、资源清单、相对路径元数据、版本与 SHA256 校验信息。
- 数据与可选附件的获取说明；本机绝对路径、用户名、密钥、令牌和账户配置不得进入发布版本。

NAVQA、NExT-QA、OpenEQA 保持各自目录、字段和时间语义。相对资源路径应由显式配置的数据根目录解析，不依赖原电脑的盘符或工作目录。配置模板只给出占位路径；本地配置不直接公开。

source_manifest 不能因文件小就原样进入仓库：需检查其中的本机路径、第三方内容、原始 QA 或答案及来源信息。原始 source 和 QA 在本地完整保留；许可和发布范围核实前，不将整个 source 目录或 QA 原表作为默认公开包。可先发布经过审查的资源描述、适配代码和上游获取链接。

## 2. 受控网盘资源

以下大型资源不推入 Git。网盘访问方式由维护者按授权范围单独提供；仓库不保存分享 URL、提取码或其他访问凭据。

| 资源 | 当前交付方式 | 配套信息 |
| --- | --- | --- |
| 图记忆 | 用户确认已上传：30 个图包 | 729 份 canonical＋729 份 temporal、必要时间索引／报告、unit binding、文件与 SHA256 清单 |
| HSGG 输出 | 用户此前已上传，保持旧分包对应 | 权威逐帧记录、实际帧覆盖、缺失／失败状态及来源 |
| 图片／视频帧 | 用户此前已上传，保持旧分包对应 | 帧映射、单元 ID、原帧名及时间语义；不声称包含所有原始视频 |
| 视觉特征 | 用户确认已上传：30 个特征包 | SigLIP FAISS、配套 metadata、原／局部行 ID 与图绑定 |
| QA／参考答案／模型回答及评分 | GitHub 提供 ID、版本 SHA256、论文 C 汇总和时间窗口元数据；三个外部综合包已在本地完成，尚未确认云端上传 | 综合包含基础 QA 与论文冻结 C 回答及历史分数；当前主包与旧参考备份的区别见 [QA 资料说明](QA_RESOURCES_CN.md) |

30 组为 NaVQA 7、NExT-QA 6、HM3D 7、ScanNet 10。图／特征新增 60 包共 13,940,403,950 字节；每包小于 4,000,000,000 字节。`catalog/aligned_packages.json`、`aligned_units.json` 提供组、文件及 729 个单元的对应。用户确认上传不替代下载后的 SHA256 比对。

## 3. 特征与索引作为可选附件

已制作并上传的视觉附件复用 1,611,961 个 768 维 float32 SigLIP 向量，不重新编码、降精度或归一化。向量的原 float32 字节、metadata 及行 ID 对应已核验；图正文保持原字节。

- NaVQA／HM3D／ScanNet 是分组 FAISS＋JSONL metadata，保留原全局行 ID 与新局部行 ID。NExT-QA 保留每视频六类索引、原 metadata 及便携 metadata。
- 旧 SQLite 查询接口、路径、全局索引范围与新包不同；`unit_binding.json` 是映射说明。使用 [便携读取示例](ALIGNED_READER_CN.md) 显式绑定实际资源，不声称解压后旧主线就能直接运行。
- 旧图覆盖少于当前权威 HSGG：NaVQA 少 2 帧，HM3D 少 37 帧；NaVQA 已有对象向量也不是当前全部对象的完整特征集。缺失项与无 canonical 绑定原样保留。
- tracklet、temporal window、event boundary、before/after pair 是旧图派生特征，不能作为新方法的身份真值；基础对象向量在语义与预处理一致时可复用。
- MiniLM 文本向量、历史混合 embedding SQLite、证据层 SQLite、anchor 等其他缓存不在这 60 包内。附件不是原项目所有索引和缓存的完整镜像。
- 原图内历史路径需要显式绑定；NaVQA 旧 JPEG 为平铺结构，七图内部 `video_id` 同名，按 `dataset + seq` 区分。

不随仓库发布虚拟环境、pip 缓存、无关模型缓存。模型通过名称、revision、获取链接和适用许可证说明恢复；模型权重的分发条件单独核对。

## 4. 来源与许可准备

本项目自有代码的 LICENSE 尚待作者确定，本次不擅自选择 MIT、Apache 或其他许可证。引用或随包分发的第三方代码保留其 LICENSE、NOTICE、来源和必要修改说明；第三方代码许可不自动覆盖本项目全部内容或原始数据。

目前可作为核对依据的来源：

- HSGG 本地源码包含 Apache-2.0 许可证：参见项目中的 `HSGG/LICENSE`、`HSGG-main/LICENSE`。其依赖与模型权重仍按各自条款核对。
- NAVQA / ReMEmbR：[上游评测说明](https://github.com/NVIDIA-AI-IOT/remembr/blob/main/eval.md)、[上游许可证](https://github.com/NVIDIA-AI-IOT/remembr/blob/main/LICENSE.md)；其 CODa 来源另见 [CODa 数据说明与条款](https://github.com/ut-amrl/coda-devkit/blob/main/docs/DATA_REPORT.md)。
- NExT-QA：[官方仓库及来源说明](https://github.com/doc-doc/NExT-QA)、[仓库许可证](https://github.com/doc-doc/NExT-QA/blob/main/LICENSE)。视频源的 VidOR / YFCC 使用条件另行核对，见 [VidOR 官方页面](https://xdshang.github.io/docs/vidor.html)。
- OpenEQA：[官方仓库](https://github.com/facebookresearch/open-eqa)、[仓库许可证](https://github.com/facebookresearch/open-eqa/blob/main/LICENSE)。底层 HM3D 与 ScanNet 数据另见 [HM3D 官方入口](https://github.com/matterport/habitat-matterport-3dresearch)、[Matterport 学术使用条款](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data)、[ScanNet 使用条款](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf)。

上述链接用于保留来源和核对要求，不等于已经确认所有 QA、帧、HSGG、图或向量均可任意公开。派生数据按数据集分别保留来源与适用条件，不能因不含图片就统一套用代码许可证。

## 5. 验收范围

- GitHub 保留代码、配置与轻量资源清单；大型图、图片、HSGG、FAISS 和 QA 答案正文不进入代码仓库。
- 资料按三个数据集独立管理，OpenEQA 内保留两个分片；时间语义和 QA 协议不能混用。
- 对齐图／特征包已经逐成员 SHA256、gzip EOF/CRC、向量行与分组业务校验。云端上传由用户确认，未将其表述为云端重下载校验。
- 新分组包的读取和路径绑定示例与原项目完整运行区分。元数据校验不检查在线模型、全部旧缓存依赖或完整 QA 正确率。
- 源数据保持只读，独立发布副本承载适配；不将原项目历史迁移、清理台账或环境目录作为常规下载前提。
- QA／答案只用于声明的提问与评分阶段，不混入图构建或身份匹配。
- 当前文件及推送的可达 Git 历史均需检查网盘访问凭据；具体清理结果记录在本次发布验证中。

本项目自有代码许可证仍由作者确定；数据和模型来源条件分别保留。首轮人工回看短名单按用户要求暂不制作。完整旧主线的索引导入与在线评测未因资料整理而自动完成。
