# GitHub 发布范围与验收说明

日期：2026-10-27。本文记录发布准备范围，不代表已经打包、上传或取得第三方数据的再分发许可。

## 1. 主仓库范围

GitHub 主仓库以复现代码和资源说明为主，采用明确的文件白名单导出：

- 本项目代码、运行与构建脚本、必要测试、配置模板、环境与依赖说明。
- 方法说明、数据适配说明、QA 与评测入口、选定主线的运行命令及结果摘要。
- 经检查并脱敏的 registry、资源清单、相对路径元数据、版本与 SHA256 校验信息。
- 数据与可选附件的获取说明；本机绝对路径、用户名、密钥、令牌和账户配置不得进入发布版本。

NAVQA、NExT-QA、OpenEQA 保持各自目录、字段和时间语义。相对资源路径应由显式配置的数据根目录解析，不依赖原电脑的盘符或工作目录。配置模板只给出占位路径；本地配置不直接公开。

source_manifest 不能因文件小就原样进入仓库：需检查其中的本机路径、第三方内容、原始 QA 或答案及来源信息。原始 source 和 QA 在本地完整保留；许可和发布范围核实前，不将整个 source 目录或 QA 原表作为默认公开包。可先发布经过审查的资源描述、适配代码和上游获取链接。

## 2. 网盘资源与待填链接

以下大型资源不推入 Git。用户已确定图、HSGG 和图片采用网盘方式提供；具体公开范围仍需结合各来源许可确认。

| 资源 | 获取入口 | 发布时需附带的信息 |
| --- | --- | --- |
| 图记忆与旧方法图产物（canonical / temporal 等） | 待填：网盘链接、提取方式 | 数据集、主线或历史版本、构建配置、输入版本、文件清单与校验值 |
| HSGG 输出（原始 sidecars、时间表、状态清单等） | 待填：网盘链接、提取方式 | 上游来源、实际帧覆盖、缺失或失败状态、许可与引用 |
| 图片 / 视频帧 | 待填：网盘链接或上游下载入口 | 原数据集来源、使用条件、帧映射与时间语义 |
| QA / 答案 / 评测标注 | 待核：许可与允许分发的文件范围 | 原始来源、版本、划分、协议；与图构建输入分开 |

链接未补齐时，应明确标为“待提供”，不能写成资源已可下载。第三方原始数据若不适合重新托管，提供官方入口与准备脚本。

## 3. 特征与索引作为可选附件

对象 embedding SQLite、FAISS 索引、检索 metadata SQLite、必要 NPZ 等可作为免重算的可选附件；不推入 Git，本次准备不复制这些文件。是否制作附件、存放位置和下载链接另行确定。

- FAISS 必须与正确的行 ID 映射、观测 / 帧绑定和 metadata 配套，不能只发布索引本体。
- 记录模型名称与 revision、特征维度、dtype、裁剪与 bbox 规则、归一化方式、源版本及校验值。
- 旧图、IR、检索结果、预测和评测日志属于基线产物，应标清主线 / 历史用途，不作为新图记忆方法的原始输入。
- 可再生的 anchor、tracklet、projection 等缓存单列；不能把“文件存在”当成“当前版本有效”。
- OpenEQA 当前仍有 runtime 元数据引用 original 中 FAISS 的情况；NPZ 本体也有仅存于 original 的情况。附件清单应按实际引用闭合，不能直接忽略整个 original 目录。
- NAV embedding SQLite 约 3.216 GiB。若附件平台要求每文件小于 2 GiB，可无损分卷并提供重组说明、分卷及原文件 SHA256，或使用适合大文件的网盘。不要以降精度、量化或裁表代替完整发布。

不随仓库发布虚拟环境、pip 缓存、无关模型缓存。模型通过名称、revision、获取链接和适用许可证说明恢复；模型权重的分发条件单独核对。

## 4. 来源与许可准备

本项目自有代码的 LICENSE 尚待作者确定，本次不擅自选择 MIT、Apache 或其他许可证。引用或随包分发的第三方代码保留其 LICENSE、NOTICE、来源和必要修改说明；第三方代码许可不自动覆盖本项目全部内容或原始数据。

目前可作为核对依据的来源：

- HSGG 本地源码包含 Apache-2.0 许可证：参见项目中的 `HSGG/LICENSE`、`HSGG-main/LICENSE`。其依赖与模型权重仍按各自条款核对。
- NAVQA / ReMEmbR：[上游评测说明](https://github.com/NVIDIA-AI-IOT/remembr/blob/main/eval.md)、[上游许可证](https://github.com/NVIDIA-AI-IOT/remembr/blob/main/LICENSE.md)；其 CODa 来源另见 [CODa 数据说明与条款](https://github.com/ut-amrl/coda-devkit/blob/main/docs/DATA_REPORT.md)。
- NExT-QA：[官方仓库及来源说明](https://github.com/doc-doc/NExT-QA)、[仓库许可证](https://github.com/doc-doc/NExT-QA/blob/main/LICENSE)。视频源的 VidOR / YFCC 使用条件另行核对，见 [VidOR 官方页面](https://xdshang.github.io/docs/vidor.html)。
- OpenEQA：[官方仓库](https://github.com/facebookresearch/open-eqa)、[仓库许可证](https://github.com/facebookresearch/open-eqa/blob/main/LICENSE)。底层 HM3D 与 ScanNet 数据另见 [HM3D 官方入口](https://github.com/matterport/habitat-matterport-3dresearch)、[Matterport 学术使用条款](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data)、[ScanNet 使用条款](https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf)。

上述链接用于保留来源和核对要求，不等于已经确认所有 QA、帧、HSGG、图或向量均可任意公开。派生数据按数据集分别保留来源与适用条件，不能因不含图片就统一套用代码许可证。

## 5. 导出与发布验收

- [ ] 明确自有代码许可证；第三方 LICENSE / NOTICE、数据来源、引用和限制保留完整。
- [ ] 确定主仓库白名单；数据、特征与历史产物按独立资源包列出，不默认整目录上传。
- [ ] 检查密钥、令牌、账户配置、本机绝对路径及可能包含敏感输出的日志。
- [ ] 配置模板支持独立的数据根目录；三个数据集各自入口、时间语义与 QA 协议可查。
- [ ] 资源包声明主线 / 历史版本、完整依赖和缺失状态；下载清单包含大小、文件数与校验值。
- [ ] QA / 答案仅用于其声明的评测阶段；不将其混入图构建或身份匹配输入。
- [ ] 用干净目录验证安装、导入、配置解析与一个最小运行示例；缺少网盘资源时给出清楚提示。
- [ ] 补齐 README 中的网盘 / 官方下载链接，检查访问方式和解包后的相对目录结构。
- [ ] 检查导出仓库的当前文件及历史均不包含排除的大文件、凭据或未授权数据。

不要直接将现有旧 Git 历史推送到新 GitHub 仓库：历史中包含约 90 MB 的归档文件，仅在当前工作区忽略或移走不能移除历史对象。新仓库建议从核验后的白名单进行干净导出，再建立新的 Git 历史；原项目和原历史保留在本地。本文档不执行导出、建仓、复制、上传或删除。
