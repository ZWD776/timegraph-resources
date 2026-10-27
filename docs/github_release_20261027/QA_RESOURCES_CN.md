# QA 配套资源与核验范围

三个数据集已按 2026-09-04 论文冻结 C 分别制作完整 QA 综合包，放入资源目录的 `QA_NAVQA/`、`QA_NEXTQA/`、`QA_OPENEQA/`。综合包同时包含官方问题、参考答案、模型实际回答、逐题历史评分、汇总、策略来源和资源绑定。GitHub 只记录版本、哈希、指标和使用说明，不包含题文、选项、答案或逐题 Judge 正文；访问方式由维护者单独提供。当前已完成本地准备与落盘校验，未确认云端上传。

## 当前主包与论文结果

| 数据集 | 推荐主包 | 字节数 | 论文冻结 C |
| --- | --- | ---: | --- |
| NaVQA | `qa_navqa_paper_frozen_c_v1.tar.gz` | 191,060 | 102/139，73.3812949640% |
| NExT-QA | `qa_nextqa_paper_frozen_c_v1.tar.gz` | 5,345,066 | 3,334/4,996，66.7333867094% |
| OpenEQA | `qa_openeqa_paper_frozen_c_v1.tar.gz` | 957,875 | 53.4993887531，0–100 归一化 Judge 分数 |

三包共 6,494,001 字节。精确 SHA256 和原字节来源指纹见 `catalog/qa_resources.json`、`catalog/source_versions.json` 及包内 manifest。每个 QA 目录的 `CURRENT_PACKAGE.json` 指向推荐主包。

旧 `qa_navqa_v2.tar.gz`、`qa_nextqa_v1.tar.gz`、`qa_openeqa_v1.tar.gz` 只整理了基础问题、参考答案、时间窗口和映射。本次综合包已包含旧包资料，再加入论文冻结 C；不需要两套重复解压。旧包保留备份，旧外层说明保存在 `reference_only_metadata/`。

`qa_navqa_v2` 的 v2 是交付修订号：第二版补充了派生时间标签未核实标记及负值审查清单，没有改动官方题目、原始参考答案或历史分数。它不是 NaVQA 官方版本号。`paper_frozen_c_v1` 是本次综合交付的独立版本名。

## 解包与读取

先创建自行选择的空目录 `<UNPACK_ROOT>`，再解压主包。包内根分别为 `navqa/qa/`、`nextqa/qa/`、`openeqa/qa/`，并非旧项目的 `datasets/...` 运行目录。例如（`-C` 后的目录必须已经存在）：

```text
tar -xzf qa_nextqa_paper_frozen_c_v1.tar.gz -C /chosen/unpack-root
```

```text
<dataset>/qa/
  official/                       原始官方标注，原字节保留
  runtime/ 或 evaluation/         提问视图与评分侧标签
  history/                        NaVQA 的七序列窗口
  metadata/                       QA ID、源版本和单元绑定
  paper_frozen_c/
    predictions.jsonl             模型已有回答，每行一题
    historical_scores.jsonl       历史逐题判定／Judge 分
    question_bindings.jsonl       每题对应 HSGG、图片、图、特征分包
    summary.json                  论文 C 汇总、分组统计和实验身份
    original/                     冻结来源原字节文件
    provenance/                   增量来源清单和核验记录
    protocol/                     OpenEQA 历史 Judge 提示词及来源
    selected_judge_details.jsonl   OpenEQA 所选历史 Judge 理由与原响应
  manifest.json / SHA256SUMS.txt / verification.json
```

`history/` 仅 NaVQA 提供，`paper_frozen_c/protocol/` 和 `selected_judge_details.jsonl` 为 OpenEQA 附件；各包的 QA 根下还可保留基础 `protocol/` 快照，不要混为同一目录。NExT-QA 的既有运行问题位于 `evaluation/questions.runtime.json`。各 JSONL 以 `question_id` 对齐；NaVQA 另有序列 ID，NExT-QA 为 `video_id`，OpenEQA 为 `episode_history`。逐题绑定保存四类配对包名及 `group_id`，图和媒体正文不在 QA 包重复复制。

无需模型即可逐行读取，例如：

```python
import json
from pathlib import Path

path = Path('/chosen/unpack-root/nextqa/qa/paper_frozen_c/predictions.jsonl')
with path.open(encoding='utf-8') as handle:
    record = json.loads(next(handle))
print(record['question_id'], record['predicted_choice_index'])
```

完整原预测与便携视图有意区分：NExT-QA JSONL 保留常用答案、推理和模型字段，完整原预测另外按原字节保存。NaVQA 冻结记录没有独立的 model/provider 字段，不依据当前默认配置猜测。增量生成清单和核验报告位于 `paper_frozen_c/provenance/`；内层旧附件说明所称“基础 QA 包”已包含在综合包中。

## 实验身份与评分

论文最终 C 按固定的持久化路线选择冻结策略，不按每题正误挑选答案。本次整理只重新核算已有分数，没有重新回答、调用 Judge 或修正历史判定。

- NaVQA：139 题的历史 official-style 判定，102 正确；binary 精确 yes/no，time／duration 按历史 2 分钟阈值，text 使用已保存的文本 Judge 判定。原始参考题集仍有 210 题，71 道 position 没有本包论文 C 回答。
- NExT-QA：五选一 exact-choice accuracy，3,334 正确；不能换成旧 3,322 版本。冻结评分标签有历史运行 SHA256 直接对应。
- OpenEQA：1–5 原分转换为 `(score - 1) * 25`，按全部 1,636 题平均。最终 C 含 B 来源的 274 题和 routed C 来源的 1,362 题；保存了对应的历史 `judge-v2` prompt、来源 SHA 和所选逐题 Judge 记录。不能称为选择题准确率，也不与旧 `mmbench` 评分混用。

原注册表／旧运行入口保留了不同快照：NaVQA C=101/139、OpenEQA C=53.2549。论文冻结 C 分别为 102/139、53.4994；这里只更新资料交付说明，不修改旧运行入口、历史文件或实验结果。

## 原始 QA 与答案版本

| 数据集 | 范围 | 原始与运行资料 | 评分侧注意事项 |
| --- | --- | --- | --- |
| NaVQA | 官方 210 题；主线 139 道 non-position，7 序列 | 原官方 CSV、七份 qa_unfilled、便携问题视图、评分侧标签、139 题筛选、窗口与单元绑定 | 保留官方原字节；位置题不伪造坐标。30 条 time 数值来自旧 adapter，全部标为未核实，其中 1 条为负、必须人工确认；9 条 duration 单位为分钟 |
| NExT-QA | validation 4,996 题，570 视频 | 原 val.csv、已有 runtime questions、evaluator-only labels、视频及四类资源配对 | 五选一准确率；逐 ID、视频、类型、正确答案一致。9 个题目行仅有空白差异：问题文本 5 行、选项 4 行，二者无交集 |
| OpenEQA | 1,636 题，152 episodes；HM3D 557／ScanNet 1,079 | 原 open-eqa-v0.json、独立提问和评分侧视图、episode 绑定、项目代码快照及论文 C 实际 Judge 证据 | 开放式回答及 LLM Judge；263 题有 extra_answers。旧基础包的当前默认配置与新增的历史 prompt／summary 分开保存；不冒称官方逐字提示词 |

各包内官方文件与原始运行资料按字节保留，manifest 记录原项目相对来源、大小和 SHA256。派生便携视图单独注明，不冒充新的人工标注。包内问题视图和评分侧标签分开；参考答案不进入图构建、视觉特征或身份关联。

## NaVQA 的时间限制

`qa_unfilled.start_time/end_time` 是题目窗口元数据；官方答案的事件时刻不是窗口截止。旧适配器将部分字符串事件时刻转换为“距问题结束多少分钟”，涉及旧时区候选选择，不能因可计算就认定是已核实的数值真值。

本次保留原始标注和旧算法输出，不截断负值、不猜测正确时区。全部 30 条 time 派生记录标有 `legacy_adapter_derived_unverified` 和 `may_use_as_verified_ground_truth=false`；异常记录另有 `review_required`、人工确认标记和清单。正式重新评分前必须确认适用的时间解释；这里保存的论文历史分数不因本次审查被重写。既有 gold 仅覆盖 seq3／seq6／seq16，放在 partial provenance 中；不能将这三序列资料称为七序列的完整人工 gold。

71 道 position 题不在 139 道主线之内，派生坐标保持空值。旧检索加载完整序列图后应用检索窗口，并非所有路线统一执行严格历史前缀截断。

## 题目历史与新分包的绑定

NaVQA 保留逐题窗口，NExT-QA 使用选中整段视频，OpenEQA 使用对应 episode 的提供历史；后两者没有逐题不同 cutoff 字段，不新增虚构截止点。QA 包内单元绑定使用已有 group_id 和视频／episode／序列 ID 对应图片、HSGG、图、视觉特征分包，不重新按 ID 字典序分组。

原版 SHA256 可同时对照 `catalog/source_versions.json`。三包经过成员字节、manifest、SHA256 账、完整 gzip EOF/CRC 和独立逐题对齐检查，源文件保持不变；模型回答、评分和绑定覆盖分别为 139／4,996／1,636 条。该核验不代表已重新运行在线 Judge、模型或恢复全部旧 QA 主线，也不代替云端下载字节校验。

原数据集及标注来源见 [发布范围与来源说明](RELEASE_SCOPE_CN.md)。第三方使用条件仍分别适用；本仓库保持私有协作资源，不将资料准备解释为统一的公开再分发授权。
