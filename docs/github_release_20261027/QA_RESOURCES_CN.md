# QA 配套资源与核验范围

三个数据集分别准备独立 QA 压缩包，供维护者放入与图、HSGG 和图片相同的受控资源目录。GitHub 只记录版本、哈希、映射与使用说明，不包含题目正文、选项或参考答案；访问方式由维护者单独提供。`catalog/qa_resources.json` 给出最终包名、大小、SHA256 和准备状态。当前仅完成本地准备，未确认云端上传。

| 数据集 | 范围 | 原始与运行资料 | 评分侧注意事项 |
| --- | --- | --- | --- |
| NaVQA | 官方 210 题；主线 139 道 non-position，7 序列 | 原官方 CSV、七份 qa_unfilled、便携问题视图、评分侧标签、139 题筛选、窗口与单元绑定 | 保留官方原字节；位置题不伪造坐标。30 条 time 数值来自旧 adapter，全部标为未核实，其中 1 条为负、必须人工确认；9 条 duration 单位为分钟 |
| NExT-QA | validation 4,996 题，570 视频 | 原 val.csv、已有 runtime questions、evaluator-only labels、视频及四类资源配对 | 五选一准确率；逐 ID、视频、类型、正确答案一致。9 个题目行仅有空白差异：问题文本 5 行、选项 4 行，二者无交集 |
| OpenEQA | 1,636 题，152 episodes；HM3D 557／ScanNet 1,079 | 原 open-eqa-v0.json、独立提问和评分侧视图、episode 绑定、项目 Judge prompt／代码快照 | 开放式回答及 LLM Judge；263 题有 extra_answers。项目 prompt 与当前默认配置不代表官方逐字提示词或论文冻结运行设置 |

各包内官方文件与原始运行资料按字节保留，manifest 记录原项目相对来源、大小和 SHA256。派生便携视图单独注明，不冒充新的人工标注。包内问题视图和评分侧标签分开；参考答案不进入图构建、视觉特征或身份关联。

## NaVQA 的时间限制

`qa_unfilled.start_time/end_time` 是题目窗口元数据；官方答案的事件时刻不是窗口截止。旧适配器将部分字符串事件时刻转换为“距问题结束多少分钟”，涉及旧时区候选选择，不能因可计算就认定是已核实的数值真值。

本次保留原始标注和旧算法输出，不截断负值、不猜测正确时区。全部 30 条 time 派生记录标有 `legacy_adapter_derived_unverified` 和 `may_use_as_verified_ground_truth=false`；异常记录另有 `review_required`、人工确认标记和清单。正式评分前必须确认适用的时间解释。既有 gold 仅覆盖 seq3／seq6／seq16，放在 partial provenance 中；不能将这三序列资料称为七序列的完整人工 gold。

71 道 position 题不在 139 道主线之内，派生坐标保持空值。旧检索加载完整序列图后应用检索窗口，并非所有路线统一执行严格历史前缀截断。

## 题目历史与新分包的绑定

NaVQA 保留逐题窗口，NExT-QA 使用选中整段视频，OpenEQA 使用对应 episode 的提供历史；后两者没有逐题不同 cutoff 字段，不新增虚构截止点。QA 包内单元绑定使用已有 group_id 和视频／episode／序列 ID 对应图片、HSGG、图、视觉特征分包，不重新按 ID 字典序分组。

原版 SHA256 可同时对照 `catalog/source_versions.json`。三包经过成员字节、manifest、SHA256 账及完整 gzip EOF/CRC 检查，源文件保持不变；这不代表已运行在线 Judge、模型或恢复全部旧 QA 主线。

原数据集及标注来源见 [发布范围与来源说明](RELEASE_SCOPE_CN.md)。第三方使用条件仍分别适用；本仓库保持私有协作资源，不将资料准备解释为统一的公开再分发授权。
