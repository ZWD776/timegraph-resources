# TimeGraph 代码与资源清单

本私有仓库整理已有 TimeGraph 项目的代码、配置和资源对应关系，供实验复现及后续研究参考。
三个数据集分别管理：NaVQA、NExT-QA、OpenEQA；OpenEQA 内保留 HM3D 与 ScanNet 分片。
这里的旧轨迹与图是算法产物，不是人工身份真值，也不代表新图记忆方法已经实现。

## 数据获取

HSGG 与对应图片通过用户提供的百度网盘分享：

- [benchmark_data](WITHHELD)，提取码 `WITHHELD`。
- 图记忆网盘链接待补充；对象特征和索引的获取位置也待补充。
- 完整资源登记见 `catalog/external_resources.template.json`。网盘内容未在本次发布中下载或重新核验。

本仓库不含图片、视频、HSGG 正文、图正文、embedding、FAISS、数据库或模型权重。
官方 QA 正文及参考答案也不随包复制；`catalog/source_versions.json` 提供精确版本与 SHA256，获取和使用遵循各上游条款。

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

`--data-root` 是包含 `datasets/` 的目录。网盘压缩包的直接解压布局不一定与旧项目规范化目录相同，应按清单对照准备，不能只更换盘符。

| 数据集 | 图片映射 | 当前完成 HSGG | 官方 QA ID | 单元 |
| --- | ---: | ---: | ---: | ---: |
| NaVQA | 9,889 | 9,759 | 210（主线 non-position 为 139） | 7 |
| NExT-QA | 23,381 | 23,044 | 4,996 | 570 |
| OpenEQA | 31,742 | 31,034 | 1,636 | 152 |

`image_available`、`hsgg_available` 表示导出时的原始资源状态，不表示新克隆已下载这些文件。
所有文件引用相对于明确选定的数据根目录；清单内不含本机绝对路径。

## 时间、身份和版本说明

- NaVQA 保留采样序号、图帧号、原帧号、Unix 时间和经过秒数；不能将文件名前缀当秒数。
- NextQA 保留最终选中的 570 个视频版本；秒数来自帧号与 FPS，不是容器 PTS。四表需按 `hsgg_record_key` 一起读取。
- OpenEQA 的 `ordinal_time` 是旧图排序值，不是真实秒数；缺失旧图映射的 745 帧保持空值。
- NaVQA QA 窗口使用 `start_time/end_time`，与答案事件时间分开；旧检索不保证统一硬截止。NextQA/OpenEQA 使用整段提供的视频或 episode 历史，没有逐题截止字段。
- 当前 HSGG 与冻结旧图的覆盖不同：NaVQA 9,759 对 9,757；OpenEQA 31,034 对 30,997。NextQA 图的 23,381 帧计数包含 skipped 时间槽。

详见 [资源准备说明](docs/github_release_20261027/RESOURCE_PREPARATION_CN.md) 和 [发布范围](docs/github_release_20261027/RELEASE_SCOPE_CN.md)。

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
- `publication_inventory.json` 记录发布文件校验值和相对来源；适配仅发生在独立发布副本。
- `release_validation.json` 记录本次离线验收及原项目保护校验；不代表网盘资源或在线模型实验已验证。
- 旧删除、迁移和封存工具只用于相应历史工作流，不是常规数据准备入口；本仓库没有携带其依赖的历史 archive 台账。

本仓库为私有协作资源。自有代码许可证由作者确定；第三方来源、许可和引用应保留，不能将所有数据统一视为同一代码许可证。
