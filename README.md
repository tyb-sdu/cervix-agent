# CervixAgent

CervixAgent 是一个终端优先的科研工作流执行智能体。本仓库从零开始，不依赖
MarineAgent、MolGlueDB 或其他未提供的既有代码。

当前版本的目标是建立可靠的工程底座：

- 将原方案的三阶段流程固化为只读协议；
- 提供终端命令、项目状态、环境检查和人工审批基础；
- 不修改研究创新点、靶点假设或体内外实验顺序；
- 不使用未授权的 Schrödinger 软件；
- 后续通过合法开源适配器接入共价对接和 GROMACS。

## 当前可用命令

```powershell
python -m cervixagent --version
python -m cervixagent workflow
python -m cervixagent doctor
python -m cervixagent init C:\path\to\research-project --name cervical-hpv
python -m cervixagent status C:\path\to\research-project
python -m cervixagent data sources
python -m cervixagent data fetch structures --project C:\path\to\research-project
python -m cervixagent data fetch natural-products --project C:\path\to\research-project
python -m cervixagent data build-test --project C:\path\to\research-project --size 500
python -m cervixagent audit baseline --project C:\path\to\research-project
python -m cervixagent audit verify --project C:\path\to\research-project
python -m cervixagent ingest test --project C:\path\to\research-project
python -m cervixagent ingest verify --project C:\path\to\research-project
python -m cervixagent ingest contract
python -m cervixagent ingest stage-public --project C:\path\to\research-project
python -m cervixagent ingest stage-verify --project C:\path\to\research-project
```

安装为系统命令后可使用：

```powershell
cervixagent doctor
cervixagent workflow
```

## 本地开发

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[chem]"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

化学环境的实际解析版本记录在 `requirements-chem.lock`。RDKit 使用 BSD-3-Clause
许可证；本项目不安装或调用 Schrödinger 软件。

## 工作流边界

外层工作流由 `workflow.lock.json` 控制，保持三阶段顺序：

1. CervixAgent 构建与亲电天然产物虚拟筛选；
2. 体外双靶点共价机制验证；
3. TC-1 体内联合治疗验证与 CC-IRS 构建。

LLM 未来只能在当前步骤调用白名单工具，不得自行增加、删除、重排研究步骤。
物理实验、候选物采购和正式科学结论必须通过人工审批。

`data build-test` 产生的是用于解析器、去重和后续开源工具适配的工程联调数据；
它不是原方案的正式天然产物库，未做化学结构标准化，也不能用于形成科研结论。

`audit baseline` 会封存环境、检查结果、许可策略和数据源状态，并用逐文件 SHA-256
检测后续修改。它是本地“可检测篡改”记录，不等同于 WORM 存储或带私钥的数字签名。

`ingest test` 只执行 RDKit SMILES 解析、清理和规范异构 SMILES 序列化。它保留原始
记录、盐/多片段、互变异构体、质子化状态和立体信息；发现规范结构重复时只标记，
不删除。该命令不计算或应用 Michael 受体、Lipinski、PAINS 等 P1-04 规则。

`ingest stage-public` 对下载清单中的 COCONUT 与 LOTUS 文件重新核对 SHA-256 后，
使用流式 RDKit 解析和 SQLite 事务暂存全部记录。运行期间使用项目锁防止两个入库任务
同时写入；成功结果必须通过 SQLite `quick_check` 和逐文件封存校验。输出仍明确标记为
`two_source_public_snapshot_staging`，不能据此完成 P1-02。

ECNPDB 当前记录为 `unresolved_not_substituted`，在确认其完整名称、官方来源、许可和
数据定义前，不用其他数据库替代，也不把 P1-02 标记为完成。

## 许可策略

方案提及的 Glide CovDock 需要合法 Schrödinger 授权。本项目当前不包含、不调用、
也不模拟该商业软件。后续开源实现将作为独立工具适配器接入，并在每次运行中如实
记录实际使用的软件、版本和参数。
