# 开发规范与任务分解（docs/05_dev_plan.md）

> v1，2026-09-15。配套：`02_poc_design.md`(v3.1)、`03_execution_plan.md`(v3.1)。
> 开发方式：**任务制**——一次只干一个 task，验收标准 = 该 task 的单元测试全绿 + 全量测试不回归；分阶段集成。

## 1. 开发机器与代码同步

| 角色 | 机器 | 干什么 |
|---|---|---|
| **开发机** | 本机（repo: `/home/tina/test/LLM_RL_cpu_poc`，有外网，24 核） | 写代码、跑**全部纯 Python 单元测试** |
| **集成/实验机** | 节点 `10.239.23.91`（`/workspace/LLM_RL_cpu_poc` → `/mnt/nvme0`，256 核，无外网） | 跑标注 **[node]** 的集成任务与实验（llama.cpp 引擎、unshare、大规模并发） |

同步：本机 `git push` → 节点 `git pull`；节点无外网时 `git bundle create repo.bundle main` + scp + 节点 `git clone repo.bundle`。

## 2. 代码存放路径与模块划分

```
LLM_RL_cpu_poc/
├── docs/                        # 01-05 文档
├── rl_sim/                      # 核心包，11 个模块
│   ├── __init__.py
│   ├── types.py                 # 数据结构          ← slime/utils/types.py
│   ├── monitor.py               # 计时/指标          ← 日志监控体系
│   ├── reward.py                # rule-based RM     ← slime/rollout/rm_hub
│   ├── sandbox.py               # SandboxPool       ← DSec / 环境交互
│   ├── data_source.py           # 任务数据源         ← slime/rollout/data_source.py
│   ├── engine.py                # 引擎协议+三实现    ← sglang_engine.py
│   ├── router.py                # HTTP router       ← SGLang router / TITO
│   ├── tokenizer.py             # BPE 封装           ← TITO tokenize 环节
│   ├── rollout_manager.py       # asyncio 流水线     ← slime/ray/rollout.py
│   ├── trainer.py               # mock 训练器        ← actor_group.py
│   └── weight_sync.py           # 权重同步           ← update_weight/
├── tests/                       # 每模块一个 test_*.py，stdlib unittest
├── train.py                     # 同步闭环入口       ← slime train.py
├── train_async.py               # 异步入口           ← slime train_async.py
├── run_experiments.py           # E0-E6 实验入口
├── vendor/                      # 模型/venv/语料（.gitignore，节点侧）
├── results/                     # 实验输出（.gitignore）
└── README.md
```

**模块数：11 个库模块 + 3 个入口脚本 + 1 个测试包。任务数：21 个 task，4 个集成门（gate）。**

## 3. 开发规范

1. **运行时零第三方依赖**（Python stdlib only）；测试用 **stdlib unittest**（节点无外网装不了 pytest；sandbox B 档的 pytest 在预烘焙 venv 里，属被测对象非测试框架）。
2. **一次一个 task**；完成标准 = 该 task 对应的 `tests/test_*.py` 全绿，且 `python -m unittest discover -s tests` 全量不回归。
3. 每个 task 一个 commit，信息格式 `T07: data_source schema + task fixtures`；完成一个 push 一个。
4. 标注 **[node]** 的 task：本机开发 → push → 节点 pull 后执行验收。
5. 模块 docstring 首行标注 slime 对照组件（如 `"""SandboxPool ← DeepSeek DSec / slime agentic env"""`）。
6. `vendor/`、`results/`、密钥/token 不入库（.gitignore）。
7. 代码风格：简洁、类型标注、无魔法数；配置集中 `rl_sim/config.py`（如需）。
8. 集成门未过不得开始下一阶段。

## 4. Task 列表（21 个）

### Phase A：基础组件（本机）

| # | Task | 内容 | 验收（单测） |
|---|---|---|---|
| T01 | 骨架 + types.py | 包结构、Sample/SampleStatus、.gitignore | `test_types.py`：序列化/默认值/状态枚举 |
| T02 | monitor.py | StageTimer（wall/CPU 双列）、百分位统计、report 格式化 | `test_monitor.py`：计时正确性、p50/p99 计算、输出格式 |
| T03 | reward.py | pass rate、数值匹配、精确答案三类 verifier | `test_reward.py`：对构造答案评分正确 |
| T04 | sandbox.py 核心 | 单任务执行：tmpdir、rlimits、runner 协议（`__RESULT__`JSON）、状态映射 | `test_sandbox.py`：passed/failed/语法错/timeout/oom 五用例 |
| T05 | sandbox.py 池化 | 固定 worker 池、burst/poisson 到达、三时间戳、队列指标 | `test_sandbox_pool.py`：并发上限、queue_depth、launch_overhead 可测 |
| T06 | sandbox 隔离 [node] | `unshare -n` 禁网 + 降权 nobody + 分档资源 profile | `test_sandbox_isolation.py`：socket 连接失败、文件属主 nobody、限额生效 |
| T07 | data_source.py A/L 任务 | 任务 schema、A 档代码题、L1 SWE-mini 仓库 fixture、L2/L3 任务、组采样 | `test_data_source.py`：取组逻辑、n_samples 复制、epoch 循环 |
| T08 | 多模态任务 | V1/V2 题目 + 图表 PNG 生成脚本（vendor 输出）、答案可验证 | `test_mm_tasks.py`：图片存在、答案校验器正确 |
| T09 | engine 协议 + MockEngine | Engine 抽象（generate/update_weights/pause/resume）、MockEngine 模板生成 | `test_engine_mock.py`：接口契约、logprob 字段、版本推进 |
| T10 | router.py | HTTP 转发、session-affinity 一致性哈希、后端注册/心跳 | `test_router.py`：路由亲和性、转发正确、后端掉线剔除 |

**集成门 G1**：`python -m unittest discover` 全绿，组件套件完成。

### Phase B：引擎与流水线

| # | Task | 内容 | 验收 |
|---|---|---|---|
| T11 | LocalEngine [node] | llama-server 多实例管理（拉起/绑核/slots/心跳/重启） | `test_engine_local.py`：实例拉起、logprob 返回、tool calling 一轮 |
| T12 | tokenizer.py | 优先走引擎 `/tokenize` 端点，本地 BPE 兜底；计时接入 monitor | `test_tokenizer.py`：与引擎结果一致、耗时被记录 |
| T13 | rollout_manager.py | asyncio 有界队列、组屏障、abort/requeue、advantage 归一、packing | `test_rollout_manager.py`（MockEngine）：组屏障行为、abort 回 buffer、advantage 数值对拍 |
| T14 | trainer.py | GRPO surrogate、clip、TIS（真实 logprob 输入）、mock 耗时、version+1 | `test_trainer.py`：loss/clip/TIS 数值对拍手算用例 |
| T15 | weight_sync.py | pause→reload→resume 时序、reload 计时、staleness 记录 | `test_weight_sync.py`：时序断言、版本一致 |

**集成门 G2**：Phase B 单测全绿 + T13–T15 联调（MockEngine 全链路数据流）通过。

### Phase C：闭环集成

| # | Task | 内容 | 验收 |
|---|---|---|---|
| T16 | train.py [node 联调] | sync 五步主循环、slime 风格参数、ckpt 落盘 | `test_train_smoke.py`：mock 引擎 3 轮闭环、控制流事件日志可追 |
| T17 | train_async.py [node 联调] | one-step lookahead、update-weights-interval、staleness 统计 | `test_train_async.py`：异步事件顺序、staleness 分布输出 |
| T18 | monitor 报告集成 | §5.6 格式全字段输出（engine/sandbox/driver/train 四段） | `test_report.py`：字段齐全、数值自洽 |

**集成门 G3**：`train.py --engine mock --num-rollout 3` 与 `--engine local --num-rollout 2`（节点）双双跑通，报告格式符合设计 §5.6。

### Phase D：实验与归档

| # | Task | 内容 | 验收 |
|---|---|---|---|
| T19 | run_experiments.py [node] | E0 基线、E1 并发扫描、E3 driver 饱和、E4 sync/async、E6 引擎画像；数据落盘 CSV/JSON | `test_experiments.py`：参数扫描编排正确 + 节点实跑出数 |
| T20 | VLM 端到端 [node] | Qwen2.5-VL-3B 实例接入、V1/V2 任务线闭环 | `test_vlm_e2e.py`：图像输入→回答→reward 全链路 |
| T21 | README + 实验报告 | 组件对照表、CPU 分工讲解、实验结论、归档 push | 审查通过 |

**集成门 G4（DoD 终审）**：设计 §7 全项。

## 5. 依赖映射（design → tasks）

- §5.1 sandbox → T04/T05/T06；§5.2 到达过程 → T05/T13/T17；§5.3 引擎 → T09/T11/T20；
- §5.4 trainer → T14；§5.5 weight sync → T15；§5.6 monitor → T02/T18；
- §2 任务线 → T07（A/L1/L2/L3）、T08（V1/V2）、T09（mock 备选）；
- §6 实验 → T19；M0 环境 → T00（本计划外，已部分完成）。

## 6. 当前进度

- [x] T00/M0：节点免密、磁盘选型、llama.cpp 节点编译、GGUF 模型分发（Qwen3-4B Q4/Q8 + Qwen2.5-VL-3B + mmproj）、llama-server 冒烟（logprob 真实返回）
- [x] T01–T10（Phase A，G1 过）：types / monitor / reward / sandbox（核心+池化+隔离）/ data_source / 多模态 / engine / router
- [x] T11–T15（Phase B）：engine_local（节点验收过）/ tokenizer / rollout_manager / trainer / weight_sync
- [x] T16–T18（Phase C，G3 过）：train.py / train_async.py / 全字段报告；节点 local 闭环 mean_reward=1.000、eval 0.833
- [x] T19：run_experiments.py（E0/E1/E4 实现，E3/E6 节点侧）
- [x] T21：README 组件对照表
- [x] T20：VLM 端到端（节点验收）
- 全量单测 105+ 绿；过程中排掉三个真实 bug（Qwen3 thinking 耗尽 token / CentOS nobody gid / pkill -f 自杀）
