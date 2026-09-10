# PoC 设计：CPU-only Agentic RL 后训练模拟（rl_sim）

> 状态：v2，复审修订版（修订记录见文末 §10）。配套：`01_research_report.md`（调研依据）、`03_execution_plan.md`（执行计划）。

## 1. 目标（复审后重新锚定）

本 PoC **不以"学会 RL 算法"或"演示 reward 上升"为目标**，而是：

1. **研究 RL 后训练的控制流**：谁调谁、同步点在哪、backpressure 怎么传导（以 slime/GLM-5.2 栈为蓝本，对照 DeepSeek V4/Miles）；
2. **画像 CPU 负载**：CPU 在流程里跑什么 workload、驱动/sandbox/数据处理的资源特征，重点是 **agentic RL 的 sandbox 怎么被驱动、跑什么 workload、启动与运行资源特征**；
3. 约束：单 CPU 节点（256 核 / 705GB，共享机器）+ 模型 API，无 GPU —— GPU 计算全部替换/模拟。

**设计原则（复审共识）**：

> **闭环假可以接受，负载失真不能接受。**

- Mock 的 reward 数值是演的，但由 reward 驱动的**控制流必须真**：DAPO 零方差组过滤、达标 abort 在途请求、partial rollout 回 buffer、weight version 推进与 staleness——它们直接改变 sandbox 到达过程和负载形状。假数值 + 真调度，负载就真。
- 计时必须分列 **wall time vs CPU time**：API 生成的 sleep/网络等待不计入 CPU 画像。
- 实验产出是**扫参数曲线**，不是 reward 曲线。

## 2. Task 选择：代码生成任务族（三档 workload）

主线任务仍是代码生成（reward 真实可算、API 天然适合、agentic RL 最小完整形态），但**从"20 道 fizzbuzz"扩展为三档混合负载**——只有混合负载才能看到 exec p50/p99、长尾拖住整组 rollout 这个 RL 特有现象：

| 档 | workload | 资源特征 | 真实对照 |
|---|---|---|---|
| **A：单步代码执行** | HumanEval 风格单函数实现 + 单测 | CPU 10–100ms，内存 MB 级 | 单步 verifier |
| **B：SWE-mini** | 多文件小仓库改 bug + 跑 pytest 套件（依赖用预烘焙只读 venv） | CPU 秒–分钟级，大量文件 IO，内存 GB 级，长尾严重 | GLM-5 10K+ SWE 环境、DSec Container 档 |
| **C：terminal/search 风格** | bash 多步 + 阻塞 IO + 超时（禁网环境下用本地 mock 服务/sleep 模拟多跳等待） | CPU 不高但**占 worker 槽位很久** | 数千 terminal 环境、多跳 search |

另加**故障注入用例**（死循环→timeout、爆内存→OOM、"装依赖"超时）专门压 sandbox 的异常路径。A 档保留少量题目仅作连通性验证。

## 3. CPU 在 RL 里干什么（设计基石）

GPU 只干两件事：rollout 前向、training 反向。剩下全是 CPU（agentic 场景 CPU 占比更高）：

```
driver(编排) → router/网关 → tokenizer → rollout 并发发射
  → sandbox 环境执行(绝对大头，多轮 agent 每 step 进一次)
  → verifier/reward(judge/单测)
  → advantage 归一/打包/切 DP → 权重同步协调 → 日志/WAL/ckpt staging
```

画像重点排序：① sandbox 执行；② 编排+路由+并发控制（决定 sandbox 到达过程）；③ tokenize/reward/packing（小但高频，决定 driver 是否成为瓶颈）；④ **driver 进程自身的 CPU 开销**（序列化/对象 churn/GC——真实系统用 Rust 重写这一层，如 DSec）。

| 环节 | 真实集群位置 | PoC 中的处理 |
|---|---|---|
| 驱动循环 / 编排调度 | CPU driver | **真实实现**（asyncio），并**测量 driver 自身 CPU%** |
| Rollout 前向 | GPU (SGLang) | **替换为模型 API**；HTTP 经本地 router 进程，IO/序列化开销真实 |
| 环境交互 / sandbox | **CPU 集群** | **真实实现**，核心观测对象（§5.1） |
| Reward / verifier | CPU | **真实实现**（test pass rate） |
| Tokenize / detokenize | CPU | **真实 BPE**（vendored vocab 离线可用；验证假设：tokenize 不是瓶颈，detokenize+对象构造才是） |
| advantage / packing / 切 DP | CPU | **真实实现**（GRPO: r − group_mean 等标量运算） |
| Training 前向+反向 | GPU (Megatron) | **mock**：按 token 数模拟耗时、版本号+1；clip/TIS surrogate 数值真实算 |
| 权重同步协调 | CPU 协调 + GPU 传输 | **mock**：记 version 广播 + pause 窗口；**不声称对照 NCCL/RDMA 保真** |
| 监控 / WAL / ckpt staging | CPU | **真实实现**：分阶段 wall/CPU 双列 + sandbox 池指标 + IO 量 |

## 4. 架构设计（组件逐一对照 slime 源码）

```
rl_sim/
├── train.py                 # 同步基线驱动循环      ← slime train.py
├── train_async.py           # 异步流水线(lookahead+abort/requeue) ← slime train_async.py
├── rl_sim/
│   ├── types.py             # Sample/SampleStatus   ← slime/utils/types.py
│   ├── data_source.py       # RolloutDataSource：三档任务集 + 故障注入用例
│   ├── router.py            # 本地 HTTP router 进程  ← SGLang router / TITO gateway
│   ├── engine.py            # APIEngine + MockEngine（均经 router）← SGLangEngine
│   ├── tokenizer.py         # 真实 BPE（vendored vocab，离线）
│   ├── sandbox.py           # SandboxPool：三档资源 profile + unshare 隔离 ← DSec
│   ├── reward.py            # rule-based RM         ← slime/rollout/rm_hub
│   ├── rollout_manager.py   # RolloutManager：asyncio 有界队列+组屏障+abort/requeue
│   │                        #   ← slime/ray/rollout.py（Ray→asyncio，理由见 §5.2）
│   ├── trainer.py           # MockMegatronTrainer   ← actor_group.py + megatron_utils/actor.py
│   ├── weight_sync.py       # WeightUpdater（mock）  ← UpdateWeightFromDistributed（仅时序对照）
│   └── monitor.py           # wall/CPU 双列 + driver CPU% + IO + sandbox 画像
├── tests/                   # sandbox 隔离/故障注入用例
└── README.md                # 组件对照表 + CPU 分工讲解 + 实验报告模板
```

**主循环**（同步基线，对齐 slime train.py 五步）：

```
for rollout_id in range(num_rollout):
    1. rollout_manager.generate(rollout_id)   # prompt组 → router → 生成 → sandbox → reward → group advantage
    2. trainer.async_train(rollout_data)      # mock 训练：真实算 advantage/clip/TIS，模拟 GPU 耗时，version+1
    3. save_model（每 N 步：mock checkpoint）
    4. weight_sync.update_weights()           # pause_generation → 版本广播 → continue_generation（记 pause 窗口）
    5. rollout_manager.eval（每 N 步：held-out pass rate）
```

**异步对照**（train_async.py，对齐 slime train_async.py）：训练当前 batch 时提前发起下一轮 generate（one-step lookahead），有界 data buffer，被 abort 的半成品回 buffer 续生成，`--update-weights-interval` 控制同步频率。

参数风格对齐 slime quick start，并保持约束 `rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout`。

## 5. 关键模块设计

### 5.1 SandboxPool（核心观测对象）

- **隔离方案**：subprocess + **`unshare -n`（network namespace，真禁网）** + rlimits + 降权到 `nobody` + 独立 tmpdir（放 /mnt/nvme0）。保留 process 级 ~ms 启动优势，同时获得 Container 档的网络隔离语义；启动耗时对比（process vs unshare vs 理论 container 100ms–1s）本身作为画像数据记录。
  - 备选降级：若环境不允许 unshare（userns 被禁），退化为纯 rlimits + 无网络任务集，并在报告中标注。
- **分档资源 profile**（一刀切限额是失真源）：
  - A：`RLIMIT_AS` 512MB / `RLIMIT_CPU` 5s / wall timeout 10s；
  - B：`RLIMIT_AS` 4GB / `RLIMIT_CPU` 120s / wall timeout 300s（pytest+numpy 在 512MB 下起不来）；依赖用**预烘焙只读 venv**，"装依赖超时"用故障注入模拟（禁网后真实 pip install 必然失败）；
  - C：小 CPU 限额 + 长 wall timeout（阻塞 IO 占槽特征）。
- **协议**：runner stdout 输出 `__RESULT__{json}`：status(passed/failed/error/oom/timeout) + per-test 通过数 + `ru_utime/ru_stime`（CPU 时间）+ `ru_maxrss`（峰值 RSS）。
- **指标**：每任务记录 启动耗时 / 排队时长 / 执行 wall / CPU 时间 / 峰值 RSS / **tmpdir IO 量（/proc/pid/io）**；每 rollout 输出 池利用率、峰值队列深度、exec p50/p99（**分档**）、状态分布。

### 5.2 到达过程（画像的灵魂）

- 并发模型用 **asyncio 有界队列**，不用 concurrent.futures（无法表达积压/背压），也**不引入 Ray**：单节点上 Ray 的 raylet/GCS 引入 GB 级内存与常驻 CPU 噪声，会污染画像对象；Ray 的编排成本作为真实系统成本写入文档说明。
- 必须实现的三件事，否则 queue_depth/利用率全是噪声：
  1. **组屏障**：group-of-n 等最慢样本（sync 模式下 sandbox 空转的直接成因）；
  2. **abort/requeue**：达标后 abort 在途请求，半成品回 buffer（对应 slime partial rollout）；
  3. **高并发到达**：>1k 并发 rollout 可配，配合 C 档长尾任务制造真实积压。
- **sync 基线 vs async 对照**：sync 低水位不是要回避的缺陷，而是实验发现——两组对照展示组屏障造成的 worker 空转，以及 async（lookahead+requeue）的修复效果。

### 5.3 Router / Engine / Tokenizer

- **router.py**：本地独立 HTTP router 进程（stdlib 实现，零依赖），driver 与 engine 间所有流量经它转发——对应 SGLang router/TITO gateway，让 HTTP/JSON 序列化开销真实化，并作为独立测量点。**MockEngine 也走 router**。
- **APIEngine**：OpenAI 兼容接口（env 配置），指数退避重试，code block 提取。
- **MockEngine**：离线兜底，内置 canonical solution + 随机变异；**不追求 reward 曲线**（那是演的），其职责是驱动控制流（版本推进、过滤、abort）。
- **tokenizer.py**：真实 BPE（vendored vocab 文件随仓库分发，离线可用；节点无外网）。

### 5.4 MockMegatronTrainer（GPU 替身）

- 真实计算（本来就是 CPU 标量活）：GRPO advantage、PPO clip surrogate 数值、模拟 train/infer mismatch 的 TIS/IcePop mask 比例（σ 随 staleness 增大）；
- 模拟：按 token 数 sleep 模拟 GPU 耗时（速率可配）；`weight_version += 1`。

### 5.5 WeightUpdater（mock，不做传输保真声明）

对齐 `UpdateWeightFromDistributed` 的**时序**：`pause_generation()` → 模拟传输耗时（参数GB/带宽GBps 估算，截断实际 sleep）→ 版本更新 → `continue_generation()`；记录 pause 窗口与 rollout 侧 staleness 分布。**明确不声称对照 NCCL/RDMA。**

### 5.6 Monitor（负载可观测性）

- **双列计时**：每阶段 wall time 与 CPU time 分列（API 网络等待不进 CPU 画像）；
- **driver 自测量**：driver 进程 CPU%（/proc/self/stat、os.times）、每阶段对象数/字节数；
- **IO 画像**：per-task 读写字节、sandbox scratch tmpfs vs nvme0 对比、WAL/ckpt staging 的 fsync 成本；
- 输出示例：

```
[rollout 3] wall: orchestrate 0.4s | router 1.1s | api_gen 12.3s | sandbox 11.8s | reward 0.2s | adv+pack 0.1s | train(mock) 2.0s | weight_sync 0.1s
           cpu : orchestrate 0.3s | router 0.9s |          -      | sandbox  9.2s | reward 0.2s | adv+pack 0.1s |      -        |      -
[sandbox tierA] busy 24/32 | q_max 41 | exec p50 0.08s p99 0.9s | cpu p50 0.05s | rss p99 88MB | passed 71% failed 24% timeout 4% oom 1%
[sandbox tierB] busy  8/32 | q_max 12 | exec p50 6.2s  p99 88s  | cpu p50 4.1s | rss p99 2.1GB | io p99 340MB
[sandbox tierC] busy  5/32 | q_max  9 | exec p50 3.1s  p99 30s  | cpu p50 0.2s | rss p99 210MB
[driver] cpu 38% | json_ser 0.6s | samples 128 (A96/B16/C16) | wal_fsync 12ms
[train] weight_version 4 | tis_masked 3.2% | zero_var_groups 2 | staleness p50 0 p99 1
```

## 6. 实验设计（扫参数，不看 reward 曲线）

E0. **基线噪声测量**：空载采集节点 CPU/IO 背景负载（共享机，705G 已用 276G），实验用 `taskset` 固定核集合，报告标注环境负载——不做这步，画像数据全是别人的噪声。

E1. **sandbox 并发度扫描**：workers = 4 / 32 / 128 × 三档混合任务 → 吞吐 vs 队列等待 vs 利用率曲线（核心产出，直接回答"sandbox 负载情况"）。

E2. **任务复杂度画像**：A/B/C 分档的 CPU-time / RSS / IO 分布与 p50/p99。

E3. **API 并发度 vs driver CPU 占用**：找到 Python driver 饱和点（对应真实系统改用 Rust router/gateway 的动机）。

E4. **sync vs async 对照**：组屏障空转 vs abort/requeue 效果（ slime train.py vs train_async.py 的直观演示）。

E5（可选）. **scratch 介质对比**：tmpdir 放 tmpfs vs /mnt/nvme0 的 IO/尾延迟差异。

## 7. 验证标准（DoD，v2）

1. 离线闭环跑通：`python train.py --engine mock --num-rollout 10`（同步基线）与 `train_async.py` 各跑通，控制流事件（过滤/abort/requeue/version 推进）日志可追；
2. sandbox 隔离：死循环→timeout、爆内存→oom、语法错→error、**禁网生效（sandbox 内 socket 连接失败）**；均不逃逸、不拖垮池；
3. 双列计时、driver CPU%、IO 量、分档 sandbox 画像全部有输出，格式符合 §5.6；
4. E0–E4 实验数据落盘（CSV/JSON），E1/E4 有结论曲线；
5. 真实 API 跑通 ≥2 rollout（API 由使用方配置，可跳过并标注）；
6. README：slime/DSec/Miles 组件对照表 + CPU 分工讲解 + 实验报告。

## 8. 非目标（第一版）

- 不实现 SAO/critic/OPD（结构上预留 weight_version/staleness/DIS 钩子，扩展路径见 README）；
- 不引入 Ray 与任何第三方 Python 依赖（stdlib only；tokenizer vocab 文件 vendor 进仓库）；
- 不做真实大模型反传；不追求 reward 数值的任何意义；
- 权重同步不做 NCCL/RDMA 保真对照。

## 9. 扩展路径（第二版候选）

- group size=1 + DIS + mock critic（SAO）；OPD 模拟（第二 API 当教师）；多轮 agentic 任务（loss_mask 区分模型/环境 token）；容器级隔离对照（若环境允许，补 container/microVM 档启动成本实测）。

## 10. 修订记录

**v2（2026-09-10）**：依据 Muse Spark 复审（/tmp/opencode/review_for_kimi.md）+ 作者增量意见修订：

1. 目标重锚定：从"跑通流程/演示 reward 上升"改为"控制流真实 + CPU 负载可测量"；确立"闭环假可以接受，负载失真不能接受"原则；
2. 任务集从单一简单题扩展为 A/B/C 三档 workload + 故障注入（复审 §2a）；B 档依赖改用预烘焙只读 venv（解决禁网与装依赖矛盾）；
3. sandbox 隔离从纯 rlimits 升级为 subprocess + `unshare -n` 真禁网 + 降权 nobody + 分档资源 profile（复审 §2b + 作者增量）；
4. 并发模型从 concurrent.futures 改为 asyncio 有界队列 + 组屏障 + abort/requeue；明确不引 Ray（单节点噪声污染画像）（复审 §2c + 作者增量）；
5. 新增本地 HTTP router 进程（MockEngine 也走 router）与真实 BPE tokenizer（vendored 离线）（复审 §3.2 + 作者增量）；
6. Monitor 改 wall/CPU 双列计时，新增 driver CPU% 自测量与 IO 画像（复审 §3.1 + 作者增量）；
7. 实验从"看 reward 上升"改为 E0–E5 扫参数，新增共享机基线噪声控制（taskset 固定核）（复审 §3.4 + 作者增量）；
8. 权重同步保留 mock，明确不做 NCCL/RDMA 保真声明（复审 §3.5）；
9. 保留 sync 基线并与 async 做对照实验（组屏障空转 vs 修复效果），作为实验发现而非缺陷（作者增量）。
