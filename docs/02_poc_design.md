# PoC 设计：CPU-only Agentic RL 后训练模拟（rl_sim）

> 状态：v3（修订记录见文末 §10）。配套：`01_research_report.md`（调研依据）、`03_execution_plan.md`（执行计划）。
> v3 变更：rollout 引擎从"远程模型 API"改为 **本地 CPU 小模型推理（llama.cpp + Qwen3-4B）为首选**，远程 API 降为备选——logprob 由此从模拟值变为真实值。

## 1. 目标

本 PoC **不以"学会 RL 算法"或"演示 reward 上升"为目标**，而是：

1. **研究 RL 后训练的控制流**：谁调谁、同步点在哪、backpressure 怎么传导（以 slime/GLM-5.2 栈为蓝本，对照 DeepSeek V4/Miles）；
2. **画像 CPU 负载**：CPU 在流程里跑什么 workload、驱动/sandbox/数据处理的资源特征，重点是 **agentic RL 的 sandbox 怎么被驱动、跑什么 workload、启动与运行资源特征**；
3. 约束：单 CPU 节点（256 核 Xeon 6767P / 705GB，共享机器）+ 无 GPU —— **rollout 前向用本地 CPU 小模型真实推理**，training 反向 mock。

**设计原则（复审共识）**：

> **闭环假可以接受，负载失真不能接受。**

- Mock 的 reward 数值是演的，但由 reward 驱动的**控制流必须真**：DAPO 零方差组过滤、达标 abort 在途请求、partial rollout 回 buffer、weight version 推进与 staleness——它们直接改变 sandbox 到达过程和负载形状。假数值 + 真调度，负载就真。
- 计时必须分列 **wall time vs CPU time**：推理/网络等待不计入 CPU 画像。
- 实验产出是**扫参数曲线**，不是 reward 曲线。

## 2. Task 选择：代码生成任务族（三档 workload）

主线任务是代码生成 + 工具调用（reward 真实可算、agentic RL 最小完整形态），**三档混合负载**——只有混合负载才能看到 exec p50/p99、长尾拖住整组 rollout 这个 RL 特有现象：

| 档 | workload | 资源特征 | 真实对照 |
|---|---|---|---|
| **A：单步代码执行** | HumanEval 风格单函数实现 + 单测 | CPU 10–100ms，内存 MB 级 | 单步 verifier |
| **B：SWE-mini** | 多文件小仓库改 bug + 跑 pytest 套件（依赖用预烘焙只读 venv） | CPU 秒–分钟级，大量文件 IO，内存 GB 级，长尾严重 | GLM-5 10K+ SWE 环境、DSec Container 档 |
| **C：terminal/search 风格** | bash 多步 + 阻塞 IO + 超时（禁网环境下用本地 mock 服务/sleep 模拟多跳等待）；**模型经 tool calling 驱动多轮交互** | CPU 不高但**占 worker 槽位很久** | 数千 terminal 环境、多跳 search |

另加**故障注入用例**（死循环→timeout、爆内存→OOM、"装依赖"超时）专门压 sandbox 的异常路径。A 档保留少量题目仅作连通性验证。

## 3. CPU 在 RL 里干什么（设计基石）

GPU 只干两件事：rollout 前向、training 反向。剩下全是 CPU（agentic 场景 CPU 占比更高）：

```
driver(编排) → router/网关 → tokenizer → rollout 并发发射
  → sandbox 环境执行(绝对大头，多轮 agent 每 step 进一次)
  → verifier/reward(judge/单测)
  → advantage 归一/打包/切 DP → 权重同步协调 → 日志/WAL/ckpt staging
```

画像重点排序：① sandbox 执行；② 编排+路由+并发控制（决定 sandbox 到达过程）；③ tokenize/reward/packing（小但高频，决定 driver 是否成为瓶颈）；④ **driver 进程自身的 CPU 开销**（序列化/对象 churn/GC——真实系统用 Rust 重写这一层，如 DSec）。v3 新增：⑤ **rollout 推理本身的 CPU 消耗**（本地小模型前向是真实 CPU 负载，且与 sandbox/driver 抢核——这正是真实 colocate 部署的核心矛盾）。

| 环节 | 真实集群位置 | PoC 中的处理 |
|---|---|---|
| 驱动循环 / 编排调度 | CPU driver | **真实实现**（asyncio），并**测量 driver 自身 CPU%** |
| Rollout 前向 | GPU (SGLang) | **本地 llama.cpp + Qwen3-4B 真实 CPU 推理（首选）** / 远程 OpenAI 兼容 API（备选） / MockEngine（离线冒烟） |
| 环境交互 / sandbox | **CPU 集群** | **真实实现**，核心观测对象（§5.1） |
| Reward / verifier | CPU | **真实实现**（test pass rate） |
| Tokenize / detokenize | CPU | **真实 BPE**（引擎侧与 driver 侧分别计时；验证假设：tokenize 不是瓶颈，detokenize+对象构造才是） |
| rollout logprob | GPU 推理附带 | **真实值**（llama.cpp per-token logprobs）→ TIS/Keep Sampling Mask 从"捏"变"测" |
| advantage / packing / 切 DP | CPU | **真实实现**（GRPO: r − group_mean 等标量运算） |
| Training 前向+反向 | GPU (Megatron) | **mock**：按 token 数模拟耗时、版本号+1；clip/TIS surrogate 数值真实算 |
| 权重同步协调 | CPU 协调 + GPU 传输 | **半真实**：pause→**真实 GGUF reload**（mmap/page-in 成本实测）→resume；不声称 NCCL/RDMA 保真 |
| 监控 / WAL / ckpt staging | CPU | **真实实现**：分阶段 wall/CPU 双列 + sandbox 池指标 + IO 量 |

## 4. 架构设计（组件逐一对照 slime 源码）

```
rl_sim/
├── train.py                 # 同步基线驱动循环      ← slime train.py
├── train_async.py           # 异步流水线(lookahead+abort/requeue) ← slime train_async.py
├── rl_sim/
│   ├── types.py             # Sample/SampleStatus   ← slime/utils/types.py
│   ├── data_source.py       # RolloutDataSource：三档任务集 + 故障注入用例
│   ├── router.py            # 本地 HTTP router 进程，session-affinity ← SGLang router / TITO gateway
│   ├── engine.py            # 引擎协议 + 三实现：
│   │                        #   LocalEngine（首选）：llama.cpp llama-server 多实例=rollout DP ranks
│   │                        #   APIEngine（备选）：远程 OpenAI 兼容 API
│   │                        #   MockEngine（冒烟）：离线模板
│   │                        #   引擎管理（拉起/心跳/重启）← sglang_engine.py + RolloutHealthMonitor
│   ├── tokenizer.py         # 真实 BPE（Qwen3 vocab，vendored 离线）
│   ├── sandbox.py           # SandboxPool：三档资源 profile + unshare 隔离 ← DSec
│   ├── reward.py            # rule-based RM         ← slime/rollout/rm_hub
│   ├── rollout_manager.py   # RolloutManager：asyncio 有界队列+组屏障+abort/requeue
│   │                        #   ← slime/ray/rollout.py（Ray→asyncio，理由见 §5.2）
│   ├── trainer.py           # MockMegatronTrainer   ← actor_group.py + megatron_utils/actor.py
│   ├── weight_sync.py       # WeightUpdater（半真实 reload）← UpdateWeightFromDistributed（仅时序对照）
│   └── monitor.py           # wall/CPU 双列 + driver CPU% + IO + sandbox/引擎画像
├── vendor/                  # llama.cpp 源码包、GGUF 模型、tokenizer、预烘焙 venv（离线分发）
├── tests/                   # sandbox 隔离/故障注入用例
└── README.md                # 组件对照表 + CPU 分工讲解 + 实验报告模板
```

**主循环**（同步基线，对齐 slime train.py 五步）：

```
for rollout_id in range(num_rollout):
    1. rollout_manager.generate(rollout_id)   # prompt组 → router → 本地推理 → (tool call→sandbox 多轮) → reward → group advantage
    2. trainer.async_train(rollout_data)      # mock 训练：真实算 advantage/clip/TIS，模拟 GPU 耗时，version+1
    3. save_model（每 N 步：mock checkpoint）
    4. weight_sync.update_weights()           # pause → GGUF reload（真实成本）→ continue
    5. rollout_manager.eval（每 N 步：held-out pass rate）
```

**异步对照**（train_async.py，对齐 slime train_async.py）：one-step lookahead，有界 data buffer，被 abort 半成品回 buffer，`--update-weights-interval` 控制同步频率。

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

### 5.3 Rollout 引擎（v3 核心变更）：llama.cpp + Qwen3-4B

**选型**：`llama.cpp llama-server` + **Qwen3-4B-Instruct-2507 GGUF（Q4_K_M，约 2.5GB）**。

| 需求 | llama.cpp 支持 |
|---|---|
| CPU 推理 | 原生最强：AVX512/AMX 路径（节点 Xeon 6767P Granite Rapids），GGUF Q4 量化 |
| **logprob 输出** | OpenAI 兼容接口 `logprobs: true` 返回 per-token logprob——远程 API 给不了、TIS 必需 |
| **KV cache** | per-slot KV cache + prefix 复用，slot 状态可查、命中率可测 |
| **agentic tool calling** | `--jinja` chat template + OpenAI `tools` schema + grammar 约束 JSON；多轮 chat 上下文保持 |
| 轻量 | 单静态二进制、零 Python 依赖、独立进程拓扑（与 slime 中 SGLangEngine 独立服务形态一致） |

**不选 vLLM CPU / SGLang CPU 的理由**：vLLM 语义最接近 SGLang（continuous batching、PagedAttention、`prompt_logprobs`、`n>1`），但 pip 依赖重、离线安装难、小模型发挥不出优势；llama.cpp 单文件部署 + CPU 性能最好。vLLM CPU 留作第二版可选对照。

**模型备选**：Qwen3-1.7B / 0.6B（E1 高并发扫描压并发上限）；**另留 Q8 副本**：rollout 用 Q4、trainer 侧 scoring 用 Q8——量化差异本身就是真实的训推不一致源。

**部署拓扑**（镜像 slime rollout 侧）：

```
driver ──► router（本地 HTTP 进程，session-affinity 一致性哈希：同一 agent id → 固定实例）
             ├── llama-server 实例 0（taskset 绑核，--parallel N slots） = rollout DP rank 0
             ├── llama-server 实例 1                                     = rollout DP rank 1
             └── ...
```

- 多实例 = slime 的多 SGLangEngine；session-affinity 对应 GLM DP-aware routing，**prefix cache 命中率可测**（Miles 参考值 96%）；
- 引擎管理（`engine.py` 内 ServerManager）：实例拉起/心跳/重启 ← 对照 `sglang_engine.py` + `RolloutHealthMonitor`。

**logprob 真实化的三个设计红利**：

1. **TIS/训推不一致从"捏"变"测"**：rollout logprob（Q4）全真；trainer mock 重算 logprob 时对同一序列用 Q8 再 scoring 一遍（或施加不同 sampling mask）→ 比率分布实测；Keep Sampling Mask 真实实现；
2. **weight sync 半真实**：pause slots → 重新 mmap/reload GGUF → resume——权重内容不变（mock trainer），但 **reload 的 CPU/IO 成本（page-in 时间）真实可测**，对应 `update_weights_from_disk` 的引擎重载窗口；
3. **多轮 agentic 全真**：tool call → sandbox 执行 → 结果拼回上下文再生成，KV cache 复用真实发生。

**性能预估**（256 核 Granite Rapids，Q4 4B）：prefill 聚合数百~上千 tok/s，decode 单流 20–60 tok/s，多 slot 聚合更高；一轮 rollout（16 prompt × 8 样本 × ~500 token）分钟级。**预估值，M3 实测写入报告。**

### 5.4 MockMegatronTrainer（GPU 替身）

- 真实计算（本来就是 CPU 标量活）：GRPO advantage、PPO clip surrogate 数值、TIS/IcePop 比率与 mask 比例——**v3 起输入为真实 rollout logprob + Q8 rescoring logprob**；
- 模拟：按 token 数 sleep 模拟 GPU 耗时（速率可配）；`weight_version += 1`。

### 5.5 WeightUpdater（半真实，不做传输保真声明）

对齐 `UpdateWeightFromDistributed` 的**时序**：`pause_generation()` → 模拟传输耗时（参数GB/带宽GBps 估算，截断实际 sleep）→ **真实 GGUF reload（mmap/page-in 计时）** → `continue_generation()`；记录 pause 窗口、reload 耗时与 rollout 侧 staleness 分布。**明确不声称对照 NCCL/RDMA。**

### 5.6 Monitor（负载可观测性）

- **双列计时**：每阶段 wall time 与 CPU time 分列（推理/网络等待不进 CPU 画像）；
- **driver 自测量**：driver 进程 CPU%（/proc/self/stat、os.times）、每阶段对象数/字节数；
- **引擎画像**：每实例 slot 占用、tok/s、prefix cache 命中率、reload 耗时；
- **IO 画像**：per-task 读写字节、sandbox scratch tmpfs vs nvme0 对比、WAL/ckpt staging 的 fsync 成本；
- 输出示例：

```
[rollout 3] wall: orchestrate 0.4s | router 1.1s | infer 38.3s | sandbox 11.8s | reward 0.2s | adv+pack 0.1s | train(mock) 2.0s | weight_sync 1.1s
           cpu : orchestrate 0.3s | router 0.9s | infer 36.1s | sandbox  9.2s | reward 0.2s | adv+pack 0.1s |      -        | reload 0.9s
[engine] inst 4×8 slots busy 91% | prefill 720 tok/s decode 41 tok/s/inst | prefix_hit 63% | reload 0.9s
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

E3. **rollout 并发度 vs driver CPU 占用**：找到 Python driver 饱和点（对应真实系统改用 Rust router/gateway 的动机）。

E4. **sync vs async 对照**：组屏障空转 vs abort/requeue 效果（slime train.py vs train_async.py 的直观演示）。

E5（可选）. **scratch 介质对比**：tmpdir 放 tmpfs vs /mnt/nvme0 的 IO/尾延迟差异。

E6（v3 新增）. **引擎画像**：llama-server 实例数 × slots 扫描（聚合 tok/s、prefix cache 命中率随 session-affinity 开关的变化）；**推理 vs sandbox/driver 的核争抢**（colocate 矛盾的真实再现）；Q4 rollout vs Q8 scoring 的 logprob 差分布（训推不一致实测）。

## 7. 验证标准（DoD，v3）

1. 离线闭环跑通：`python train.py --engine local --num-rollout 10`（同步基线）与 `train_async.py` 各跑通，控制流事件（过滤/abort/requeue/version 推进）日志可追；
2. sandbox 隔离：死循环→timeout、爆内存→oom、语法错→error、**禁网生效（sandbox 内 socket 连接失败）**；均不逃逸、不拖垮池；
3. **logprob 真实**：样本携带 llama.cpp per-token logprob；TIS 比率由 Q4/Q8 双 scoring 实测；E6 产出 logprob 差分布图；
4. 双列计时、driver CPU%、IO 量、分档 sandbox 画像、引擎画像全部有输出，格式符合 §5.6；
5. E0–E4、E6 实验数据落盘（CSV/JSON），E1/E4 有结论曲线；
6. 远程 API（备选引擎）跑通 ≥2 rollout（API 由使用方配置，可跳过并标注）；MockEngine 冒烟通过；
7. README：slime/DSec/Miles 组件对照表 + CPU 分工讲解 + 实验报告。

## 8. 非目标（第一版）

- 不实现 SAO/critic/OPD（结构上预留 weight_version/staleness/DIS 钩子，扩展路径见 README）；
- 不引入 Ray；除 llama.cpp（C++ 独立二进制）外 Python 侧保持 stdlib only；
- 不做真实大模型反传；不追求 reward 数值的任何意义；
- 权重同步不做 NCCL/RDMA 保真对照；vLLM CPU 对照留第二版。

## 9. 扩展路径（第二版候选）

- group size=1 + DIS + mock critic（SAO）；OPD 模拟（Q8 当教师，`advantage -= coef × (student_logp − teacher_logp)`，logprob 全真后可行）；多轮 agentic 任务加深（真实 SWE 仓）；容器级隔离对照（若环境允许，补 container/microVM 档启动成本实测）；vLLM CPU 引擎对照。

## 10. 修订记录

**v3（2026-09-10）**：rollout 引擎本地化（用户决策）：

1. rollout 前向从"远程模型 API"改为 **llama.cpp + Qwen3-4B GGUF 本地 CPU 推理为首选**（logprob/KV cache/tool calling 全真实），远程 API 降备选、MockEngine 留作冒烟；新增 §5.3 选型论证与部署拓扑；
2. rollout logprob 由模拟值变真实值 → TIS/IcePop/Keep Sampling Mask 改为实测（Q4 rollout vs Q8 scoring 双精度对照）；§5.4 相应更新；
3. weight sync 从纯 mock 升级为半真实：真实 GGUF reload，page-in 成本可测（§5.5）；
4. 画像重点新增第五项：rollout 推理本身的 CPU 消耗及其与 sandbox/driver 的核争抢（§3）；实验新增 E6 引擎画像（§6）；DoD 新增第 3/6 条（§7）；
5. monitor 新增引擎画像行（实例/slots/tok/s/prefix_hit/reload）。

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
