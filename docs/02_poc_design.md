# PoC 设计：CPU-only Agentic RL 后训练模拟（rl_sim）

> 状态：待审查。配套文档：`01_research_report.md`（调研依据）、`03_execution_plan.md`（执行计划）。

## 1. 目标

1. **理解 RL 后训练全流程**：以 slime（GLM-5.2 同款栈）为蓝本，端到端跑通 rollout → reward → advantage → train → weight sync 闭环；
2. **理解 CPU 侧负载**：真实实现并观测编排调度、sandbox 执行、reward、数据处理、权重同步协调等 CPU 操作，重点是 **sandbox 负载画像**；
3. **约束**：单 CPU 节点 + 模型 API，无 GPU —— GPU 计算（rollout 前向、training 反向）全部替换/模拟；
4. **不做**：不追求真实训练效果，不引入 GPU 依赖，第一版不实现异步流水线（结构上预留扩展点）。

## 2. Task 选择：代码生成 + 单测 sandbox 执行

选 **HumanEval 风格代码生成任务**（函数签名 + docstring → 实现代码 → sandbox 跑单测 → pass rate 作 reward）。理由：

- sandbox 有**真实 CPU 负载**：执行模型生成的代码、跑测试、超时/OOM 处理，直接对应 DeepSeek DSec 沙箱与 GLM-5 的 SWE 环境；
- reward **真实可算**（rule-based test pass rate），无需 reward model；
- 模型 API 天然适合代码生成；
- 是 agentic RL 的最小完整形态（生成→环境交互→验证→奖励）。

内置 20 道小题（reverse_string / fizzbuzz / is_palindrome 级别），16 道训练 + 4 道 held-out 评估。

## 3. 核心认知：CPU vs GPU 分工（设计基石）

| 环节 | 真实集群位置 | PoC 中的处理 |
|---|---|---|
| 驱动循环 / 编排调度 | CPU (driver) | **真实实现** `train.py` |
| Rollout 前向（生成） | GPU (SGLang/vLLM) | **替换为模型 API**（HTTP 调用本身是真实 CPU 网络 IO）+ 离线 MockEngine 兜底 |
| 环境交互 / sandbox 执行 | **CPU**（DSec/SWE 环境集群） | **真实实现**，重点观测对象 |
| Reward / verifier | CPU | **真实实现**（test pass rate） |
| Tokenize / detokenize | CPU | **真实实现**（简化 whitespace 计数，注释说明真实为 BPE） |
| 组内 advantage / 归一化 | CPU | **真实实现**（GRPO: r − group_mean，纯标量） |
| Buffer / packing / 按 DP 切分 | CPU | **真实实现**（简化版） |
| Training 前向+反向 | GPU (Megatron) | **mock**：按 token 数模拟耗时、版本号 +1；clip/TIS 的 surrogate 数值真实计算（标量） |
| 权重同步协调 | CPU 协调 + GPU 传输 | **模拟**：pause → 版本广播 → resume；记录 staleness |
| 监控 / 日志 / 心跳 | CPU | **真实实现**：分阶段耗时 + sandbox 池指标 |

## 4. 架构设计（组件逐一对照 slime 源码）

```
rl_sim/
├── train.py                 # 驱动循环          ← slime train.py
├── rl_sim/
│   ├── types.py             # Sample/SampleStatus ← slime/utils/types.py
│   ├── data_source.py       # RolloutDataSource   ← slime/rollout/data_source.py
│   ├── engine.py            # RolloutEngine 协议   ← SGLangEngine + HTTP /generate
│   │                        #   APIEngine：OpenAI 兼容 chat/completions
│   │                        #   MockEngine：离线兜底（见 §5.3）
│   ├── sandbox.py           # SandboxPool          ← DeepSeek DSec / agentic 环境交互
│   ├── reward.py            # rule-based RM        ← slime/rollout/rm_hub（deepscaler/dapo）
│   ├── rollout_manager.py   # RolloutManager       ← slime/ray/rollout.py（Ray→concurrent.futures）
│   ├── trainer.py           # MockMegatronTrainer  ← slime/ray/actor_group.py + megatron_utils/actor.py
│   ├── weight_sync.py       # WeightUpdater        ← update_weight/UpdateWeightFromDistributed
│   └── monitor.py           # StageTimer/SandboxMetrics ← RolloutHealthMonitor + 日志体系
└── README.md
```

**主循环**（对齐 slime train.py 五步）：

```
for rollout_id in range(num_rollout):
    1. rollout_manager.generate(rollout_id)   # prompt组 → 并发API生成 → sandbox执行 → reward → group advantage
    2. trainer.async_train(rollout_data)      # mock 训练：真实算 advantage/clip/TIS 数值，模拟 GPU 耗时，weight_version+1
    3. save_model（每 N 步：mock checkpoint = version + 指标历史 JSON）
    4. weight_sync.update_weights()           # pause_generation → 版本广播 → continue_generation
    5. rollout_manager.eval（每 N 步：held-out 题 pass rate）
```

参数风格对齐 slime quick start：`--rollout-batch-size --n-samples-per-prompt --num-rollout --eps-clip 0.2 --eps-clip-high 0.28 --use-tis --sandbox-workers --engine {api,mock}`，并保持约束 `rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout`。

## 5. 关键模块设计

### 5.1 SandboxPool（核心观测对象）

- **形态**：固定大小 worker 池（默认 min(8, nproc)，目标节点 256 核可开大），单 worker 串行执行——模拟真实"一实例一容器"；
- **隔离**：每个任务独立 tmpdir，写入 `solution.py` + `test_runner.py`，`subprocess` 子进程内设 `resource.setrlimit`：
  - `RLIMIT_CPU` = 5s（硬 CPU 时间）、`RLIMIT_AS` = 512MB（地址空间）、`RLIMIT_NOFILE` = 64、`RLIMIT_FSIZE` = 16MB；
  - 外层 `timeout` kill 兜底；
- **协议**：runner 执行全部 assert，stdout 输出一行 `__RESULT__{json}`：status(passed/failed/error/oom) + per-test 通过数 + `ru_utime+ru_stime`（CPU 时间）+ `ru_maxrss`（峰值内存）；
- **指标**（对应" sandbox 负载情况"的调研目标）：每任务记录 排队时长 / 执行 wall / CPU 时间 / 峰值 RSS / 退出状态；每 rollout 输出 池利用率、峰值队列深度、执行时间 p50/p99、状态分布（passed/failed/timeout/oom/error）。

### 5.2 RolloutManager

- 从 RolloutDataSource 取 prompt 组（每组 `n_samples_per_prompt` 个 Sample，同 slime 的组复制逻辑）；
- 两级并发：API 生成并发池（默认 16）→ SandboxPool；
- 组归一化 advantage（V3.2 风格只减均值；零方差组计数——对应 DAPO dynamic sampling）；
- 打包训练 batch：tokens / rollout_logprobs / rewards / advantages / loss_mask / weight_version。

### 5.3 两种 RolloutEngine

- **APIEngine**：OpenAI 兼容接口（env: `RL_SIM_API_BASE/RL_SIM_API_KEY/RL_SIM_API_MODEL`），指数退避重试，从 markdown code block 提取代码；logprob 用模拟值（多数 chat API 不返回 token logprob，注释说明 slime 用 SGLang `return_logprob`）；
- **MockEngine**（离线兜底）：内置各题 canonical solution，按 `p_correct(v) = min(0.95, 0.15 + 0.08·weight_version)` 概率生成正确实现，否则施加随机变异（off-by-one、运算符翻转等）；reward 曲线随训练真实上升，用于演示"训练有效"与全流程闭环；同时模拟 5–20ms 生成延迟使调度行为真实。

### 5.4 MockMegatronTrainer（GPU 计算的替身）

- **真实计算**（本来就是 CPU 标量活）：GRPO surrogate loss 数值、PPO clip 生效比例、模拟 train/infer mismatch（`train_logp = rollout_logp + N(0, σ)`，σ 随 staleness 增大）、**TIS/IcePop 比率与 mask 比例**（`exp(train_logp − rollout_logp)` 落在 [1/β, β] 外即 mask）；
- **模拟**：按 token 数 sleep 模拟 GPU 训练耗时（速率可配）；`weight_version += 1`；输出 loss/kl/tis_masked 指标。

### 5.5 WeightUpdater（权重同步模拟）

对齐 `UpdateWeightFromDistributed` 时序：`pause_generation()` → 模拟传输耗时（按 参数GB/带宽GBps 计算并截断实际 sleep）→ 引擎侧版本更新 → `continue_generation()`；记录每次同步耗时与 rollout 侧观测到的 staleness 分布（`当前版本 − 样本生成时版本`）。

### 5.6 Monitor（负载可观测性）

每步输出示例：

```
[rollout 3] stage wall: orchestrate 0.4s | api_gen 12.3s | sandbox_queue 3.1s | sandbox_exec 8.7s | reward 0.2s | adv+pack 0.1s | train(mock_gpu) 2.0s | weight_sync 0.1s
[sandbox] busy peak 8/8 | queue_depth max 14 | exec p50 0.8s p99 4.2s | passed 61% failed 28% timeout 9% oom 2%
[train] weight_version 4 | samples 128 | mean_reward 0.58 (+0.07) | tis_masked 3.2% | zero_var_groups 2
```

结束输出汇总报告（各阶段占比、sandbox 负载画像、reward 曲线）。

## 6. 验证标准（DoD）

1. `python train.py --engine mock --num-rollout 10` 在目标节点离线跑通，mean_reward 单调上升（MockEngine 设计保证），各阶段指标合理；
2. sandbox 隔离用例：死循环 → timeout kill；爆内存分配 → oom；语法错误 → error；均不逃逸、不拖垮池；
3. `python train.py --engine api` 用真实模型 API 跑通 ≥2 个 rollout（API 由使用方配置）；
4. 每步输出 §5.6 格式的 CPU 阶段分解与 sandbox 负载报告；
5. README 给出与 slime/DSec/Miles 的组件对照表。

## 7. 非目标（第一版）

- 不实现异步流水线 / SAO（结构上已预留 `weight_version`/staleness 字段与 DIS 钩子，扩展路径见 README）；
- 不引入 Ray（concurrent.futures 零依赖替代，README 说明对应关系）；
- 不做真实大模型反传；无第三方 Python 依赖（纯 stdlib）。

## 8. 扩展路径（第二版候选）

- `train_async.py`：one-step lookahead + `--update-weights-interval` + partial rollout buffer（对应 slime train_async.py 与 GLM-5 异步 Agentic RL）；
- group size = 1 + DIS + mock critic（对应 SAO）；
- 多轮 agentic 任务（loss_mask 区分模型 token / 环境 token）；
- OPD 模拟（第二个 API 当教师，`advantage -= coef × (student_logp − teacher_logp)`）。
