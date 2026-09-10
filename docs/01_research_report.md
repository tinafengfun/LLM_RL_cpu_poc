# 调研报告：DeepSeek V4 / GLM-5.2 RL 后训练方法与开源复现框架

> 调研日期：2026-09-10。所有关键结论均附一手来源链接。

## 0. TL;DR

- **DeepSeek V4**：后训练 = 领域专家培养（SFT + **GRPO** RL）+ **On-Policy Distillation (OPD)** 合并统一模型。OPD 取代了 V3.2 的 mixed RL 阶段（V4 并非没有 RL，RL 用在专家培养阶段）。
- **GLM-5.2**：后训练 = SFT → Reasoning RL（GRPO + IcePop）→ Agentic RL → General RL → 跨阶段蒸馏；生产环境已部署 **SAO（Single-rollout Asynchronous Optimization）** 新范式，配套框架 **slime** 已开源（THUDM，GLM-5.2 同款后训练栈）。
- **开源复现框架**：slime / Miles / verl / AReaL / OpenRLHF / TRL 等。本 PoC 选 **slime** 为蓝本（架构最贴合目标、组件边界清晰）。
- **对 PoC 最重要的认知**：RL 系统中 GPU 只做两件事（rollout 前向 + training 反向），**其余全部是 CPU 负载**——编排调度、HTTP 路由、tokenize、reward/verifier、sandbox 环境执行、data buffer/packing、权重同步协调。

---

## 1. DeepSeek V4 后训练

来源：[DeepSeek-V4 论文 arXiv:2606.19348 §5](https://arxiv.org/html/2606.19348)

### 1.1 两阶段 pipeline

论文原文："a two-stage paradigm: the independent cultivation of domain-specific experts, followed by unified model consolidation via on-policy distillation"。

**阶段一：Specialist Training（§5.1.1，沿用 V3.2 管线）**
- Base model 先在领域专属高质量数据上 SFT；
- 再用 **GRPO** 做领域 RL（超参与 DeepSeek-R1 / V3.2 一致），奖励为领域特定 prompt 与 reward signal；
- 领域覆盖：mathematics、coding、agent、instruction following 等；
- **Reasoning Effort 分档**：Non-think / Think High / Think Max 三档，RL 时每档施加不同 length penalty 和 context window（8K / 128K / 384K）；
- **Generative Reward Model (GRM)**：难验证任务弃用传统 scalar RM，改用 rubric-guided 数据 + GRM 评分；关键创新是**直接对 GRM 本身施加 RL——actor 网络原生充当 GRM**，生成与评判联合优化。

**阶段二：On-Policy Distillation（§5.1.2，替代 V3.2 的 mixed RL）**
- 用 10+ 个领域专家教师蒸馏出**一个**统一 student。

### 1.2 OPD 机制

- 目标函数（Eq. 29）：`L_OPD(θ) = Σ_i w_i · D_KL(π_θ ‖ π_E_i)`，π_E_i 为第 i 个专家教师，w_i 为教师权重；
- 计算 reverse KL 必须**从 student 采样轨迹**（on-policy）：student 自己生成，教师提供该轨迹上的输出分布；
- **关键工程决策：full-vocabulary logit distillation**。先前工作（含开源 Miles 的 OPD 实现）把 full-vocab KL 简化为 token 级估计（把 `sg[log π_E/π_θ]` 当 per-token advantage 代入 policy loss），DeepSeek 明确指出该做法**梯度方差高、训练不稳定**，因此采用完整词表 logits 蒸馏；
- 方法出处：Thinking Machines "On-Policy Distillation" 博客（2025）与 MiniLLM（ICLR 2024, arXiv:2306.08543）。

### 1.3 为什么用 OPD 替换 mixed RL

规避权重合并/混合 RL 常见的性能退化（"practically circumventing the performance degradation often encountered in traditional weight-merging or mixed RL techniques"），同时回避多域 mixed RL 的平衡难题。

### 1.4 被替换的对象：DeepSeek-V3.2 的 RL

来源：[V3.2 论文 arXiv:2512.02556 §3](https://arxiv.org/html/2512.02556v1)

- 算法 **GRPO**；管线 = 6 领域 specialist distillation → **Mixed RL**（reasoning + agent + alignment 合并进单一 RL 阶段，规避多阶段灾难性遗忘）；
- 奖励：reasoning/agent = rule-based outcome reward + length penalty + language consistency reward；通用任务 = generative RM（每 prompt 自带 rubric）；
- RL 算力预算 **超过 pre-training 成本的 10%**；
- **GRPO 规模化稳定技术（§3.1）**：
  - **Unbiased KL Estimate**：IS ratio 修正 K3 estimator；数学等领域弱 KL 甚至不加更好；
  - **Off-Policy Sequence Masking**：负 advantage 且新旧策略 KL 超阈值 δ 的序列整体 mask；
  - **Keep Routing**：保留 rollout 时 MoE 专家路由，训练时强制复现；
  - **Keep Sampling Mask**：保留 top-p/top-k 截断 mask，保证动作子空间一致。

### 1.5 DeepSeek 内部 infra（V4 论文 §5.2，CPU 侧负载的重要参考）

- **教师调度**：教师权重 offload 集中式存储，按需加载 + ZeRO 分片；不物化 full logits，只缓存教师最后一层 hidden states，训练时即时重建；样本按教师索引排序派发；
- **可抢占 rollout 服务**：token 粒度 WAL；强调从头重新生成未完成请求在数学上不正确（引入 length bias）；
- **百万 token RL 的数据处理**：rollout 数据拆成轻量 metadata（全局 shuffle/packing）+ 重型 per-token 字段（shared-memory loader 按 mini-batch 即用即释放）——典型的 CPU 侧数据工程；
- **DSec 沙箱**：Rust 三组件（Apiserver/Edge/Watcher）+ 3FS，单集群数十万并发沙箱，四种执行底座（Function Call / Container(EROFS) / microVM(Firecracker) / fullVM(QEMU)），trajectory log 支持抢占安全恢复；
- **FP4 QAT**：post-training 全程 MXFP4 QAT，rollout 直接用原生 FP4 权重保证训推一致。

### 1.6 开源侧对照：Miles（SGLang/RadixArk 团队）

来源：[DeepSeek-V4 Day-0 博客（LMSYS 2026-04-25）](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)、[Miles v0.1 技术报告 arXiv:2609.08368](https://arxiv.org/html/2609.08368v1)

- RL loop 三段式：**SGLang rollout → Megatron/FSDP training → 每 step 权重同步回 rollout**；
- 三种权重同步：Broadcast（NCCL，Kimi K2 1T 约 1 分钟）/ **P2P RDMA 直写**（53.3s→7.2s）/ **Disk-delta**（只传变化字节 + checksum）；
- session-affinity 路由保 prefix cache（参考命中率 96%）；TITO session server（服务端拥有 tokenization）；R3（Rollout Routing Replay）；TIS/MIS loss 校正训推不一致；
- 数值对齐验证：285B 模型 32×GB300 上 DAPO 训练，step-0 rollout/训练 log-prob 漂移仅 ~0.023；
- Miles 的 OPD 走 token 级/top-K 估计路线（与 DeepSeek 内部 full-vocab 形成方法学对照）。

---

## 2. GLM-5.2 RL 后训练

来源：[GLM-5 论文 arXiv:2602.15763](https://arxiv.org/pdf/2602.15763)、[SAO 论文 arXiv:2607.07508](https://arxiv.org/abs/2607.07508)、[z.ai/blog/glm-5.2](https://z.ai/blog/glm-5.2)

### 2.1 整体 pipeline

SFT → **Reasoning RL** → **Agentic RL** → **General RL** → **On-Policy Cross-Stage Distillation**（防多阶段遗忘）。

### 2.2 Reasoning RL：GRPO + IcePop

- **IcePop** 处理 training/inference mismatch：定义 `ρ = π_train_θold / π_infer_θold`，用 `pop(ρ, 1/β, β)` 把 ρ 落在 [1/β, β] 之外的 token 直接 mask 成 0（β=2，ε_low=0.2，ε_high=0.28）；完全 on-policy，**group size 32**；
- DSA 架构稳定性：SGLang 的非确定性 CUDA top-k 会导致 RL 崩溃 + 熵骤降 → 改确定性 `torch.topk` 且 **RL 全程冻结 indexer 参数**；
- 四个域（math/science/code/TIR）联合训练，每域专属 judge model 产出 binary outcome reward。

### 2.3 Agentic RL：全异步、解耦（CPU 侧架构的极佳参考）

- 训练引擎与推理引擎部署在不同 GPU 上完全解耦；推理引擎持续生成，攒够阈值送训练；**每 K 次梯度更新把权重推回推理引擎**；每次推理侧权重更新后 **reset optimizer**；
- **Multi-Task Rollout Orchestrator**：中央编排器 + 任务微服务注册（自带 rollout/reward 逻辑），>1k 并发 rollout，动态调整任务采样比；
- **TITO（Token-in-Token-out）Gateway**：训练侧直接消费推理引擎产出的精确 token 流，不 re-tokenize；
- **Direct Double-sided IS**：`r_t(θ) = exp(log π_θ − log π_rollout)`，丢弃 π_θold；信任域外 token **完全 mask**；
- **过旧样本丢弃**：每条 response 记录模型版本序列 (w_0,...,w_k)，`w' − w_0 > τ` 则丢弃整条；GRPO 组内剔除后有效样本过半则重复 pad，否则整组丢弃；
- **DP-aware routing**：同一 agent 实例 consistent hashing 到同一 DP rank，保 KV cache；
- 环境规模：10K+ 真实 SWE 环境（9 种语言）、数千 terminal 环境、多跳 search 任务——**全部 CPU 负载**。

### 2.4 SAO（Single-rollout Asynchronous Optimization）

2026-07-15 发布，已部署于 GLM-5.2（750B/744B-A40B）生产训练。三个核心机制：

1. **Single-rollout（group size = 1）**：每 prompt 只采一条，完成即训，消除"组内等最慢"的同步壁垒；在线/agentic 场景组采样结构性不可行；
2. **DIS**：同 2.3 的 direct double-sided IS；超参 TIR `ε_low=0.3, ε_high=5.0`，coding agent `ε_low=0.8, ε_high=3.0`；
3. **Critic 回归**（single-rollout 方差大，重新引入 value model）：
   - TTUR：每 1 次 actor 更新，critic 更新 2 次；
   - Frozen-attention：value 不稳定主要来自 Full Attention 层 → 冻结 attention 只训 MoE 投影；
   - Skip-Observation GAE：observation token 不参与 TD（advantage=0），Bellman target 直接桥接相邻 action；length-adaptive λ；
- 结果：稳定训练 ~1000 步（vanilla GRPO ~160 步崩溃，VAPO ~90 步）；AIME 2025 97.3% vs GRPO 84.2%（二手解读，方向与论文一致）。

---

## 3. 开源框架格局与选型

| 框架 | 出品 | 训练后端 | Rollout | 编排 | 特点 | 与本 PoC 的关系 |
|---|---|---|---|---|---|---|
| **slime** | THUDM / Z.ai | Megatron | SGLang | Ray | GLM-5.2 同款栈，async-first，MoE 高性能 | **PoC 蓝本** |
| Miles | SGLang/RadixArk | Megatron/FSDP | SGLang | Ray | DeepSeek V4 Day-0，企业级 MoE RL | 对照参考 |
| verl | ByteDance | FSDP/Megatron | vLLM/SGLang | Ray | 生态最大，HybridFlow dataflow | 参考 |
| AReaL | 清华/蚂蚁 | FSDP/Megatron | SGLang | Ray | fully async | SAO 对照 |
| OpenRLHF | 社区 | DeepSpeed | vLLM | Ray | 经典 PPO/RLHF | 参考 |
| TRL | HuggingFace | HF Trainer | HF | 无 | 算法实验，非生产 | 不适合 |

**选型结论**：以 **slime** 为蓝本。理由：① GLM-5.2 生产同款，调研结论可直接对照；② 架构极简（只有 training/rollout/data buffer 三模块）；③ 组件命名与边界清晰，mock 时可逐一对照真实源码。

参考对比资料：[Anyscale: Open Source RL Libraries](https://www.anyscale.com/blog/open-source-rl-libraries-for-llms)、[Anatomy of RL Frameworks](https://www.hanifleo.com/anatomy-of-rl-frameworks/)、[RL Libraries Overview (mid-2026)](https://ai-infrastructure.net/rl-libraries-llms/)

---

## 4. slime 架构详解（PoC 直接对照的骨架）

来源：[GitHub THUDM/slime](https://github.com/THUDM/slime) README 与源码（train.py、slime/ray/rollout.py、slime/ray/actor_group.py、slime/backends/megatron_utils/ 等）

### 4.1 三大模块（Ray 编排）

- **training (Megatron)**：主训练进程组，从 Data Buffer 读数据，训完同步参数给 rollout；
- **rollout (SGLang + router)**：生成数据（含 reward/verifier），自定义 generate 函数可包多轮/工具调用/环境交互；
- **Data Buffer**：prompt 初始化、自定义数据、rollout 生成方法。

默认 **disaggregated**（actor 与 rollout 各占独立 GPU），`--colocate` 共享。rollout 数据经 `ray.put` ObjectRef 传训练侧；SGLang server 以标准 HTTP `/generate` 暴露（payload 带 `input_ids`、`return_logprob: True`）。

### 4.2 主训练 loop（train.py，每个 rollout_id）

1. `rollout_manager.generate.remote(rollout_id)` —— 收集一批 rollout 数据；
2. `actor_model.async_train(rollout_id, rollout_data_ref)` —— Megatron 训练（有 critic 时先训 critic 拿 values 作 `external_data`）；
3. 周期性 `save_model`；
4. `actor_model.update_weights()` —— 新权重推回 SGLang；
5. 周期性 `rollout_manager.eval.remote(rollout_id)`。

`train_async.py`：one-step lookahead 异步（训练当前 batch 时提前发起下一轮 generate），`--update-weights-interval` 控制同步频率；`examples/fully_async/` 更激进（常驻并发池 + 被 abort 样本回 buffer）。

### 4.3 Rollout 侧关键类

- `RolloutManager`（@ray.remote）：启动 router、加载 data_source / generate_rollout / custom RM；`generate()` → `_get_rollout_data` → `_convert_samples_to_train_data`（**组归一化：`(r − group_mean)/(group_std + 1e-6)`**，组装 tokens/rewards/loss_masks/rollout_log_probs）→ `_split_train_data_by_dp`；
- 默认 `generate_rollout`：`rollout_batch_size` 为目标循环，每组 `n_samples_per_prompt` 个 Sample，asyncio 并发 → HTTP POST router → 拿回 `output_token_logprobs` → `async_rm` 算 reward；支持 dynamic sampling filter（DAPO 零方差组过滤）、`--partial-rollout` 半成品续生成（`--mask-offpolicy-in-partial-rollout`）；
- `Sample`（types.py）：prompt/tokens/response/reward/loss_mask/rollout_log_probs/status∈{PENDING,COMPLETED,TRUNCATED,ABORTED}；
- RM hub：内置 rule-based RM（deepscaler/dapo/math/f1/gpqa）；`--custom-rm-path` 自定义 `async def (args, sample) -> float`；`remote_rm` POST 外部 RM 服务。

### 4.4 训练侧关键类

- `RayTrainGroup`：一个角色的训练进程组（每 GPU 一个 Ray actor），`async_train()` 广播、`update_weights()` 触发同步，支持 actor/critic role；
- `MegatronTrainRayActor`：TensorBackuper 在 CPU 侧维护多份权重副本（actor/ref/teacher/old_actor/rollout_actor）；`train_actor` = 算 ref/teacher/old log-probs → `compute_advantages_and_returns` → Megatron train；
- loss.py：PPO clip（ε_low=0.2/ε_high=0.28）；advantage estimator 支持 grpo/gspo/cispo/ppo/reinforce++；**TIS**（`tis = exp(train_logp − rollout_logp)` clamp 后作 pg_loss 权重）与 **icepop**（域外置 0）；`compute_advantages_and_returns` 支持 GAE 与自定义 advantage function；
- OPD：`apply_opd_kl_to_advantages`，`advantage -= opd_kl_coef × (student_logp − teacher_logp)`。

### 4.5 权重同步（train → SGLang）

- `UpdateWeightFromDistributed`（默认）：`pause_generation` → `flush_cache` → 建 device process group → TP/EP all-gather 收拢参数 → 转 HF 格式 → 分桶 `dist.broadcast` → 引擎接收 → `continue_generation`；
- `update_weight_from_disk / _delta`：全量或 delta 经共享文件系统；`update_weight_from_tensor`：colocate 直传。

### 4.6 关键配置约束

```
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
```
GLM-5.2 官方示例（256×H100）：训练 TP4×PP8×CP8×EP32，BF16 训练 + FP8 rollout，PD 分离 + EAGLE MTP 投机解码。

---

## 5. 对本 PoC 的启示：CPU vs GPU 分工

| 环节 | 真实集群位置 | 依据 |
|---|---|---|
| 驱动循环/编排调度 | CPU driver | slime train.py 单进程 driver + Ray remote |
| Rollout 前向 | GPU (SGLang) | — |
| 环境交互/sandbox 执行 | **CPU 集群** | DSec 沙箱、GLM-5 10K+ SWE 环境 |
| Reward/verifier | CPU | rule-based RM / judge 调用 |
| Tokenize/detokenize | CPU | TITO 之前的必经环节 |
| 组内 advantage/归一化 | CPU | 纯标量运算 |
| Buffer/packing/按 DP 切分 | CPU | slime `_split_train_data_by_dp`、V4 §5.2.4 |
| Training 前向+反向 | GPU (Megatron) | — |
| 权重同步协调/序列化 | CPU 协调 + GPU/NVLink/RDMA 传输 | slime update_weight 三模式 |
| 监控/日志/心跳/WAL | CPU | RolloutHealthMonitor、V4 §5.2.3 |

**结论**：用"模型 API 替换 SGLang（GPU）+ mock trainer 替换 Megatron（GPU）"后，剩余全部环节都可以在单 CPU 节点上真实运行，且正是工业 RL 系统中 CPU 侧的真实负载轮廓。

## 6. 主要来源

- DeepSeek-V4：https://arxiv.org/html/2606.19348 ；V3.2：https://arxiv.org/html/2512.02556v1
- GRPO 出处：https://arxiv.org/abs/2402.03300 ；R1：https://arxiv.org/abs/2501.12948
- GLM-5：https://arxiv.org/pdf/2602.15763 ；SAO：https://arxiv.org/abs/2607.07508 ；GLM-5.2 官方：https://z.ai/blog/glm-5.2
- slime：https://github.com/THUDM/slime ；Miles：https://arxiv.org/html/2609.08368v1 ；Day-0 博客：https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
- OPD：https://thinkingmachines.ai/blog/on-policy-distillation ；MiniLLM：https://arxiv.org/abs/2306.08543
