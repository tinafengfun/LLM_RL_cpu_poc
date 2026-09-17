# LLM_RL_cpu_poc

在无 GPU 的 CPU 节点上，研究 RL 后训练（post-training）的**控制流**与 **CPU 负载画像**。

以 **slime**（THUDM，GLM-5.2 同款后训练栈）为蓝本，对照 DeepSeek V4 / Miles：rollout 前向用 **llama.cpp + Qwen3-4B 本地 CPU 推理**（logprob / KV cache / tool calling 全真实，远程模型 API 为备选），training 反向（GPU）mock，其余全部 CPU 环节真实实现并可观测——重点是 **agentic RL sandbox 的驱动方式、workload 与资源特征**。

设计原则：**闭环假可以接受，负载失真不能接受。**（假 reward 数值 + 真控制流 + 真实 CPU 负载测量）

## 文档（终审查用）

| 文档 | 内容 |
|---|---|
| [docs/01_research_report.md](docs/01_research_report.md) | 调研报告：DeepSeek V4 后训练（GRPO→OPD）、GLM-5.2 RL（GRPO+IcePop / SAO）、开源框架格局、slime 架构详解 |
| [docs/02_poc_design.md](docs/02_poc_design.md) | PoC 设计 v3.1：两大真实场景任务（长程 agentic / 多模态）、unshare 禁网隔离、asyncio 到达过程、双列计时、E0–E6 实验、验证标准（§10 附修订记录） |
| [docs/03_execution_plan.md](docs/03_execution_plan.md) | 执行计划 v3.1：目标节点（10.239.23.91）环境、/workspace 磁盘选型、里程碑 M0–M6、验证方案、风险对策 |
| [docs/05_dev_plan.md](docs/05_dev_plan.md) | 开发规范与任务分解：21 个 task、4 个集成门、单测验收制、开发机/节点分工 |

## 组件对照表（PoC ↔ 真实系统）

| rl_sim 模块 | slime / 真实系统 | 说明 |
|---|---|---|
| `train.py` | `train.py`（slime driver） | 同步五步主循环 |
| `train_async.py` | `train_async.py` | one-step lookahead + 权重同步间隔 + staleness 丢弃 |
| `rl_sim/types.py` | `slime/utils/types.py` | Sample / SampleStatus |
| `rl_sim/data_source.py` | `slime/rollout/data_source.py` | epoch 组采样、held-out split |
| `rl_sim/engine_local.py` | `slime/backends/sglang_utils/sglang_engine.py` | llama-server 多实例 = rollout DP ranks；**真实 logprob** |
| `rl_sim/router.py` | SGLang router / GLM TITO gateway | rendezvous session-affinity、failover |
| `rl_sim/tokenizer.py` | TITO tokenize 环节 | 引擎 `/tokenize` 端点（server-side BPE） |
| `rl_sim/sandbox.py` | **DeepSeek DSec** / GLM SWE 环境 | prlimit 限额 + `unshare -n` 禁网 + nobody 降权；启动/CPU/RSS/IO 全记录 |
| `rl_sim/reward.py` | `slime/rollout/rm_hub` | rule-based：pass rate / numeric / exact |
| `rl_sim/rollout_manager.py` | `slime/ray/rollout.py` | asyncio 有界队列、**组屏障**、abort/requeue、GRPO advantage |
| `rl_sim/trainer.py` | `slime/backends/megatron_utils/` | mock GPU：真实 GRPO/clip/TIS 标量，模拟耗时，version+1 |
| `rl_sim/weight_sync.py` | `update_weight/UpdateWeightFromDistributed` | pause→transfer→resume 时序（不声称 NCCL 保真） |
| `rl_sim/monitor.py` | RolloutHealthMonitor + 日志体系 | wall/CPU 双列、百分位、全字段 rollout 报告 |

## 快速开始

```bash
# 无模型冒烟（任意机器，零依赖）
python train.py --engine mock --num-rollout 5

# 节点真实闭环（需 llama.cpp + GGUF，见 docs/03 M0）
python train.py --engine local --engine-instances 2 --engine-slots 4 \
    --rollout-batch-size 2 --n-samples-per-prompt 2 --sandbox-workers 4

# 异步对照
python train_async.py --engine mock --update-weights-interval 2 --max-staleness 0

# 实验
python run_experiments.py --exp E0            # 共享机基线噪声
python run_experiments.py --exp E1 --workers 4,32,128 --n-tasks 256   # sandbox 扫描
python run_experiments.py --exp E4            # sync vs async

# 测试（验收标准）
python -m unittest discover -s tests
```

## 已知限制

- llama.cpp 无 prompt_logprobs 端点 → trainer 侧 rescore 用**序列级（GSPO 风格）**重要性比率，非 per-token TIS；
- Qwen3 默认 thinking 已禁用（`chat_template_kwargs.enable_thinking=false`）；
- 权重同步只模拟时序与 reload 窗口，不声称 NCCL/RDMA 保真；
- L1/L2 多轮 agentic 闭环在 T20 集成；V1/V2 需 VLM 引擎（Qwen2.5-VL-3B）。
