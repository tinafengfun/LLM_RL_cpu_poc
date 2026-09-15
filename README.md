# LLM_RL_cpu_poc

在无 GPU 的 CPU 节点上，研究 RL 后训练（post-training）的**控制流**与 **CPU 负载画像**。

以 **slime**（THUDM，GLM-5.2 同款后训练栈）为蓝本，对照 DeepSeek V4 / Miles：rollout 前向用 **llama.cpp + Qwen3-4B 本地 CPU 推理**（logprob / KV cache / tool calling 全真实，远程模型 API 为备选），training 反向（GPU）mock，其余全部 CPU 环节真实实现并可观测——重点是 **agentic RL sandbox 的驱动方式、workload 与资源特征**。

设计原则：**闭环假可以接受，负载失真不能接受。**（假 reward 数值 + 真控制流 + 真实 CPU 负载测量）

## 文档（终审查用）

| 文档 | 内容 |
|---|---|
| [docs/01_research_report.md](docs/01_research_report.md) | 调研报告：DeepSeek V4 后训练（GRPO→OPD）、GLM-5.2 RL（GRPO+IcePop / SAO）、开源框架格局、slime 架构详解 |
| [docs/02_poc_design.md](docs/02_poc_design.md) | PoC 设计 v2：三档 sandbox workload、unshare 禁网隔离、asyncio 到达过程、wall/CPU 双列计时、E0–E5 扫参实验、验证标准（§10 附复审修订记录） |
| [docs/03_execution_plan.md](docs/03_execution_plan.md) | 执行计划 v2：目标节点（10.239.23.91）环境、/workspace 磁盘选型、里程碑 M0–M6、验证方案、风险对策（§8 附修订记录） |

当前状态：**v2 文档已合并复审意见，待终审**。终审通过后按 03 计划 M0→M6 执行。
