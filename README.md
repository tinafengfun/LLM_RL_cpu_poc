# LLM_RL_cpu_poc

在无 GPU、只有模型 API 的 CPU 节点上，模拟 RL 后训练（post-training）全流程的 PoC。

以 **slime**（THUDM，GLM-5.2 同款后训练栈）为蓝本，对照 DeepSeek V4 / Miles 的 RL 系统设计：rollout 前向（GPU）替换为模型 API，training 反向（GPU）mock，其余全部 CPU 环节真实实现并可观测——重点是 **sandbox 负载画像**与 **CPU 侧调度/数据处理**。

## 文档（审查用）

| 文档 | 内容 |
|---|---|
| [docs/01_research_report.md](docs/01_research_report.md) | 调研报告：DeepSeek V4 后训练（GRPO→OPD）、GLM-5.2 RL（GRPO+IcePop / SAO）、开源框架格局、slime 架构详解 |
| [docs/02_poc_design.md](docs/02_poc_design.md) | PoC 设计：task 选择（代码生成+单测 sandbox）、CPU/GPU 分工表、模块设计（逐一对照 slime 源码）、验证标准 |
| [docs/03_execution_plan.md](docs/03_execution_plan.md) | 执行计划：目标节点（10.239.23.91）环境、/workspace 磁盘选型、里程碑 M0–M6、验证方案、风险对策 |

当前状态：**文档审查阶段，代码尚未实现**（审查通过后按 03 计划执行）。
