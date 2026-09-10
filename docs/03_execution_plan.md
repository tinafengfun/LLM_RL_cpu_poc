# 执行计划：rl_sim PoC 落地步骤

> 状态：待审查。审查通过后再开始执行。配套：`01_research_report.md`、`02_poc_design.md`。

## 1. 目标环境（已摸底，2026-09-10）

- 节点：`10.239.23.91`（hostname `sh12156002s3013`），root 访问，**已配置免密 SSH**（本机 `~/.ssh/id_ed25519` → 节点 `authorized_keys`）；
- 硬件：**256 核 CPU / 705GB RAM**（sandbox 池可远大于默认 8）；
- 软件：Python 3.12.12（无 torch，本 PoC 纯 stdlib 不需要）、git 2.52；
- 磁盘选型（`df -hT` 实测）：

| 挂载点 | 容量 | 可用 | 使用率 | 结论 |
|---|---|---|---|---|
| `/` (cs-root) | 69G | 14G | 81% | 避免 |
| `/home` | 1.8T | 254G | 86% | 避免 |
| `/mnt/nvme0` | 1.8T | **1.8T** | **2%** | ✅ **选作 /workspace** |
| `/mnt/nvme2` | 3.5T | 2.2T | 40% | 备选（有其他数据） |
| `/mnt/nvme3` | 1.8T | 361G | 80% | 避免 |

## 2. 环境搭建步骤（M0）

```bash
ssh root@10.239.23.91
mkdir -p /mnt/nvme0/workspace
ln -sfn /mnt/nvme0/workspace /workspace
cd /workspace && git clone https://github.com/tinafengfun/LLM_RL_cpu_poc.git
cd LLM_RL_cpu_poc && python3 -m venv .venv && source .venv/bin/activate
# 无第三方依赖；APIEngine 仅用 stdlib urllib
```

验收：`/workspace` 指向 nvme0 且可写；`python3 --version` = 3.12+；仓库可拉取。

## 3. 里程碑

| 里程碑 | 内容 | 验收标准 | 预估 |
|---|---|---|---|
| **M0** | 节点环境搭建（§2） | `/workspace` 就绪，仓库 clone 成功 | 0.5h |
| **M1** | 数据与骨架：`types.py` / `data_source.py`（20 题）/ `reward.py` | 数据源按组出 Sample；reward 对构造的正/错代码评分正确 | 2h |
| **M2** | `sandbox.py` SandboxPool + 隔离用例 | 死循环→timeout、爆内存→oom、语法错→error，全部按预期 kill 且无逃逸；指标收集完整 | 3h |
| **M3** | `engine.py`：MockEngine + APIEngine | MockEngine 离线生成（正确率随版本上升）；APIEngine 连通真实 API 并提取代码 | 2h |
| **M4** | `rollout_manager.py` + `trainer.py` + `weight_sync.py` | 组 advantage / clip / TIS 数值正确（对拍手算用例）；版本同步时序正确 | 3h |
| **M5** | `monitor.py` + `train.py` 主循环 + 端到端验证 | 设计文档 §6 的 DoD 全项通过 | 3h |
| **M6** | README（组件对照表 + CPU 分工讲解 + 扩展路径）+ 结果归档 | 审查通过，push 回 GitHub | 2h |

总计约 2 个工作日。M2/M4 是最有价值的部分（sandbox 负载、RL 数据流），优先保证质量。

## 4. 验证方案（对应设计文档 §6 DoD）

```bash
# V1 离线闭环（主验证）
python train.py --engine mock --num-rollout 10 --rollout-batch-size 16 \
    --n-samples-per-prompt 8 --sandbox-workers 32
# 期望：mean_reward 随 rollout 上升；每步输出 CPU 阶段分解 + sandbox 负载报告

# V2 sandbox 隔离
python -m pytest tests/test_sandbox.py   # 死循环/爆内存/语法错/正常 用例

# V3 真实 API（需配置 env）
export RL_SIM_API_BASE=... RL_SIM_API_KEY=... RL_SIM_API_MODEL=...
python train.py --engine api --num-rollout 2
```

## 5. 运行规模建议（256 核节点）

- `--sandbox-workers 32~64`：故意制造队列积压以观测 sandbox 饱和行为（利用率、queue depth、p99）；
- 可对比实验：sandbox-workers = 4 / 32 / 128 三档，画"sandbox 并发度 vs rollout 吞吐 vs 队列等待"曲线——直接回答"sandbox 负载情况"；
- 模拟训练耗时系数、生成延迟均可配，便于演示不同 CPU/GPU 速度配比下的瓶颈迁移（对应真实系统 rollout 占 80–90% 时间的结论）。

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 节点无外网 / GitHub 不可达 | 本地打包 `git bundle` scp 过去；pip 不依赖外网（零依赖） |
| 模型 API 未提供或不稳定 | MockEngine 兜底完成全部验证；APIEngine 留好重试与超时 |
| RLIMIT_AS 影响 Python 解释器启动 | 512MB 已验证充裕；如异常上调至 1GB |
| 节点是共享机器（705G 内存已用 276G） | 限制 sandbox 池规模；运行前后检查负载；产物全部放 /workspace |
| 对话中出现过的 GitHub token 泄露风险 | **推送后立即轮换 token**；token 不写入任何仓库文件 |

## 7. 审查检查点（本次请求审查的点）

1. 调研报告结论是否准确（§01）；
2. PoC 范围是否合适：task 选择、sandbox 设计、mock 边界（§02）；
3. 节点与磁盘选型、里程碑划分是否同意（§03）；
4. 确认后按 M0→M6 执行；如需调整范围（如加入异步/SAO 第一版就做），在审查意见中说明。
