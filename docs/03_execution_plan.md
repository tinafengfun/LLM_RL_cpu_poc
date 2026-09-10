# 执行计划：rl_sim PoC 落地步骤

> 状态：v2，复审修订版（修订记录见文末 §8）。配套：`01_research_report.md`、`02_poc_design.md`（v2）。

## 1. 目标环境（已摸底，2026-09-10）

- 节点：`10.239.23.91`（hostname `sh12156002s3013`），root 访问，**已配置免密 SSH**（本机 `~/.ssh/id_ed25519` → 节点 `authorized_keys`）；
- 硬件：**256 核 CPU / 705GB RAM**（注意：**共享机器**，摸底时已用 276G 内存——实验需 E0 基线噪声控制，见设计 §6）；
- 软件：Python 3.12.12（本 PoC 纯 stdlib，不需要 torch）、git 2.52；
- 磁盘选型（`df -hT` 实测）：

| 挂载点 | 容量 | 可用 | 使用率 | 结论 |
|---|---|---|---|---|
| `/` (cs-root) | 69G | 14G | 81% | 避免 |
| `/home` | 1.8T | 254G | 86% | 避免 |
| `/mnt/nvme0` | 1.8T | **1.8T** | **2%** | ✅ **选作 /workspace 与 sandbox scratch** |
| `/mnt/nvme2` | 3.5T | 2.2T | 40% | 备选（有其他数据） |
| `/mnt/nvme3` | 1.8T | 361G | 80% | 避免 |

## 2. 环境搭建步骤（M0）

```bash
ssh root@10.239.23.91
mkdir -p /mnt/nvme0/workspace /mnt/nvme0/sandbox_scratch
ln -sfn /mnt/nvme0/workspace /workspace
cd /workspace && git clone https://github.com/tinafengfun/LLM_RL_cpu_poc.git
cd LLM_RL_cpu_poc && python3 -m venv .venv && source .venv/bin/activate

# v2 新增检查项：
unshare -n true && echo "userns OK"          # sandbox 禁网依赖；失败则走降级方案(设计 §5.1)
id nobody                                     # 降权用户存在性
taskset -c 0-3 python3 -c 'pass'              # E0 核绑定可用性
python3 -m pytest --version 2>/dev/null || echo "pytest 需 vendor"   # B 档预烘焙 venv 用
```

- **预烘焙 venv**（B 档依赖）：在构建机（有网）制作含 pytest/numpy 的 venv 打包 scp 至 `/mnt/nvme0/workspace/vendor/`，sandbox 内只读引用；
- **tokenizer vocab**：vendor 进仓库 `vendor/tokenizer/`（节点无外网，不可现下）；
- **E0 基线**：空载采集 10min 节点 CPU/IO 背景负载存档 `results/baseline/`。

验收：`/workspace` 指向 nvme0 且可写；unshare/taskset 检查有结论；仓库可拉取；基线数据落盘。

## 3. 里程碑（v2）

| 里程碑 | 内容 | 验收标准 | 预估 |
|---|---|---|---|
| **M0** | 节点环境 + 预烘焙 venv + tokenizer vendor + E0 基线（§2） | 验收项全过 | 2h |
| **M1** | `types.py` / `data_source.py`（**A/B/C 三档任务集 + 故障注入用例**）/ `reward.py` | 三档任务按组出 Sample；reward 对构造正/错代码评分正确 | 3h |
| **M2** | `sandbox.py`：unshare 禁网 + 降权 + **分档资源 profile** + 指标收集（启动耗时/ru_*/RSS/IO） | 隔离用例全过：死循环→timeout、爆内存→oom、语法错→error、**socket 连接失败验证禁网**；指标完整 | 4h |
| **M3** | `router.py` + `engine.py`（Mock/API 均经 router）+ `tokenizer.py`（真 BPE） | router 转发与延迟可测；MockEngine 离线生成；APIEngine 连通真实 API；tokenize/detokenize 计时接入 monitor | 3h |
| **M4** | `rollout_manager.py`（**asyncio 有界队列 + 组屏障 + abort/requeue**）+ `trainer.py` + `weight_sync.py` | 组 advantage/clip/TIS 数值对拍正确；>1k 并发到达下队列积压可见；abort 半成品回 buffer 可追 | 4h |
| **M5** | `train.py`（sync 基线）+ `train_async.py` + `monitor.py`（**wall/CPU 双列 + driver CPU% + IO**）+ 实验 E1–E4 | 设计 §7 DoD 全项通过 | 5h |
| **M6** | README（对照表 + CPU 分工讲解 + **实验报告**）+ 结果归档 push | 终审通过 | 3h |

总计约 3 个工作日（比 v1 多 1 天，主要加在三档任务、asyncio 流水线与实验）。**M2/M4/M5 是核心价值**，优先保证质量。

## 4. 验证方案（对应设计 §7 DoD）

```bash
# V1 离线闭环（同步基线 + 异步对照）
taskset -c 0-63 python train.py --engine mock --num-rollout 10 --rollout-batch-size 16 \
    --n-samples-per-prompt 8 --sandbox-workers 32
taskset -c 0-63 python train_async.py --engine mock --num-rollout 10 ... --update-weights-interval 2

# V2 sandbox 隔离（含禁网验证）
python -m pytest tests/ -v   # 死循环/爆内存/语法错/禁网/三档资源 profile

# V3 实验扫描（E1/E3/E4 曲线数据落盘 results/）
python run_experiments.py --exp E1 --workers 4,32,128
python run_experiments.py --exp E3 --api-concurrency 1,8,64
python run_experiments.py --exp E4   # sync vs async

# V4 真实 API（可选，需 env）
export RL_SIM_API_BASE=... RL_SIM_API_KEY=... RL_SIM_API_MODEL=...
python train.py --engine api --num-rollout 2
```

## 5. 运行规模建议（256 核共享节点）

- 实验核集：taskset 固定 64 核（如 0-63），避开其他租户抖动，报告标注核集与环境负载；
- `--sandbox-workers 4/32/128` 三档扫描：故意制造队列积压以观测饱和行为（E1 核心曲线）；
- 模拟训练耗时系数、生成延迟可配：演示不同 CPU/GPU 速度配比下的瓶颈迁移（对应真实系统 rollout 占 80–90% 时间的结论）；
- sandbox scratch 默认 `/mnt/nvme0/sandbox_scratch`，E5 可选对比 tmpfs。

## 6. 风险与对策（v2）

| 风险 | 对策 |
|---|---|
| **共享机噪声污染画像** | E0 基线先行；taskset 固定核集；报告标注背景负载；关键实验跑 2 次取一致结果 |
| unshare/userns 被内核禁用 | M0 先检查；降级为纯 rlimits + 无网络任务集，报告标注隔离减弱（设计 §5.1） |
| 节点无外网 / GitHub 不可达 | 本地 `git bundle` scp；tokenizer vocab 与预烘焙 venv 全部 vendor，零运行时下载 |
| 预烘焙 venv 体积/兼容性 | 构建机与节点同 glibc 大版本下制作；体积控制在 <500MB；M0 验证 pytest 可跑 |
| RLIMIT_AS 与解释器启动冲突 | 分档 profile（A 512MB / B 4GB）；B 档 M2 单独验证 pytest+numpy 起得来 |
| 模型 API 未提供或不稳定 | MockEngine 经 router 完成全部验证；APIEngine 重试+超时 |
| 节点共享负载挤占 | 避开高峰；sandbox 池上限 128；运行前后检查负载；产物全部放 /mnt/nvme0 |
| 对话中出现过的 GitHub token 泄露风险 | **推送后立即轮换 token**；token 不写入任何仓库文件 |

## 7. 审查检查点（本次请求终审的点）

1. v2 修订是否完整吸收复审意见（设计 §10 修订记录逐条可溯）；
2. 三档任务集与 unshare 禁网方案是否同意（设计 §2、§5.1）；
3. 里程碑增量（M0 基线/vendor、M4 asyncio 流水线、M5 实验）与 3 天工期是否同意；
4. 确认后按 M0→M6 执行；如需调整（如砍掉 E5、压缩任务集规模），在终审意见中说明。

## 8. 修订记录

**v2（2026-09-10）**：配套设计文档 v2 修订：

1. M0 新增：unshare/taskset 可用性检查、预烘焙 venv 制作、tokenizer vocab vendor、**E0 基线噪声采集**（共享机画像前提）；
2. M1 任务集改三档 + 故障注入；M2 改 unshare 禁网 + 分档资源 profile + IO 指标；M3 新增 router 与真 BPE tokenizer；M4 改 asyncio 有界队列 + 组屏障 + abort/requeue；M5 新增 train_async 对照与 E1–E4 实验，monitor 改双列计时 + driver CPU%；
3. 工期从 2 天调整为 3 天；
4. 风险表新增共享机噪声、userns 禁用、venv 兼容性三项；
5. 验证方案改为 taskset 固定核 + sync/async 对照 + 实验扫描脚本。
