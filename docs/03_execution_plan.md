# 执行计划：rl_sim PoC 落地步骤

> 状态：v3.1（修订记录见文末 §8）。配套：`01_research_report.md`、`02_poc_design.md`（v3.1）。
> v3.1 变更：任务集真实场景化（两大真实 RL 场景），M0/M1/M3 相应更新。
> v3 变更：rollout 引擎改为本地 llama.cpp + Qwen3-4B（首选），M0 新增引擎构建与模型分发，M3 重写。

## 1. 目标环境（已摸底，2026-09-10）

- 节点：`10.239.23.91`（hostname `sh12156002s3013`），root 访问，**已配置免密 SSH**；
- 硬件：**256 核 Xeon 6767P（Granite Rapids，AVX512+AMX）/ 705GB RAM**（共享机器，摸底时已用 276G——实验需 E0 基线噪声控制，见设计 §6）；
- 软件：Python 3.12.12、git 2.52、**g++/cmake/make 已装（llama.cpp 可节点本地编译）**、glibc 2.39、CentOS Stream 10；
- 磁盘选型（`df -hT` 实测）：

| 挂载点 | 容量 | 可用 | 使用率 | 结论 |
|---|---|---|---|---|
| `/` (cs-root) | 69G | 14G | 81% | 避免 |
| `/home` | 1.8T | 254G | 86% | 避免 |
| `/mnt/nvme0` | 1.8T | **1.8T** | **2%** | ✅ **选作 /workspace、sandbox scratch、模型目录** |
| `/mnt/nvme2` | 3.5T | 2.2T | 40% | 备选（有其他数据） |
| `/mnt/nvme3` | 1.8T | 361G | 80% | 避免 |

## 2. 环境搭建步骤（M0）

```bash
ssh root@10.239.23.91
mkdir -p /mnt/nvme0/workspace /mnt/nvme0/sandbox_scratch /mnt/nvme0/models
ln -sfn /mnt/nvme0/workspace /workspace
cd /workspace && git clone https://github.com/tinafengfun/LLM_RL_cpu_poc.git
cd LLM_RL_cpu_poc && python3 -m venv .venv && source .venv/bin/activate

# 基础能力检查：
unshare -n true && echo "userns OK"          # sandbox 禁网依赖；失败则走降级方案(设计 §5.1)
id nobody                                     # 降权用户存在性
taskset -c 0-3 python3 -c 'pass'              # E0 核绑定可用性

# v3 新增：llama.cpp 构建与模型分发（节点无外网，全部 vendor）
# 本地（有网机）：
git clone --depth 1 https://github.com/ggml-org/llama.cpp && tar czf llama.cpp.tar.gz llama.cpp
#   下载 Qwen3-4B-Instruct-2507 Q4_K_M + Q8 GGUF、Qwen3-1.7B Q4_K_M、
#   Qwen2.5-VL-3B-Instruct Q4_K_M GGUF + mmproj（多模态引擎，v3.1 新增）（HuggingFace/ModelScope）
scp llama.cpp.tar.gz *.gguf root@10.239.23.91:/mnt/nvme0/models/
# 节点：
cd /mnt/nvme0/models && tar xzf llama.cpp.tar.gz && cd llama.cpp
cmake -B build -DGGML_NATIVE=ON && cmake --build build --config Release -j 64   # AVX512/AMX 原生优化
./build/bin/llama-server --model /mnt/nvme0/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
    --port 8080 --parallel 8 -t 16 --jinja   # 冒烟：logprobs + tool calling + slots
```

- **预烘焙 venv**（B 档依赖）：构建机制作含 pytest/numpy 的 venv 打包 scp 至 `/mnt/nvme0/workspace/vendor/`，sandbox 内只读引用；
- **tokenizer vocab**：随 GGUF/仓库 vendor（llama.cpp 自带 BPE，driver 侧计时用它）；
- **E0 基线**：空载采集 10min 节点 CPU/IO 背景负载存档 `results/baseline/`。

验收：`/workspace` 就绪；unshare/taskset 有结论；**llama-server 冒烟通过（logprobs 输出 + tool calling + 多 slot 并发）**；基线数据落盘。

## 3. 里程碑（v3）

| 里程碑 | 内容 | 验收标准 | 预估 |
|---|---|---|---|
| **M0** | 节点环境 + **llama.cpp 编译 + GGUF 分发 + 引擎冒烟** + 预烘焙 venv + E0 基线（§2） | 验收项全过 | 3h |
| **M1** | `types.py` / `data_source.py`（**两大真实场景任务集：L1 SWE-mini 仓库 / L2 terminal / L3 多跳问答 / V1-V2 多模态题 + 预渲染图片 vendor** + 故障注入 + mock 备选）/ `reward.py` | 各任务线按组出 Sample；reward 对构造正/错答案评分正确 | 4h |
| **M2** | `sandbox.py`：unshare 禁网 + 降权 + 分档资源 profile + 指标收集（启动耗时/ru_*/RSS/IO） | 隔离用例全过：死循环→timeout、爆内存→oom、语法错→error、**socket 连接失败验证禁网** | 4h |
| **M3** | `engine.py`（**LocalEngine：llama-server 多实例管理/心跳/绑核/slots，文本 Qwen3-4B + 多模态 Qwen2.5-VL-3B 双引擎** + APIEngine + MockEngine）+ `router.py`（session-affinity）+ `tokenizer.py` | 多实例拉起与心跳正常；logprob 真实返回；图像输入通路正常；tool calling 多轮可驱动 L1/L2 任务；router 转发与延迟可测 | 4h |
| **M4** | `rollout_manager.py`（asyncio 有界队列 + 组屏障 + abort/requeue）+ `trainer.py`（**真实 logprob 输入的 TIS**）+ `weight_sync.py`（**真实 GGUF reload**） | 组 advantage/clip/TIS 数值对拍正确；>1k 并发到达下队列积压可见；abort 半成品回 buffer 可追；reload 耗时可测 | 4h |
| **M5** | `train.py`（sync 基线）+ `train_async.py` + `monitor.py`（双列计时 + driver CPU% + IO + **引擎画像**）+ 实验 E1–E4、E6 | 设计 §7 DoD 全项通过 | 5h |
| **M6** | README（对照表 + CPU 分工讲解 + 实验报告）+ 结果归档 push | 终审通过 | 3h |

总计约 3.5 个工作日。**M2/M3/M4/M5 是核心价值**，优先保证质量。

## 4. 验证方案（对应设计 §7 DoD）

```bash
# V1 离线闭环（同步基线 + 异步对照）
taskset -c 0-63 python train.py --engine local --num-rollout 10 --rollout-batch-size 16 \
    --n-samples-per-prompt 8 --sandbox-workers 32 --engine-instances 4 --engine-slots 8
taskset -c 0-63 python train_async.py --engine local ... --update-weights-interval 2

# V2 sandbox 隔离（含禁网验证）
python -m pytest tests/ -v   # 死循环/爆内存/语法错/禁网/三档资源 profile

# V3 实验扫描（E1/E3/E4/E6 曲线数据落盘 results/）
python run_experiments.py --exp E1 --workers 4,32,128
python run_experiments.py --exp E3 --infer-concurrency 1,8,64
python run_experiments.py --exp E4   # sync vs async
python run_experiments.py --exp E6 --engine-instances 1,2,4 --slots 4,8,16   # 引擎画像 + 核争抢

# V4 远程 API（备选引擎，可选）
export RL_SIM_API_BASE=... RL_SIM_API_KEY=... RL_SIM_API_MODEL=...
python train.py --engine api --num-rollout 2

# V5 MockEngine 冒烟（无模型文件也能跑通流程）
python train.py --engine mock --num-rollout 2
```

## 5. 运行规模建议（256 核共享节点）

- 实验核集：taskset 固定 64 核（如 0-63），避开其他租户抖动，报告标注核集与环境负载；
- **核分配是三方的**：推理实例（llama-server 每实例 8–16 核）+ sandbox 池 + driver/router，E6 专门扫描配比，再现 colocate 核争抢；
- `--sandbox-workers 4/32/128` 三档扫描：故意制造队列积压（E1 核心曲线）；
- 模拟训练耗时系数可配：演示不同 CPU/GPU 速度配比下的瓶颈迁移（对应真实系统 rollout 占 80–90% 时间的结论）；
- sandbox scratch 默认 `/mnt/nvme0/sandbox_scratch`，E5 可选对比 tmpfs。

## 6. 风险与对策（v3）

| 风险 | 对策 |
|---|---|
| **共享机噪声污染画像** | E0 基线先行；taskset 固定核集；报告标注背景负载；关键实验跑 2 次取一致结果 |
| **llama.cpp 编译失败/AMX 路径异常** | 先 `-DGGML_NATIVE=ON` 默认构建；失败退 `-DGGML_AVX512=ON` 保守路径；M0 冒烟卡住则用预编译 release 二进制兜底 |
| **模型下载体积/渠道** | Q4_K_M 约 2.5GB + Q8 约 4.5GB + 1.7B 约 1.2GB，本机 HF/ModelScope 下载后 scp；下载失败用 ModelScope 镜像 |
| **小模型 tool calling 质量不足** | C 档任务 scaffold 做宽松解析（grammar 约束 + 重试 1 次）；仍不足则 C 档降级为脚本化多轮（控制流不变） |
| **推理与 sandbox 抢核导致画像混杂** | taskset 分区固定；E6 单独扫描；这正是要观测的现象，记录而非消除 |
| unshare/userns 被内核禁用 | M0 先检查；降级为纯 rlimits + 无网络任务集，报告标注隔离减弱 |
| 节点无外网 / GitHub 不可达 | 本地 `git bundle` scp；llama.cpp 源码/GGUF/venv 全部 vendor，零运行时下载 |
| 预烘焙 venv 体积/兼容性 | 构建机与节点同 glibc 大版本下制作；<500MB；M0 验证 pytest 可跑 |
| 模型 API 未提供或不稳定 | LocalEngine 为主不依赖远程 API；APIEngine 仅备选 |
| 节点共享负载挤占 | 避开高峰；sandbox 池上限 128；运行前后检查负载；产物全部放 /mnt/nvme0 |
| 对话中出现过的 GitHub token 泄露风险 | **推送后立即轮换 token**；token 不写入任何仓库文件 |

## 7. 审查检查点（本次请求终审的点）

1. v3 引擎选型是否同意：llama.cpp + Qwen3-4B 为首选（设计 §5.3）；
2. v2 复审意见吸收是否完整（设计 §10 v2 记录逐条可溯）；
3. 三档任务集与 unshare 禁网方案是否同意（设计 §2、§5.1）；
4. 里程碑增量（M0 引擎构建、M3 重写、工期 3.5 天）是否同意；
5. 确认后按 M0→M6 执行；如需调整（如砍掉 E5、换模型档位），在终审意见中说明。

## 8. 修订记录

**v3.1（2026-09-15）**：任务集真实场景化（配套设计 v3.1）：

1. M1 重写：任务集改两大真实场景（L1 SWE-mini 仓库 / L2 terminal / L3 多跳问答 / V1-V2 多模态题 + 预渲染图片 vendor + mock 备选），预估 3h→4h；
2. M0 模型分发新增 Qwen2.5-VL-3B-Instruct GGUF + mmproj（多模态引擎）；
3. M3 新增 VLM 引擎实例管理（与文本引擎并存）。

**v3（2026-09-10）**：rollout 引擎本地化（用户决策）：

1. M0 新增：llama.cpp 源码 vendor + 节点本地编译（AVX512/AMX）+ GGUF 模型分发（Q4 主推理 / Q8 scoring / 1.7B 高并发备选）+ llama-server 冒烟验收；节点确认自带 g++/cmake/make；
2. M3 重写：LocalEngine（llama-server 多实例管理/心跳/绑核/slots）为首选引擎，APIEngine 降备选，MockEngine 留冒烟；router 加 session-affinity；
3. M4：TIS 输入改真实 logprob；weight_sync 改真实 GGUF reload；
4. 验证方案 V1/V3 改 `--engine local`，新增 V5 MockEngine 冒烟；实验新增 E6 引擎画像与核争抢扫描；
5. 风险表新增：编译失败兜底、模型下载渠道、tool calling 质量、推理与 sandbox 抢核；
6. 工期从 3 天调整为 3.5 天。

**v2（2026-09-10）**：配套设计文档 v2 修订：

1. M0 新增：unshare/taskset 可用性检查、预烘焙 venv 制作、tokenizer vocab vendor、**E0 基线噪声采集**（共享机画像前提）；
2. M1 任务集改三档 + 故障注入；M2 改 unshare 禁网 + 分档资源 profile + IO 指标；M3 新增 router 与真 BPE tokenizer；M4 改 asyncio 有界队列 + 组屏障 + abort/requeue；M5 新增 train_async 对照与 E1–E4 实验，monitor 改双列计时 + driver CPU%；
3. 工期从 2 天调整为 3 天；
4. 风险表新增共享机噪声、userns 禁用、venv 兼容性三项；
5. 验证方案改为 taskset 固定核 + sync/async 对照 + 实验扫描脚本。
