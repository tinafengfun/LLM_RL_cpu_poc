# 09 双节点部署设计：node1 训练侧 + node2 SWE-bench sandbox 专用节点

> 2026-10-18。在单节点 PoC（T01–T21 完成）基础上，把 sandbox 层拆到独立节点，逼近 slime `coding_agent_rl` 例子的真实部署形态（GPU 节点群 + 独立 sandbox 节点群）。
> 结构：§1 背景与目标 → §2 决策记录 → §3 节点档案 → §4 总体架构 → §5 新组件设计 → §6 SWE-bench 任务线 → §7 失真声明 → §8 与真实系统对照 → §9 开发任务拆分 → §10 验收标准 → §11 修订记录。

## 1. 背景与目标

单节点 PoC 里 sandbox 是 node1 本机 subprocess（prlimit + `unshare -n` + setpriv），**RL 框架与 sandbox 之间没有网络边界**。真实系统（slime coding_agent_rl 例子、DeepSeek DSec）里这是一个跨节点服务边界：

- slime `coding_agent_rl`：6 GPU 节点（slime+Megatron+SGLang，进 Ray）+ 2 sandbox 节点（Sunabako，**不进 Ray**，SSH 访问）；agent CLI 装在 sandbox 内，回调 GPU 节点的 adapter HTTP 拿模型生成。
- DSec（DeepSeek，arXiv 2609.22978）：训练框架经 libdsec SDK → apiserver → edge 调 sandbox；agent loop 与 agent sandbox 都在 GPU 池之外。

**目标**：node1 部署 rl_sim 全套（slime 模拟控制面 + 所有模拟 GPU 的部分），node2 只部署 sandbox 与 agent 执行，task 换为 SWE-bench。观测对象不变——CPU 负载画像，但边界真实化：跨节点 RPC 的开销、两侧 CPU 分工、突发到达在独立 sandbox 节点上的积压形态，全部可测。

**非目标**：不追求 SWE-bench 官方复现精度（无 docker、无 per-instance conda）；不跑真 Codex/Claude Code CLI；不改造 rl_sim 已有单节点模式（保留为回归基线）。

## 2. 决策记录

| # | 决策点 | 选项 | 结论 | 理由 |
|---|---|---|---|---|
| D1 | node1→node2 调用路径 | HTTP sandboxd / 纯 SSH / 两者 | **HTTP sandboxd** | 1 RTT/次调用、监控集中（DSec edge 角色）、可做池化/TTL/准入；纯 SSH 留作二期对照 |
| D2 | agent loop 位置 | node1 / **node2** | **node2** | 贴 slime 例子真实形态：agent 在 sandbox 节点，模型调用跨网回调 node1 adapter；node2 CPU 画像包含 agent 进程本身（真实开销项） |
| D3 | SWE-bench 起步规模 | **1 repo** / 3-5 repo | **先 sympy 1 个 repo 端到端** | 纯 Python、环境最简单、pytest 直接可跑；验收后再扩 django/sphinx |
| D4 | node1 的 "slime" | **rl_sim** / 真 slime 仓 | **rl_sim** | 真 slime 需要 GPU 跑不起来；如需代码对照单独 checkout，不进入运行路径 |

D2 选了 slime 忠实方案，关键含义：**跨网流量从"tool call"翻转为"模型调用"**——每次 LLM 生成 = 1 次 node2→node1 RPC（24ms RTT），sandbox 内命令执行留在 node2 本地。这正是生产系统的真实数据流方向。

## 3. 节点档案（实测）

| | node1（训练侧） | node2（sandbox 侧） |
|---|---|---|
| IP | 10.239.23.91 | 172.16.116.170 |
| CPU | 256C Xeon 6767P（GNR，AVX512/AMX） | 128C Xeon 8352Y（ICX，2.2GHz） |
| 内存 | 705GB | 1007GB |
| OS | CentOS Stream 10 | openEuler 24.03（内核 6.6） |
| Python | 3.12 | 3.11 |
| 隔离能力 | unshare/prlimit/setpriv 已验证 | `unshare -n` 已验证可用，cgroup v2 |
| docker | 无 | 无（→ 不走 OCI 镜像路线） |
| 外网 | 可拉 GitHub | **github/pypi/HF 全通**（数据可直接下载） |
| 工作盘 | /mnt/nvme0/workspace | 根盘余 289G，用 /workspace |
| 节点间 RTT | **~24ms**（跨机房级，生产同机房 <1ms，失真但放大 RPC 可见性） | ←同左 |
| 共享纪律 | 共享机：taskset 固核 + E0 基线 | 共享机（/home 有 873G 他人数据）：同纪律 |

双向连通性：node1→node2 ping 通；**node2→node1 方向为 T22 预检项**（agent 回调 adapter 依赖此方向）。

## 4. 总体架构

```
node1 = 10.239.23.91（slime 模拟 + 模拟 GPU）          node2 = 172.16.116.170（sandbox 专用）
┌──────────────────────────────────────────────┐      ┌──────────────────────────────────────────────┐
│ train_async.py（driver 主循环）               │      │ sandboxd（HTTP 守护进程，stdlib 零依赖）      │
│ rollout_manager（组屏障/abort/优势计算）       │      │  ├─ /sandbox create|exec|stop（本地执行）     │
│                                              │      │  ├─ /trajectory start|status|abort            │
│  ┌─ adapter service（新，HTTP） ◄────────────│──模型调│─ agent_runner（每轨迹一个协程/进程）        │
│  │  ├─ 转发 router → llama-server ×N         │  用 RPC│     ├─ 多轮 loop：生成→tool call→本地执行   │
│  │  ├─ 留存 token IDs + logprobs（TIS 用）   │      │     ├─ git diff 抽 patch                    │
│  │  └─ loss_mask 标记（模型输出=1）          │      │     └─ 触发 grader                          │
│  └─ router → llama-server ×N（Q4 GGUF）      │      │  ├─ grader：新鲜 sandbox 评分               │
│       = 模拟 GPU 推理（rollout DP ranks）     │      │  │   （apply test_patch + FAIL_TO_PASS）    │
│ MockMegatronTrainer（GPU 替身）              │      │  └─ monitor agent（CPU/IO/队列画像）         │
│ WeightUpdater（GGUF reload 计时）            │      │                                              │
│ monitor（node1 侧 + 拉取 node2 汇总）        │      │ 数据：/workspace/swebench/                   │
└──────────────────────────────────────────────┘      │  ├─ instances.jsonl（sympy 子集）            │
        ▲ 轨迹下发（start）/ 状态轮询（status）│      │  ├─ envs/<repo>.tar.zst（预烘焙 venv+快照）  │
        └─────────────────────────────────────│──────│  └─ runs/<traj_id>/（工作区+结果）           │
                                              └──────────────────────────────────────────────┘
```

数据流（一次轨迹）：

1. node1 rollout_manager 展开组采样 → `POST node2/trajectory/start {instance_id, group_id, traj_id, adapter_url, budgets}`
2. node2 sandboxd 起 agent_runner：解包环境 → 建 agent sandbox → 多轮 loop（每轮 `POST node1/adapter/generate` 拿生成；tool call 走 `POST localhost/sandbox/exec` 本地执行）
3. loop 结束（模型自停/预算尽）→ `git diff` 抽 patch → **新鲜 sandbox** 评分（apply patch + test_patch，跑 FAIL_TO_PASS/PASS_TO_PASS pytest）→ reward
4. node1 轮询 `GET /trajectory/{id}/status` 收结果（reward、stats、消息日志引用）；token 级数据已在 node1 adapter 侧
5. 组齐 → advantage → mock 训练 → 权重同步，流程不变

## 5. 新组件设计

### 5.1 adapter service（node1，新）

slime 例子里 adapter 是"sandbox 内 CLI 与 SGLang 之间的桥"，留存精确 token 流。我们的实现：

- HTTP 接口 `POST /generate`：`{traj_id, messages, tools, max_tokens, sampling}` → 经 router 打到 llama-server → 返回 `{text, tool_calls, finish_reason, usage}`
- **留存**：按 traj_id 记录 input/output token IDs + per-token logprobs（llama.cpp `logprobs:true`），写 `adapter_log/<traj_id>.jsonl`；loss_mask：模型输出 token=1，observation=0
- 会话亲和沿用 router 一致性哈希（同 traj_id → 固定引擎实例，保 prefix cache）
- 不引入新依赖，stdlib `http.server` 线程模型即可（并发上限 = 轨迹数，远小于引擎 slot 压力）

### 5.2 sandboxd（node2，新）

DSec edge 的微缩对照，stdlib HTTP 守护进程：

| 端点 | 语义 |
|---|---|
| `POST /sandbox/create` | 分配 sandbox id + tmpdir，应用资源档（A/B/C profile 沿用现有定义） |
| `POST /sandbox/exec` | 在指定 sandbox 跑命令：prlimit + `unshare -n` + setpriv 降权；返回 stdout/exit/`ru_utime`/`ru_maxrss`/IO |
| `POST /sandbox/stop` | 杀进程树、回收 tmpdir |
| `POST /trajectory/start` | 起 agent_runner（异步，立即返回 traj_id） |
| `GET /trajectory/{id}/status` | running/done/aborted + 结果引用 |
| `POST /trajectory/{id}/abort` | 杀 runner + 其全部 sandbox（对应组屏障 abort） |
| `GET /stats` | 池利用率、队列深度、exec p50/p99 分档、sandbox 存活数 |

- 执行原语**复用现有 `rl_sim/sandbox.py`**（node2 经 git 部署同一份代码），sandboxd 只是把它包成服务
- 准入控制：节点级并发上限 + 内存预算 cap（对照 Sunabako 的 memory admission cap），超限拒绝 → node1 侧见排队/重试
- TTL：sandbox 闲置超时自动回收（对照 DSec ttl_running_stop）

### 5.3 agent_runner（node2，新）

自研轻量 agent loop（**不是**真 Codex/Claude Code CLI，见 §7 失真声明）：

- 输入：instance spec（repo、base_commit、problem_statement）+ adapter_url + 预算（agent/grader/whole-rollout 三级超时，默认对照 slime 例子 3600/600/7200s，PoC 可调小）
- loop：拼 system+problem_statement → adapter 生成 → 解析 tool call（bash/edit 两个工具起步）→ 本地 sandboxd exec → observation 截断拼装 → 再生成；上下文超预算做简单截断（compaction 留二期）
- 终止：模型输出 `submit` / 预算尽 / abort
- 产出：`runs/<traj_id>/` 下 patch（git diff）、消息日志、每轮耗时/等待分解、sandbox 调用统计

### 5.4 grader（node2，新）

- **新鲜 sandbox**（独立 tmpdir + 干净环境解包，对照 slime 例子"grading runs in a fresh sandbox"）
- apply 模型 patch → apply 官方 test_patch → 跑 FAIL_TO_PASS + PASS_TO_PASS 指定的 pytest 节点 → 全过 reward=1 否则 0
- 输出 per-test 通过数 + 执行统计（CPU 时间/峰值 RSS），reward 数值回 node1

### 5.5 双侧监控汇总

node2 monitor agent 周期采样（CPU busy、IO、load、sandboxd /stats）；node1 monitor 定时拉取，与 node1 数据按 wall clock 对齐进同一份报告（两机时钟差记入失真声明）。

## 6. SWE-bench 任务线

- **数据集**：SWE-bench-verified（HF 可直下），先过滤 `repo=="sympy/sympy"` 取 ~50 实例；字段：instance_id、base_commit、problem_statement、patch（金标准，只用于 sanity check）、test_patch、FAIL_TO_PASS、PASS_TO_PASS
- **环境烘焙**（env_baker，一次性离线）：sympy @ 各 base_commit 的源码快照 + 按实例 python 版本建 venv（pytest + sympy 依赖）→ 打成 `envs/sympy.tar.zst`；实例间 commit 差异用 git 对象库覆盖（单 repo 多 commit 存一份 .git，检出到 tmpdir）
- **评分**：grader 按 §5.4；不跑官方 `swebench` 包（依赖 conda/docker），自实现最小 pytest 判分，失真见 §7
- **到达模式**：沿用 rollout_manager 组采样（group-of-n 同实例多条轨迹），突发规模可配（对照 slime 例子 168 并发 agent slot，PoC 从 8-32 起步扫参）

## 7. 失真声明（必须随报告出现）

| # | 失真 | 真实系统 | 影响 |
|---|---|---|---|
| F1 | 节点间 RTT ~24ms | 同机房 <1ms | 模型调用 RPC 开销被放大 ~24 倍；好处是 RPC 成本可见性高，结论按"上限"读 |
| F2 | per-repo venv | per-instance conda + docker 镜像 | 环境启动成本偏低（无镜像分发/挂载），DSec 的 on-demand loading 层完全没有 |
| F3 | unshare+prlimit 进程级隔离 | PRoot rootfs / 容器 / microVM | 隔离开销偏低；启动耗时对比本身作为画像数据 |
| F4 | 自研 agent loop | 真 Codex/Claude Code CLI | node2 少了 CLI 进程（node.js 等）的 CPU 常驻开销；二期可补 |
| F5 | 单 sandbox 节点 | 2+ 节点 + 调度器 | 无放置调度/集群视图（power-of-k-choices 等） |
| F6 | 环境预烘焙 tarball | OCI 镜像 + 按需加载 | 无镜像拉取突发流量（DSec 的核心优化点之一） |
| F7 | 两机各自 NTP | 同步时钟 | 双侧时间轴对齐有毫秒级误差 |

## 8. 与真实系统对照

| 组件 | 本 PoC | slime coding_agent_rl | DSec |
|---|---|---|---|
| sandbox 服务 | sandboxd（HTTP） | Sunabako（SSH 二进制调用） | apiserver→edge→aether/chronus |
| agent 位置 | agent_runner 在 node2 | Codex/CC CLI 在 sandbox 内 | agent sandbox + worker container |
| 模型桥 | adapter（留存 logprob） | adapter → SGLang（token 级） | 训练框架内 |
| 隔离 | unshare+prlimit+降权 | patched PRoot + cgroup | 容器/microVM/VM + AppArmor/eBPF |
| 资源限额 | 分档 rlimits | 8GiB/1024 pids + 准入 cap | 分后端限额 + 高密度超卖 |
| 评分 | 新鲜 sandbox pytest | 新鲜 sandbox 官方测试 | 任务自带 verifier |
| 三级超时 | agent/grader/rollout | 3600/600/7200s | TTL + 训练协同 pause |

## 9. 开发任务拆分（续 05 规范：一次一 task、单测验收、每 task 一 commit）

| Task | 内容 | 验收（单测/门） | 位置 |
|---|---|---|---|
| T22 | node2 初始化：免密、/workspace、git clone、预检（双向连通/带宽/unshare） | `tools/preflight_node2.sh` 全绿 | node2 |
| T23 | SWE-bench 数据加载器：instances.jsonl 解析 → data_source 新任务线 | 本地 fixture 单测 | dev→双节点 |
| T24 | env_baker：sympy 快照+venv 烘焙、environments.json | 烘焙产物自校验脚本 | node2 |
| T25 | sandboxd：§5.1 表全部端点（执行原语复用 sandbox.py） | dev 机起 daemon 跑单测 | dev→node2 |
| T26 | adapter service：/generate + logprob 留存 + loss_mask | mock engine 单测 | dev→node1 |
| T27 | agent_runner：多轮 loop + tool 解析 + patch 抽取 | fake adapter + fake sandboxd 单测 | dev→node2 |
| T28 | grader：新鲜 sandbox 评分 | fixture repo + 假 patch 单测 | dev→node2 |
| T29 | rollout_manager 接 remote backend：start/poll/abort/组屏障 | fake sandboxd 单测 | dev→node1 |
| T30 | 双侧监控汇总报告 | 单测（假数据对齐） | dev |
| T31 | **G4 集成门**：1 个 sympy 实例端到端（真 llama.cpp + 真 sandbox + 真评分）+ 画像报告 + 06 文档补实验 | 双节点实跑 | 双节点 |

代码位置：`rl_sim/adapter.py`、`rl_sim/sandboxd.py`、`rl_sim/agent_runner.py`、`rl_sim/grader.py`、`rl_sim/swebench.py`、`tools/preflight_node2.sh`、`tools/bake_swe_env.py`，单测对应进 `tests/`。node2 部署 = git pull 同一份仓 + 配置指向。

## 10. 验收标准（DoD）

1. T22–T30 全部单测绿（dev 机）；T31 双节点实跑通：1 个 sympy 实例 ≥1 组（n≥2）轨迹闭环，reward 由真 pytest 评出 ✅ 2026-10-19
2. 画像报告含：node1/node2 双侧 CPU busy 时间轴、模型调用 RPC 耗时分布（含 24ms RTT 分解）、sandbox exec 分档 p50/p99、组屏障空转、sandboxd 队列深度 ✅ 见 docs/06 G4 节（exec 分档以 turn_stats 逐轮数据呈现；组屏障空转 79s 实测）
3. 单节点模式（local backend）回归不破坏：T01–T21 全部单测继续绿 ✅ 163 passed
4. 06 文档补双节点实验章节；本文件随代码推送 GitHub 终审 ✅

## 11. 修订记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-10-18 | 初版。决策 D1–D4 由用户拍板（HTTP sandboxd / loop 在 node2 / sympy 先行 / rl_sim 即 slime） |
| v1.1 | 2026-10-19 | T22–T31 全部完成，G4 门通过（docs/06 G4 节）。新增 `rl_sim/netutil.py`（集群内 HTTP 绕代理）与 `train_swe.py`（双节点 driver）；G4 实跑钓出并修复 5 个集成缺陷（代理 403 / git safe.directory / problem_statement 载荷 / ctx_slot 均分溢出 / router 4xx 投毒） |
