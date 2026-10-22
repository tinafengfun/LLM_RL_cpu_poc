# VALIDATION.md — 双节点 SWE-bench RL rollout 手工验证手册

本文档让一个人类操作者能在 **node1 + node2** 上手工启动完整的双节点 RL rollout 实验，并理解数据如何流动。全部命令实选自当前生产配置（2026-10-22 在线运行过）。

- **node1** `10.239.23.91`（256C Xeon GNR，共享机）：driver、adapter、2×llama.cpp 推理引擎
- **node2** `172.16.116.170`（128C Xeon ICX）：sandboxd（agent loop + 评分都在这台）
- **dev 开发机**：代码主仓 `/home/tina/test/LLM_RL_cpu_poc`，改动经 rsync 部署到两个节点的 `/workspace/LLM_RL_cpu_poc`

---

## 1. RL 数据流向（完整链路）

```
        node1 (10.239.23.91)                          node2 (172.16.116.170)
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│ train_swe.py driver              │      │ sandboxd :18101                  │
│  └ RolloutManager                │      │  └ trajectory_worker ×N (子进程)  │
│      │ ① POST /trajectory/start ─┼──────┼─→ ② 起 worker + 创建 sandbox      │
│      │   (swe载荷+adapter_url)   │      │     (unshare隔离/降权nobody/      │
│      │                           │      │      git clone --shared 镜像)     │
│      │                           │      │  ③ agent loop 多轮:               │
│ adapter :18090                   │      │     messages ──────────┐          │
│  └ Router(HRW亲和,按traj_id)     │      │                        │          │
│      │ ④ POST /generate ◄────────┼──────┼────────────────────────┘          │
│      ▼                           │      │  ⑤ 仅文本回传; bash工具调用        │
│ llama-server ×2 (:18081/:18082)  │      │     → sandbox exec (p50 0.01s)    │
│  (Qwen3-4B, taskset固核, 4槽/个) │      │  ⑥ submit → git diff 抽 patch     │
│                                  │      │  ⑦ grader: 新鲜sandbox           │
│ adapter_log/<traj_id>.jsonl      │      │     clone+patch+test_patch        │
│  (token级logprob, 不出node1)     │      │     → pytest F2P/P2P → reward 0/1 │
│      │ ⑧ GET /trajectory/status ─┼──────┼─→ result.json                     │
│      ▼ (轮询收reward/turns)      │      │                                  │
│  ⑨ _fill_logprob 汇总            │      │ DualMonitor: node1采/proc +       │
│  ⑩ 组屏障→advantage=reward-组均值 │      │ 拉 node2 /sysstats /stats 对齐时间轴│
│     (GRPO); zero_var组丢弃(DAPO) │      │                                  │
└──────────────────────────────────┘      └──────────────────────────────────┘
```

逐环节说明：

1. **下发**：driver 从 `/workspace/swebench/instances.jsonl` 读 11 个 sympy 实例，`RolloutDataSource` 按 n 采样分组；每个样本分配全局唯一 `traj_id`，连同样本载荷（题面、base_commit、test_patch、F2P/P2P——**gold patch 不出门**）POST 到 node2。
2. **轨迹拉起**：sandboxd 为每条轨迹起独立 `trajectory_worker` 子进程，创建隔离 sandbox（unshare 命名空间、降权 nobody、TTL 租约=agent_timeout+600s），从本地 bare 镜像 `git clone --shared` 出工作副本并 checkout 到 issue 对应的 base commit。
3. **agent loop**：worker 里的 `AgentRunner` 跑多轮 ReAct 循环——每轮把对话历史发给 node1 adapter；工具只有 `bash` 和 `submit` 两个；观测超 8000 字符截断，旧观测剪枝（保留最近 2 条完整）。
4. **跨节点生成**：adapter 按 `traj_id` 做 HRW 亲和路由到一台引擎（前缀缓存复用），转发 llama.cpp `/v1/chat/completions`（带 `logprobs:true`）。**跨网往返只带文本**。
5. **logprob 留存（本 PoC 的核心设计）**：token 级 logprob 不落网，直接写在 node1 本地 `adapter_log/<traj_id>.jsonl`——这就是 RL 训练要的 `loss_mask=1` 数据来源（工具观测不经过这里，天然 mask=0）。
6. **评分（verifier）**：submit（或 max_turns/timeout）后，grader 在**全新 sandbox** 里重放：clone → 打模型 patch → 打官方 test_patch → 跑解析后的 pytest 节点 → `rc==0` 则 reward=1。评分环境与被评环境完全隔离。
7. **回收**：driver 按 `poll_interval_s=2s` 轮询轨迹状态；拿到 reward/turns/状态后，从 adapter_log 汇总该轨迹的 logprob 总和与 token 数。
8. **组屏障与 advantage**：同组 n 个样本全部终局后才算 advantage = reward − 组均值（GRPO 风格）；全同 reward 的组按 DAPO 丢弃（画像运行时我们关掉了丢弃以保留全部数据）。
9. **画像**：DualMonitor 每 2s 对齐采集双侧 CPU/内存/sandbox 统计，写进 `summary.json`。

对应真实系统：`train_swe.py`≈slime rollout driver；adapter+router≈SGLang router/推理网关；llama.cpp≈策略引擎（vLLM/SGLang 的 CPU 替身）；sandboxd≈DSec 式 agent 执行环境；grader≈rule-based verifier；trainer 侧（Megatron 模拟、权重同步、staleness/TIS）目前只在单节点线实现，双节点线暂为画像采集（见 §6 已知边界）。

---

## 2. 前置条件检查清单

| 位置 | 检查项 | 命令/路径 |
|---|---|---|
| node1 | 推理二进制 | `/mnt/nvme0/models/llama.cpp/build/bin/llama-server` |
| node1 | 模型 | `/mnt/nvme0/models/Qwen3-4B-Q4_K_M.gguf` |
| node1 | 任务数据 | `/workspace/swebench/instances.jsonl`（11 行） |
| node2 | bare 镜像 | `/workspace/swebench/repo/sympy.git` |
| node2 | 预烘焙 venv | `/workspace/swebench/envs/py311`（原地使用，不可移动） |
| 双节点 | 代码 | `/workspace/LLM_RL_cpu_poc`（与 dev 仓同步） |

代码有改动时从 dev 部署（**两个节点都要**）：

```bash
cd /home/tina/test/LLM_RL_cpu_poc
rsync -aq --delete --exclude='__pycache__' --exclude='*.pyc' rl_sim/ root@10.239.23.91:/workspace/LLM_RL_cpu_poc/rl_sim/
rsync -q train_swe.py root@10.239.23.91:/workspace/LLM_RL_cpu_poc/train_swe.py
rsync -aq --delete --exclude='__pycache__' --exclude='*.pyc' rl_sim/ root@172.16.116.170:/workspace/LLM_RL_cpu_poc/rl_sim/
```

---

## 3. 手工启动步骤

> **两条纪律**（血泪教训，见 docs/06）：
> ① SSH 起远程后台服务必须用 `cd X; setsid nohup ... >log 2>&1 </dev/null &`（分号不是 `&&`，否则通道挂死）；
> ② kill 和重启**绝不放在同一条远程命令里**（pkill 模式串会匹配到携带该串的包装 shell 自身）。pkill 一律用 `[x]xx` 方括号写法。

### 3.0 先看服务是否已在跑（在跑则跳到 §4）

```bash
ssh root@10.239.23.91 'curl -s -m 2 -o /dev/null -w "e0=%{http_code} " http://127.0.0.1:18081/health; curl -s -m 2 -o /dev/null -w "e1=%{http_code} " http://127.0.0.1:18082/health; curl -s -m 2 -o /dev/null -w "adapter=%{http_code}\n" http://127.0.0.1:18090/healthz'
ssh root@172.16.116.170 'curl -s -m 2 http://127.0.0.1:18101/healthz'
```

### 3.1 node1：起双引擎（各占 16 固核，4 槽，64k 总上下文 = 16384/槽）

```bash
ssh root@10.239.23.91 'cd /workspace/LLM_RL_cpu_poc; mkdir -p logs/engines adapter_log; setsid nohup taskset -c 16-31 /mnt/nvme0/models/llama.cpp/build/bin/llama-server --model /mnt/nvme0/models/Qwen3-4B-Q4_K_M.gguf --port 18081 --parallel 4 --jinja --ctx-size 65536 > logs/engines/t31-e0.log 2>&1 < /dev/null & echo E0_OK'
ssh root@10.239.23.91 'cd /workspace/LLM_RL_cpu_poc; setsid nohup taskset -c 32-47 /mnt/nvme0/models/llama.cpp/build/bin/llama-server --model /mnt/nvme0/models/Qwen3-4B-Q4_K_M.gguf --port 18082 --parallel 4 --jinja --ctx-size 65536 > logs/engines/t31-e1.log 2>&1 < /dev/null & echo E1_OK'
```

验证（健康 + 启动日志确认 `n_ctx_slot = 16384`）：

```bash
ssh root@10.239.23.91 'for i in $(seq 1 90); do H0=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:18081/health); H1=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:18082/health); [ "$H0" = 200 ] && [ "$H1" = 200 ] && { echo HEALTHY; break; }; sleep 1; done; grep -h n_ctx_slot /workspace/LLM_RL_cpu_poc/logs/engines/t31-e0.log /workspace/LLM_RL_cpu_poc/logs/engines/t31-e1.log'
```

### 3.2 node1：起 adapter

```bash
ssh root@10.239.23.91 'cd /workspace/LLM_RL_cpu_poc; setsid nohup python3 -m rl_sim.adapter --port 18090 --bind 0.0.0.0 --log-dir adapter_log --backend eng0=http://127.0.0.1:18081 --backend eng1=http://127.0.0.1:18082 > logs/adapter-t31.log 2>&1 < /dev/null & echo ADAPTER_OK'
```

### 3.3 node2：起 sandboxd

```bash
ssh root@172.16.116.170 'cd /workspace/LLM_RL_cpu_poc; setsid nohup python3 -m rl_sim.sandboxd --port 18101 --scratch /workspace/swebench/scratch --mem-cap-mb 200000 --max-concurrent-exec 64 > /workspace/swebench/sandboxd.log 2>&1 < /dev/null & echo SANDBOXD_OK'
```

### 3.4 跨节点冒烟（关键路径：node2 → node1 adapter → 引擎 → logprob 留存）

```bash
ssh root@172.16.116.170 'curl -s -m 60 -X POST http://10.239.23.91:18090/generate -H "Content-Type: application/json" -d "{\"traj_id\":\"smoke-1\",\"turn\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: pong\"}],\"max_tokens\":16}"'
ssh root@10.239.23.91 'cat /workspace/LLM_RL_cpu_poc/adapter_log/smoke-1.jsonl'
```

期望：node2 拿到 `{"text":"pong",...}`；node1 的 `smoke-1.jsonl` 里有 `out_tokens`（含每个 token 的 logprob）。

---

## 4. 跑实验

### 4.1 小闭环验证（1 实例 × 2 轨迹，约 10–15 分钟）

```bash
ssh root@10.239.23.91 'cd /workspace/LLM_RL_cpu_poc; setsid nohup python3 train_swe.py --limit 1 --n-samples-per-prompt 2 --rollout-batch-size 1 --num-rollout 1 --out results/validation_g4 > logs/validation_g4.log 2>&1 < /dev/null & echo RUN_OK'
```

### 4.2 规模化画像（11 实例 × 44 轨迹 burst，约 35–45 分钟）

```bash
ssh root@10.239.23.91 'cd /workspace/LLM_RL_cpu_poc; setsid nohup python3 train_swe.py --limit 11 --n-samples-per-prompt 4 --rollout-batch-size 11 --max-turns 20 --num-rollout 1 --out results/t32_scale > logs/t32_scale.log 2>&1 < /dev/null & echo RUN_OK'
```

### 4.3 守望（完成自动输出结尾摘要）

```bash
ssh root@10.239.23.91 'while pgrep -f "[t]rain_swe.py" >/dev/null; do sleep 30; done; tail -60 /workspace/LLM_RL_cpu_poc/logs/validation_g4.log'
```

### 4.4 运行中观测

```bash
# node2 全景：活跃轨迹/sandbox/exec/拒绝/内存
ssh root@172.16.116.170 'curl -s -m 3 http://127.0.0.1:18101/stats'
# node1 adapter：调用数/token 数/后端分布/延迟样本
ssh root@10.239.23.91 'curl -s -m 3 http://127.0.0.1:18090/stats'
# 引擎错误（应为 0）
ssh root@10.239.23.91 'grep -c send_error /workspace/LLM_RL_cpu_poc/logs/engines/t31-e0.log /workspace/LLM_RL_cpu_poc/logs/engines/t31-e1.log'
```

---

## 5. 结果解读

- 摘要：`node1:/workspace/LLM_RL_cpu_poc/results/<out>/summary.json`
  - `history[]`：组数/样本数/kept/mean_reward/wall
  - `sample_rows[]`：每条轨迹的 reward/turns/终态/logprob/token 数/sandbox 明细
  - `dual_monitor`：双侧 CPU 均值峰值、sandbox 峰值活跃、拒绝数、失联采样数
- 单轨迹详情：`node2:/workspace/swebench/scratch/runs/<traj_id>/`（result.json = turn_stats 逐轮延迟、messages.jsonl、worker.log）
- logprob 原始数据：`node1:/workspace/LLM_RL_cpu_poc/adapter_log/<traj_id>.jsonl`
- 通过标准（G4）：轨迹能到 `submitted`/`max_turns` 并被**真 pytest 评分**（grade 字段含 rc/n_nodes）；reward=0 属预期（4B 模型修不动真 sympy bug，画像不训练）。

---

## 6. 已知边界（如实声明）

- 双节点线目前是 **rollout+reward+画像**，trainer 语义（TIS 修正、权重更新、staleness 丢弃）只在单节点线（train.py/train_async.py）实现；
- abort 重跑是全新轨迹，无 DSec 式 pause/resume；
- 仅 sympy 单 repo（11 实例）；reward 为 0/1 二值，无部分分。

---

## 7. 关闭与清理

```bash
# 停实验 driver（在跑的话）— kill 与后续重启分开执行！
ssh root@10.239.23.91 'pkill -f "[t]rain_swe.py"'
# 清 node2 上可能残留的孤儿轨迹（driver 死后 worker 会继续吃引擎槽位）
ssh root@172.16.116.170 'for id in $(pgrep -af "rl_sim.trajectory_worker" | grep -oP "(?<=--traj-id )\S+"); do curl -s -m 5 -X POST http://127.0.0.1:18101/trajectory/abort -H "Content-Type: application/json" -d "{\"traj_id\":\"$id\"}"; done'
# 停服务
ssh root@10.239.23.91 'pkill -f "[r]l_sim.adapter"'
ssh root@10.239.23.91 'pkill -f "[l]lama-server"'
ssh root@172.16.116.170 'pkill -f "[r]l_sim.sandboxd"'
```

---

## 8. 故障速查表（本系统实踩过的坑 → 处置）

| 症状 | 根因 | 处置（已内建，遇到先核对代码版本） |
|---|---|---|
| driver 起手 `ConnectionResetError` | 服务器 listen backlog=5，burst RST | netutil `BurstThreadingHTTPServer`（backlog 256） |
| 一条 start 失败整个 rollout 崩 | 无容错 | rollout_manager 已加重试+降级 |
| `router: no alive backends` | 超时风暴摘环不复活 / 旧版 4xx 投毒 | 环 60s 冷却复活；4xx 不摘环；backend-timeout 1800s |
| `adapter_error: HTTP 502 …context size` | prompt 超每槽 ctx | 引擎 `-c 65536`；agent 观测剪枝 |
| 轨迹中途 `unknown sandbox` 404 | TTL 回收（默认 600s < 排队生成耗时） | sandbox 租约=agent_timeout+600 |
| 轨迹 `adapter_error: TimeoutError` ~660s | 客户端超时 < adapter 后端超时 | 阶梯：agent 2000s > adapter 1800s |
| worker 报 git `safe.directory` | nobody 克隆 root 镜像 | 已内建 HOME 注册豁免 |
| node1 访问 node2 403 | 企业代理 `no_proxy` 不含 172.16/12 | netutil 内部调用绕代理 |
| SSH 起远程服务命令挂死 | `A && B &` 子 shell 攥住通道 | 用 `;` + setsid（§3 命令已正确） |
| `pkill` 把自己的远程 shell 杀了 | 模式串匹配包装 shell | `[x]xx` 方括号写法；kill/重启分两条命令 |
