# 实验结果快照（docs/06_experiment_results.md）

> 2026-09-17，节点 10.239.23.91（256 核 Xeon 6767P / 705GB，taskset 0-63）。
> 随代码迭代持续补充；实验定义见 `docs/02_poc_design.md` §6。

## E0 共享机基线（2min 快采）

- 60 样本：CPU busy 均值 **0.5%**、峰值 9.4%；load1 均值 2.40、峰值 5.31（其他租户低强度活动）
- 结论：基线窗口机器近空闲，实验数据可用；正式实验仍需 taskset 固核 + 标注背景

## E1 sandbox 并发扫描（256 个 A 档任务 burst，mock 负载基线）

| workers | wall (s) | 吞吐 (task/s) | queue_wait p50 | queue_wait p99 | peak_busy | peak_queue |
|---|---|---|---|---|---|---|
| 4 | 1.86 | 137 | 0.907s | 1.829s | 4 | 252 |
| 32 | 0.46 | 552 | 0.147s | 0.331s | 32 | 217 |
| 128 | 0.55 | 466 | 0.002s | 0.008s | 44 | 11 |

**结论（经典的池化三段曲线）**：
1. 4 workers：池饱和，队列积压（peak_queue 252 = 几乎全体排队），p99 等待 1.8s；
2. 32 workers：吞吐 4×，排队大幅缓解——本负载的甜点区；
3. 128 workers：吞吐反降 16%——短任务（~50ms）下进程启动开销主导，burst 排空速度跟不上 worker 启动速度（peak_busy 仅 44/128），**并发不是越大越好**。

对应真实系统：sandbox 池规模需匹配任务时长分布；短任务高 churn 场景下启动开销是硬约束（DSec 用 Rust + 预热池的动机）。

## G3 真实闭环（llama.cpp + Qwen3-4B Q4_K_M，2 rollout）

- 真实 logprob（per-token）、sandbox 执行、权重同步、held-out eval 全链路跑通；
- **eval pass_rate 0.833**（真模型真单测）；
- **mean_reward 1.000 → 全组 zero_var 被 DAPO 过滤 → kept 0**：强模型+简单题下 GRPO 无梯度的真实展示——这正是 DAPO dynamic sampling 存在的理由；
- 单样本生成延迟 p50 ~12s（4B Q4 @ 32 核/实例），一轮 rollout（4 样本）~30–47s。

## T20 VLM 端到端（Qwen2.5-VL-3B + mmproj）

- 图像输入→回答→logprob（-0.19，真实）→rule-based reward 全链路 OK；
- 模型把 6 根柱数成 5（reward=0）——3B VLM 的计数能力边界，PoC 验证通路不验证分数。

## 过程中排掉的典型 bug（画像素材）

1. **Qwen3 默认 thinking**：max_tokens 全烧在 reasoning_content，content 为空 → `chat_template_kwargs.enable_thinking=false`；
2. **CentOS `nobody` 是 uid/gid 99**（Debian 是 65534/nogroup）→ setpriv 动态解析；
3. **`pkill -f llama-server` 会杀掉携带该串的远程 shell 自身** → 用 `pkill -x`；
4. 硬 CPU 限额触发的是 **SIGKILL 而非 SIGXCPU**（soft==hard 时内核直接杀）；
5. llama.cpp 无 prompt_logprobs 端点 → trainer rescore 采用序列级（GSPO 风格）比率。

## G4 双节点 SWE-bench 端到端（2026-10-19，node1 10.239.23.91 + node2 172.16.116.170）

拓扑：node1 跑 `train_swe.py` driver + adapter（:18090）+ 2×llama.cpp（Qwen3-4B Q4_K_M，taskset 16-31/32-47，`--parallel 4 --ctx-size 65536` → 16384/槽）；node2 跑 sandboxd（:18101），agent loop 与 grader 都在 node2。任务：SWE-bench_Verified `sympy__sympy-23262` × n=2 组采样，max_turns=40。

### 闭环结果（wall 876s）

| 轨迹 | turns | 终止状态 | 评分（真 pytest，62 节点） | reward | logprob tokens |
|---|---|---|---|---|---|
| …-6d1268e7 | 19 | submitted（真提交 patch） | rc=1，F2P `test_issue_14941` 未过（1 failed / 61 passed） | 0.0 | 1704 |
| …-6405cd10 | 40 | max_turns | 同上，正常进入评分 | 0.0 | 2412 |

reward=0 是 **4B 模型能力**（真 sympy bug 修不动），不是链路问题——链路每一环都是真的：多轮 tool calling、patch 抽取、新鲜 sandbox 起 pytest、F2P/P2P 节点解析、诚实判负。

### 负载画像（DualMonitor，376 采样点，零失联）

| 指标 | node1（引擎侧） | node2（sandbox 侧） |
|---|---|---|
| CPU busy 均值 | 6.42% | **0.06%** |
| CPU busy 峰值 | 45.88%（整机含合租背景） | 2.17% |
| sandbox 峰值活跃 | — | 2 |
| exec 总数 / 拒绝 | — | 79 / 0 |

- **生成主导 wall**：两条轨迹 gen_lat 793s/868s ≈ 各自 wall 的 99.7%/99.4%；sandbox exec 合计 1.1s/4.5s。每轮生成 ~22–42s（CPU 推理 ~2.6 tok/s/槽）。
- **RTT 可忽略**：24ms × 全程 ~60 次跨节点调用 ≈ 1.4s，占 wall 0.2%。
- **组屏障空转**：组 wall 876.3s ≈ max(795.1, 873.8)+2.5s 调度开销；先完成的轨迹在屏障等了 ~79s——n=2 小组下 straggler 代价的直观样本。
- node2 画像与 DSec 生产数据（90% sandbox CPU ≤5%）一致：sandbox CPU 不是瓶颈，瓶颈在生成侧。

### G4 钓出的 5 个真集成缺陷（均已修复+回归测试，见 git log）

1. 企业代理 403：`no_proxy` 不含 172.16/12 → 新增 `rl_sim/netutil.py`，集群内 HTTP 全部绕代理；
2. sandbox 降权 nobody 克隆 root 镜像触发 git safe.directory（`-c` 不覆盖 upload-pack 子进程）→ sandbox 内 HOME 注册全局豁免；
3. `to_task` 载荷缺 `problem_statement` → agent 起手 KeyError；
4. llama.cpp `--parallel 4` 均分 ctx（8192→2048/槽）第一轮溢出 → 引擎升 `-c 65536`；
5. router 把 4xx（请求自身问题）当后端故障摘环 → 一条超长请求杀死全部后端；修复为 4xx 不摘环不 failover，并全链路透传后端错误体。配套 agent_runner 上下文剪枝（保留最近 2 条完整观测，旧的压成 300 字符标记），40 轮 prompt 峰值压到 5946（此前第 8 轮就 8884 溢出）。

> 运维教训记档：kill 与重启不可合并进同一远程命令（pkill 模式串会匹配携带该串的包装 shell 自身）。
