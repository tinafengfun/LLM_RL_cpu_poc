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
