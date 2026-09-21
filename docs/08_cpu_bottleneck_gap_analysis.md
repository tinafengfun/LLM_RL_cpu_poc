# 08 CPU 负载瓶颈、工业解法与差距分析

> 2026-09-21。回答三个问题：RL 后训练中 CPU 干什么、什么会成为瓶颈、工业界怎么解决；并对照本 PoC 找差距，给出下一步工作方向。
> 结构：§1 CPU 工作清单 → §2 瓶颈点 → §3 工业解法（调研证据）→ §4 与 PoC 的差距矩阵 → §5 下一步方向 → §6 参考来源。

## 1. CPU 在 RL 后训练中干什么（工作清单）

按数据流顺序，CPU 侧的全部工作（GPU 只做 rollout 前向 + training 前向/反向两件）：

| # | 工作 | 说明 | PoC 对应模块 |
|---|---|---|---|
| C1 | 驱动编排 | driver 主循环、依赖调度、错误恢复 | train.py |
| C2 | 路由/网关 | 请求分发、session 亲和、负载均衡 | router.py |
| C3 | tokenize/detokenize | BPE 编解码（TITO 之前必经） | tokenizer.py |
| C4 | rollout 并发发射 | 组采样展开、并发控制、超时管理 | rollout_manager.py |
| C5 | **sandbox 环境执行** | 代码执行/单测/工具调用，agentic 场景的绝对大头 | sandbox.py |
| C6 | reward/verifier | 规则判分、judge 调用编排 | reward.py |
| C7 | advantage/打包/切 DP | 组归一化、loss_mask、序列 packing | rollout_manager.py |
| C8 | 权重同步协调 | pause/收拢参数/分桶/广播/恢复 | weight_sync.py |
| C9 | 数据落盘 | WAL、checkpoint staging、日志 | （部分） |
| C10 | 监控心跳 | 引擎健康、指标采集 | monitor.py |
| C11 | **推理引擎 CPU 侧** | 本 PoC 特有：llama.cpp 推理本身吃 CPU（真实系统里是 GPU） | engine_local.py |

## 2. 哪些会成为瓶颈（机理分析）

按"什么时候卡死整条流水线"排序：

**B1. sandbox 长尾 + 组屏障（agentic 场景第一瓶颈）**
GRPO 组采样要求同组 n 条全部完成才算 advantage。一条 90 秒的 SWE 轨迹拖住 7 条 5 秒的轨迹，整组等待 = max 而非 mean。sandbox 池在此期间低水位空转。多轮交互把到达率再放大 5-10 倍。

**B2. rollout-train 串行气泡（同步模式固有）**
同步五步循环里，训练时推理引擎完全空闲，生成时训练器空闲。rollout 占 70-90% 时间时，算力利用率被腰斩。

**B3. 权重同步暂停窗**
每次训练 step 后 pause→传输→resume。大模型全量广播分钟级，期间 rollout 停摆；频率越高（on-policy 要求越强）损失越大。

**B4. driver/网关 CPU 饱和**
Python driver 的 JSON 序列化、per-sample 对象构造、GC 在高并发下先饱和——DeepSeek 用 Rust 重写 DSec 网关、GLM 用独立 TITO gateway 就是证据。

**B5. 训推不一致（精度/算子层面）**
推理引擎（FP8、batch 大、kernel 不同）和训练框架（BF16）对同一序列算出的 logprob 不同 → 重要性比率偏离 1 → 不校正则训练发散。这是正确性瓶颈，但校正机制（重算/掩码）本身有开销。

**B6. 数据搬运与序列化**
rollout 数据（含 per-token logprob 大数组）在进程间搬运：Ray object store、pickle、内存带宽。百万 token 上下文时尤为严重。

**B7. 冷启动与镜像分发**
sandbox/引擎实例启动：镜像拉取、venv 准备、模型 mmap page-in。突发高并发时启动风暴（我们 E1 已观测到 128 workers 吞吐反降）。

## 3. 工业界解法（调研证据）

### 3.1 时间分布：rollout 主导是实测共识

- 同步 RL 中 **rollout 约占 70% 总时间**，batch 内最长 response 是中位数的 **25–32×**（RollPacker, [arXiv:2509.21009](https://arxiv.org/abs/2509.21009)）；
- **<1% 的长尾样本吃掉 >50% 的 generation 时间**（RLHFuse, [arXiv:2409.13221](https://arxiv.org/abs/2409.13221)）；
- verl/HybridFlow：分离部署示例中 GPU **1/3 时间空闲**（[arXiv:2409.19256](https://arxiv.org/html/2409.19256v1)）；
- AReaL 全异步：同等 GPU 端到端 **最高 2.77×**（[arXiv:2505.24298](https://arxiv.org/abs/2505.24298)）；
- SkyRL-Agent：async pipeline 让 generation 阶段 GPU 利用率稳定 **~90%**；CPU 侧 runtime 初始化/reward 计算是空泡来源（[arXiv:2511.16108](https://arxiv.org/html/2511.16108v1)）；
- Miles 实测 GLM-5.2 744B agentic RL：**中位 step 263s**；128 并发上限下 90–100 在生成，**其余在等 tool call——工具等待本身就是调度空洞**（[arXiv:2609.08368](https://arxiv.org/html/2609.08368v1)）。

### 3.2 长尾/组屏障的解法与收益

| 技术 | 收益 | 来源 |
|---|---|---|
| Partial rollout（半成品续生成 + off-policy mask） | 消除组屏障等待 | [slime docs](https://thudm-slime.mintlify.app/concepts/rollout-and-reward) |
| DAPO dynamic sampling（过滤零方差组） | AIME 42→50；不增 wall time（长尾主导下采样免费） | [arXiv:2503.14476](https://arxiv.org/html/2503.14476v1) |
| RollPacker tail batching（长尾集中成长轮次） | 端到端 **2.03–2.56×** | [arXiv:2509.21009](https://arxiv.org/abs/2509.21009) |
| StreamRL 长度预测调度 | 吞吐 **最高 2.66×** | [arXiv:2504.15930](https://arxiv.org/abs/2504.15930) |
| SkyRL-Agent 三段有界队列（init/run/reward 分离） | **1.55×** vs 朴素 async | [arXiv:2511.16108](https://arxiv.org/abs/2511.16108) |
| Laminar relay-worker 细粒度权重同步 + 长尾 repack | 1024 GPU 上 **5.48×** | [arXiv:2510.12633](https://arxiv.org/abs/2510.12633) |
| PipelineRL in-flight 权重更新 | 学习速度 **~2×** | [arXiv:2509.19128](https://arxiv.org/abs/2509.19128) |
| Miles **sample 粒度补位**（完成一条补一条，组仅作训练单位） | 消除组间等待 | [Miles §2.2](https://arxiv.org/html/2609.08368v1) |

### 3.3 权重同步

- 1T 模型各通道对比：**磁盘 ~分钟级；NCCL broadcast ~50s；RDMA P2P ~7s**（K2: 53.3s→7.2s，7.37×；GLM-5 744B: 58.3s→8.5s）（[LMSYS P2P 博客](https://www.lmsys.org/blog/2026-04-29-p2p-update/)）；
- **关键教训：权重注册的 CPU 开销（数十秒）曾是最大时间槽**——解法是把权重副本 staging 在 CPU（每训练 rank +32GB CPU 内存）+ 零拷贝 RDMA；
- disk-delta：只写变化字节 + checksum，**只有最后 reload 一步暂停生成**，写读文件与 rollout 并行；colocate 下 optimizer streaming 把 offload 24s→5.2s（[Miles §3.2/§4](https://arxiv.org/html/2609.08368v1)）。

### 3.4 数据通路

- **TITO 的动机**：re-tokenize 静默改变 token 边界 → importance ratio 漂移 → 梯度针对"从未发生的轨迹"更新（[LMSYS TITO 博客](https://www.lmsys.org/blog/2026-05-13-no-token-left-behind/)）；
- Miles-Diffusion：base64/JSON 反序列化改**裸字节 + 独立进程池解包**，rollout 157.4→87.6s/step，总 step 321.9→252.1s（[Miles §6](https://arxiv.org/html/2609.08368v1)）；
- R3 路由记录的代价：(tokens−1)×layers×k 个 int32，32K token×60 层×k=8 ≈ **60MB/条轨迹**——训推一致性与数据通路的直接 trade-off；
- DeepSeek V4：token 粒度 WAL（抢占重放，**从头重生成数学上不正确——引入 length bias**）；百万 token 数据拆 metadata/per-token 字段分离搬运；TileLang host codegen 把 kernel 参数校验从数十~数百 µs 降到 <1µs（[V4 §5.2](https://arxiv.org/html/2606.19348v1)）。

### 3.5 sandbox 舰队工程（CPU 控制面的真实事故与解法）

- **INTELLECT-3**（最详实的一手数据）：朴素 K8s 路径在数千并发下**每条命令延迟飙到 2.5s（etcd 写锁饱和）**；解法：Rust Gateway 直连 Pod + `nsenter` 注入 + webhook 就绪推送 → 任意镜像冷启动 **<10s 与负载无关**；256 沙箱/节点 bin-packing；训练实测 **>4000 并发沙箱**（[arXiv:2512.16144 §2.3](https://arxiv.org/html/2512.16144v1)）；
- **Prime Intellect 环境中心**：23 taskset / ~365,000 任务 / **~135,000 预构建镜像**；明确指出"千级并发 rollout 会立刻撞 Docker Hub rate limit"（[博客](https://www.primeintellect.ai/blog/scaling-agentic-rl)）；
- **DeepSeek DSec**：Rust 三组件，单集群**数十万并发沙箱**，四档底座（Function/Container/microVM/fullVM），overlaybd CoW 毫秒级快照，全序 trajectory log 支持抢占 replay（[V4 §5.2.5](https://arxiv.org/html/2606.19348v1)）；
- GLM-5：Multi-Task Rollout Orchestrator >1k 并发，任务微服务注册 + 动态采样比（[GLM-5 §4](https://arxiv.org/html/2602.15763v1)）。

### 3.6 Router/网关

- sgl-router 用 Rust 的原因：比 Python 版快 **2×**；cache-aware 路由吞吐 **1.9×**、prefix 命中率 **20%→75%**；**未优化的推理引擎可在 CPU 调度上花掉一半时间**（[SGLang v0.4 博客](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)）；
- Miles session-affinity：prefix cache 命中率 **96%**；无路由 key 的请求直接报错（防静默掉命中率）。

### 3.7 训推不一致的量级

- 同权重下 vLLM 与 FSDP 的 **per-token 概率最大差可达 1.0**；INT8 rollout 不校正则**熵塌缩、训练崩**（[slime/TIS 实测博客](https://fengyao.notion.site/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56)）；
- Miles GLM-5.2 744B 实测 rollout/trainer logprob 平均偏差 **0.0369**（100 步，TIS 校正下稳定）；极端方案 true-on-policy 可做到**逐位为 0**（吞吐有代价）；
- IcePop 双侧 mask 治 MoE 路由敏感 + 长 CoT 累积发散（[Ring-1T, arXiv:2510.18855](https://arxiv.org/html/2510.18855v2)）。

### 3.8 两个诚实的空白点

- **CPU 推理进工业 RL 生产管线：零公开实例。** 工业栈里 CPU 承担权重 staging、offload、数据通路、沙箱；我们的 llama.cpp rollout 是研究代理，不是工业现状。
- **CPU-NUMA 效应无公开实测**——我们的独占大节点正好可补。

## 4. 与 PoC 的差距矩阵（修订版）

文献证据下真实系统 CPU 瓶颈排序：**sandbox 控制面（K8s/etcd 2.5s/命令级事故）→ 数据通路（tokenize、反序列化、60MB 路由张量）→ 权重同步的 CPU staging → 路由/网关 → 组屏障与 staleness 调度**。

| 维度 | 真实系统 | 本 PoC 现状 | 差距 |
|---|---|---|---|
| sandbox 执行体 | DSec 四档/gVisor/container | process+unshare+rlimits | 中（缺隔离成本对比） |
| sandbox 控制面 | Rust 网关 + 预热池 + 镜像流式分发 | 无控制面（本地 fork） | **大**（冷启动/分发未模拟） |
| 到达过程 | >1k 并发 + 中央编排 + sample 补位 | asyncio 队列 + burst/poisson/group | 小 |
| 组屏障/长尾 | 25-32× 长尾实测 | 已实现，待 L1 真实长尾 | 小→中 |
| partial rollout | 续生成 + 旧段 loss_mask=0 | 整段 requeue 重生成 | 中 |
| rollout 前向 | GPU | CPU 小模型（真实 logprob） | 尺度失真，可接受 |
| 权重同步 | NCCL/P2P/disk-delta，CPU staging 是关键 | 时序 mock + 真实 reload | 中（缺真实字节搬运） |
| 数据通路 | TITO/裸字节/object store/60MB 路由张量 | HTTP JSON（未测字节量） | 中 |
| 训推不一致 | per-token TIS，实测偏差 0.037 | 序列级（llama.cpp 限制） | 中 |
| driver/网关 | Rust | Python（正是观测对象） | 设计内保留 |
| 多轮 agentic | L1/L2/L3 全多轮 | L1/L2 闭环未集成 | **大** |
| 容错/WAL | token 级 WAL、抢占 replay | 无 | 大 |
| NUMA | 无公开数据 | 未测 | 空白（可补） |

## 5. 下一步工作方向（按优先级）

**P0 — L1/L2 多轮 agentic 闭环（把 B1 做实）**
rollout_manager 加 agent loop：engine 多轮对话 + tool call 解析 + sandbox 执行 + 结果拼回。多轮把 sandbox 到达率放大 5-10×，组屏障×长尾效应才会真实出现。对应 T22。

**P1 — E6 引擎画像 + 核争抢（把 C11/B2 量化）**
llama-server 实例数 × slots 扫描；推理与 sandbox/driver 的核配比扫描；产出"rollout 侧 CPU 怎么花掉"的完整曲线。

**P2 — partial rollout 续生成（对齐 slime 真机制）**
abort 半成品保留已生成段，下轮续生成 + 旧段 loss_mask=0；对比整段重生成的吞吐差。替代现在的整段 requeue。

**P3 — 数据通路字节量画像（B6）**
统计每 rollout 的 JSON 序列化字节量/耗时；加 WAL（per-sample JSONL + fsync 成本）；测量 base64 vs 裸字节差异（对照 Miles-Diffusion 157→87.6s 的发现）。

**P4 — 权重同步 disk-delta 模式（B3）**
真实写变化字节到共享盘 + reload 计时；与全量 broadcast 模式对比 pause 窗口。对照 Miles disk-delta"只有 reload 暂停"的设计。

**P5 — SAO/异步完整实验（B1+B2 的解药验证）**
group=1 + DIS + mock critic；产出 sync vs async 的 sandbox 利用率、staleness 分布、吞吐对比（对照 AReaL 2.77× 的量级感）。

**P6 — NUMA 效应测量（文献空白）**
节点是双路/多 NUMA 大机器，测 sandbox 池跨 NUMA vs 绑 NUMA 的 p99 差异——公开文献没有的数据。

**P7 — sandbox 冷启动与预热池（B7）**
burst 下 launch_overhead 分布（我们已有指标）；实现预热池（pre-warmed idle workers）对比冷启动；对照 INTELLECT-3 的 <10s 与"预热池瞬时"。

**P8 — VLM 图像预处理 CPU 成本（多模态线补全）**
图像 decode/resize/encode base64 的 CPU 占比；对照真实多模态 RL 的预处理开销。

## 6. 参考来源

- RollPacker: https://arxiv.org/abs/2509.21009 ；RLHFuse: https://arxiv.org/abs/2409.13221 ；verl/HybridFlow: https://arxiv.org/html/2409.19256v1
- AReaL: https://arxiv.org/abs/2505.24298 ；SkyRL-Agent: https://arxiv.org/abs/2511.16108 ；StreamRL: https://arxiv.org/abs/2504.15930 ；Laminar: https://arxiv.org/abs/2510.12633 ；PipelineRL: https://arxiv.org/abs/2509.19128 ；DAPO: https://arxiv.org/html/2503.14476v1
- Miles 技术报告: https://arxiv.org/html/2609.08368v1 ；LMSYS P2P: https://www.lmsys.org/blog/2026-04-29-p2p-update/ ；TITO: https://www.lmsys.org/blog/2026-05-13-no-token-left-behind/
- DeepSeek V4: https://arxiv.org/html/2606.19348v1 ；GLM-5: https://arxiv.org/html/2602.15763v1 ；SAO: https://arxiv.org/abs/2607.07508
- INTELLECT-3: https://arxiv.org/html/2512.16144v1 ；Prime Intellect 环境中心: https://www.primeintellect.ai/blog/scaling-agentic-rl
- sgl-router: https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/ ；TIS 实测: https://fengyao.notion.site/Your-Efficient-RL-Framework-Secretly-Brings-You-Off-Policy-RL-Training-237721e3f6c48094ad67dad3ac091c56 ；IcePop/Ring-1T: https://arxiv.org/html/2510.18855v2
- 未验证项记录：Miles Day-0 博客 step-0 漂移 0.023 为二手转述；APRIL +44% 仅有聚合源；V4 §5.2 部分细节经 HF 博客交叉确认。
