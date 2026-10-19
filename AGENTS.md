# AGENTS.md

本文件记录在本仓库工作必须遵守的铁律。

## 铁律：不放过任何一个 warning 和 log

任何一次命令输出、测试运行、服务日志里的 **warning、异常日志、非零退出码、超时**，都必须追查到底、彻底搞清楚根因。**如果是问题，就真实修复**。

具体要求：

1. **禁止用 workaround 冒充修复**。"本地超时不代表远程没起来，另行验证"这类说法是绕过，不是修复。把任何坑记入"已排掉"之前，必须回答：根因是什么？修了吗？为什么不修？
2. **修复必须反向验证**。修复前能稳定复现问题，修复后同一操作稳定不复现；有条件就做 A/B 对照实验隔离变量，用数据说话。
3. **warning 只有两种归宿：修复，或定性**。定性必须写明来源、影响、不处理的理由，记入下方"已知 warning 清单"。不允许默默忽略，不允许"反正成功了"就放过。
4. 测试和长跑日志里出现任何一条**新** warning/ERROR，停下来先追查，再继续推进任务。

### 案例：sandboxd 远程启动挂通道（2026-10-19）

旧启动命令 `cd X && setsid nohup python3 -m rl_sim.sandboxd ... & disown` 长期被当作"已修复"记录，实际每次都挂满超时。根因：`A && B &` 会把整个 AND 列表包进一个后台子 shell，子 shell 继承 SSH 通道的 stdout/stderr 并活到服务进程结束，sshd 永远收不到 EOF，客户端挂死。真实修复：把 `&&` 改成 `;`，只后台化三条 fd 已全部重定向的那条简单命令。A/B 对照（sleep 30）证实：旧模式 exit=124 挂死，新模式 0.22s 干净退出。

## 已知 warning 清单（已定性，非问题，勿再重复追查）

- `** WARNING: connection is not using a post-quantum key exchange algorithm`：仅出现在 ssh 到 node2（172.16.116.170）。来源：dev 客户端 OpenSSH 10.2 的默认提示；node2（openEuler 24.03）的 sshd 仅提供 curve25519 / dh-group-exchange 经典密钥交换，不支持 mlkem768x25519-sha256（node1 为 OpenSSH 9.9，支持，无此 warning）。影响为 "store now, decrypt later" 级远期理论风险，实验内网环境可接受。不处理的理由：修复需改动共享机 node2 的 sshd/crypto policy，超出授权范围。若 node2 日后升级 sshd 或放开 kex 策略，此条作废。
- `Authorized users only. All activities may be monitored and reported.`：node2 的登录 banner（motd），不是 warning。
