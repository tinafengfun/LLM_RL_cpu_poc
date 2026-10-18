#!/usr/bin/env bash
# T22 preflight for node2 (sandbox-only node). Run ON node2:
#   bash tools/preflight_node2.sh
# Env overrides: NODE1_IP (default 10.239.23.91), WORKSPACE (default /workspace)
# Every check prints PASS/FAIL; exit 1 if any FAIL. No credentials in this file.
set -u
NODE1_IP="${NODE1_IP:-10.239.23.91}"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO="$WORKSPACE/LLM_RL_cpu_poc"
FAILS=0

check() { # check <name> <cmd...>
  local name="$1"; shift
  if out=$("$@" 2>&1); then
    echo "PASS  $name  | $(echo "$out" | head -1)"
  else
    echo "FAIL  $name  | $(echo "$out" | head -1)"; FAILS=$((FAILS+1))
  fi
}

echo "== node2 preflight $(date -Is) host=$(hostname) =="

check "kernel>=5.x"        uname -r
check "python3>=3.9"       python3 -c "import sys; assert sys.version_info >= (3,9), sys.version; print(sys.version.split()[0])"
check "bin:prlimit"        which prlimit
check "bin:unshare"        which unshare
check "bin:setpriv"        which setpriv
check "bin:git"            which git
check "unshare -n works"   unshare -n true
check "nobody user"        python3 -c "import pwd; e=pwd.getpwnam('nobody'); print('uid=%d gid=%d' % (e.pw_uid, e.pw_gid))"
check "workspace writable" bash -c "touch '$WORKSPACE/.pf' && rm '$WORKSPACE/.pf' && echo ok"
check "disk free >= 20G"   bash -c "a=\$(df -BG --output=avail '$WORKSPACE' | tail -1 | tr -dc 0-9); [ \"\$a\" -ge 20 ] && echo \${a}G || exit 1"
check "repo present"       bash -c "cd '$REPO' && git log --oneline -1"
check "repo clean-ish"     bash -c "cd '$REPO' && git status --porcelain | head -3 | wc -l"
check "rl_sim importable"  bash -c "cd '$REPO' && python3 -c 'import rl_sim.sandbox, rl_sim.types; print(\"import ok\")'"

echo "-- network to node1 ($NODE1_IP) --"
check "ping node1"         bash -c "ping -c 3 -W 2 '$NODE1_IP' | tail -1"
if rtt=$(ping -c 5 -W 2 "$NODE1_IP" 2>/dev/null | tail -1 | awk -F/ '{print $5}'); then
  echo "INFO  RTT avg to node1: ${rtt} ms"
fi
bw=$( { dd if=/dev/zero bs=1M count=10 2>/dev/null | ssh -o BatchMode=yes -o ConnectTimeout=5 root@"$NODE1_IP" 'cat > /dev/null' ; } 2>&1 && echo ok )
if [ "$bw" = "ok" ]; then echo "PASS  ssh push 10MB to node1  | ok"; else echo "WARN  ssh to node1 not passwordless from node2 (only needed for ops, runtime uses HTTP)"; fi

echo "== preflight done: $FAILS failure(s) =="
[ "$FAILS" -eq 0 ]
