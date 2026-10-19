"""Internal cluster HTTP helpers — always proxy-free.

node1/node2 both export the corporate http_proxy (proxy.ims.intel.com) and
their no_proxy covers 10.0.0.0/8 but NOT 172.16.0.0/12: a node1 -> node2
(172.16.116.170) urllib call went through the proxy and got 403 Forbidden
(T31 first run). All node-to-node / node-to-localhost service calls go
through this module, which pins an empty ProxyHandler so behaviour does not
depend on the shell environment. External model-API traffic
(engine.py / router.py / tokenizer.py) deliberately keeps the default env
proxy handling.
"""

from __future__ import annotations

import json
import urllib.request

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post_json(url: str, obj: dict, timeout_s: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with _opener.open(req, timeout=timeout_s) as r:
        return json.loads(r.read())


def get_json(url: str, timeout_s: float) -> dict:
    with _opener.open(url, timeout=timeout_s) as r:
        return json.loads(r.read())
