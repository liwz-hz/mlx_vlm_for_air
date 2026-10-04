"""Minimal KV-quant drift eval (thermal-friendly): 5 prefix positions on one
~12K-char text; per config ~5K tokens of total prefill work.

  python3 tools/kv_quant_eval.py collect kv8.json
  # (restart server with other kv config)
  python3 tools/kv_quant_eval.py collect bf16.json
  python3 tools/kv_quant_eval.py compare kv8.json bf16.json
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "mlx-community/Qwen3.8-27B-4bit"
POSITIONS = [2000, 4500, 7000, 9500, 12000]


def corpus():
    return open("docs/local-serve-guide.md", encoding="utf-8").read()[:12000]


def top20(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + "\n继续写一个词："}],
        "max_tokens": 1, "temperature": 0,
        "logprobs": True, "top_logprobs": 20,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        lp = json.load(resp)["choices"][0].get("logprobs")
    e = lp["content"][0]
    return {"top1": e["token"], "logprob": e["logprob"],
            "top": [(t["token"], t["logprob"]) for t in (e.get("top_logprobs") or [])]}


def collect(path):
    text = corpus()
    out = []
    for p in POSITIONS:
        t0 = time.perf_counter()
        r = top20(text[:p])
        out.append({"pos": p, **r})
        print(f"pos={p} {time.perf_counter()-t0:.1f}s top1={r['top1']!r}", flush=True)
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print("saved", path)


def compare(a_path, b_path):
    A, B = json.load(open(a_path)), json.load(open(b_path))
    same = dlp = jac = n = nj = 0
    for a, b in zip(A, B):
        n += 1
        if a["top1"] == b["top1"]:
            same += 1
            dlp += abs(a["logprob"] - b["logprob"])
        sa, sb = {t for t, _ in a["top"]}, {t for t, _ in b["top"]}
        if sa or sb:
            jac += len(sa & sb) / len(sa | sb)
            nj += 1
    print(f"top1 一致: {same}/{n}")
    print(f"top1 相同时平均 |Δlogprob|: {dlp/max(same,1):.4f}")
    print(f"top20 Jaccard 平均: {jac/max(nj,1):.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
