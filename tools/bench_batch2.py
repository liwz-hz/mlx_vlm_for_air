"""Batch=2 decode throughput test: solo vs 2 concurrent streams.

Uses short unique prompts (fast prefill), measures steady decode rate
per stream. APC warm so TTFT ~0; decode timing starts at first token.
"""
import json
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "mlx-community/Qwen3.8-27B-4bit"


def stream(prompt, max_tokens, result, key):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    toks = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if line.startswith(b"data:") and line.strip() != b"data: [DONE]":
                if first is None:
                    first = time.perf_counter()
                toks += 1
    result[key] = (first - t0, toks, time.perf_counter() - first)


def rate(prompt, max_tokens=150, concurrent=1):
    result = {}
    threads = []
    for i in range(concurrent):
        p = prompt if concurrent == 1 else f"[变体{i}] " + prompt
        t = threading.Thread(target=stream, args=(p, max_tokens, result, i))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rates = []
    for i in range(concurrent):
        ttft, toks, dur = result[i]
        rates.append(toks / dur if dur > 0 else 0)
    total = sum(rates)
    return rates, total


PROMPT = "写一个关于森林的短故事，一直写下去。"

# warm APC for both prompt variants
for i in range(2):
    p = PROMPT if i == 0 else f"[变体{i}] " + PROMPT
    r = {}
    stream(p, 4, r, 0)

r1, t1 = rate(PROMPT, concurrent=1)
print(f"solo      : {r1[0]:5.1f} tok/s  (total {t1:.1f})")

r2, t2 = rate(PROMPT, concurrent=2)
print(f"batch=2   : 每流 {['%.1f' % r for r in r2]} tok/s  (total {t2:.1f})")

print(f"总吞吐提升: {t2/t1:.2f}x   单流保留: {r2[0]/r1[0]*100:.0f}%/{r2[1]/r1[0]*100:.0f}%")
