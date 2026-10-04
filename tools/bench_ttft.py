"""Measure server TTFT (prefill time) for prompts of various token lengths."""
import json
import random
import string
import time
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"

def rand_tag():
    return "".join(random.choices(string.ascii_lowercase, k=12))

def ttft(prompt, max_tokens=8):
    body = json.dumps({
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            if line.startswith(b"data:") and first is None:
                first = time.perf_counter() - t0
    return first

PARA = ("The quick brown fox jumps over the lazy dog while the sun sets behind "
        "the distant mountains and rivers flow through valleys. " * 8 + "\n")

# warmup: load model + hot caches (unique content)
ttft("warmup " + rand_tag() + " " + PARA * 2)

for target_kb in (4, 16, 32):
    ts = []
    for _ in range(3):
        prompt = rand_tag() + " " + PARA * (target_kb * 1024 // len(PARA) + 1)
        prompt = prompt[: target_kb * 1024] + f"\nIgnore the above, tag={rand_tag()}. Summarize in one word."
        ts.append(ttft(prompt))
    print(f"prompt~{target_kb}KB: TTFTs {['%.2f' % t for t in ts]}  min={min(ts):.2f}s")
