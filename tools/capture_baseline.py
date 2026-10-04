"""Capture baseline greedy outputs for parity checking (unique prompts, temp=0)."""
import json
import random
import string
import urllib.request

URL = "http://127.0.0.1:8080/v1/chat/completions"

def gen(prompt, max_tokens=200):
    body = json.dumps({
        "model": "mlx-community/Qwen3.8-27B-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]

random.seed(20261004)
PROMPTS = [
    "Give me a detailed plan for a 3-day trip to Kyoto with a focus on temples and local food.",
    "Explain the mathematics of the transformer attention mechanism, then write a Python function implementing it.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle, in the style of magical realism.",
    "What are the tradeoffs between row-split and column-split parallelism for hybrid CPU/GPU GEMM execution? Be technical.",
    "Write a Python class implementing an LRU cache with O(1) operations, including tests.",
]
outputs = [gen(p) for p in PROMPTS]
with open("/tmp/baseline_outputs.json", "w") as f:
    json.dump({"prompts": PROMPTS, "outputs": outputs}, f, indent=1)
print("saved", len(outputs), "outputs; lengths:", [len(o) for o in outputs])
