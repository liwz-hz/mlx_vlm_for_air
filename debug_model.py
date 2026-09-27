"""Qwen3.8-27B 前向传播单步调试脚本。

调试前先停掉 8080 服务（32GB 内存只够加载一份 15.2GB 权重）:
    lsof -ti:8080 | xargs kill

推荐断点位置 (mlx_vlm/models/qwen3_5/language.py):
    LanguageModel.__call__      :1573  入口，token -> logits 全流程
    Qwen3_5Model.__call__       :1179  64 层循环
    Qwen3_5DecoderLayer         :1140  单层调度 (线性/全注意力混合)
    Qwen3_5GatedDeltaNet        :1047  线性注意力 (本模型 3/4 的层)
    Qwen3_5Attention            :857   全注意力 (每 4 层 1 个)
    Qwen3_5MLP                  :978   FFN
"""
import os

import mlx.core as mx

from mlx_vlm.utils import load

MODEL = os.path.expanduser(
    "~/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-4bit"
)

print("加载模型 (~8s)...")
model, processor = load(MODEL)
lm = model.language_model
tok = processor.tokenizer

prompt = "用一句话介绍你自己"
inputs = mx.array([tok(prompt)["input_ids"]])
print("输入:", inputs.shape, inputs.tolist())

cache = lm.make_cache()

# 阶段 1: prefill (整段 prompt 一次前向, 断点观察各层 x 的形状)
logits = lm(inputs, cache=cache)
mx.eval(logits)
print("prefill logits:", logits.shape)

# 阶段 2: greedy decode (每步 1 token, 断点观察增量解码与 cache 更新)
cur = mx.argmax(logits[0, -1, :])
out_text = ""
for _ in range(10):
    out = lm(cur[None], cache=cache)
    mx.eval(out)
    cur = mx.argmax(out[0, -1, :])
    out_text += tok.decode([int(cur)])
    print("生成中:", out_text)
