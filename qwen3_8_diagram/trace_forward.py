"""Qwen3.8-27B-4bit 首次前向传播(prefill)真实形状追踪。

加载本地 ModelScope 权重,以「一张 512x512 图片 + 提示词」为输入,
记录 tokenization、视觉塔、64 层解码器、lm_head 每个关键节点的
输入/输出形状与 dtype,输出 trace_output.json 供架构图使用。

运行前确保 8080 服务已停止(32GB 内存仅够一份权重):
    lsof -ti:8080 | xargs kill
"""
import json
import os
import sys

# 强制使用仓库代码,而非 pip 安装的旧版 mlx_vlm
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

MODEL_DIR = os.path.expanduser(
    "~/.cache/modelscope/hub/models/mlx-community/Qwen3.8-27B-4bit"
)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "trace_output.json")

# ---------------------------------------------------------------------------
# 1. 追踪机制:类级 __call__ 补丁 + id 注册表
# ---------------------------------------------------------------------------
TRACE = []
REG = {}
FUNC_HITS = {}


def shp(x):
    if isinstance(x, mx.array):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (list, tuple)):
        return [shp(i) for i in x]
    if isinstance(x, dict):
        return {k: shp(v) for k, v in x.items()}
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return type(x).__name__


def _traced_call(cls):
    if "__call__" not in cls.__dict__:
        return
    orig = cls.__dict__["__call__"]

    def __call__(self, *args, **kwargs):
        label = REG.get(id(self))
        if label is None:
            return orig(self, *args, **kwargs)
        entry = {
            "n": label,
            "in": [shp(a) for a in args],
            "kw": {k: shp(v) for k, v in kwargs.items()},
        }
        out = orig(self, *args, **kwargs)
        entry["out"] = shp(out)
        TRACE.append(entry)
        return out

    cls.__call__ = __call__


def _patch_method(cls, name, max_hits=1):
    orig = getattr(cls, name)

    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        key = f"{cls.__name__}.{name}"
        FUNC_HITS[key] = FUNC_HITS.get(key, 0) + 1
        if FUNC_HITS[key] <= max_hits:
            TRACE.append(
                {
                    "n": f"FN::{key}#{FUNC_HITS[key]}",
                    "in": [shp(a) for a in args],
                    "kw": {k: shp(v) for k, v in kwargs.items()},
                    "out": shp(out),
                }
            )
        return out

    setattr(cls, name, wrapped)


def _patch_func(module, fname, max_hits=1, stack=False):
    orig = getattr(module, fname)

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        FUNC_HITS[fname] = FUNC_HITS.get(fname, 0) + 1
        if FUNC_HITS[fname] <= max_hits:
            entry = {
                "n": f"FN::{fname}#{FUNC_HITS[fname]}",
                "in": [shp(a) for a in args],
                "kw": {k: shp(v) for k, v in kwargs.items()},
                "out": shp(out),
            }
            if stack:
                import traceback

                entry["stack"] = [
                    f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}"
                    for f in traceback.extract_stack()[-6:-1]
                ]
            TRACE.append(entry)
        return out

    setattr(module, fname, wrapped)


GENERIC = (
    nn.Linear,
    nn.QuantizedLinear,
    nn.Embedding,
    nn.QuantizedEmbedding,
    nn.RMSNorm,
    nn.LayerNorm,
    nn.Conv1d,
    nn.Conv3d,
)
for c in GENERIC:
    _traced_call(c)

from mlx_vlm.models.qwen3_5 import language as q35_lang  # noqa: E402
from mlx_vlm.models.qwen3_vl import vision as q3v_vis  # noqa: E402
from mlx_vlm.models.rope_utils import MRoPERotaryEmbedding  # noqa: E402

CUSTOM = (
    q3v_vis.Attention,
    q3v_vis.MLP,
    q3v_vis.Qwen3VLMoEVisionBlock,
    q3v_vis.PatchEmbed,
    q3v_vis.PatchMerger,
    q3v_vis.VisionRotaryEmbedding,
    q3v_vis.VisionModel,
    q35_lang.Qwen3_5Attention,
    q35_lang.Qwen3_5MLP,
    q35_lang.Qwen3_5GatedDeltaNet,
    q35_lang.Qwen3_5DecoderLayer,
    q35_lang.Qwen3_5RMSNormGated,
    MRoPERotaryEmbedding,
)
for c in CUSTOM:
    _traced_call(c)

_patch_method(q3v_vis.VisionModel, "fast_pos_embed_interpolate", 1)
_patch_method(q3v_vis.VisionModel, "rot_pos_emb", 1)
_patch_method(MRoPERotaryEmbedding, "apply_rotary", 1)

_patch_func(q35_lang, "gated_delta_update", 2)
_patch_func(q35_lang, "scaled_dot_product_attention", 3, stack=True)
_patch_func(q35_lang, "apply_multimodal_rotary_pos_emb", 1)
_patch_func(q3v_vis, "ensure_fused_sdpa", 1)


def register(mod, path):
    REG[id(mod)] = path
    for name, child in mod.children().items():
        if isinstance(child, (list, tuple)):
            for i, c in enumerate(child):
                if isinstance(c, nn.Module):
                    register(c, f"{path}.{name}.{i}")
        elif isinstance(child, nn.Module):
            register(child, f"{path}.{name}")


def tree_bytes(d):
    tot = 0

    def rec(x):
        nonlocal tot
        if isinstance(x, mx.array):
            tot += x.nbytes
        elif isinstance(x, dict):
            for v in x.values():
                rec(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                rec(v)

    rec(d)
    return tot


# ---------------------------------------------------------------------------
# 2. 构造输入并执行首次 forward
# ---------------------------------------------------------------------------
def main():
    from PIL import Image, ImageDraw

    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load, prepare_inputs

    print("加载模型 (~1min)...")
    model, processor = load(MODEL_DIR)
    register(model, "model")
    tok = processor.tokenizer
    cfg = model.config

    # --- 测试图片:512x512 手绘场景(蓝天、太阳、房子) ---
    img = Image.new("RGB", (512, 512), color=(135, 206, 235))
    d = ImageDraw.Draw(img)
    d.ellipse([400, 60, 480, 140], fill=(255, 215, 0))
    d.rectangle([140, 300, 380, 470], fill=(205, 133, 63))
    d.polygon([(120, 300), (260, 200), (400, 300)], fill=(178, 34, 34))
    d.rectangle([180, 360, 230, 410], fill=(70, 130, 180))
    d.rectangle([300, 360, 350, 470], fill=(90, 90, 90))
    img_path = os.path.join(HERE, "sample_input.png")
    img.save(img_path)

    PROMPT = "请描述这张图片"
    formatted = apply_chat_template(processor, cfg, PROMPT, num_images=1)
    inputs = prepare_inputs(
        processor,
        images=[img],
        prompts=[formatted],
        image_token_index=cfg.image_token_index,
        add_special_tokens=False,
    )
    print("inputs keys:", list(inputs.keys()))

    input_ids = inputs["input_ids"]
    mask = inputs.get("attention_mask")
    if mask is None:
        mask = inputs.get("mask")
    pixel_values = inputs.get("pixel_values")
    grid = inputs.get("image_grid_thw")

    ids = input_ids.tolist()
    ids = ids[0] if isinstance(ids[0], list) else ids
    L = len(ids)

    vs = cfg.vision_start_token_id
    ve = cfg.vision_end_token_id
    it = cfg.image_token_id
    span = [i for i, t in enumerate(ids) if t == vs or t == ve]
    img_span = [i for i, t in enumerate(ids) if t == it]
    prefix_ids = ids[: span[0] + 1] if span else ids
    suffix_ids = ids[span[-1]:] if span else []

    input_report = {
        "prompt": PROMPT,
        "formatted_prompt": formatted,
        "image_path": img_path,
        "image_size": [512, 512],
        "input_ids_shape": list(input_ids.shape),
        "seq_len": L,
        "prefix_text": tok.decode(prefix_ids),
        "prefix_tokens": len(prefix_ids),
        "suffix_text": tok.decode(suffix_ids),
        "suffix_tokens": len(suffix_ids),
        "n_image_tokens": len(img_span),
        "vision_span": [span[0], span[-1]] if span else None,
        "first_ids": ids[:8],
        "last_ids": ids[-8:],
        "special_token_ids": {
            "vision_start": vs,
            "vision_end": ve,
            "image": it,
            "bos": 248044,
            "eos": 248046,
        },
        "pixel_values": shp(pixel_values) if pixel_values is not None else None,
        "grid_thw": grid.tolist() if grid is not None else None,
    }
    print("seq_len:", L, "image tokens:", len(img_span), "grid:", grid.tolist())

    # --- 首次 forward ---
    TRACE.clear()
    cache = model.language_model.make_cache()

    features = model.get_input_embeddings(
        input_ids, pixel_values, image_grid_thw=grid, mask=mask
    )
    inputs_embeds = features.inputs_embeds
    position_ids = features.position_ids
    rope_deltas = features.rope_deltas
    mx.eval(inputs_embeds)
    print(
        "features:",
        {
            k: (list(v.shape) if hasattr(v, "shape") else v)
            for k, v in features.to_dict().items()
            if v is not None
        },
    )

    out = model.language_model(
        input_ids,
        mask=mask,
        cache=cache,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        rope_deltas=rope_deltas,
        image_grid_thw=grid,
        capture_layer_ids=list(range(cfg.text_config.num_hidden_layers)),
        return_hidden=True,
    )
    logits = out.logits
    mx.eval(logits)
    print("logits:", logits.shape)

    next_id = int(mx.argmax(logits[0, -1, :]).item())
    next_text = tok.decode([next_id])

    flow = {
        "inputs_embeds": shp(inputs_embeds),
        "position_ids": shp(position_ids),
        "rope_deltas": shp(rope_deltas),
        "hidden_states_shapes": [list(h.shape) for h in out.hidden_states],
        "logits": shp(logits),
        "next_token_id": next_id,
        "next_token_text": next_text,
    }

    # --- 权重/量化信息 ---
    def qw(layer, name):
        m = getattr(layer, name)
        info = {}
        for attr in ("weight", "scales", "biases"):
            v = getattr(m, attr, None)
            if v is not None:
                info[attr] = shp(v)
        for attr in ("bits", "group_size"):
            v = getattr(m, attr, None)
            if v is not None:
                info[attr] = v
        return info

    layers = model.language_model.model.layers
    lin0 = layers[0]
    fa3 = layers[3]
    weights = {
        "embed_tokens": shp(model.language_model.model.embed_tokens.weight),
        "embed_tokens_class": type(model.language_model.model.embed_tokens).__name__,
        "lm_head": qw(model.language_model, "lm_head"),
        "linear_in_proj_qkv": qw(lin0.linear_attn, "in_proj_qkv"),
        "linear_conv1d": shp(lin0.linear_attn.conv1d.weight),
        "linear_A_log": shp(lin0.linear_attn.A_log),
        "linear_dt_bias": shp(lin0.linear_attn.dt_bias),
        "fa_q_proj": qw(fa3.self_attn, "q_proj"),
        "fa_o_proj": qw(fa3.self_attn, "o_proj"),
        "mlp_gate_proj": qw(lin0.mlp, "gate_proj"),
        "vision_patch_embed": shp(model.vision_tower.patch_embed.proj.weight),
        "vision_pos_embed": shp(model.vision_tower.pos_embed.weight),
        "vision_block0_attn_qkv": shp(model.vision_tower.blocks[0].attn.qkv.weight),
        "merger_fc1": shp(model.vision_tower.merger.linear_fc1.weight),
    }

    report = {
        "model": {
            "path": MODEL_DIR,
            "model_type": cfg.model_type,
            "num_hidden_layers": cfg.text_config.num_hidden_layers,
            "hidden_size": cfg.text_config.hidden_size,
            "intermediate_size": cfg.text_config.intermediate_size,
            "num_attention_heads": cfg.text_config.num_attention_heads,
            "num_key_value_heads": cfg.text_config.num_key_value_heads,
            "head_dim": cfg.text_config.head_dim,
            "vocab_size": cfg.text_config.vocab_size,
            "layer_types": [
                "linear_attention" if l.is_linear else "full_attention"
                for l in model.language_model.model.layers
            ],
            "full_attention_interval": cfg.text_config.full_attention_interval,
            "rope_parameters": cfg.text_config.rope_parameters,
            "linear": {
                "num_key_heads": cfg.text_config.linear_num_key_heads,
                "num_value_heads": cfg.text_config.linear_num_value_heads,
                "key_head_dim": cfg.text_config.linear_key_head_dim,
                "value_head_dim": cfg.text_config.linear_value_head_dim,
                "conv_kernel": cfg.text_config.linear_conv_kernel_dim,
            },
            "vision": {
                "depth": cfg.vision_config.depth,
                "hidden_size": cfg.vision_config.hidden_size,
                "num_heads": cfg.vision_config.num_heads,
                "intermediate_size": cfg.vision_config.intermediate_size,
                "patch_size": cfg.vision_config.patch_size,
                "temporal_patch_size": cfg.vision_config.temporal_patch_size,
                "spatial_merge_size": cfg.vision_config.spatial_merge_size,
                "out_hidden_size": cfg.vision_config.out_hidden_size,
                "num_position_embeddings": cfg.vision_config.num_position_embeddings,
            },
            "quantization": cfg.quantization,
        },
        "input": input_report,
        "flow": flow,
        "weights": weights,
        "memory": {
            "vision_tower_bytes": tree_bytes(model.vision_tower.parameters()),
            "language_model_bytes": tree_bytes(model.language_model.parameters()),
        },
        "func_hits": FUNC_HITS,
        "trace": TRACE,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("trace entries:", len(TRACE))
    print("已写出:", OUT_JSON)
    print("下一个 token:", next_id, repr(next_text))


if __name__ == "__main__":
    main()
