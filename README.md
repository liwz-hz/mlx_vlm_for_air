# mlx_vlm_for_air

基于 [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 的增强 fork，面向 Apple Silicon 本地推理服务。
实测基准：Qwen3.8-27B-4bit + MTP 投机解码，MacBook Air（M5, 32GB，无风扇）decode **20 tok/s**（冷机）/ ~11 tok/s（持续）。

核心增强：

- **MTP 投机解码**（Qwen3.5/3.8 原生 MTP 头，`--draft-model`）
- **KV cache 8bit 量化**（`--kv-bits 8`）
- **APC 自动前缀缓存**（`APC_ENABLED=1`，agent 系统提示免重算 prefill）
- CPU/GPU prefill 协同（实验性，默认关闭，`MLX_VLM_CPU_PREFILL`）
- 完整部署、调优与实测文档：[docs/local-serve-guide.md](docs/local-serve-guide.md)

## 1. 下载模型（ModelScope）

```bash
pip install modelscope
modelscope download --model mlx-community/Qwen3.8-27B-4bit
modelscope download --model mlx-community/Qwen3.8-27B-MTP-4bit   # MTP drafter（238MB）
```

注意：modelscope 会把模型名里的 `.` 转成 `___`，实际路径为
`~/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-4bit`。

## 2. 安装

```bash
git clone https://github.com/liwz-hz/mlx_vlm_for_air.git
cd mlx_vlm_for_air
pip install -e .
pip install mlx==0.32.2 mlx-metal==0.32.2   # 0.32.2 的 skinny GEMV 比 0.31.2 快 67%
```

## 3. 启动服务（推荐配置）

```bash
APC_ENABLED=1 APC_NUM_BLOCKS=4096 \
nohup python -m mlx_vlm.server \
  --model ~/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-4bit \
  --draft-model ~/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-MTP-4bit \
  --draft-block-size 4 \
  --host 127.0.0.1 --port 8080 \
  --kv-bits 8 \
  --max-kv-size 98304 \
  > server.log 2>&1 & disown
```

在仓库根目录下执行（Python 优先加载当前目录的包）；`nohup ... &` 之后必须 `disown`，否则后台服务可能随终端会话退出。

## 4. 参数说明

| 参数 | 作用 | 实测效果 |
|---|---|---|
| `APC_ENABLED=1` | 自动前缀缓存（默认关闭） | 重复前缀 TTFT 4.8s → 0.2s |
| `APC_NUM_BLOCKS=4096` | APC 块池（块 16 token，4096 块 = 64K token 前缀） | 需缓存更长前缀可调大 |
| `--draft-model` | MTP 投机解码 drafter（须与主模型同源 checkpoint） | decode 7.8 → 16.8 tok/s |
| `--draft-block-size 4` | 每轮 verify 块 = 锚点 + 3 草稿 | 16.8 → 19.1 tok/s（≥6 接受率崩塌） |
| `--kv-bits 8` | KV cache 8bit 量化 | KV 内存减半（64 → 32KB/token） |
| `--max-kv-size 98304` | 上下文上限 96K token | 满载 KV ≈ 3GB，峰值内存 ≈ 24GB |

## 5. 使用要点

- **性能测试前确认 `pmset -g | grep lowpower` 为 0 且接通电源**：低电量模式 decode 20 → 5.9 tok/s
- API 为 OpenAI 兼容（`http://127.0.0.1:8080/v1/chat/completions`）；`model` 字段必须用 `/v1/models` 返回的完整路径
- `max_kv_size` 是 live knob，改后无需重载模型
- MTP 验证为强贪心（target argmax）：开 MTP 时温度/top-p 采样失效，需要采样多样性请去掉 `--draft-model`
- 响应 `timings` 字段含 `prompt_per_second` / `predicted_per_second` / `peak_memory` / `cached_tokens` / `draft_n_accepted`，可直接用于性能诊断

完整文档（硬件实测、优化记录、坑清单）：[docs/local-serve-guide.md](docs/local-serve-guide.md)
