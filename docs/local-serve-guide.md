# mlx-vlm 本地推理服务：部署、接入与调优实战指南

> 实测环境：MacBook Air (M5, 32GB, macOS) / Qwen3.8-27B-4bit + MTP 投机解码
> 本文所有数据均为实测，可作为同类机型的性能预期参考。

---

## 1. 项目简介

**mlx-vlm** 是基于 Apple MLX 框架的视觉-语言模型推理库（本仓库为增强 fork），核心能力：

- **200+ 模型架构**支持（Qwen3.5、DeepSeek、GLM、Kimi、Llama 等主流系列）
- **OpenAI 兼容 API 服务器**（`mlx_vlm.server`）：连续批处理（continuous batching）、流式输出、tool calls
- **APC 自动前缀缓存**（Automatic Prefix Caching）：重复前缀（如 agent 的系统提示词）免重算 prefill
- **KV cache 量化**：`--kv-bits 8` 将 KV 内存减半
- **投机解码**：支持 MTP / EAGLE3 / dflash 三类 drafter，decode 提速 1.5~2.5×
- 另含 Anthropic 兼容接口、embeddings、rerank 等端点

适用平台：Apple Silicon（M 系列）。CPU+GPU 共享统一内存，模型加载后常驻内存。

## 2. 环境准备

### 2.1 运行本地仓库（避免跑系统已安装的 mlx_vlm）

在**仓库根目录**下执行 `python -m mlx_vlm.server` 即可，Python 会优先加载当前目录的包。验证：

```bash
python -c "import mlx_vlm; print(mlx_vlm.__file__)"
# 输出应为 <仓库路径>/mlx_vlm/__init__.py，而非 site-packages
```

其他方式：`PYTHONPATH=/path/to/mlx-vlm python -m mlx_vlm.server`，或 `pip install -e .`。

### 2.2 模型下载（modelscope）

```bash
modelscope download --model mlx-community/Qwen3.8-27B-4bit
modelscope download --model mlx-community/Qwen3.8-27B-MTP-4bit   # MTP drafter（238MB）
```

**注意目录名转义**：modelscope 会把 `.` 转成 `___`，即
`Qwen3.8-27B-4bit` → `~/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-4bit`。

### 2.3 显存/内存预算（关键！）

| 组件 | 占用 |
|---|---|
| 27B 4bit 权重 | ~15.2GB（每 token 解码全量读取） |
| KV cache（fp16，64 层/GQA 4 头/head_dim 256） | 262KB/token |
| KV cache（`--kv-bits 8`） | 131KB/token |

32GB 机器上，权重 + KV + 系统必须有充足余量，否则见 §6.3 内存压力。

## 3. 启动推理服务

### 3.1 基础启动

```bash
python -m mlx_vlm.server \
  --model ~/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-4bit \
  --host 127.0.0.1 --port 8080
```

### 3.2 推荐配置（全优化，实测有效）

```bash
APC_ENABLED=1 APC_NUM_BLOCKS=4096 \
python -m mlx_vlm.server \
  --model ~/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-4bit \
  --draft-model ~/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-MTP-4bit \
  --draft-block-size 4 \
  --host 127.0.0.1 --port 8080 \
  --kv-bits 8 \
  --max-kv-size 49152
```

| 参数 | 作用 | 实测效果 |
|---|---|---|
| `APC_ENABLED=1` | 自动前缀缓存（**默认关闭**） | 重复前缀 TTFT 4.8s → 0.2s |
| `--kv-bits 8` | KV cache 8bit 量化 | KV 内存减半 |
| `--max-kv-size N` | KV token 上限，防长会话内存失控 | 峰值从 22.1GB 得到控制 |
| `--draft-model` + 自动识别 `--draft-kind mtp` | MTP 投机解码 | decode 2.9 → 5.1 tok/s |
| `--draft-block-size 4` | 每轮验证 4 个 draft token（默认 3） | decode 5.1 → **5.9 tok/s**（block=6 超出 drafter 训练深度，接受率崩塌，勿用） |

## 4. API 调用

```bash
# 查看可用模型（loaded: true 的才是已加载的）
curl -s http://127.0.0.1:8080/v1/models

# Chat（模型 ID 必须用完整路径！）
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$HOME'/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-4bit",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 256
  }'
```

**重要坑**：服务端把未知模型名当作 HF repo 在线拉取（会 401 失败），**且该失败请求会把已加载的模型挤出内存**。务必使用 `/v1/models` 返回的完整路径作为 model ID。

响应里的 `timings` 字段是性能诊断金矿：`prompt_per_second`（prefill 速率）、`predicted_per_second`（decode 速率）、`peak_memory`、`cached_tokens`、`draft_rounds/draft_n_accepted`（投机解码接受率）。

## 5. opencode 接入

`~/.config/opencode/opencode.json` 配置（OpenAI 兼容 provider）：

```json
{
  "model": "mlx-local//Users/<user>/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-4bit",
  "provider": {
    "mlx-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX Local Server",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "local"
      },
      "models": {
        "/Users/<user>/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-4bit": {
          "name": "Qwen3.8 27B 4bit (local MLX)",
          "modalities": { "input": ["text"], "output": ["text"] },
          "limit": { "context": 49152, "output": 16384 }
        }
      }
    }
  }
}
```

三个关键点：

1. **双斜杠是正确的**：opencode 按**第一个** `/` 拆分 provider/model，模型 ID 本身含路径时形如 `mlx-local//Users/...`。
2. **limits 必须与服务端对齐**：服务端静态校验 `prompt_tokens + max_tokens ≤ MAX_KV_SIZE`，opencode 的 `limit.output` 直接决定请求的 max_tokens。若 `context` 虚报过大（如 262144），opencode 不会及时 compact，会撞服务端 `Bad Request`。
3. 配置修改后需**重启 opencode** 生效；provider 调用是本地 loopback，不产生外网流量。

## 6. 性能调优实战（方法论 + 数据）

### 6.1 两个本质瓶颈

| 阶段 | 瓶颈类型 | 原因 | 优化方向 |
|---|---|---|---|
| **prefill** | **算力**（GPU FLOPS） | 27B 每 token ≈ 54 GFLOPs；权重整个 batch 只读一遍（摊销） | APC 免重算、精简 prompt、换小模型 |
| **decode** | **带宽 + 算力强度不足** | batch=1 GEMV 内存延迟受限，GPU 算力大量闲置 | 投机解码（提高每步并行度） |

实测（M5 Air）：prefill 60~113 tok/s（有效算力 3.2~6.1 TFLOPS，无风扇机属正常）；decode 基线 2.9 tok/s。

**decode 慢的真正机制（powermetrics 实测）**：GPU 活跃度 100%、功耗仅 2.5W、频率被 DVFS 压到 486~636MHz（峰值 1578MHz）——**不是热降频（2.5W 远未触顶）、不是换页（wired limit 实测无效）、不是调度间隙（活跃度 100%）**，而是 batch=1 GEMV 算力强度太低，GPU 在等数据，DVFS 随即降频。结论：**一切能提高"每次访存的并行计算量"的手段都是这类硬件的正解**——MTP 投机解码正是如此（batch-1 GEMV → batch-4 GEMM 验证，有效带宽 45 → 77+ GB/s）。

### 6.2 调优手段优先级（实测收益）

| 优先级 | 手段 | 收益 | 成本 |
|---|---|---|---|
| 1 | **MTP 投机解码** | decode 2.9 → 5.1 tok/s | +238MB drafter |
| 2 | **`--draft-block-size 4`** | 5.1 → 5.9 tok/s（+15%） | 0 |
| 3 | **APC 前缀缓存** | 重复前缀 TTFT 4.8s→0.2s（12K 系统 prompt 只付一次） | 0 |
| 4 | **KV 8bit 量化 + max-kv-size 封顶** | 防内存失控 | 0 |
| 5 | 换小模型（8B 级） | prefill ~3.4×、decode 数倍 | 精度 |

**已验证无效的方向**（避免重复踩坑）：
- `mx.set_wired_limit`（权重锁页）：对照实验 2.8 vs 2.9 tok/s，无增益——换页不是当前瓶颈
- 整步 `mx.compile`：GPU 活跃度已 100%（无调度空隙），融合无收益空间
- CPU 分担算子：统一内存共享带宽/功耗，负优化
- draft-block-size ≥ 6：超出 drafter 训练深度，接受率 79%→29% 崩塌
- 自定义 skinny-M kernel（`tools/qmv_skinny.py` + `mlx_vlm/fast_qmv.py`，默认关闭）：微基准 M=2 gate/up 1.53×，但端到端 A/B 无差异——**根因：投机解码的 verify 热路径走 `mlx_vlm/models/quantized_verifier.py` 的专用融合 kernel，绕过 nn.QuantizedLinear**；nn 层 patch 只能拦到 drafter 层（~5% 轮时）。要优化 verify 需直接改 quantized_verifier.py 内部的 kernel
- 脏 APC 磁盘缓存会传染崩溃：drafter 损坏或异常退出后，`~/.cache/mlx-vlm/apc` 的缓存会让后续所有服务接受率崩塌，需删除后重启

**突发/持续功耗墙（M5 无风扇 Air 的决定性约束）**：
微基准实测同一 kernel：冷启动突发 **44.7 GB/s**，持续负载 100ms 后跌至 **~25 GB/s 并锁死**（powermetrics 可见频率 486-636MHz）。所有"理论性能"（9 TFLOPS 大 GEMM、60+ GB/s GEMV、0.21ms/token 带宽地板）都是突发窗口数字，**LLM 持续解码只能用持续态带宽（~25-35 GB/s 有效）**。两个独立 kernel 实现（MLX qmv_wide 与定制版）在持续态都撞同一堵墙——这是功耗墙不是 kernel 墙。结论：MTP block=4 的 5.9 tok/s 已贴近本机持续态物理上限；再往上只有降字节数（3bit 量化/小模型）或改善散热。

### 6.3 内存压力诊断（长会话性能衰减时排查）

症状：decode 持续低于 2.5 tok/s、重启服务不恢复。

诊断命令：

```bash
sysctl vm.swapusage          # swap 已用 >0 且持续增长 = 压力
vm_stat | grep "Pages free"  # free 只剩几百 MB = 危险
footprint <server_pid>       # 服务进程真实占用（含 Metal，ps RSS 不准）
echo '密码' | sudo -S powermetrics --samplers gpu_power -i 500 -n 3  # GPU 频率/功耗/降频判断
```

原理：CPU/GPU **共享同一内存控制器带宽和 SoC 功耗预算**。极端压力下（free < 500MB）swap 换页会抢带宽，但这与 DVFS 降频是两个独立机制，需用 powermetrics 区分（GPU 功耗高=热/功耗限制；功耗低+频率低=算力强度不足；功耗低+频率正常=换页抢带宽）。

### 6.4 投机解码说明

- Qwen3.5/3.8 架构原生带 MTP（config 中 `mtp_num_hidden_layers: 1`），但 mlx-community 主模型转换**按设计不含 MTP 权重**，drafter 分片单独发布（本例 `Qwen3.8-27B-MTP-4bit`，238MB，block_size=3）
- 本仓库 `mlx_vlm/speculative/drafters/qwen3_5_mtp/` 有完整实现，`--draft-kind mtp` 从 `model_type: qwen3_5_mtp` 自动识别
- drafter 与 target 必须出自同一原始 checkpoint

## 7. 常见坑速查

| 现象 | 原因与解决 |
|---|---|
| 请求 401/Repository Not Found，之后模型变慢或掉内存 | model 名不是完整路径 → 服务端去 HF 拉取失败，**并挤掉已加载模型**。用完整路径 |
| `Request needs N context tokens, but MAX_KV_SIZE is M` | prompt+max_tokens 超过 `--max-kv-size`。调大该值或调小客户端 output limit |
| `cached_tokens` 恒为 0 | APC 默认关闭，需 `APC_ENABLED=1` |
| 重启服务后速度仍慢 | 内存压力是系统级的，清理 app / `sudo purge` |
| modelscope 目录名对不上 | `.` 被转义为 `___` |
| opencode 报模型找不到 | provider/model 引用需双斜杠 `mlx-local//Users/...` |
| decode 忽快忽慢 | 检查是否有并发请求（连续批处理会分摊算力）及内存压力 |

## 8. 性能预期参考（M5 Air 32GB / Qwen3.8-27B-4bit）

| 场景 | 数值 |
|---|---|
| 首次 prefill（8K tokens，无 APC） | ~60 tok/s，139s TTFT |
| APC 命中后 TTFT | ~0.2s |
| decode 基线（无投机解码） | ~2.9 tok/s（355ms/token，有效带宽 ~45GB/s） |
| decode + MTP block 3 | 5.1 tok/s |
| **decode + MTP block 4（最终配置）** | **5.2~5.9 tok/s**（代码任务 5.2，高可预测任务 5.9） |
| GPU 运行状态（decode 时） | 486~636MHz / 2.5W / 活跃度 100% |
| 峰值内存（权重+KV+激活） | 16.4GB 起，随上下文增长 |
| 27B agent 场景体感 | 慢但可用；要流畅请用 8B 级（20+ tok/s） |
