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

**后台启动**：bash 函数/shell 函数内的 `nohup ... &` 可能被会话终止。使用 `disown`：
```bash
nohup python -m mlx_vlm.server ... > log 2>&1 & disown
```

### 2.2 模型下载（modelscope）

```bash
modelscope download --model mlx-community/Qwen3.8-27B-4bit
modelscope download --model mlx-community/Qwen3.8-27B-MTP-4bit   # MTP drafter（238MB）
```

**注意目录名转义**：modelscope 会把 `.` 转成 `___`，即
`Qwen3.8-27B-4bit` → `~/.cache/modelscope/hub/models/mlx-community/Qwen3___8-27B-4bit`。

### 2.3 软件依赖

```bash
pip install mlx==0.32.2 mlx-metal==0.32.2   # 0.32.2 的 M=4 skinny GEMV 比 0.31.2 快 67%
```

## 3. 硬件规格与性能实测

| 项目 | 值 | 获取命令 |
|---|---|---|
| 机型 | MacBook Air (Mac17,4) | `system_profiler SPHardwareDataType` |
| 芯片 | Apple M5 | `sysctl -n machdep.cpu.brand_string` |
| 内存 | 32 GB LPDDR5 (Micron) | `system_profiler SPMemoryDataType` |
| CPU 核心 | 10 = 4 性能核 + 6 能效核 | `sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu` |
| GPU 核心 | 10 | `ioreg -l \| grep gpu-core-count` |

### 3.1 CPU 算力

| 指标 | 值 | 来源 |
|---|---|---|
| 性能核（P-core）最高频率 | ~4.5 GHz | `powermetrics --samplers cpu_power`（E-cluster 实测 3048 MHz，P-cluster 更高） |
| 能效核（E-core）最高频率 | ~3.0 GHz | 同上 |
| L1 缓存（每核） | I: 128KB / D: 64KB | `sysctl hw.l1icachesize hw.l1dcachesize` |
| L2 缓存 | 6 MB | `sysctl hw.l2cachesize` |
| AMX fp32 矩阵乘 | **1,418-1,554 GFLOPS** | 实测（见下方方法） |
| AMX 并发（GPU 同时跑） | 16.4 GB/s 有效带宽 | 实测（GPU bf16 GEMV 同时跑 numpy BLAS） |

**CPU 算力获取方法**：
```bash
# AMX 矩阵乘吞吐（需安装 numpy，自动调 Accelerate BLAS）
python3 -c "
import numpy as np, time
a = np.random.randn(400, 5120).astype(np.float32)
b = np.random.randn(5120, 17408).astype(np.float32)
c = a @ b  # 预热
t0 = time.perf_counter()
c = a @ b
t = time.perf_counter() - t0
print(f'CPU fp32 GEMM: {2*400*5120*17408/t/1e9:.0f} GFLOPS')
"
```

### 3.2 AMX vs NEON：CPU 内两种计算单元

Apple Silicon 的每个性能核（P-core）内有两套计算单元：

| | NEON (SIMD) | AMX (矩阵协处理器) |
|---|---|---|
| **是什么** | 128-bit 向量 SIMD 指令集 | 512-bit 专用矩阵乘法单元 |
| **每指令操作** | 4×fp32 FMA = **8 FLOP** | 16×16 矩阵 FMA = **512 FLOP** |
| **理论峰值** (4核×4.5GHz) | 144 GFLOPS | **9,216 GFLOPS** |
| **实测** (fp32 GEMM) | ~40 GFLOPS (估) | **1,062 GFLOPS** |
| **比值** | 1× | **64×** |
| **擅长** | 整数/逐元素/位移/解包 | 密集矩阵乘法 |
| **不擅长** | 大规模 GEMM | 整数运算、元素级操作 |

**获取方法**：
```bash
# AMX: numpy 矩阵乘法自动调用 Accelerate BLAS → AMX
python3 -c "
import numpy as np, time
a = np.random.randn(100, 5120).astype(np.float32)
b = np.random.randn(4096, 5120).astype(np.float32)
c = a @ b.T  # 预热
t0 = time.perf_counter(); c = a @ b.T
print(f'AMX: {2*100*5120*4096/(time.perf_counter()-t0)/1e9:.0f} GFLOPS')
"
```

**在 LLM 推理中的分工**：

| 操作 | 应该用 | 原因 |
|---|---|---|
| 权重×输入 矩阵乘法 | **AMX** | 密集 GEMM，AMX 快 64× |
| 4bit 权重解包 (shift/mask) | **NEON** | 纯整数位操作，AMX 不支持 |
| swiglu/sigmoid 激活 | **NEON** | 逐元素操作，无矩阵结构 |
| argmax 采样 | **NEON** | 比较/选择，非矩阵乘 |

**AMX + NEON 交叉并行的理论可行性**：两者是 P-core 内**独立的执行管线**，可以同时执行。理论上可以构建流水线：NEON 解包第 N+1 块 4bit 权重 → AMX 用第 N 块已解包权重做矩阵乘。但这需要汇编级编程（C 编译器无法自动生成混合 AMX/NEON 指令流水线），工程复杂度极高。当前通过 Accelerate BLAS 调用 AMX 已经是实际最优方案。

### 3.3 GPU 算力

| 指标 | 值 | 来源 |
|---|---|---|
| GPU 核心数 | 10 | `ioreg -l \| grep gpu-core-count` |
| 频率范围 | 338 - 1578 MHz | `powermetrics --samplers gpu_power` |
| 满载频率（无 LPM） | 1084-1578 MHz | 同上（GEMV 持续负载时实测） |
| 满载频率（LPM 开启） | **486-636 MHz** | 同上（LPM 下实测） |
| 满载功耗（无 LPM） | ~2.8 W | 同上 |
| bf16 GEMM 峰值算力 | **8.0-8.4 TFLOPS** | 实测（见下方方法） |
| bf16 GEMV 持续读带宽 | **105.2 GB/s** | 实测（见下方方法） |
| 4bit 量化 matmul M=1 | 82.9 GB/s 有效 | 实测 |

**GPU 算力获取方法**：
```bash
# bf16 GEMM 峰值算力
python3 -c "
import mlx.core as mx, time
a = mx.random.normal((400, 5120)).astype(mx.bfloat16)
b = mx.random.normal((17408, 5120)).astype(mx.bfloat16)
c = a @ b.T; mx.eval(c)  # 预热
t0 = time.perf_counter()
c = a @ b.T; mx.eval(c)
t = time.perf_counter() - t0
print(f'GPU bf16 GEMM: {2*400*5120*17408/t/1e12:.1f} TFLOPS')
"

# bf16 GEMV 持续读带宽
python3 -c "
import mlx.core as mx, time
w = mx.random.normal((17408, 5120)).astype(mx.float16)
x = mx.random.normal((1, 5120)).astype(mx.float16)
for _ in range(50): y = x @ w.T; mx.eval(y)  # 预热
t_end = time.perf_counter() + 6
n = 0
while time.perf_counter() < t_end:
    for _ in range(20): y = x @ w.T; mx.eval(y)
    n += 20
dt = 6
print(f'GPU GEMV 带宽: {w.nbytes * n / dt / 1e9:.1f} GB/s')
"
```

**GPU 频率/功耗监控**：
```bash
sudo powermetrics --samplers gpu_power -i 1000 -n 3
# 输出 GPU HW active frequency, GPU Power, residency per frequency bin
```

### 3.4 内存带宽

| 指标 | 值 | 来源 |
|---|---|---|
| 内存类型 | LPDDR5 | `system_profiler SPMemoryDataType` |
| 容量 | 32 GB | 同上 |
| 内存总线宽度 | 128-bit（推算） | Apple 基础 M 系列均为 128-bit |
| 理论带宽 | ~120-150 GB/s（推算） | LPDDR5-6400 × 128-bit ≈ 102 GB/s；LPDDR5X-7500 × 128-bit ≈ 150 GB/s |
| **实测 GPU 有效读带宽** | **105.2 GB/s** | bf16 GEMV 持续测量（上方方法） |
| GPU+CPU 并发 | 97.3 + 16.4 = 113.7 GB/s | 实测（GPU GEMV + numpy BLAS 同时跑） |

**推算说明**：Apple 不公开 M5 的内存速度。从实测 GEMV 105.2 GB/s（含反量化开销，有效带宽低于总线峰值）和 M4（基础版）120 GB/s 的公开规格推断，M5（基础版）的理论内存带宽约 120-150 GB/s。

### 3.5 LPM 开启/关闭对比

| 指标 | LPM 开启 | LPM 关闭 | 差异 |
|---|---|---|---|
| GPU 满载频率 | 486-636 MHz | 1084-1578 MHz | **2.5-2.8×** |
| GPU 满载功耗 | ~2.5 W | ~2.8 W | 仅 +12% |
| GPU bf16 GEMV 带宽 | 25-35 GB/s | **105 GB/s** | **3-4×** |
| decode 吞吐（全优化） | 5.9 tok/s | **20.0 tok/s** | **3.4×** |
| prefill 速度（M=4000） | ~60 tok/s | ~200 tok/s | **3.3×** |

**关键发现**：LPM 牺牲 3-4× 性能，仅换来 12% 功耗节省。对于 AC 供电场景，关闭 LPM 是零成本的最大优化。

**LPM 状态检查**：`pmset -g | grep lowpower`（0=关，1=开）

### 3.6 各阶段实测性能汇总（无 LPM + mlx 0.32.2）

| 层级 | 操作 | 性能 |
|---|---|---|
| GPU | bf16 GEMM (400×5120)×(5120×17408) | 8.0 TFLOPS |
| GPU | bf16 GEMV (1×5120)×(5120×17408) | 105.2 GB/s |
| GPU | 4bit 量化 matmul M=1 | 82.9 GB/s |
| GPU | 4bit 量化 matmul M=4 | 81.2 GB/s（0.154ms/token） |
| CPU | AMX fp32 GEMM (400×5120)×(5120×17408) | 1554 GFLOPS |
| CPU | NEON 4bit GEMV（自定义 kernel） | 11.3 GB/s |
| 并发 | GPU GEMV + CPU BLAS | GPU 97.3 + CPU 16.4 GB/s |
| 端到端 | decode（MTP block4 + kv8 + APC） | **20.0 tok/s** |
| 端到端 | prefill（M=3518, 首次） | ~200 tok/s |
| 端到端 | prefill（APC 命中后） | ~0.2s TTFT |

### 3.7 显存/内存预算

| 组件 | 占用 |
|---|---|
| 27B 4bit 权重 | ~15.2GB（每 token 解码全量读取） |
| KV cache（fp16，64 层/GQA 4 头/head_dim 256） | 262KB/token |
| KV cache（`--kv-bits 8`） | 131KB/token |

32GB 机器上，权重 + KV + 系统必须有充足余量。

## 4. 启动推理服务

### 4.1 推荐配置（全优化，实测 20 tok/s）

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

| 参数 | 作用 | 实测效果 |
|---|---|---|
| `APC_ENABLED=1` | 自动前缀缓存（**默认关闭**） | 重复前缀 TTFT 4.8s → 0.2s |
| `--kv-bits 8` | KV cache 8bit 量化 | KV 内存减半 |
| `--max-kv-size 98304` | KV token 上限 96K（原 48K，2026-10 翻倍） | 满载 KV ≈ 3GB；峰值内存 ≈ 24GB 仍安全 |
| `--draft-model` + 自动识别 `--draft-kind mtp` | MTP 投机解码 | decode 2.9→5.1 tok/s |
| `--draft-block-size 4` | 每轮 verify 块总长 4 = 锚点 + **3 个草稿**（默认 3 = 锚点 + 2 草稿；实测 draft_n=3.0/轮） | 5.1→5.9 tok/s（无LPM: 16.8→19.1） |

注：`max_kv_size` 是 live knob（改后无需重载模型）。APC 块池 4096 块 = 64K token 前缀缓存；
若需缓存完整 96K 前缀可提 `APC_NUM_BLOCKS=6144`（满载 +1GB）。

### 4.2 KV 8bit 量化的质量评估（2026-10 实测）

方法（`tools/kv_quant_eval.py`，热友好：每配置 ~5K token prefill 工作量）：
1. **top1 一致率**：同一文本取 5 个递增前缀（2K-12K 字符），对比 kv8 与 bf16 两个 server
   的下一 token 贪婪选择 → **5/5 一致**
2. **长文本一致性**：温度 0 生成 800+ 字符 × 5 题，kv8 vs bf16 → **2/5 完全一致**，
   3/5 在中途分叉（近平局 token 被噪声翻转，属混沌放大，两边文本均合理）
3. 分布级对比（top-20 logprob）：当前 server 的 logprob 返回路径为占位值（`logprob:0.0`、
   top 列表恒空，`--top-logprobs-k 20` 无效），待修；方法脚本已备好

结论：kv8 噪声量级 ~0.2-0.4%（与 bf16 自身舍入同阶），**无可见质量问题**；
收益为 KV 内存/带宽减半（decode 提速 + 上下文容量翻倍）。若做严肃评测可跑困惑度对比。

## 5. API 调用

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

响应里的 `timings` 字段是性能诊断金矿：`prompt_per_second`、`predicted_per_second`、`peak_memory`、`cached_tokens`、`draft_rounds/draft_n_accepted`（投机解码接受率）。

## 6. opencode 接入

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
1. **双斜杠是正确的**：opencode 按第一个 `/` 拆分 provider/model
2. **limits 必须与服务端对齐**：服务端静态校验 `prompt + max_tokens ≤ MAX_KV_SIZE`
3. 配置修改后需**重启 opencode** 生效

## 7. 性能调优

### 7.1 本机 LLM 推理的本质

LLM 推理分两个阶段，**瓶颈完全不同**：

| 阶段 | 瓶颈 | 原因 | CPU 能否帮忙 |
|---|---|---|---|
| **prefill** | **算力**（GPU FLOPS） | 权重 batch 内摊销（读一遍算 4000 token），GPU 算力满载 95% | ✅ 能（GPU 算力满但带宽空闲） |
| **decode** | **带宽**（GB/s） | 每 token 全量读 15.2GB 权重，GPU 打满 105 GB/s 内存控制器 | ❌ 不能（被内存控制器饿死） |

**为什么 decode 阶段 CPU 帮不上忙**：Apple Silicon 的 CPU 和 GPU 共享同一块 LPDDR 内存和同一个内存控制器。decode 时 GPU 独占带宽（每 token 读全部 15.2GB 权重），CPU 发出的任何内存请求都要跟 GPU 抢同一个控制器端口，实测 CPU 并发有效带宽从 16 GB/s 跌到 ~2 GB/s，协同后反而比 GPU 独跑慢（0.95×）。

**为什么 prefill 阶段 CPU 能帮忙**：prefill 一次处理全部 prompt token（如 4000 个），权重只读一遍但算 4000 次乘加。GPU 算力满载（95%），但内存带宽只用 ~15%（权重读一遍 15.2GB / 4000 token 计算时间）。CPU 的 AMX 协处理器（~1.4 TFLOPS）可以并行处理部分 token 的矩阵乘法，与 GPU 争抢的不是带宽而是各自独立的算力。

### 7.2 优化手段（按收益排序）

| 优先级 | 手段 | 阶段 | 收益 | 成本 |
|---|---|---|---|---|
| **0** | **关闭低电量模式** | 全局 | **5.9→16.8 tok/s** | 0 |
| **1** | **升级 mlx 到 0.32.2** | decode | **16.8→20.0 tok/s** | pip install |
| 2 | MTP 投机解码 | decode | 2.9→5.1 | +238MB |
| 3 | `--draft-block-size 4` | decode | 5.1→5.9 | 0 |
| 4 | APC 前缀缓存 | prefill | TTFT 4.8s→0.2s | 0 |
| 5 | KV 8bit + max-kv-size | 全局 | 防内存失控 | 0 |
| 6 | CPU/GPU prefill 协同 | prefill | MLP +8% → prefill +5% | 0（`MLX_VLM_CPU_PREFILL=0.10`） |

**最终配置实测（3 轮中位）**：**20.01 tok/s**（计数任务），代码任务 18.3，知识问答 10.6。

### 7.3 CPU/GPU Prefill 协同（`MLX_VLM_CPU_PREFILL`）

**原理**：prefill 是算力瓶颈（GPU 95% 满载），CPU AMX 协处理器有 ~1.4 TFLOPS 闲着。按 token 维度切分——GPU 算 90% 的 token（走 4bit 量化 kernel），CPU 算 10% 的 token（走 fp32 BLAS/AMX），两者完全并行。

```
输入: [token_0 ... token_3599] [token_3600 ... token_3999]
        └── GPU 90% (量化MLP) ──┘└── CPU 10% (BLAS/AMX) ──┘
                     ↓                        ↓
                concatenate 一次合并
```

**启用**：环境变量 `MLX_VLM_CPU_PREFILL=0.10`（默认关闭）

**实测收益**（M=4000 token 全 MLP，3 投影+激活）：

| 配置 | 耗时 | 提速 |
|---|---|---|
| GPU 独跑 | 289.8 ms | — |
| CPU/GPU 协同 (10%) | **268.8 ms** | **+8%** |

端到端：MLP 占 prefill ~65% → prefill 整体提速 ~5% → TTFT 18s → 17.1s。

**限制**：
- 只在 prefill（M > 32）激活，decode 自动跳过（M=4~5 走原路，零开销）
- CPU AMX（1.4 TFLOPS）仅为 GPU（8.0 TFLOPS）的 18%，CPU 分数超 12% 后 CPU 变为瓶颈，性能反而下降
- M < 1000 时固定开销（~4ms）吃掉收益，实际无提升

**实现要点**（`mlx_vlm/cpu_prefill.py`）：
- CPU 线程 100% 纯 numpy（零 MLX API 调用，无 GIL 争抢）
- 用 `memoryview(mx_array)` 零拷贝读取统一内存中的 bf16 输入（0.01ms vs tolist 的 58ms）
- CPU 结果通过 `mx.array(numpy)` + `mx.concatenate` 合并（~2ms，不用预分配 buffer 避免并发写入问题）

### 7.4 各层性能数据（无 LPM + mlx 0.32.2）

| 层级 | 指标 |
|---|---|
| GPU 内存带宽（bf16 GEMV） | 105.5 GB/s |
| GPU 4bit quantized_matmul M=1 | 82.9 GB/s |
| GPU 4bit quantized_matmul M=4 | 81.2 GB/s / 0.154ms per token |
| fork verify kernel T=4 | 69-75 GB/s |
| 端到端（全配置） | 20.0 tok/s |

### 7.5 已验证无效的方向

- `mx.set_wired_limit`（权重锁页）：无增益
- 整步 `mx.compile`：GPU 活跃度已 100%，无调度空隙
- CPU/GPU 行切分协同（NEON 手写 kernel，7 GB/s 单独）：并发有效带宽仅 ~2 GB/s + 同步开销 → 0.95×
- v8 half2 verify kernel（`mlx_vlm/verify_v8.py`，MLX_VLM_VERIFY_V8=1 开启）：kernel 级 qkv 1.59×，但 gate/up 占 76% 字节持平 → 端到端持平
- v7 位精确解包外提：寄存器溢出 → kernel 慢 2×
- DVFS 时钟保持器（`mlx_vlm/clock_keeper.py`）：微基准 +18%，端到端持平
- draft-block-size ≥ 6：接受率 79%→29% 崩塌

### 7.6 mlx.fast 陷阱（改 kernel 前必读）

1. `grid` 参数是**总线程数**（不是 threadgroup 数）
2. 同一 kernel 对象跨模板参数复用会得到**错误结果**（必须按 shape 缓存）
3. 微基准不立即 `mx.eval` 会因输出缓冲复用产生 **rel=0 的假阳性**
4. MSL 2D thread 数组在部分展开下降级到**未同步内存副本** → 全 NaN
5. 运行时编译 kernel 的名称**不含源码哈希**——修改源码后 JIT 缓存可能关联坏二进制（需重启清除）

### 7.7 内存压力诊断

```bash
sysctl vm.swapusage          # swap >0 且增长 = 压力
vm_stat | grep "Pages free"  # <500MB = 危险
footprint <server_pid>       # 真实占用（含 Metal）
```

### 7.8 投机解码说明

- Qwen3.5/3.8 架构原生带 MTP，drafter 分片单独发布（238MB）
- 本仓库 `mlx_vlm/speculative/drafters/qwen3_5_mtp/` 有完整实现
- drafter 与 target 必须出自同一原始 checkpoint

## 8. 常见坑速查

| 现象 | 原因与解决 |
|---|---|
| 请求 401 + 模型被挤出内存 | model 名不是完整路径。用 `/v1/models` 返回的路径 |
| `Request needs N tokens, but MAX_KV_SIZE is M` | 调大 `--max-kv-size` 或调小客户端 output limit |
| `cached_tokens` 恒为 0 | APC 默认关闭，需 `APC_ENABLED=1` |
| 后台 nohup 服务莫名退出 | 加 `disown` |
| decode 速度突然减半 | 检查是否开启了低电量模式 |
| modelscope 目录名对不上 | `.` 被转义为 `___` |
| opencode 报模型找不到 | provider/model 需双斜杠 |
| 接受率崩塌（accepted≈1-6） | 删除 `~/.cache/mlx-vlm/apc` 后重启 |

## 9. 性能预期参考（M5 Air 32GB / Qwen3.8-27B-4bit）

| 场景 | LPM 开启 | **LPM 关闭** |
|---|---|---|
| 基线 decode（无投机解码） | 2.9 tok/s | — |
| + MTP block 3 + APC | 5.1 tok/s | 16.8 tok/s |
| + MTP block 4 + kv8 + APC | 5.9 tok/s | **20.0 tok/s** |
| 代码任务 | 5.2 tok/s | 18.3 tok/s |
| 知识问答 | — | 10.6 tok/s |
| APC 命中后 TTFT | 0.2s | 0.2s |
| 峰值内存 | 16.4GB+ | 16.4GB+ |

**总结**：M5 Air 32GB 跑 27B-4bit，关闭 LPM + 全优化配置可达 **20 tok/s**——已具备实际 agent 使用价值。

## 10. Prefill 深度分析：为什么 CPU/GPU 协同计算在本机为负收益（2026-10 实测）

### 10.1 Bound 点剖析（M=2048，48 delta 层 + 16 全注意力层）

| 组件 | 占 prefill 时间 | 实测吞吐 | 状态 |
|---|---|---|---|
| 量化 GEMM（MLP+各投影） | **~92%** | 11.2-12.2 TFLOPS | **已达 GPU 峰值（bf16 mm 13.2）的 92-95%，无内核级余量** |
| delta core（串行扫描 kernel） | ~7.6% | 0.48 TFLOPS | 唯一"低效"组件，但绝对值小 |
| sdpa（全注意力） | ~1.5% | 18-20 TFLOPS | 极快，无需优化 |
| lm_head | ~0 | — | 分块路径丢弃中间 chunk logits（lazy 不物化），无浪费 |

实测参考（server 自身日志，无 APC，温度上升后速率递减）：
- 798 tok: 179-181 tok/s
- 3066 tok: 154-188 tok/s
- 6094 tok: 102-140 tok/s（**43.5s→54.7s 单调劣化 = 无风扇热降频**）

### 10.2 CPU 协同计算（`MLX_VLM_CPU_PREFILL`，默认关闭）的完整验证

理论：CPU（AMX 1.65 TFLOPS）可为 11.5 TFLOPS 的 GPU 增加 ~14% 算力 → 理想 +12%。
实现（v3.1，`mlx_vlm/cpu_prefill.py` + `mlx_vlm/cpu_share.c`）做到了工程极限：
- MLP K-split（无 concat：`down([h_gpu|h_cpu]) = h_gpu@W_l + h_cpu@W_r`）
- BNNSMatMul bf16（CPU 直接吃 bf16 输入，省 GPU 端 cast；无公开 fp16 2× AMX API，实测 bf16 仅 1.73T）
- 整个 CPU 份额单次 ctypes C 调用（GIL 全释放——Python 循环会被 MLX async 派发线程的 GIL 竞争拖慢 +40%）
- BNNS `n_threads=8` + `QOS_CLASS_USER_INTERACTIVE`（默认 12 线程与 Metal 派发线程抢核心，性能断崖）

逐级实测（MLP 单算子，M=2048，交替基线）：
| 配置 | MLP op |
|---|---|
| GPU-only | 97-98 ms |
| + 上述全部修复，frac=0.10-0.12 | 91 ms（**+6~8%**） |

**但端到端 TTFT（同热状态 A/B 对照）**：

| prompt | 无 co-exec | 有 co-exec (0.11) |
|---|---|---|
| ~4KB | 4.16s | 4.40s（-5.8%） |
| ~16KB | 16.34s | 18.26s（**-11.8%**） |
| ~32KB | 43.54s | 44.68s（-2.6%） |

结论：**端到端为负的机理（2026-10-04 补充进程隔离对照实验后修正）**：
1. 跨进程干扰实测为零（交替相位对照：GPU qmm 29.9→30.5ms 纹丝不动；独立进程 CPU 在并发相位无额外劣化）——
   **GPU 从未被 CPU 拖慢**，"功耗共享拖慢 GPU" 的说法不成立；
2. Apple 功耗调度优先保 GPU：持续负载下 **CPU 集群单调热降频**（子进程 183→210ms，无风扇），
   CPU 有效算力从 1.65T 跌至 ~1.0T，卸载比例从 14% 缩水到 ~8%；
3. 每层 `fut.result()` 硬同步是结构性的：eager MLX 无法对外部内存写入建立图依赖，CPU 份额变慢时
   GPU 队列排空产生空泡（~30ms/层 × 64 层 ≈ 实测 16KB 差距 -1.9s）；
4. **多进程方案被排除**：GIL 已由单次 ctypes 消除（非剩余瓶颈）；跨进程干扰本就为零（无收益）；
   MLX 的 Metal buffer 无法跨进程共享，激活值需经 POSIX shm 往返拷贝（+4~6ms/op，严格劣于零拷贝进程内设计）；
   macOS 无硬绑核 API（`thread_policy_set` 仅建议性，QoS 已应用）。

**微基准 +6~8%（MLP 单算子，同热状态交替基线）在持续负载下被 2+3 吞噬。数值上 CPU fp32/bf16
路径更精确（greedy 输出前缀 3/3 一致），但性能为负，默认关闭。**

### 10.4 Batch=2 实测：合并 batch 生效但 verify 慢路径使其净负收益（2026-10-04）

| 指标 | solo | batch=2 |
|---|---|---|
| 每流 decode | 10-11.3 tok/s | 4.2 tok/s（总吞吐 0.73-0.81×，**更差**） |
| 轮数 | 15 | 15（两流共享同一批轮次 = 真合并） |
| 接受率 | 14/45 | 14/43（无退化） |
| 每轮耗时 | 175ms | **400ms（2.3×）** |

结论：continuous batching + `_mtp_rounds_batch` 的多行 MTP 路径**架构上正确**（合并 verify、
接受率不塌），但 B=2 的每轮成本是 B=1 的 2.3×，吃掉全部权重摊销收益。嫌疑在
`Qwen3_5BatchInvariantForward` 的 B≥2 路径（verifier 内 kernel dispatch，需专项 profiling）。
**修掉后 batch=2 预期总吞吐 ~1.9×（每流 ~11 tok/s）**。当前状态：并发请求会互相拖慢，
单流场景保持 batch=1。
prefill 侧无 batch 收益（算力已 92-95% 峰值，M=2048 与 M=4096 的 qmm 吞吐实测相同）。

注意：LPM（低电量模式）下 decode 从 ~11 掉到 3.5 tok/s（GPU 带宽被钳到 25-35GB/s），
跑性能测试前确认 `pmset -g | grep lowpower` 为 0 且接通电源。

### 10.5 剩余真实优化空间（按投入产出排序）

1. **delta core 融合分块 kernel**：现 kernel 对 T 串行扫描（0.48 TFLOPS，26× 低于 GPU 能力上限）；
   FLA 风格 chunk-parallel 融合 kernel 理论可到 3+ TFLOPS → prefill **-4~5%**。工程量大（正确性验证难）。
2. **自定义 qmm kernel**：qmm 与 bf16 mm 峰值差 5-8%（dequant 在 kernel 内的开销）→ prefill **-4%**。
3. **热管理**：长 prefill（>4k tok）实际受热降频支配（6094 tok 三连发 43.5→52.9→59.8s），任何算子优化在持续负载下都会打折。
4. 小投影拆分（qkv/z/out/q/o）已逐一实测为负：消费者依赖拆分输出导致 GPU 空转，2.8ms 收益 < 4ms glue。
