# Pythia-70M CPU 推理加速

本项目针对 Pythia-70M 模型在 CPU 环境下的推理进行了无训练加速优化，实现了约 1.7 倍的生成吞吐量提升。

## 硬件
Intel Ultra5 125H, ddr5 32G

## 功能特性

- **Baseline 实现** (`baseline_cpu.py`): 基于 Hugging Face `transformers` 的标准 CPU 实现
- **优化实现** (`optimized_cpu.py`): 利用 float16 精度、channels_last 内存格式、KV Cache 和多线程优化

## 环境要求

```bash
# conda 可选
pip install torch transformers datasets numpy
```

## 快速开始

### 1. 准备数据和模型

```bash
# 下载模型
huggingface-cli download EleutherAI/pythia-70m --local-dir ./models/pythia-70m

# 准备数据集 (PG-19 或 WikiText)
# 将数据集放置在 ./datasets/ 目录下
```

### 2. 运行 Baseline（单核）

```bash
taskset -c 0 python baseline_cpu.py \
    --dataset pg19 \
    --data_dir ./datasets/pg19 \
    --max_articles 2 \
    --model_path ./models/pythia-70m
```

### 3. 运行优化版本（最佳配置）

```bash
python optimized_cpu.py \
    --dataset pg19 \
    --data_dir ./datasets/pg19 \
    --max_articles 2 \
    --model_path ./models/pythia-70m \
    --dtype float16 \
    --num_threads 4
```

## 实现原理

### Baseline 实现 (`baseline_cpu.py`)

Baseline 使用标准的 Hugging Face transformers 流程：

1. **模型加载**: 使用 `AutoModelForCausalLM.from_pretrained()` 加载 Pythia-70M
2. **精度**: 默认使用 `float32` 精度
3. **困惑度计算**: 使用滑动窗口方法计算长文本的困惑度
4. **生成基准测试**: 测量以下指标
   - **TTFT** (Time To First Token): 首个生成 token 的延迟
   - **TPOT** (Time Per Output Token): 每个生成 token 的平均时间
   - **Throughput**: 每秒生成的 token 数
   - **GFLOPs**: 估算的计算量

### 优化实现 (`optimized_cpu.py`)

优化版本应用了多项 CPU 推理加速技术：

| 优化项 | 实现方式 |
|--------|----------|
| **环境变量设置** | 在 PyTorch 导入前设置 `OMP_NUM_THREADS`、`MKL_NUM_THREADS` 等 |
| **内存格式** | 使用 `channels_last` 格式提升 CPU 缓存利用率 |
| **精度优化** | 支持 `float16`/`bfloat16`，减少内存带宽占用 |
| **注意力实现** | 可选 SDPA (Scaled Dot Product Attention) |
| **图编译** | 可选 `torch.compile` 减少开销 |
| **KV Cache** | 启用以减少自回归生成的重复计算 |
| **生成循环** | 最小化开销，优化张量管理 |

## 性能指标

### 测试配置: PG-19 数据集，2 篇文章

| 指标 | Baseline (单核) | Optimized (float16, 4线程) | 提升 |
|------|----------------|----------------------------|------|
| **困惑度 (PPL)** | 53.51 | 55.38 | |
| **首 Token 时间 (TTFT)** | 27.93 ms | 17 ms | **1.6x** |
| **每 Token 时间 (TPOT)** | 11.02 ms | **6.54 ms** | **1.7x** |
| **吞吐量** | 90.7 tok/s | **151 tok/s** | **1.7x** |
| **模型大小** | 268.7 MB | 134.3 MB | |

### 配置对比

| 精度 | 线程数 | PPL | 吞吐量 | 说明 |
|------|--------|-----|--------|------|
| float32 | 8 | 53.51 | ~145 tok/s | 最佳精度 |
| **float16** | **4** | **55.38** | **151 tok/s** | **最佳吞吐** |
| float16 | 6 | 55.40 | ~148 tok/s | 良好替代 |
| float16 | 8 | 55.38 | ~130 tok/s | 线程竞争过多 |

## 命令行参数

### Baseline 参数

```bash
python baseline_cpu.py \
    --model_path ./models/pythia-70m    # 模型路径
    --dataset pg19                      # 数据集名称 (pg19/wikitext)
    --data_dir ./datasets/pg19          # 数据目录
    --mode ppl,generation               # 运行模式
    --max_articles 2                    # 最大文章数
    --max_new_tokens 20                 # 最大生成 token 数
```

### Optimized 参数

```bash
python optimized_cpu.py \
    --model_path ./models/pythia-70m    # 模型路径
    --dataset pg19                      # 数据集名称
    --data_dir ./datasets/pg19          # 数据目录
    --dtype float16                     # 精度 (float32/bfloat16/float16)
    --num_threads 4                     # CPU 线程数
    --use_sdpa                          # 启用 SDPA 注意力
    --use_compile                       # 启用 torch.compile
    --max_articles 2                    # 最大文章数
```

