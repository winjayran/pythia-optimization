# KVPress 对比实验：Pythia-70m 基线与优化方法

## 硬件环境介绍

NVIDIA GeForce RTX 3060 12G

驱动程序版本:	32.0.15.9579
驱动程序日期:	2026/3/4
DirectX 版本:	12 (FL 12.2)

CPU 11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz

基准速度:	2.50 GHz
插槽:	1
内核:	8
逻辑处理器:	16
虚拟化:	已启用
L1 缓存:	640 KB
L2 缓存:	4.0 MB
L3 缓存:	16.0 MB


## 项目简介

本项目对比了基线（普通 HuggingFace transformers，无 KV 压缩）与 KVPress 优化方法在 Pythia-70m 模型上的性能。实验使用 WikiText 和 PG19 数据集，并评估了多种 KV 缓存压缩方法。

## KVPress 修改内容

由于 Pythia 模型使用 GPTNeoX 架构，而 kvpress 官方并不直接支持，我们需要对修改：
- `kvpress/presses/base_press.py` - 添加 GPTNeoX 支持
- `kvpress/presses/scorer_press.py` - 修复 head_dim 属性
- `kvpress/presses/snapkv_press.py` - 修复 RoPE 和 GQA 处理
- `kvpress/utils.py` - 修复 query/key 状态提取
...（详情见./kvpress/）

## 环境配置

### 基线环境（不含 kvpress）

```bash
# 创建并激活 conda 环境
conda create -n kvpress-baseline python=3.13
conda activate kvpress-baseline

# 安装基础依赖
pip install torch transformers datasets tqdm
```

### 优化环境（含 kvpress）

```bash
conda create -n nlp-kvpress python=3.13
conda activate nlp-kvpress # 此外还要重装cuda tool kits还有安装uv
git clone https://github.com/NVIDIA/kvpress.git
cd kvpress

uv sync

```

## 运行实验

### 1. 准备数据和模型

```bash
# 下载 Pythia-70m 模型
# 模型会自动下载到 ~/.cache/huggingface/

# 下载 WikiText 数据集
# 数据集会自动下载到 ~/.cache/huggingface/datasets/

# 准备 PG19 数据集（已处理）
# 位于 datasets/pg19/pg19-test-100.parquet
```

### 2. 运行基线评估

```bash
# 使用基线环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kvpress-baseline

# WikiText 评估 (100 样本)
python scripts/baseline_eval.py \
    --dataset wikitext \
    --num_samples 100 \
    --max_seq_len 2048 \
    --num_generate 256 \
    --device cuda

# PG19 评估 (100 样本)
python scripts/baseline_eval.py \
    --dataset pg19 \
    --num_samples 100 \
    --max_seq_len 2048 \
    --num_generate 256 \
    --device cuda
```

### 3. 运行 KVPress 优化评估

```bash
source ~/path_to_file/kvpress/.venv/bin/activate

# WikiText 评估 (100 样本, SnapKVPress)
python scripts/kvpress_eval.py \
    --dataset wikitext \
    --num_samples 100 \
    --max_seq_len 2048 \
    --num_generate 256 \
    --compression_ratio 0.5 \
    --device cuda

# PG19 评估 (100 样本, SnapKVPress)
python scripts/kvpress_eval.py \
    --dataset pg19 \
    --num_samples 100 \
    --max_seq_len 2048 \
    --num_generate 256 \
    --compression_ratio 0.5 \
    --device cuda
```

### 4. 对比不同 KVPress 方法

```bash
source ~/path_to_file/kvpress/.venv/bin/activate

# 运行方法对比 (WikiText)
python scripts/compare_presses.py \
    --dataset wikitext \
    --num_samples 50 \
    --max_seq_len 512 \
    --num_generate 128 \
    --device cuda \
    --presses KnormPress TOVAPress SnapKVPress StreamingLLMPress LagKVPress

# 运行方法对比 (PG19)
python scripts/compare_presses.py \
    --dataset pg19 \
    --num_samples 50 \
    --max_seq_len 512 \
    --num_generate 128 \
    --device cuda \
    --presses KnormPress TOVAPress SnapKVPress StreamingLLMPress LagKVPress
```

## 实验结果

### 100 样本对比（WikiText，GPU）

| 指标 | 基线 | KVPress (SnapKV) | 变化 |
|------|------|------------------|------|
| **困惑度 (PPL)** | 37.87 | 117.34 | +210% |
| **吞吐量** | 209.8 tok/s | 183.1 tok/s | -12.7% |
| **TTFT** | 0.79 ms | 1.28 ms | -62% |
| **TPOT** | 4.76 ms | 5.45 ms | -14.5% |
| **实际压缩率** | - | 50.1% | |

### 100 样本对比（PG19，GPU）

| 指标 | 基线 | KVPress (SnapKV) | 变化 |
|------|------|------------------|------|
| **困惑度 (PPL)** | 126.13 | 234.32 | +86% |
| **吞吐量** | 195.9 tok/s | 200.2 tok/s | +2.2% |
| **TTFT** | 12.87 ms | 7.65 ms | +41% |
| **TPOT** | 5.05 ms | 4.96 ms | +1.8% |
| **实际压缩率** | - | 50.0% | |

### 不同 KVPress 方法对比（50 样本）

#### WikiText

| 方法 | 困惑度 | 吞吐量 (tok/s) |
|------|--------|----------------|
| **KnormPress** | **170.72** | **204.26** |
| TOVAPress | 250.71 | 203.35 |
| SnapKVPress | 261.06 | 201.17 |
| LagKVPress | 281.26 | 203.25 |
| StreamingLLMPress | 285.16 | 201.88 |

#### PG19

| 方法 | 困惑度 | 吞吐量 (tok/s) |
|------|--------|----------------|
| **KnormPress** | **161.00** | **193.94** |
| StreamingLLMPress | 174.32 | 190.69 |
| LagKVPress | 174.32 | 185.30 |
| SnapKVPress | 175.70 | 166.48 |
| TOVAPress | 176.44 | 184.61 |

## 最终结论

### 1. 数据集依赖的性能表现

- **WikiText**：KVPress 由于短文本的压缩开销，吞吐量反而下降 (-12.7%)
- **PG19**：KVPress 在长序列上略快 (+2.2%)，TTFT 提升 41%

### 2. 质量与速度的权衡

- 所有 KV 压缩方法都会导致困惑度显著增加（86-210%）
- 这是 KV 缓存压缩的固有代价：减少内存以换取质量下降

### 3. **最佳方法：KnormPress**

经过对比测试，**KnormPress** 是所有测试方法中表现最好的：

| 优势 | 说明 |
|------|------|
| 最低困惑度 | WikiText: 170.72, PG19: 161.00 |
| 最高吞吐量 | WikiText: 204.26 tok/s, PG19: 193.94 tok/s |
| 简单高效 | 仅基于 key 的 L2 范数进行评分 |
| 兼容性好 | 与 GPTNeoX 架构配合良好 |

**KnormPress 相比 SnapKVPress 的优势：**
- WikiText: 困惑度降低 35% (170.72 vs 261.06)
- PG19: 困惑度降低 8% (161.00 vs 175.70)


## 参考

- [KVPress GitHub](https://github.com/NVIDIA/kvpress)
- [KVPress 论文](https://arxiv.org/abs/2510.00636)
- [Pythia 模型](https://github.com/EleutherAI/pythia)


## 其他
在 ./cpu_optimization 中做了Intel ultra5 125H上cpu的优化，详情请见该文件夹。