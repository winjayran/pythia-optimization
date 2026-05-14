import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def estimate_flops(
    num_params: int,
    seq_len: int,
    num_generated_tokens: int,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    head_dim: int,
    vocab_size: int,
) -> Dict[str, float]:
    prefill_flops = 2 * num_layers * seq_len * hidden_dim * (
        2 * hidden_dim + seq_len) * 4
    ffn_dim = 4 * hidden_dim

    qkv_proj = 3 * seq_len * hidden_dim * (num_heads * head_dim)
    attn_scores = seq_len * seq_len * (num_heads * head_dim)
    out_proj = seq_len * hidden_dim * (num_heads * head_dim)
    mlp_flops = 2 * seq_len * hidden_dim * ffn_dim + seq_len * ffn_dim * hidden_dim

    prefill_flops = num_layers * (qkv_proj + attn_scores + out_proj + mlp_flops)
    prefill_flops += seq_len * hidden_dim * vocab_size

    per_token_flops = num_layers * (
        3 * 1 * hidden_dim * (num_heads * head_dim) +
        seq_len * (num_heads * head_dim) +
        1 * hidden_dim * (num_heads * head_dim) +
        2 * 1 * hidden_dim * ffn_dim + 1 * ffn_dim * hidden_dim)
    per_token_flops += hidden_dim * vocab_size

    decode_flops = per_token_flops * num_generated_tokens

    total_flops = prefill_flops + decode_flops
    total_tokens = seq_len + num_generated_tokens
    avg_flops_per_token = total_flops / total_tokens

    return {
        "prefill_flops": prefill_flops,
        "decode_flops": decode_flops,
        "total_flops": total_flops,
        "avg_flops_per_token": avg_flops_per_token,
        "per_token_flops": per_token_flops,
    }


@torch.inference_mode()
def evaluate_baseline(
    model_name: str = "EleutherAI/pythia-70m",
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    num_samples: int = 10,
    max_seq_len: int = 2048,
    num_generate: int = 256,
    device: str = "cpu",
) -> Dict[str, Any]:
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = model.config
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Model params: {num_params:,}")
    print(f"Loading dataset: {dataset_name} ({dataset_config})")

    if dataset_name == "wikitext":
        dataset = load_dataset(dataset_name, dataset_config, split="test")
        text_column = "text"
    elif dataset_name == "pg19":
        dataset = load_dataset("parquet", data_files="datasets/pg19/pg19-test-100.parquet", split="train")
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = dataset.filter(lambda x: len(x[text_column].strip()) > 0)

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "num_samples": num_samples,
        "max_seq_len": max_seq_len,
        "num_generate": num_generate,
        "num_params": num_params,
        "samples": [],
    }

    total_prefill_time = 0.0
    total_decode_time = 0.0
    total_tokens_generated = 0
    all_perplexities = []

    for idx in tqdm(range(min(num_samples, len(dataset))), desc=f"Evaluating {dataset_name}"):
        sample = dataset[idx]
        text = sample[text_column]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_seq_len,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(device)
        input_seq_len = input_ids.shape[1]

        start_prefill = time.perf_counter()
        outputs = model(input_ids, use_cache=True)
        prefill_time = time.perf_counter() - start_prefill
        total_prefill_time += prefill_time

        past_key_values = outputs.past_key_values

        generation_times = []
        generated_tokens = []

        start_decode = time.perf_counter()
        next_token_logits = outputs.logits[:, -1, :]

        for step in range(num_generate):
            token_start = time.perf_counter()

            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())

            if step == 0:
                ttft = (time.perf_counter() - start_decode) * 1000

            inputs_with_cache = {
                "input_ids": next_token,
                "past_key_values": past_key_values,
                "use_cache": True,
            }
            outputs = model(**inputs_with_cache)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            token_time = time.perf_counter() - token_start
            generation_times.append(token_time)

        decode_time = time.perf_counter() - start_decode
        total_decode_time += decode_time
        total_tokens_generated += num_generate

        full_sequence = torch.cat([input_ids, torch.tensor(generated_tokens, device=device).unsqueeze(0)], dim=1)
        with torch.no_grad():
            outputs_full = model(full_sequence, labels=full_sequence)
            loss = outputs_full.loss
            perplexity = torch.exp(loss).item()

        all_perplexities.append(perplexity)

        sample_result = {
            "input_seq_len": input_seq_len,
            "generated_tokens": num_generate,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "ttft_ms": ttft,
            "tpot_ms": sum(generation_times[1:]) / (len(generation_times) - 1) * 1000 if len(generation_times) > 1 else 0,
            "throughput_tokens_s": num_generate / decode_time if decode_time > 0 else 0,
            "perplexity": perplexity,
        }
        results["samples"].append(sample_result)

    total_time = total_prefill_time + total_decode_time
    total_tokens_processed = sum(s["input_seq_len"] for s in results["samples"]) + total_tokens_generated

    avg_perplexity = sum(all_perplexities) / len(all_perplexities)
    avg_ttft = sum(s["ttft_ms"] for s in results["samples"]) / len(results["samples"])
    avg_tpot = sum(s["tpot_ms"] for s in results["samples"]) / len(results["samples"])
    avg_throughput = total_tokens_generated / total_decode_time if total_decode_time > 0 else 0

    avg_input_len = sum(s["input_seq_len"] for s in results["samples"]) / len(results["samples"])
    flops = estimate_flops(
        num_params=num_params,
        seq_len=int(avg_input_len),
        num_generated_tokens=num_generate,
        num_layers=config.num_hidden_layers,
        hidden_dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        vocab_size=config.vocab_size,
    )

    results["aggregate"] = {
        "avg_perplexity": avg_perplexity,
        "avg_ttft_ms": avg_ttft,
        "avg_tpot_ms": avg_tpot,
        "throughput_tokens_s": avg_throughput,
        "total_prefill_time_s": total_prefill_time,
        "total_decode_time_s": total_decode_time,
        "total_time_s": total_time,
        "total_tokens_processed": total_tokens_processed,
        "flops": flops,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation for Pythia-70m")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "pg19"], help="Dataset to use")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length for context")
    parser.add_argument("--num_generate", type=int, default=256, help="Number of tokens to generate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")

    args = parser.parse_args()

    results = evaluate_baseline(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        num_samples=args.num_samples,
        max_seq_len=args.max_seq_len,
        num_generate=args.num_generate,
        device=args.device,
    )

    print("\n" + "=" * 50)
    print("BASELINE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {results['model']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Samples: {results['num_samples']}")
    print(f"\nMetrics:")
    print(f"  Avg Perplexity: {results['aggregate']['avg_perplexity']:.4f}")
    print(f"  Avg TTFT: {results['aggregate']['avg_ttft_ms']:.2f} ms")
    print(f"  Avg TPOT: {results['aggregate']['avg_tpot_ms']:.2f} ms")
    print(f"  Throughput: {results['aggregate']['throughput_tokens_s']:.2f} tokens/s")
    print(f"\nTiming:")
    print(f"  Total Prefill Time: {results['aggregate']['total_prefill_time_s']:.2f} s")
    print(f"  Total Decode Time: {results['aggregate']['total_decode_time_s']:.2f} s")
    print(f"  Total Time: {results['aggregate']['total_time_s']:.2f} s")
    print(f"\nFLOPs:")
    print(f"  Prefill FLOPs: {results['aggregate']['flops']['prefill_flops']:,.0f}")
    print(f"  Decode FLOPs: {results['aggregate']['flops']['decode_flops']:,.0f}")
    print(f"  Total FLOPs: {results['aggregate']['flops']['total_flops']:,.0f}")
    print(f"  Avg FLOPs/Token: {results['aggregate']['flops']['avg_flops_per_token']:,.0f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
