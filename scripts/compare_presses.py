import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "kvpress"))

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from kvpress.presses.streaming_llm_press import StreamingLLMPress
from kvpress.presses.tova_press import TOVAPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress
from kvpress.presses.think_press import ThinKPress
from kvpress.presses.lagkv_press import LagKVPress
from kvpress.presses.criticalkv_press import CriticalKVPress
from kvpress.presses.pyramidkv_press import PyramidKVPress

PRESS_CONFIG = {
    "SnapKVPress": (SnapKVPress, {"compression_ratio": 0.5, "window_size": 64}),
    "KnormPress": (KnormPress, {"compression_ratio": 0.5}),
    "ExpectedAttentionPress": (ExpectedAttentionPress, {"compression_ratio": 0.5}),
    "StreamingLLMPress": (StreamingLLMPress, {"compression_ratio": 0.5}),
    "TOVAPress": (TOVAPress, {"compression_ratio": 0.5}),
    "ObservedAttentionPress": (ObservedAttentionPress, {"compression_ratio": 0.5}),
    "ThinKPress": (ThinKPress, {"key_channel_compression_ratio": 0.5, "window_size": 32}),
    "LagKVPress": (LagKVPress, {"compression_ratio": 0.5}),
    "CriticalKVPress": (lambda **kw: CriticalKVPress(KnormPress(**kw)), {}),
    "PyramidKVPress": (PyramidKVPress, {"compression_ratio": 0.5}),
}


@torch.inference_mode()
def evaluate_press(
    model,
    tokenizer,
    dataset,
    text_column: str,
    press_config: tuple,
    user_kwargs: dict,
    num_samples: int,
    max_seq_len: int,
    num_generate: int,
    device: str,
    min_window_size: int = 64,
) -> Dict[str, Any]:
    press_class, default_kwargs = press_config
    press_kwargs = {**default_kwargs, **user_kwargs}

    try:
        press = press_class(**press_kwargs)
    except Exception as e:
        return {"error": str(e), "press_class": press_class.__name__}

    results = {
        "press_class": press_class.__name__,
        "press_kwargs": press_kwargs,
        "samples": [],
    }

    total_prefill_time = 0.0
    total_decode_time = 0.0
    total_tokens_generated = 0
    all_perplexities = []
    compressed_lengths = []

    sample_count = 0
    for idx in tqdm(range(min(num_samples, len(dataset))), desc=f"{press_class.__name__}", leave=False):
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

        if input_seq_len <= min_window_size:
            continue

        cache = DynamicCache()

        try:
            start_prefill = time.perf_counter()
            with press(model):
                outputs = model(input_ids, past_key_values=cache)
            prefill_time = time.perf_counter() - start_prefill
            total_prefill_time += prefill_time

            if len(cache.layers) > 0:
                compressed_len = cache.layers[0].keys.shape[2]
            else:
                compressed_len = input_seq_len
            compressed_lengths.append(compressed_len)

            start_decode = time.perf_counter()
            next_token_logits = outputs.logits[:, -1, :]
            generated_tokens = []

            for step in range(num_generate):
                token_start = time.perf_counter()
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_tokens.append(next_token.item())

                if step == 0:
                    ttft = (time.perf_counter() - start_decode) * 1000

                inputs_with_cache = {
                    "input_ids": next_token,
                    "past_key_values": cache,
                    "use_cache": True,
                }
                outputs = model(**inputs_with_cache)
                next_token_logits = outputs.logits[:, -1, :]
                cache = outputs.past_key_values

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
                "compressed_seq_len": compressed_len,
                "actual_compression_ratio": 1 - (compressed_len / input_seq_len),
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "ttft_ms": ttft,
                "perplexity": perplexity,
            }
            results["samples"].append(sample_result)
            sample_count += 1

        except Exception as e:
            results["error"] = str(e)
            results["error_at_sample"] = idx
            break

    if sample_count == 0:
        return {"error": "No valid samples", "press_class": press_class.__name__}

    avg_perplexity = sum(all_perplexities) / len(all_perplexities)
    avg_ttft = sum(s["ttft_ms"] for s in results["samples"]) / len(results["samples"])
    avg_throughput = total_tokens_generated / total_decode_time if total_decode_time > 0 else 0
    avg_compression_ratio = sum(s["actual_compression_ratio"] for s in results["samples"]) / len(results["samples"])
    avg_compressed_len = sum(compressed_lengths) / len(compressed_lengths)
    avg_input_len = sum(s["input_seq_len"] for s in results["samples"]) / len(results["samples"])

    results["aggregate"] = {
        "num_samples": sample_count,
        "avg_perplexity": avg_perplexity,
        "avg_ttft_ms": avg_ttft,
        "throughput_tokens_s": avg_throughput,
        "total_prefill_time_s": total_prefill_time,
        "total_decode_time_s": total_decode_time,
        "avg_compression_ratio": avg_compression_ratio,
        "avg_compressed_seq_len": avg_compressed_len,
        "avg_input_seq_len": avg_input_len,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare KVPress methods")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m", help="Model name or path")
    parser.add_argument("--dataset", type=str, default="wikitext", choices=["wikitext", "pg19"], help="Dataset to use")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--num_generate", type=int, default=256, help="Number of tokens to generate")
    parser.add_argument("--compression_ratio", type=float, default=0.5, help="Compression ratio")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--presses", type=str, nargs="+",
                        default=list(PRESS_CONFIG.keys()),
                        help="Presses to compare")

    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,}")

    print(f"Loading dataset: {args.dataset} ({args.dataset_config})")
    if args.dataset == "wikitext":
        dataset = load_dataset(args.dataset, args.dataset_config, split="test")
        text_column = "text"
    elif args.dataset == "pg19":
        dataset = load_dataset("parquet", data_files="datasets/pg19/pg19-test-100.parquet", split="train")
        text_column = "text"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    min_chars = (64 + 100) * 4
    dataset = dataset.filter(lambda x: len(x[text_column].strip()) > min_chars)

    all_results = {
        "model": args.model,
        "dataset": args.dataset,
        "num_params": num_params,
        "num_samples": args.num_samples,
        "max_seq_len": args.max_seq_len,
        "num_generate": args.num_generate,
        "compression_ratio": args.compression_ratio,
        "device": args.device,
        "presses": {},
    }

    user_kwargs = {"compression_ratio": args.compression_ratio}

    print("\n" + "=" * 60)
    print("COMPARING KV PRESS METHODS")
    print("=" * 60)

    for press_name in args.presses:
        if press_name not in PRESS_CONFIG:
            print(f"Skipping unknown press: {press_name}")
            continue

        press_config = PRESS_CONFIG[press_name]
        print(f"\nTesting: {press_name}")

        result = evaluate_press(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            text_column=text_column,
            press_config=press_config,
            user_kwargs=user_kwargs,
            num_samples=args.num_samples,
            max_seq_len=args.max_seq_len,
            num_generate=args.num_generate,
            device=args.device,
        )

        all_results["presses"][press_name] = result

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            agg = result["aggregate"]
            print(f"  Perplexity: {agg['avg_perplexity']:.4f}")
            print(f"  Throughput: {agg['throughput_tokens_s']:.2f} tokens/s")
            print(f"  TTFT: {agg['avg_ttft_ms']:.2f} ms")
            print(f"  Compression: {agg['avg_compression_ratio']:.2%}")

    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Press':<25} {'PPL':<10} {'Throughput':<15} {'TTFT (ms)':<12} {'Compression'}")
    print("-" * 70)

    valid_presses = [(name, r) for name, r in all_results["presses"].items() if "error" not in r]
    valid_presses.sort(key=lambda x: x[1]["aggregate"]["avg_perplexity"])

    for press_name, result in valid_presses:
        agg = result["aggregate"]
        print(f"{press_name:<25} {agg['avg_perplexity']:<10.4f} {agg['throughput_tokens_s']:<15.2f} "
              f"{agg['avg_ttft_ms']:<12.2f} {agg['avg_compression_ratio']:.2%}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return all_results


if __name__ == "__main__":
    main()
