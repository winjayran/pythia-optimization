import argparse
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

def get_num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_ppl(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    max_length: int = 2048,
    stride: int = 512,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, int]:
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0]
    seq_len = len(input_ids)
    
    if seq_len < 2:
        return float("inf"), 0
    
    if seq_len <= max_length:
        input_ids = input_ids.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        ppl = np.exp(loss)
        return ppl, seq_len - 1
    
    nlls = []
    total_eval_tokens = 0
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        window_ids = input_ids[begin_loc:end_loc].unsqueeze(0).to(device)
        
        if begin_loc == 0:
            trg_len = end_loc - 1
        else:
            trg_len = end_loc - prev_end_loc
        
        if trg_len <= 0:
            continue
        
        labels = window_ids.clone()
        labels[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(window_ids, labels=labels)
            neg_log_likelihood = outputs.loss.item() * trg_len
        
        nlls.append(neg_log_likelihood)
        total_eval_tokens += trg_len
        prev_end_loc = end_loc
        
        if end_loc == seq_len:
            break
    
    if total_eval_tokens == 0:
        return float("inf"), 0
    
    ppl = np.exp(sum(nlls) / total_eval_tokens)
    return ppl, total_eval_tokens


def run_generation_benchmark(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    prompt_len = input_ids.size(1)
    
    t_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    ttft = time.perf_counter() - t_start
    
    print(f"  [DEBUG] Prefill + first token: {ttft:.6f}s")
    
    generated_ids = [next_token.item()]
    past_key_values = outputs.past_key_values
    current_input = next_token
    
    t_per_token_list = []
    total_gen_tokens = 0
    
    for step in range(max_new_tokens - 1):
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        t1 = time.perf_counter()
        
        dt = t1 - t0
        t_per_token_list.append(dt)
        generated_ids.append(next_token.item())
        past_key_values = outputs.past_key_values
        current_input = next_token
        total_gen_tokens += 1
        
        if (step + 1) % 10 == 0:
            print(f"  [DEBUG] Token {step+2}/{max_new_tokens}: {dt:.6f}s")
    
    total_gen_time = sum(t_per_token_list)
    avg_tpot = total_gen_time / total_gen_tokens if total_gen_tokens > 0 else 0.0
    throughput = total_gen_tokens / total_gen_time if total_gen_tokens > 0 else 0.0
    
    params = get_num_parameters(model)
    total_flops = sum(2 * params * (prompt_len + i) for i in range(max_new_tokens))
    total_gflops = total_flops / 1e9
    
    metrics = {
        "prompt_length": prompt_len,
        "new_tokens_generated": max_new_tokens,
        "TTFT_seconds": ttft,
        "TPOT_avg_seconds": avg_tpot,
        "generation_throughput_tokens_per_sec": throughput,
        "total_estimated_GFLOPs_generation": total_gflops,
    }
    
    return metrics, generated_ids


def load_wikitext_articles(
    data_dir: str,
    split: str = "test",
    min_chars: int = 500,
    max_articles: int = 10,
) -> List[str]:
    texts = []
    data_path = Path(data_dir)
    
    try:
        dataset = load_from_disk(str(data_path))
        if split in dataset:
            for sample in dataset[split]:
                text = sample.get('text', '')
                if text.strip() and len(text) >= min_chars:
                    lines = [l.strip() for l in text.split('\n') if l.strip()]
                    clean_text = ' '.join(lines)
                    texts.append(clean_text)
                    if len(texts) >= max_articles:
                        break
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        raise
    
    return texts

def load_pg19_articles(
    data_dir: str,
    min_chars: int = 500,
    max_articles: int = 10,
) -> List[str]:
    from datasets import load_dataset
    import glob
    texts = []
    
    try:
        parquet_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
        if not parquet_files:
            print(f"  Warning: No parquet files found in {data_dir}. Falling back to load from datasets Hub.")
            dataset = load_dataset('emozilla/pg19', split='validation')
        else:
            dataset = load_dataset('parquet', data_files=parquet_files, split='train')
            
        for sample in dataset:
            text = sample.get('text', '')
            if text.strip() and len(text) >= min_chars:
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                clean_text = ' '.join(lines)
                texts.append(clean_text)
                if len(texts) >= max_articles:
                    break
    except Exception as e:
        print(f"  Error loading PG19 dataset: {e}")
        raise
    
    return texts

def main():
    parser = argparse.ArgumentParser(
        description="CPU Baseline for Pythia-70M — PPL + Generation"
    )
    parser.add_argument("--model_path", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--data_dir", type=str, default="./wikitext")
    parser.add_argument("--mode", type=str, default="ppl,generation")
    parser.add_argument("--max_ppl_length", type=int, default=2048)
    parser.add_argument("--ppl_stride", type=int, default=512)
    parser.add_argument("--gen_prompt", type=str, default="The capital of France is")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--output_json", type=str, default="baseline_cpu_results.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--min_article_chars", type=int, default=0)
    parser.add_argument("--max_articles", type=int, default=100)
    args = parser.parse_args()
    
    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"{'='*60}")
    print(f"CPU Baseline for Pythia-70M")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    
    num_params = get_num_parameters(model)
    print(f"Parameters: {num_params:,}")
    
    datasets = [d.strip() for d in args.dataset.split(",")]
    modes = [m.strip() for m in args.mode.split(",")]
    
    results = {
        "model": args.model_path,
        "num_parameters": num_params,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if "ppl" in modes:
        for ds_name in datasets:
            print(f"\n{'='*60}")
            print(f"PPL Evaluation on {ds_name}")
            print(f"{'='*60}")
            
            if ds_name.lower() in ["wikitext", "wikitext-2"]:
                texts = load_wikitext_articles(
                    data_dir=args.data_dir,
                    split="test",
                    min_chars=args.min_article_chars,
                    max_articles=args.max_articles,
                )
            elif ds_name.lower() == "pg19":
                texts = load_pg19_articles(
                    data_dir=args.data_dir,
                    min_chars=args.min_article_chars,
                    max_articles=args.max_articles,
                )
            else:
                print(f"  Skipping {ds_name} (not implemented)")
                continue
            
            if not texts:
                print("  ✗ No valid articles found")
                results[f"ppl_{ds_name}"] = "N/A"
                continue
            
            print(f"  Articles loaded: {len(texts)}")
            print(f"  Article lengths (chars): {[len(t) for t in texts]}")
            
            combined = " ".join(texts)
            print(f"  Combined text: {len(combined):,} chars")
            
            t0 = time.perf_counter()
            ppl, tok_count = compute_ppl(
                model, tokenizer, combined,
                max_length=args.max_ppl_length,
                stride=args.ppl_stride,
                device=device,
            )
            elapsed = time.perf_counter() - t0
            
            print(f"\n  ✓ PPL: {ppl:.3f}")
            print(f"  ✓ Tokens evaluated: {tok_count:,}")
            print(f"  ✓ Time: {elapsed:.1f}s")
            print(f"  ✓ Tokens/second: {tok_count/elapsed:.1f}")
            
            results[f"ppl_{ds_name}"] = round(ppl, 3)
            results[f"ppl_{ds_name}_tokens"] = tok_count
            results[f"ppl_{ds_name}_time_seconds"] = round(elapsed, 1)
            results[f"ppl_{ds_name}_tokens_per_second"] = round(tok_count/elapsed, 1)
    
    if "generation" in modes:
        print(f"\n{'='*60}")
        print(f"Generation Benchmark")
        print(f"{'='*60}")
        print(f"Prompt: '{args.gen_prompt}'")
        print(f"Max new tokens: {args.max_new_tokens}")
        
        gen_metrics, gen_ids = run_generation_benchmark(
            model, tokenizer, args.gen_prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        
        print(f"\n{'─'*40}")
        print(f"Generation Metrics:")
        print(f"{'─'*40}")
        print(f"  TTFT:        {gen_metrics['TTFT_seconds']*1000:.2f} ms")
        print(f"  TPOT (avg):  {gen_metrics['TPOT_avg_seconds']*1000:.2f} ms")
        print(f"  Throughput:  {gen_metrics['generation_throughput_tokens_per_sec']:.1f} tokens/s")
        print(f"  GFLOPs:      {gen_metrics['total_estimated_GFLOPs_generation']:.2f}")
        print(f"{'─'*40}")
        
        for k, v in gen_metrics.items():
            results[f"gen_{k}"] = v if isinstance(v, (int, float)) else v
        
        enc = tokenizer(args.gen_prompt, return_tensors="pt").to(device)
        full_output = model.generate(
            enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_text = tokenizer.decode(full_output[0], skip_special_tokens=True)
        print(f"\n  Generated text:")
        print(f"  {gen_text}")
        results["gen_text"] = gen_text
    
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output_json}")
    print(f"{'='*60}")
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for key, val in results.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()