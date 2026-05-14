import os
import sys

num_threads = None
for i, arg in enumerate(sys.argv):
    if arg == "--num_threads" and i + 1 < len(sys.argv):
        try:
            num_threads = int(sys.argv[i + 1])
        except ValueError:
            pass
        break

if num_threads is None:
    num_threads = os.cpu_count() or 4

os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["KMP_BLOCKTIME"] = "0"

import argparse
import json
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

class OptimizedPythia:

    def __init__(
        self,
        model_path: str,
        dtype: str = "float32",
        use_sdpa: bool = False,
        use_compile: bool = False,
        num_threads: int = 8,
    ):
        self.device = torch.device("cpu")
        self.use_sdpa = use_sdpa
        self.num_threads = num_threads
        self.dtype_str = dtype

        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(1)

        if hasattr(torch.backends, 'opt_eager'):
            torch.backends.opt_eager.enable(True)

        print(f"CPU threads: {num_threads}")
        print(f"MKL enabled: {torch.backends.mkl.is_available()}")
        print(f"MKLDNN enabled: {torch.backends.mkldnn.is_available()}")

        self.dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = self.dtype_map.get(dtype, torch.float32)

        print(f"\n{'='*60}")
        print(f"Loading Pythia-70M (dtype={dtype}, compile={use_compile})")
        print(f"{'='*60}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": False,
        }

        if use_sdpa:
            load_kwargs["attn_implementation"] = "sdpa"

        print("Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )

        self.model = self.model.to(memory_format=torch.channels_last)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        if use_compile:
            print("Applying torch.compile...")
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                print("  torch.compile applied")
            except Exception as e:
                print(f"  Warning: torch.compile failed: {e}")

        self.num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {self.num_params:,}")
        print(f"Model size: {self._get_model_size():.1f} MB")

        print("Warming up...")
        self._warmup()

    def _get_model_size(self) -> float:
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return param_size / (1024 ** 2)

    def _warmup(self):
        dummy = torch.randint(0, 1000, (1, 32), device=self.device)
        with torch.no_grad():
            for _ in range(2):
                _ = self.model(dummy, labels=dummy)
                _ = self.model(dummy[:, -1:], use_cache=True)
        import gc
        gc.collect()

    @torch.no_grad()
    def compute_ppl(
        self,
        text: str,
        max_length: int = 2048,
        stride: int = 512,
    ) -> Tuple[float, int]:
        encodings = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids[0]
        seq_len = len(input_ids)

        if seq_len < 2:
            return float("inf"), 0

        if seq_len <= max_length:
            input_ids = input_ids.unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
            ppl = np.exp(loss)
            return ppl, seq_len - 1

        nlls = []
        total_eval_tokens = 0
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            window_ids = input_ids[begin_loc:end_loc].unsqueeze(0).to(self.device)

            if begin_loc == 0:
                trg_len = end_loc - 1
            else:
                trg_len = end_loc - prev_end_loc

            if trg_len <= 0:
                continue

            labels = window_ids.clone()
            labels[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(window_ids, labels=labels)
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

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20,
    ) -> Dict:
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)
        attention_mask = enc.attention_mask.to(self.device)
        prompt_len = input_ids.size(1)

        t_start = time.perf_counter()

        outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=True)
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        ttft = time.perf_counter() - t_start

        generated_ids = [next_token.item()]
        t_per_token_list = []
        current_input = next_token

        for _ in range(max_new_tokens - 1):
            t0 = time.perf_counter()

            outputs = self.model(
                current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            dt = time.perf_counter() - t0
            t_per_token_list.append(dt)
            generated_ids.append(next_token.item())
            current_input = next_token

        total_gen_tokens = len(t_per_token_list)
        total_gen_time = sum(t_per_token_list)
        avg_tpot = total_gen_time / total_gen_tokens if total_gen_tokens > 0 else 0.0
        throughput = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0.0

        total_flops = sum(2 * self.num_params * (prompt_len + i) for i in range(max_new_tokens))
        total_gflops = total_flops / 1e9

        full_ids = torch.cat([input_ids[0], torch.tensor(generated_ids, device=self.device)])
        generated_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)

        return {
            "prompt_length": prompt_len,
            "new_tokens_generated": max_new_tokens,
            "TTFT_seconds": ttft,
            "TPOT_avg_seconds": avg_tpot,
            "generation_throughput_tokens_per_sec": throughput,
            "total_estimated_GFLOPs_generation": total_gflops,
            "generated_text": generated_text,
            "ttft_ms": round(ttft * 1000, 2),
            "tpot_ms": round(avg_tpot * 1000, 2),
            "tpot_min_ms": round(min(t_per_token_list) * 1000, 2) if t_per_token_list else 0,
            "tpot_max_ms": round(max(t_per_token_list) * 1000, 2) if t_per_token_list else 0,
            "throughput_tokens_per_sec": round(throughput, 1),
            "total_gflops": round(total_gflops, 2),
            "tokens_generated": max_new_tokens,
        }

def load_wikitext(data_dir: str, min_chars: int = 0, max_articles: int = 10) -> str:
    from pathlib import Path
    texts = []
    try:
        dataset = load_from_disk(str(Path(data_dir)))
        if 'test' in dataset:
            for sample in dataset['test']:
                text = sample.get('text', '')
                if text.strip() and len(text) >= min_chars:
                    lines = [l.strip() for l in text.split('\n') if l.strip()]
                    clean_text = ' '.join(lines)
                    texts.append(clean_text)
                    if len(texts) >= max_articles:
                        break
    except Exception as e:
        print(f"  Error: {e}")
    return ' '.join(texts)


def load_pg19(data_dir: str, min_chars: int = 0, max_articles: int = 10) -> str:
    import glob
    texts = []
    try:
        parquet_files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
        if not parquet_files:
            from datasets import load_dataset
            dataset = load_dataset('emozilla/pg19', split='validation')
        else:
            from datasets import load_dataset
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
        print(f"  Error: {e}")
    return ' '.join(texts)


def main():
    parser = argparse.ArgumentParser(description="Optimized CPU Inference - Generation Throughput")
    parser.add_argument("--model_path", type=str, default="./models/pythia-70m")
    parser.add_argument("--dataset", type=str, default="pg19")
    parser.add_argument("--data_dir", type=str, default="./datasets/pg19")
    parser.add_argument("--mode", type=str, default="ppl,generation")
    parser.add_argument("--max_ppl_length", type=int, default=2048)
    parser.add_argument("--ppl_stride", type=int, default=512)
    parser.add_argument("--gen_prompt", type=str, default="The capital of France is")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--output_json", type=str, default="optimized_results.json")
    parser.add_argument("--min_article_chars", type=int, default=0)
    parser.add_argument("--max_articles", type=int, default=2)
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--use_sdpa", action="store_true", default=False)
    parser.add_argument("--use_compile", action="store_true", default=False)
    parser.add_argument("--num_threads", type=int, default=8)

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Optimized CPU Inference - Generation Throughput")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Dtype: {args.dtype}")
    print(f"Threads: {args.num_threads}")
    print(f"SDPA: {args.use_sdpa}")
    print(f"Compile: {args.use_compile}")

    model = OptimizedPythia(
        model_path=args.model_path,
        dtype=args.dtype,
        use_sdpa=args.use_sdpa,
        use_compile=args.use_compile,
        num_threads=args.num_threads,
    )

    results = {
        "model": args.model_path,
        "config": {
            "dtype": args.dtype,
            "sdpa": args.use_sdpa,
            "compile": args.use_compile,
            "num_threads": args.num_threads,
        },
        "num_parameters": model.num_params,
        "device": "cpu",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    modes = [m.strip() for m in args.mode.split(",")]

    if "ppl" in modes:
        print(f"\n{'='*60}")
        print(f"PPL Evaluation on {args.dataset}")
        print(f"{'='*60}")

        if args.dataset.lower() in ["wikitext", "wikitext-2"]:
            texts = load_wikitext(args.data_dir, args.min_article_chars, args.max_articles)
        elif args.dataset.lower() == "pg19":
            texts = load_pg19(args.data_dir, args.min_article_chars, args.max_articles)
        else:
            texts = ""

        if not texts:
            print("  No valid articles found")
            return

        texts_list = [texts] if isinstance(texts, str) and texts else []
        if not texts_list:
            print("  No valid articles found")
            return

        combined = " ".join(texts_list)
        print(f"  Combined text: {len(combined):,} chars")

        t0 = time.perf_counter()
        ppl, tok_count = model.compute_ppl(combined, args.max_ppl_length, args.ppl_stride)
        elapsed = time.perf_counter() - t0

        print(f"\n  PPL: {ppl:.3f}")
        print(f"  Tokens: {tok_count:,}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Speed: {tok_count/elapsed:.1f} tok/s")

        results[f"ppl_{args.dataset}"] = round(ppl, 3)
        results[f"ppl_{args.dataset}_tokens"] = tok_count
        results[f"ppl_{args.dataset}_time_seconds"] = round(elapsed, 1)
        results[f"ppl_{args.dataset}_tokens_per_second"] = round(tok_count/elapsed, 1)

    if "generation" in modes:
        print(f"\n{'='*60}")
        print(f"Generation Benchmark")
        print(f"{'='*60}")
        print(f"Prompt: '{args.gen_prompt}'")

        gen = model.generate(args.gen_prompt, args.max_new_tokens)

        print(f"\n{'─'*50}")
        print(f"  TTFT:          {gen['ttft_ms']:.2f} ms")
        print(f"  TPOT (avg):    {gen['tpot_ms']:.2f} ms")
        print(f"  TPOT (min):    {gen['tpot_min_ms']:.2f} ms")
        print(f"  TPOT (max):    {gen['tpot_max_ms']:.2f} ms")
        print(f"  Throughput:    {gen['throughput_tokens_per_sec']:.1f} tok/s")
        print(f"  Total GFLOPs:  {gen['total_gflops']:.2f}")
        print(f"  {'─'*50}")

        for k, v in gen.items():
            results[f"gen_{k}"] = v

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results: {args.output_json}")
    print(f"{'='*60}")
    print(f"\nKEY METRICS:")
    if f"ppl_{args.dataset}" in results:
        print(f"  PPL:        {results[f'ppl_{args.dataset}']}")
    if "gen_ttft_ms" in results:
        print(f"  TTFT:       {results['gen_ttft_ms']} ms")
        print(f"  TPOT:       {results['gen_tpot_ms']} ms")
        print(f"  Throughput: {results['gen_throughput_tokens_per_sec']} tok/s")


if __name__ == "__main__":
    main()
