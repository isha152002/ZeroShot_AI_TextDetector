# Copyright (c) Guangsheng Bao.
# MIT License
# Changes: stripped OpenAI/T5 generation paths; kept only load_data + 
#          HuggingFace-based generation for gpt-neo-2.7B

import json
import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ---------- data loading (used by fast_detect_gpt.py) ----------

def load_data(data_file):
    with open(data_file, "r") as f:
        data = json.load(f)
    return data


# ---------- dataset helpers ----------

def load_base_data(dataset, n_samples):
    """Returns list of (human_text, prompt) from xsum / squad / writing."""
    if dataset == "xsum":
        d = load_dataset("xsum", split="test", trust_remote_code=True)
        texts = [x["document"] for x in d]
    elif dataset == "squad":
        d = load_dataset("squad", split="validation", trust_remote_code=True)
        texts = [x["context"] for x in d]
    elif dataset == "writing":
        d = load_dataset("bookcorpus", split="train", trust_remote_code=True)
        texts = [x["text"] for x in d]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # deduplicate and take first n_samples
    seen = set()
    unique = []
    for t in texts:
        t = t.strip()
        if t not in seen:
            seen.add(t)
            unique.append(t)
        if len(unique) == n_samples:
            break
    return unique


def truncate(text, tokenizer, max_length=200):
    tokens = tokenizer.encode(text)[:max_length]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# ---------- generation ----------

def generate_samples(base_texts, tokenizer, model, device,
                     do_top_p=False, top_p=0.96,
                     do_temperature=False, temperature=0.8,
                     max_new_tokens=200, prompt_tokens=30):
    """
    For each human text:
      1. take first `prompt_tokens` tokens as prompt
      2. generate a continuation with the model
    Returns list of generated strings (continuations only, no prompt).
    """
    generated = []
    for text in tqdm(base_texts, desc="Generating samples"):
        prompt_ids = tokenizer.encode(text)[:prompt_tokens]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
        if do_top_p:
            gen_kwargs.update(do_sample=True, top_p=top_p)
        elif do_temperature:
            gen_kwargs.update(do_sample=True, temperature=temperature)
        else:
            gen_kwargs.update(do_sample=True, top_k=40)

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        # strip the prompt tokens from output
        new_tokens = out[0][inputs.input_ids.shape[1]:]
        generated.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return generated


# ---------- main ----------

def build_dataset(args):
    print(f"Loading base dataset: {args.dataset}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name, padding_side="right"
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_texts_raw = load_base_data(args.dataset, args.n_samples)

    # truncate human texts to max_length tokens
    original_texts = [truncate(t, tokenizer) for t in base_texts_raw]
    original_texts = original_texts[:args.n_samples]

    print(f"Loading generation model: {args.base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device

    sampled_texts = generate_samples(
        original_texts, tokenizer, model, device,
        do_top_p=args.do_top_p,
        do_temperature=args.do_temperature,
        temperature=getattr(args, "temperature", 0.8),
    )

    data = {"original": original_texts, "sampled": sampled_texts}
    with open(args.output_file, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(original_texts)} pairs to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--base_model_name", type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--output_file", type=str, default="./exp_main/data/xsum_gpt-neo-2.7B")
    parser.add_argument("--cache_dir", type=str, default="../cache")
    parser.add_argument("--do_top_p", action="store_true")
    parser.add_argument("--do_temperature", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    build_dataset(args)