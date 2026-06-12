# Copyright (c) Guangsheng Bao.
# MIT License
# Changes: hardcoded gpt-neo-2.7B support; added fp16 + device_map for T4x2

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_tokenizer(model_name, dataset=None, cache_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side="right",
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def load_model(model_name, device, cache_dir=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,       # fp16 to fit T4 VRAM
        device_map="auto",               # spreads across both T4s if needed
    )
    model.eval()
    return model