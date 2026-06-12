# Copyright (c) Guangsheng Bao.
# MIT License
# Changes:
#   - default model hardcoded to EleutherAI/gpt-neo-2.7B
#   - removed truncation=True from tokenizer call in experiment() — repo doesn't
#     truncate here, truncation is handled at data_builder stage
#   - added epsilon=1e-8 guard on sigma in get_sampling_discrepancy to avoid /0

import random
import numpy as np
import torch
import tqdm
import argparse
import json

from model import load_tokenizer, load_model
from data_builder import load_data
from metrics import get_roc_metrics, get_precision_recall_metrics


# ------------------------------------------------------------------ #
#  Core scoring functions — exact copy from repo                       #
# ------------------------------------------------------------------ #

def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)


def get_sampling_discrepancy(logits_ref, logits_score, labels):
    """Monte-Carlo version: samples 10k token sequences from ref distribution."""
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1

    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)

    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    sigma_tilde = sigma_tilde + 1e-8  # guard against zero std

    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()


def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    """Analytic version: computes mean/var of the scoring distribution in closed form."""
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1

    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels

    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref    = torch.softmax(logits_ref, dim=-1)

    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref  = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / (var_ref.sum(dim=-1).sqrt() + 1e-8)
    discrepancy = discrepancy.mean()
    return discrepancy.item()


# ------------------------------------------------------------------ #
#  Experiment loop — exact copy from repo, default model changed       #
# ------------------------------------------------------------------ #

def experiment(args):
    # load scoring model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model     = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()

    # load sampling model only if different
    if args.sampling_model_name != args.scoring_model_name:
        sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
        sampling_model     = load_model(args.sampling_model_name, args.device, args.cache_dir)
        sampling_model.eval()

    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])

    # choose criterion
    if args.discrepancy_analytic:
        name         = "sampling_discrepancy_analytic"
        criterion_fn = get_sampling_discrepancy_analytic
    else:
        name         = "sampling_discrepancy"
        criterion_fn = get_sampling_discrepancy

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text  = data["sampled"][idx]

        # ---- original (human) text ----
        tokenized = scoring_tokenizer(
            original_text, return_tensors="pt", padding=True,
            return_token_type_ids=False
        ).to(args.device)
        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]

        if args.sampling_model_name == args.scoring_model_name:
            logits_ref = logits_score
        else:
            tokenized_s = sampling_tokenizer(
                original_text, return_tensors="pt", padding=True,
                return_token_type_ids=False
            ).to(args.device)
            assert torch.all(tokenized_s.input_ids[:, 1:] == labels), "Tokenizer mismatch."
            logits_ref = sampling_model(**tokenized_s).logits[:, :-1]

        original_crit = criterion_fn(logits_ref, logits_score, labels)

        # ---- sampled (machine) text ----
        tokenized = scoring_tokenizer(
            sampled_text, return_tensors="pt", padding=True,
            return_token_type_ids=False
        ).to(args.device)
        labels = tokenized.input_ids[:, 1:]

        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]

        if args.sampling_model_name == args.scoring_model_name:
            logits_ref = logits_score
        else:
            tokenized_s = sampling_tokenizer(
                sampled_text, return_tensors="pt", padding=True,
                return_token_type_ids=False
            ).to(args.device)
            assert torch.all(tokenized_s.input_ids[:, 1:] == labels), "Tokenizer mismatch."
            logits_ref = sampling_model(**tokenized_s).logits[:, :-1]

        sampled_crit = criterion_fn(logits_ref, logits_score, labels)

        results.append({
            "original":      original_text,
            "original_crit": original_crit,
            "sampled":       sampled_text,
            "sampled_crit":  sampled_crit,
        })

    # DEBUG — verify score ranges and separation
    import numpy as np
    real_scores   = [x["original_crit"] for x in results]
    sample_scores = [x["sampled_crit"]  for x in results]

    print("\n=== DEBUG ===")
    print(f"Human  scores — mean: {np.mean(real_scores):.3f}  std: {np.std(real_scores):.3f}  min: {np.min(real_scores):.3f}  max: {np.max(real_scores):.3f}")
    print(f"AI     scores — mean: {np.mean(sample_scores):.3f}  std: {np.std(sample_scores):.3f}  min: {np.min(sample_scores):.3f}  max: {np.max(sample_scores):.3f}")
    print(f"Separation (AI mean - Human mean): {np.mean(sample_scores) - np.mean(real_scores):.3f}")
    print(f"Inf count  — human: {sum(np.isinf(real_scores))}  AI: {sum(np.isinf(sample_scores))}")
    print(f"NaN count  — human: {sum(np.isnan(real_scores))}  AI: {sum(np.isnan(sample_scores))}")
    print(f"Sample human scores (first 5): {real_scores[:5]}")
    print(f"Sample AI     scores (first 5): {sample_scores[:5]}")
    print("=== END DEBUG ===\n")

    # metrics
    predictions = {
        "real":    [x["original_crit"] for x in results],
        "samples": [x["sampled_crit"]  for x in results],
    }
    print(
        f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, "
        f"Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}"
    )

    fpr, tpr, roc_auc = get_roc_metrics(predictions["real"], predictions["samples"])
    p, r, pr_auc      = get_precision_recall_metrics(predictions["real"], predictions["samples"])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    results_file = f"{args.output_file}.{name}.json"
    output = {
        "name": f"{name}_threshold",
        "info": {"n_samples": n_samples},
        "predictions": predictions,
        "raw_results": results,
        "metrics":    {"roc_auc": roc_auc, "fpr": fpr, "tpr": tpr},
        "pr_metrics": {"pr_auc": pr_auc, "precision": p, "recall": r},
        "loss": 1 - pr_auc,
    }
    with open(results_file, "w") as fout:
        json.dump(output, fout)
    print(f"Results written into {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file",         type=str,  default="./exp_main/results/xsum_gpt-neo-2.7B")
    parser.add_argument("--dataset",             type=str,  default="xsum")
    parser.add_argument("--dataset_file",        type=str,  default="./exp_main/data/xsum_gpt-neo-2.7B")
    parser.add_argument("--sampling_model_name", type=str,  default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--scoring_model_name",  type=str,  default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--discrepancy_analytic", action="store_true")
    parser.add_argument("--seed",                type=int,  default=0)
    parser.add_argument("--device",              type=str,  default="cuda")
    parser.add_argument("--cache_dir",           type=str,  default="../cache")
    args = parser.parse_args()
    experiment(args)