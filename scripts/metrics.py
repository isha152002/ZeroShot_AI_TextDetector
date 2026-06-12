# Copyright (c) Guangsheng Bao.
# MIT License
# Exact copy of repo metrics.py — no changes needed.

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np


def get_roc_metrics(real_preds, sample_preds):
    """
    real_preds:   list of criterion scores for human texts
    sample_preds: list of criterion scores for machine texts
    Returns (fpr, tpr, roc_auc) — fpr/tpr are lists for the curve.
    """
    preds = real_preds + sample_preds
    # label 0 = human, 1 = machine
    labels = [0] * len(real_preds) + [1] * len(sample_preds)
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), roc_auc


def get_precision_recall_metrics(real_preds, sample_preds):
    """
    Returns (precision, recall, pr_auc) — precision/recall are lists.
    """
    preds = real_preds + sample_preds
    labels = [0] * len(real_preds) + [1] * len(sample_preds)
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), pr_auc