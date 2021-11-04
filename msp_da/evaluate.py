"""
Evaluation metrics for MSP-DA.

- Event Detection (ACE-05): micro P/R/F1 ignoring the null class (label 0)
- Sentiment Analysis (FDU-MTL): binary accuracy
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_ed_metrics(
    all_preds: List[int],
    all_labels: List[int],
) -> Dict[str, float]:
    """Precision, Recall, F1 for event detection (ignores null class 0)."""
    tp = fp = tn = fn = 0.0
    for pred, label in zip(all_preds, all_labels):
        if pred == 0:
            if label == 0:
                tn += 1
            else:
                fn += 1
        else:
            if pred == label:
                tp += 1
            else:
                fp += 1
    precision = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_sa_metrics(
    all_preds: List[int],
    all_labels: List[int],
) -> Dict[str, float]:
    """Binary accuracy for sentiment analysis."""
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = 100.0 * correct / len(all_labels) if all_labels else 0.0
    return {"accuracy": acc}


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: torch.device,
    task: str = "ed",
) -> Dict[str, float]:
    """Run inference and compute metrics.

    Args:
        model:      MSPDAModel instance
        dataloader: evaluation DataLoader
        device:     torch device
        task:       "ed" (event detection) or "sa" (sentiment analysis)

    Returns:
        dict with evaluation metrics
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        anchor_positions = batch.get("anchor_positions")
        if anchor_positions is not None:
            anchor_positions = anchor_positions.to(device)
        labels = batch["labels"].to(device)

        pooler_output, _ = model.encode(
            input_ids, attention_mask, token_type_ids, anchor_positions
        )
        logits = model.classify(pooler_output)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    if task == "ed":
        return compute_ed_metrics(all_preds, all_labels)
    else:
        return compute_sa_metrics(all_preds, all_labels)
