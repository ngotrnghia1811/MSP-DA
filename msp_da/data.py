"""
Data loading for MSP-DA.

Supports two tasks:
- Event Detection (ACE-05): multi-class classification over 34 event types
- Sentiment Analysis (FDU-MTL): binary classification across 16 product review domains

ACE-05 data format (JSON per domain split):
    { domain: [ {"sent": str, "anc_pos": int, "label": int}, ... ] }

FDU-MTL data format (tab-separated text files):
    labeled:   <label>\t<text>
    unlabeled: <text>
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# ACE-05 Event Detection
# ---------------------------------------------------------------------------

def word_to_char_span(sentence: str, anchor_pos: int) -> Tuple[int, int]:
    """Convert word-level anchor position to character span."""
    tokens = sentence.split(" ")
    char_start = len(" ".join(tokens[:anchor_pos]))
    char_end = char_start + len(tokens[anchor_pos])
    if anchor_pos > 0:
        char_start += 1
        char_end += 1
    return char_start, char_end


def read_ace05(
    data_dir: str,
    split: str,
    domains: List[str],
    source_domains: List[str],
) -> Tuple[List[str], List[Tuple[int, int]], List[int], List[int]]:
    """Load ACE-05 data from JSON files.

    Args:
        data_dir:       directory containing JSON split files
        split:          one of "train", "eval", "test"
        domains:        list of domain keys to include
        source_domains: list of domain keys considered source (label 0), rest are target (label 1)

    Returns:
        texts, char_spans, event_labels, domain_labels
    """
    texts, spans, labels, dom_labels = [], [], [], []
    for fname in os.listdir(data_dir):
        if split not in fname:
            continue
        with open(os.path.join(data_dir, fname), "r") as f:
            data = json.load(f)
        for domain, examples in data.items():
            if domain not in domains:
                continue
            dom_label = 0 if domain in source_domains else 1
            for ex in examples:
                texts.append(ex["sent"])
                labels.append(ex["label"])
                span = word_to_char_span(ex["sent"], ex["anc_pos"])
                spans.append(span)
                dom_labels.append(dom_label)
    return texts, spans, labels, dom_labels


class AceEDDataset(Dataset):
    """ACE-05 Event Detection dataset."""

    def __init__(
        self,
        texts: List[str],
        spans: List[Tuple[int, int]],
        labels: List[int],
        dom_labels: List[int],
        tokenizer,
        max_length: int = 56,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels
        self.dom_labels = dom_labels
        self.anchor_positions = self._resolve_anchor_positions(spans)

    def _resolve_anchor_positions(self, spans: List[Tuple[int, int]]) -> List[int]:
        positions = []
        for i, (start, _) in enumerate(spans):
            tok_pos = self.encodings.char_to_token(i, start)
            if tok_pos is None:
                tok_pos = 1
            positions.append(tok_pos)
        return positions

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["dom_labels"] = torch.tensor(self.dom_labels[idx])
        item["anchor_positions"] = torch.tensor(self.anchor_positions[idx])
        return item


# ---------------------------------------------------------------------------
# FDU-MTL Sentiment Analysis
# ---------------------------------------------------------------------------

def read_fdu_mtl(
    data_dir: str,
    split: str,
    domains: Optional[List[str]] = None,
) -> Tuple[List[str], List[int], List[int]]:
    """Load FDU-MTL sentiment analysis data.

    File naming convention: <domain>.task.<split>
    Labeled format: <label>\t<text>
    Unlabeled format: <text>

    Args:
        data_dir: directory containing .task.train / .task.test / .task.unlabel files
        split:    "train", "test", or "unlabel"
        domains:  list of domain names; None = load all

    Returns:
        texts, sentiment_labels (−1 for unlabeled), domain_id_labels
    """
    texts, labels, dom_labels = [], [], []
    all_files = sorted(os.listdir(data_dir))
    available_domains = sorted(set(
        f.split(".task.")[0] for f in all_files if ".task." in f
    ))
    if domains is None:
        domains = available_domains

    for dom_id, dom in enumerate(available_domains):
        if dom not in domains:
            continue
        fname = f"{dom}.task.{split}"
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split("\t")
                if split in ("train", "test") and len(parts) >= 2:
                    texts.append(parts[1])
                    labels.append(int(parts[0]))
                elif split == "unlabel":
                    texts.append(parts[0])
                    labels.append(-1)
                dom_labels.append(dom_id)
    return texts, labels, dom_labels


class MTLDataset(Dataset):
    """FDU-MTL Sentiment Analysis dataset."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        dom_labels: List[int],
        tokenizer,
        max_length: int = 200,
    ):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels
        self.dom_labels = dom_labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["dom_labels"] = torch.tensor(self.dom_labels[idx])
        return item


# ---------------------------------------------------------------------------
# Combined source+target sampler (maintains target ratio per batch)
# ---------------------------------------------------------------------------

class SrcTgtBatchSampler(torch.utils.data.Sampler):
    """Interleaves source and target indices to maintain a fixed target ratio per batch."""

    def __init__(
        self,
        src_size: int,
        tgt_size: int,
        batch_size: int,
        target_ratio: float = 0.2,
    ):
        self.src_size = src_size
        self.tgt_size = tgt_size
        self.batch_size = batch_size
        self.tgt_per_batch = max(1, int(batch_size * target_ratio))
        self.src_per_batch = batch_size - self.tgt_per_batch

    def __iter__(self):
        src_perm = torch.randperm(self.src_size).tolist()
        tgt_perm = (torch.randperm(self.tgt_size) + self.src_size).tolist()

        src_batches = [src_perm[i:i + self.src_per_batch]
                       for i in range(0, len(src_perm), self.src_per_batch)]
        tgt_batches = [tgt_perm[i:i + self.tgt_per_batch]
                       for i in range(0, len(tgt_perm), self.tgt_per_batch)]

        for i, src_b in enumerate(src_batches):
            tgt_b = tgt_batches[i % len(tgt_batches)]
            yield src_b + tgt_b

    def __len__(self) -> int:
        return self.src_size // self.src_per_batch


def build_tokenizer(model_name: str = "bert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)
