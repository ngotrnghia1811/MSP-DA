"""
Entry point for MSP-DA evaluation.

Usage:
    python evaluate.py --config configs/ace05_bc.yaml --checkpoint checkpoints/best_model.pt
"""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from msp_da.config import MSPDAConfig
from msp_da.data import (
    AceEDDataset,
    MTLDataset,
    build_tokenizer,
    read_ace05,
    read_fdu_mtl,
)
from msp_da.evaluate import evaluate as eval_fn
from msp_da.model import MSPDAModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MSP-DA checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    args = parser.parse_args()

    cfg = MSPDAConfig.from_yaml(args.config)
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = build_tokenizer(cfg.model.bert_model_name)
    model = MSPDAModel(cfg)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    dc = cfg.data
    if cfg.data.task == "ed":
        test_texts, test_spans, test_labels, test_dom = read_ace05(
            dc.test_file, "test", [dc.target_domain], dc.source_domains
        )
        test_ds = AceEDDataset(test_texts, test_spans, test_labels, test_dom,
                               tokenizer, dc.max_seq_length)
    else:
        test_texts, test_labels, test_dom = read_fdu_mtl(dc.test_file, "test")
        test_dom_labels = [1] * len(test_texts)
        test_ds = MTLDataset(test_texts, test_labels, test_dom_labels, tokenizer, dc.max_seq_length)

    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2)
    metrics = eval_fn(model, test_loader, device, task=cfg.data.task)
    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2f}")


if __name__ == "__main__":
    main()
