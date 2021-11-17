"""
Entry point for MSP-DA training.

Usage:
    python train.py --config configs/ace05_bc.yaml
    python train.py --config configs/fdu_mtl_MR.yaml
"""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from msp_da.config import MSPDAConfig
from msp_da.data import (
    AceEDDataset,
    MTLDataset,
    SrcTgtBatchSampler,
    build_tokenizer,
    read_ace05,
    read_fdu_mtl,
)
from msp_da.model import MSPDAModel
from msp_da.train import meta_train, source_pretrain


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_ace05_loaders(cfg: MSPDAConfig, tokenizer):
    dc = cfg.data
    src_texts, src_spans, src_labels, src_dom = read_ace05(
        dc.train_file, "train", dc.source_domains, dc.source_domains
    )
    tgt_texts, tgt_spans, tgt_labels, tgt_dom = read_ace05(
        dc.train_file, "train", [dc.target_domain], dc.source_domains
    )
    eval_texts, eval_spans, eval_labels, eval_dom = read_ace05(
        dc.dev_file, "eval", dc.source_domains, dc.source_domains
    )
    test_texts, test_spans, test_labels, test_dom = read_ace05(
        dc.test_file, "test", [dc.target_domain], dc.source_domains
    )

    src_ds = AceEDDataset(src_texts, src_spans, src_labels, src_dom, tokenizer, dc.max_seq_length)
    tgt_ds = AceEDDataset(tgt_texts, tgt_spans, tgt_labels, tgt_dom, tokenizer, dc.max_seq_length)
    eval_ds = AceEDDataset(eval_texts, eval_spans, eval_labels, eval_dom, tokenizer, dc.max_seq_length)
    test_ds = AceEDDataset(test_texts, test_spans, test_labels, test_dom, tokenizer, dc.max_seq_length)

    return src_ds, tgt_ds, eval_ds, test_ds


def build_mtl_loaders(cfg: MSPDAConfig, tokenizer):
    dc = cfg.data
    all_doms = None

    src_texts, src_labels, src_dom = read_fdu_mtl(dc.train_file, "train")
    src_texts = [t for t, d in zip(src_texts, src_dom)
                 if _resolve_dom_name(dc.train_file, d) != dc.target_domain]
    src_dom_f = [d for d in src_dom
                 if _resolve_dom_name(dc.train_file, d) != dc.target_domain]
    src_labels_f = [l for l, d in zip(src_labels, src_dom)
                    if _resolve_dom_name(dc.train_file, d) != dc.target_domain]
    src_dom_labels = [0] * len(src_texts)

    tgt_train_texts, _, tgt_dom = read_fdu_mtl(dc.train_file, "train")
    tgt_unlabel_texts, _, _ = read_fdu_mtl(dc.train_file, "unlabel")
    tgt_texts = []
    tgt_labels = []
    for t, d in zip(tgt_train_texts + tgt_unlabel_texts,
                    [d for d in tgt_dom] + [d for d in tgt_dom]):
        if _resolve_dom_name(dc.train_file, d) == dc.target_domain:
            tgt_texts.append(t)
            tgt_labels.append(-1)
    tgt_dom_labels = [1] * len(tgt_texts)

    test_texts, test_labels, _ = read_fdu_mtl(dc.test_file, "test")
    test_dom_labels = [1] * len(test_texts)

    src_ds = MTLDataset(src_texts, src_labels_f, src_dom_labels, tokenizer, dc.max_seq_length)
    tgt_ds = MTLDataset(tgt_texts, tgt_labels, tgt_dom_labels, tokenizer, dc.max_seq_length)
    test_ds = MTLDataset(test_texts, test_labels, test_dom_labels, tokenizer, dc.max_seq_length)

    return src_ds, tgt_ds, test_ds, test_ds


def _resolve_dom_name(data_dir, dom_id):
    import os
    from glob import glob
    files = sorted(glob(os.path.join(data_dir, "*.task.train")))
    doms = [os.path.basename(f).replace(".task.train", "") for f in files]
    if dom_id < len(doms):
        return doms[dom_id]
    return str(dom_id)


def main():
    parser = argparse.ArgumentParser(description="Train MSP-DA")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip source pre-training (load from checkpoint)")
    parser.add_argument("--pretrain_ckpt", default=None,
                        help="Path to pre-trained checkpoint to load before meta-training")
    args = parser.parse_args()

    cfg = MSPDAConfig.from_yaml(args.config)
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = build_tokenizer(cfg.model.bert_model_name)
    model = MSPDAModel(cfg)

    if cfg.data.task == "ed":
        src_ds, tgt_ds, eval_ds, test_ds = build_ace05_loaders(cfg, tokenizer)
    else:
        src_ds, tgt_ds, eval_ds, test_ds = build_mtl_loaders(cfg, tokenizer)

    if not args.skip_pretrain:
        src_loader = DataLoader(src_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=2)
        eval_loader = DataLoader(eval_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2)
        source_pretrain(model, src_loader, eval_loader, cfg, device)
        import os
        os.makedirs(cfg.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(cfg.output_dir, "pretrain.pt"))
        print("Pre-training complete. Checkpoint saved.")

    if args.pretrain_ckpt:
        state = torch.load(args.pretrain_ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded pre-trained weights from {args.pretrain_ckpt}")

    combined_ds = ConcatDataset([src_ds, tgt_ds])
    batch_sampler = SrcTgtBatchSampler(
        src_size=len(src_ds),
        tgt_size=len(tgt_ds),
        batch_size=cfg.train.batch_size,
        target_ratio=cfg.train.target_ratio,
    )
    train_loader = DataLoader(combined_ds, batch_sampler=batch_sampler, num_workers=2)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2)

    meta_train(model, train_loader, eval_loader, cfg, device)

    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2)
    from msp_da.evaluate import evaluate as eval_fn
    import os
    best_ckpt = os.path.join(cfg.output_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"Loaded best model from {best_ckpt}")
    final_metrics = eval_fn(model, test_loader, device, task=cfg.data.task)
    print("Final test metrics:", final_metrics)


if __name__ == "__main__":
    main()
