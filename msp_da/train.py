"""
MSP-DA Training Loop.

Two-phase training:
1. Source Pre-training: warm up the adapter and classification head on labeled source data.
2. Meta Self-Paced Training: the main MSP-DA loop following Algorithm 1 from the paper.

Meta-training step (per mini-batch):
    (a) Forward pass through BERT+adapter on the full batch.
    (b) Compute per-sample CE losses on source examples.
    (c) Neural-SPL: split source into meta-source (easy, loss ≤ age) and meta-target (hard, loss > age).
    (d) Optionally: generate pseudo-labels for target examples; add easy pseudo-labeled samples to meta-source.
    (e) Compute layer-wise DANN weights via MetaWeightNet (MWN) on meta-source adapter representations.
    (f) Meta-train step: one inner gradient step on [SPL-weighted CE loss + weighted DANN loss] using meta-source.
    (g) Meta-val step: compute CE loss on meta-target using updated (inner-stepped) parameters.
    (h) SPL + MWN update: backprop meta-val loss → update age λ_a and MWN weights.
    (i) Combined outer step: update adapter + heads using L_total = L_tr + β * L_val.
"""

import copy
import gc
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from transformers import get_linear_schedule_with_warmup

from .config import MSPDAConfig
from .evaluate import evaluate
from .model import MSPDAModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_batch_indices(dom_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (src_idx, tgt_idx) from a dom_labels tensor (0=source, 1=target)."""
    src_idx = (dom_labels == 0).nonzero(as_tuple=True)[0]
    tgt_idx = (dom_labels == 1).nonzero(as_tuple=True)[0]
    return src_idx, tgt_idx


def _inner_step(
    model: MSPDAModel,
    pooler_output: torch.Tensor,
    adapter_downs,
    labels: torch.Tensor,
    dom_labels: torch.Tensor,
    easy_index: torch.Tensor,
    tgt_index: torch.Tensor,
    spl_weights: torch.Tensor,
    layer_weights: torch.Tensor,
    inner_lr: float,
    num_steps: int = 1,
    pseudo_easy_index: Optional[torch.Tensor] = None,
    spl_weights_pseudo: Optional[torch.Tensor] = None,
) -> "MSPDAModel":
    """Perform num_steps inner gradient updates (meta-train step).

    Returns a cloned, updated model (first-order approximation: detach graph).
    """
    learner = copy.deepcopy(model)
    learner.train()

    params = [p for p in learner.encoder.adapters.parameters()] + \
             [p for p in learner.classifier.parameters()] + \
             [p for p in learner.dann_heads.parameters()]
    inner_opt = AdamW(params, lr=inner_lr, weight_decay=0.0)

    for _ in range(num_steps):
        inner_opt.zero_grad()

        ce_loss, _ = learner.compute_ce_losses(
            pooler_output[easy_index],
            labels[easy_index],
            sample_weights=spl_weights[easy_index] if spl_weights is not None else None,
        )

        dann_labels = torch.cat([dom_labels[easy_index], dom_labels[tgt_index]], dim=0)
        dann_downs = [
            torch.cat([adapter_downs[l][easy_index], adapter_downs[l][tgt_index]], dim=0)
            for l in range(learner.num_layers)
        ]
        dann_loss = learner.compute_dann_loss(dann_downs, dann_labels, layer_weights)

        mtr_loss = ce_loss + dann_loss
        if pseudo_easy_index is not None and spl_weights_pseudo is not None:
            pl_loss, _ = learner.compute_ce_losses(
                pooler_output[pseudo_easy_index],
                labels[pseudo_easy_index],
                sample_weights=spl_weights_pseudo,
            )
            mtr_loss = mtr_loss + pl_loss

        mtr_loss.backward()
        inner_opt.step()

    for lp, mp in zip(learner.parameters(), model.parameters()):
        lp.data = lp.data.detach()

    return learner


# ---------------------------------------------------------------------------
# Source Pre-training
# ---------------------------------------------------------------------------

def source_pretrain(
    model: MSPDAModel,
    src_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    cfg: MSPDAConfig,
    device: torch.device,
) -> None:
    """Train only on labeled source data (warm-up phase)."""
    adapter_params = list(model.encoder.adapters.parameters())
    head_params = list(model.classifier.parameters())
    optimizer = AdamW(adapter_params + head_params, lr=cfg.train.learning_rate,
                      weight_decay=cfg.train.weight_decay)

    total_steps = len(src_dataloader) * cfg.train.src_pretrain_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    model.to(device)
    model.train()

    print(f"[SRC PRE-TRAIN] {cfg.train.src_pretrain_epochs} epochs on source domain")
    for epoch in range(cfg.train.src_pretrain_epochs):
        total_loss = 0.0
        for batch in src_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            anchor_positions = batch.get("anchor_positions")
            if anchor_positions is not None:
                anchor_positions = anchor_positions.to(device)
            labels = batch["labels"].to(device)
            dom_labels = batch["dom_labels"].to(device)

            pooler_output, _ = model.encode(
                input_ids, attention_mask, token_type_ids, anchor_positions
            )
            src_idx, _ = _get_batch_indices(dom_labels)
            if len(src_idx) == 0:
                continue

            loss, _ = model.compute_ce_losses(pooler_output[src_idx], labels[src_idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(src_dataloader), 1)
        metrics = evaluate(model, eval_dataloader, device, task=cfg.data.task)
        metric_str = "  ".join(f"{k}={v:.2f}" for k, v in metrics.items())
        print(f"  Epoch {epoch+1:3d}/{cfg.train.src_pretrain_epochs}  loss={avg_loss:.4f}  {metric_str}")

    model.train()


# ---------------------------------------------------------------------------
# Meta Self-Paced Training
# ---------------------------------------------------------------------------

def meta_train(
    model: MSPDAModel,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    cfg: MSPDAConfig,
    device: torch.device,
) -> None:
    """Main MSP-DA meta-training loop."""
    model.to(device)
    model.train()

    spl_params = list(model.spl_head.parameters())
    mwn_params = list(model.mwn.parameters())
    model_params = (
        list(model.encoder.adapters.parameters()) +
        list(model.classifier.parameters()) +
        list(model.dann_heads.parameters())
    )

    outer_optimizer = AdamW(model_params, lr=cfg.train.learning_rate,
                            weight_decay=cfg.train.weight_decay)
    spl_mwn_optimizer = AdamW(spl_params + mwn_params, lr=cfg.train.meta_val_lr,
                               weight_decay=cfg.train.weight_decay)

    os.makedirs(cfg.output_dir, exist_ok=True)

    global_step = 0
    best_metric = 0.0
    best_ckpt = None
    task = cfg.data.task

    print(f"[META TRAIN] {cfg.train.num_epochs} epochs")
    for epoch in range(cfg.train.num_epochs):
        for batch in train_dataloader:
            model.train()
            outer_optimizer.zero_grad()
            spl_mwn_optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            anchor_positions = batch.get("anchor_positions")
            if anchor_positions is not None:
                anchor_positions = anchor_positions.to(device)
            labels = batch["labels"].to(device)
            dom_labels = batch["dom_labels"].to(device)

            src_idx, tgt_idx = _get_batch_indices(dom_labels)
            if len(src_idx) < 2 or len(tgt_idx) == 0:
                continue

            # (a) Forward pass
            pooler_output, adapter_downs = model.encode(
                input_ids, attention_mask, token_type_ids, anchor_positions
            )

            # (b) Per-sample CE losses on source examples
            with torch.no_grad():
                _, src_losses = model.compute_ce_losses(
                    pooler_output[src_idx].detach(), labels[src_idx]
                )

            # (c) Neural-SPL: split source into easy (meta-source) and hard (meta-target)
            spl_weights, age_pct = model.compute_spl_weights(
                src_losses, age_clamp=cfg.train.age_clamp
            )
            easy_mask = spl_weights != 0.5
            hard_mask = ~easy_mask
            easy_rel = easy_mask.nonzero(as_tuple=True)[0]
            hard_rel = hard_mask.nonzero(as_tuple=True)[0]
            easy_index = src_idx[easy_rel]
            hard_index = src_idx[hard_rel]

            if len(easy_index) == 0 or len(hard_index) == 0:
                continue

            # (d) Pseudo-labels for target
            pseudo_easy_index = None
            spl_w_pseudo = None
            if cfg.train.pseudo_label:
                with torch.no_grad():
                    pseudo_logits = model.classify(pooler_output[tgt_idx].detach())
                    pseudo_labels_tgt = pseudo_logits.argmax(dim=-1)
                labels_clone = labels.clone()
                labels_clone[tgt_idx] = pseudo_labels_tgt
                _, tgt_losses = model.compute_ce_losses(
                    pooler_output[tgt_idx].detach(), labels_clone[tgt_idx]
                )
                tgt_spl_w, tgt_age_pct = model.compute_spl_weights(tgt_losses)
                tgt_easy_rel = (tgt_spl_w != 0.5).nonzero(as_tuple=True)[0]
                pseudo_easy_index = tgt_idx[tgt_easy_rel]
                spl_w_pseudo = tgt_spl_w[tgt_easy_rel]
                labels = labels_clone

            # (e) Layer-wise DANN weights
            if cfg.train.dann_wtype == "mwn":
                layer_weights = model.compute_mwn_weights(adapter_downs, easy_index)
            elif cfg.train.dann_wtype == "uniform":
                layer_weights = torch.ones(model.num_layers, device=device) / model.num_layers
            elif cfg.train.dann_wtype == "anneal_up":
                w = (1.25 ** torch.arange(model.num_layers, dtype=torch.float, device=device)) * 0.01844
                layer_weights = w / w.sum()
            elif cfg.train.dann_wtype == "anneal_dw":
                w = (1.25 ** torch.arange(model.num_layers, dtype=torch.float, device=device)) * 0.01844
                layer_weights = w.flip(0) / w.sum()
            else:
                layer_weights = torch.ones(model.num_layers, device=device) / model.num_layers

            # (f) Meta-train: inner step on easy (meta-source) samples
            learner = _inner_step(
                model, pooler_output.detach(), adapter_downs,
                labels, dom_labels,
                easy_index, tgt_idx,
                spl_weights, layer_weights,
                inner_lr=cfg.train.meta_train_lr,
                num_steps=cfg.train.num_meta_train_steps,
                pseudo_easy_index=pseudo_easy_index,
                spl_weights_pseudo=spl_w_pseudo,
            )

            # (g) Meta-val: CE on hard (meta-target) samples using updated learner
            learner.eval()
            learner_pooler, _ = learner.encode(
                input_ids, attention_mask, token_type_ids, anchor_positions
            )
            mval_loss, _ = learner.compute_ce_losses(
                learner_pooler[hard_index].detach(), labels[hard_index]
            )

            # (h) Update SPL + MWN via meta-val gradient
            mval_loss.backward()
            spl_mwn_optimizer.step()
            spl_mwn_optimizer.zero_grad()

            # (i) Outer update: combined loss on main model
            ce_tr_loss, _ = model.compute_ce_losses(
                pooler_output[easy_index],
                labels[easy_index],
                sample_weights=spl_weights[easy_rel],
            )
            dann_labels = torch.cat([dom_labels[easy_index], dom_labels[tgt_idx]], dim=0)
            dann_downs = [
                torch.cat([adapter_downs[l][easy_index], adapter_downs[l][tgt_idx]], dim=0)
                for l in range(model.num_layers)
            ]
            dann_tr_loss = model.compute_dann_loss(dann_downs, dann_labels, layer_weights.detach())

            ce_val_out, _ = model.compute_ce_losses(pooler_output[hard_index], labels[hard_index])

            total_loss = ce_tr_loss + dann_tr_loss + cfg.train.meta_test_beta * ce_val_out
            if pseudo_easy_index is not None:
                pl_loss, _ = model.compute_ce_losses(
                    pooler_output[pseudo_easy_index],
                    labels[pseudo_easy_index],
                    sample_weights=spl_w_pseudo,
                )
                total_loss = total_loss + pl_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(model_params, 1.0)
            outer_optimizer.step()
            outer_optimizer.zero_grad()

            del learner
            gc.collect()
            global_step += 1

            if global_step % cfg.train.eval_steps == 0:
                metrics = evaluate(model, eval_dataloader, device, task=task)
                key = "f1" if task == "ed" else "accuracy"
                score = metrics.get(key, 0.0)
                metric_str = "  ".join(f"{k}={v:.2f}" for k, v in metrics.items())
                print(f"  Step {global_step:5d}  {metric_str}  age={model.spl_head.age.item():.4f}")

                if score > best_metric:
                    best_metric = score
                    best_ckpt = os.path.join(cfg.output_dir, "best_model.pt")
                    torch.save(model.state_dict(), best_ckpt)
                    print(f"  → New best {key}={best_metric:.2f}, saved to {best_ckpt}")

        print(f"Epoch {epoch+1}/{cfg.train.num_epochs} done")

    if best_ckpt:
        print(f"\nBest checkpoint: {best_ckpt}  ({key}={best_metric:.2f})")
