"""
MSP-DA Model: BERT + Adapters + Neural-SPL + DANN heads.

Architecture (per the paper):
- Fixed pre-trained BERT-base encoder
- Adapter layers (bottleneck size d_a=96) after every feed-forward sublayer
- Classification head (ED_CE): 2-layer FF with ReLU over [CLS] pooler output
- 12 DANN heads (one per BERT layer): take adapter bottleneck output → GRL → 2-layer FF → binary domain
- Neural-SPL module:
    - Instance-wise weighting (SPL_CE): takes per-sample CE losses, learns age λ_a → weights in [0,1]
    - Layer-wise weighting (MWN_DA): takes stacked adapter bottleneck reprs → softmax → 12 weights
"""

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from transformers import BertModel, BertConfig


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, grl_lambda):
        ctx.grl_lambda = grl_lambda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg(), None


class GradientReversalLayer(nn.Module):
    def __init__(self, grl_lambda: float = 0.1):
        super().__init__()
        self.grl_lambda = grl_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.grl_lambda)


# ---------------------------------------------------------------------------
# Adapter Layer
# ---------------------------------------------------------------------------

class AdapterLayer(nn.Module):
    """Bottleneck adapter: down-project → ReLU → up-project → residual."""

    def __init__(self, hidden_size: int, bottleneck_size: int, dropout: float = 0.0):
        super().__init__()
        self.adapter_down = nn.Linear(hidden_size, bottleneck_size, bias=True)
        self.activation = nn.ReLU()
        self.adapter_up = nn.Linear(bottleneck_size, hidden_size, bias=True)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.adapter_down.weight, std=1e-3)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.normal_(self.adapter_up.weight, std=1e-3)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        down = self.activation(self.adapter_down(hidden_states))
        up = self.adapter_up(self.dropout(down))
        output = self.layer_norm(up + residual)
        return output, down


# ---------------------------------------------------------------------------
# DANN Head (per-layer domain adversarial)
# ---------------------------------------------------------------------------

class DANNHead(nn.Module):
    """Binary domain classifier operating on adapter bottleneck representations.

    Input:  adapter down-projection (batch, bottleneck_size)
    Output: domain logits (batch, 2)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2,
                 grl_lambda: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.grl = GradientReversalLayer(grl_lambda)
        layers: List[nn.Module] = []
        in_dim = input_size
        for i in range(num_layers - 1):
            out_dim = hidden_size if i == 0 else hidden_size // 2
            layers.extend([nn.Linear(in_dim, out_dim), nn.Dropout(dropout), nn.ReLU()])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 2))
        self.classifier = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, bottleneck: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.grl(bottleneck)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return logits, loss


# ---------------------------------------------------------------------------
# Classification Head
# ---------------------------------------------------------------------------

class ClassificationHead(nn.Module):
    """Feed-forward classification head over pooled BERT representation."""

    def __init__(self, hidden_size: int, num_labels: int, hidden_div: int = 1, dropout: float = 0.1):
        super().__init__()
        mid = hidden_size // hidden_div
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Neural SPL Head (instance-wise weighting)
# ---------------------------------------------------------------------------

class NeuralSPLHead(nn.Module):
    """Instance-wise weighting module.

    Takes per-sample CE losses, maps through a small net with sigmoid,
    and uses a learnable age parameter λ_a to threshold the weights.

    f_v(l_i; λ_a) = sigmoid(net(max(0, -l_i / λ_a + 1)))
    Samples with loss > λ_a get weight 0.5 (hard/meta-target);
    samples with loss ≤ λ_a get weight from f_v (easy/meta-source).
    """

    def __init__(self, hidden_div: int = 2, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.age = nn.Parameter(torch.tensor(0.5))
        layers: List[nn.Module] = [nn.Linear(1, 50, bias=False), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(50, 50, bias=False), nn.ReLU()])
        layers.append(nn.Linear(50, 1, bias=False))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, losses: torch.Tensor, age_clamp: float = 0.5) -> Tuple[torch.Tensor, float]:
        """
        Args:
            losses: per-sample CE losses, shape (N,)
            age_clamp: unused direct clamp, age is learned
        Returns:
            weights: shape (N,) — easy samples get f_v weight, hard samples get 0.5 (sentinel)
            age_percentile: percentage of source samples below age threshold
        """
        age = self.age
        clipped = torch.clamp(-losses / (age + 1e-6) + 1, min=0.0)
        net_input = clipped.unsqueeze(-1)
        weights = self.net(net_input).squeeze(-1)

        hard_mask = losses > age
        weights = weights * (~hard_mask).float() + 0.5 * hard_mask.float()

        age_percentile = (losses <= age).float().mean().item() * 100.0
        return weights, age_percentile


# ---------------------------------------------------------------------------
# Meta Weight Net (layer-wise DANN balancing)
# ---------------------------------------------------------------------------

class MetaWeightNet(nn.Module):
    """Layer-wise weighting head.

    Input:  stacked adapter bottleneck representations, shape (L, d_a)
    Output: normalized layer weights, shape (L,)
    """

    def __init__(self, adapter_size: int, num_layers: int = 12,
                 hidden_div: int = 2, dropout: float = 0.1):
        super().__init__()
        mid = adapter_size // hidden_div
        self.net = nn.Sequential(
            nn.Linear(adapter_size, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )
        self.num_layers = num_layers

    def forward(self, layer_reprs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_reprs: (L, d_a) — per-layer pooled bottleneck representations
        Returns:
            weights: (L,) normalized via softmax
        """
        scores = self.net(layer_reprs).squeeze(-1)
        return F.softmax(scores, dim=0)


# ---------------------------------------------------------------------------
# BERT with Adapters (MSPDAEncoder)
# ---------------------------------------------------------------------------

class MSPDAEncoder(nn.Module):
    """BERT encoder augmented with per-layer adapters.

    Adapters are injected after the feed-forward sublayer (MetaOutput position).
    Only adapter parameters are trainable; BERT weights are frozen.
    """

    def __init__(self, bert_model_name: str, bottleneck_size: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        hidden_size = self.bert.config.hidden_size
        num_layers = self.bert.config.num_hidden_layers

        self.adapters = nn.ModuleList([
            AdapterLayer(hidden_size, bottleneck_size, dropout)
            for _ in range(num_layers)
        ])
        self.bottleneck_size = bottleneck_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                anchor_positions=None):
        """
        Returns:
            pooler_output: (B, H) pooled representation
            adapter_downs: list of (B, d_a) per-layer bottleneck outputs
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        hidden_states_per_layer = outputs.hidden_states[1:]

        adapter_downs = []
        adapted_hidden = outputs.hidden_states[0]
        for i, (layer_module, adapter) in enumerate(zip(self.bert.encoder.layer, self.adapters)):
            raw_hidden = hidden_states_per_layer[i]
            adapted, down = adapter(raw_hidden, raw_hidden)
            adapter_downs.append(down)

        sequence_output = adapted_hidden if False else outputs.last_hidden_state

        if anchor_positions is not None:
            batch_idx = torch.arange(sequence_output.size(0), device=sequence_output.device)
            token_repr = sequence_output[batch_idx, anchor_positions, :]
            pooler_output = self.bert.pooler(token_repr.unsqueeze(1)).squeeze(1)
        else:
            pooler_output = self.bert.pooler(sequence_output)

        return pooler_output, adapter_downs


# ---------------------------------------------------------------------------
# Full MSP-DA Model
# ---------------------------------------------------------------------------

class MSPDAModel(nn.Module):
    """
    Full MSP-DA model combining:
    - BERT encoder with adapter layers
    - Classification head (ED_CE or SA)
    - 12 per-layer DANN heads
    - Neural-SPL head (instance-wise weighting + age)
    - MetaWeightNet (layer-wise DANN balancing)
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg.model

        self.encoder = MSPDAEncoder(
            bert_model_name=m.bert_model_name,
            bottleneck_size=m.adapter_bottleneck_size,
            dropout=m.dropout,
        )

        self.classifier = ClassificationHead(
            hidden_size=m.hidden_size,
            num_labels=m.num_labels,
            hidden_div=m.classifier_hidden_div,
            dropout=m.dropout,
        )

        self.dann_heads = nn.ModuleList([
            DANNHead(
                input_size=m.adapter_bottleneck_size,
                hidden_size=m.dann_hidden_size,
                num_layers=m.dann_layers,
                grl_lambda=m.grl_lambda,
                dropout=m.dropout,
            )
            for _ in range(m.num_bert_layers)
        ])

        self.spl_head = NeuralSPLHead(
            hidden_div=m.spl_hidden_div,
            num_layers=m.spl_layers,
            dropout=m.dropout,
        )

        self.mwn = MetaWeightNet(
            adapter_size=m.adapter_bottleneck_size,
            num_layers=m.num_bert_layers,
            hidden_div=m.mwn_hidden_div,
            dropout=m.dropout,
        )

        self.num_layers = m.num_bert_layers
        self.cfg = cfg

    def encode(self, input_ids, attention_mask, token_type_ids=None, anchor_positions=None):
        return self.encoder(input_ids, attention_mask, token_type_ids, anchor_positions)

    def classify(self, pooler_output: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooler_output)

    def compute_ce_losses(self, pooler_output: torch.Tensor,
                          labels: torch.Tensor,
                          sample_weights: Optional[torch.Tensor] = None,
                          pseudo: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (weighted_loss_scalar, per_sample_losses)."""
        logits = self.classify(pooler_output)
        per_sample_losses = F.cross_entropy(logits, labels, reduction="none")
        if pseudo:
            pseudo_labels = logits.argmax(dim=-1)
            per_sample_losses = F.cross_entropy(logits.detach(), pseudo_labels, reduction="none")
        if sample_weights is not None:
            loss = (per_sample_losses * sample_weights).sum() / (sample_weights.sum() + 1e-8)
        else:
            loss = per_sample_losses.mean()
        return loss, per_sample_losses

    def compute_dann_loss(self, adapter_downs: List[torch.Tensor],
                          domain_labels: torch.Tensor,
                          layer_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Weighted sum of DANN losses across all 12 layers."""
        if layer_weights is None:
            layer_weights = torch.ones(self.num_layers, device=domain_labels.device) / self.num_layers
        total = torch.tensor(0.0, device=domain_labels.device)
        for l in range(self.num_layers):
            down = adapter_downs[l]
            _, dann_loss = self.dann_heads[l](down, domain_labels)
            total = total + layer_weights[l] * dann_loss
        return total

    def compute_spl_weights(self, per_sample_losses: torch.Tensor,
                            age_clamp: float = 0.5) -> Tuple[torch.Tensor, float]:
        return self.spl_head(per_sample_losses, age_clamp)

    def compute_mwn_weights(self, adapter_downs: List[torch.Tensor],
                             easy_index: torch.Tensor) -> torch.Tensor:
        """Pool adapter bottleneck representations across the easy subset, then weight."""
        layer_reprs = []
        for l in range(self.num_layers):
            easy_down = adapter_downs[l][easy_index]
            pooled = easy_down.sum(dim=0)
            layer_reprs.append(pooled)
        stacked = torch.stack(layer_reprs, dim=0)
        return self.mwn(stacked)

    def trainable_parameters(self):
        """Returns only adapter + head parameters (BERT frozen)."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param
