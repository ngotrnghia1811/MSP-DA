from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    bert_model_name: str = "bert-base-uncased"
    adapter_bottleneck_size: int = 96
    num_labels: int = 34
    num_bert_layers: int = 12
    hidden_size: int = 768
    dropout: float = 0.1

    classifier_hidden_div: int = 1
    dann_hidden_size: int = 200
    dann_hidden_div: int = 4
    dann_layers: int = 2
    grl_lambda: float = 0.1

    spl_hidden_div: int = 2
    spl_layers: int = 2

    mwn_hidden_div: int = 2
    mwn_layers: int = 2


@dataclass
class TrainConfig:
    learning_rate: float = 1e-4
    meta_train_lr: float = 1e-4
    meta_val_lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 100
    target_ratio: float = 0.2
    num_epochs: int = 200
    num_meta_train_steps: int = 1
    meta_test_beta: float = 2.0
    first_order: bool = True
    eval_steps: int = 200
    save_steps: int = 1000
    seed: int = 42

    src_pretrain_epochs: int = 20
    pseudo_label: bool = True
    dann_wtype: str = "mwn"

    age_clamp: float = 0.5


@dataclass
class DataConfig:
    task: str = "ed"
    train_file: str = ""
    dev_file: str = ""
    test_file: str = ""
    source_domains: List[str] = field(default_factory=lambda: ["bn", "nw"])
    target_domain: str = "bc"
    max_seq_length: int = 56


@dataclass
class MSPDAConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "checkpoints/"

    @classmethod
    def from_yaml(cls, path: str) -> "MSPDAConfig":
        import yaml
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        cfg = cls()
        if "model" in raw:
            for k, v in raw["model"].items():
                setattr(cfg.model, k, v)
        if "train" in raw:
            for k, v in raw["train"].items():
                setattr(cfg.train, k, v)
        if "data" in raw:
            for k, v in raw["data"].items():
                setattr(cfg.data, k, v)
        if "output_dir" in raw:
            cfg.output_dir = raw["output_dir"]
        return cfg
