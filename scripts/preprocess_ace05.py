"""
Preprocess raw ACE-05 data into the JSON format expected by MSP-DA.

The script expects raw ACE-05 data with standard splits and outputs per-domain
JSON files used by msp_da/data.py.

Output format (one file per split: train.json, eval.json, test.json):
    {
        "domain_key": [
            {"sent": str, "anc_pos": int, "label": int},
            ...
        ]
    }

Label mapping (34 classes including O=0):
    0: O (no event)
    1-33: ACE-05 event subtypes

Usage:
    python scripts/preprocess_ace05.py \
        --input_dir /path/to/raw/ace2005 \
        --output_dir data/ace2005/
"""

import argparse
import json
import os
from collections import defaultdict

ACE05_DOMAINS = ["bn", "nw", "bc", "cts", "un", "wl"]

ACE05_EVENT_SUBTYPES = [
    "O",
    "Be-Born", "Marry", "Divorce", "Injure", "Die",
    "Transport", "Transfer-Ownership", "Transfer-Money",
    "Start-Org", "End-Org", "Declare-Bankruptcy", "Merge-Org",
    "Attack", "Demonstrate",
    "Meet", "Phone-Write",
    "Start-Position", "End-Position", "Nominate", "Elect",
    "Arrest-Jail", "Release-Parole", "Charge-Indict", "Trial-Hearing",
    "Convict", "Sentence", "Execute", "Acquit", "Pardon", "Appeal",
    "Fine", "Extradite",
    "Sue",
]

LABEL2ID = {label: idx for idx, label in enumerate(ACE05_EVENT_SUBTYPES)}


def preprocess_ace05(input_dir: str, output_dir: str) -> None:
    """Convert raw ACE-05 annotations to per-split JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    splits = {"train": defaultdict(list), "eval": defaultdict(list), "test": defaultdict(list)}

    for domain in ACE05_DOMAINS:
        for split_name in splits:
            domain_split_dir = os.path.join(input_dir, domain, split_name)
            if not os.path.isdir(domain_split_dir):
                continue
            for fname in os.listdir(domain_split_dir):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(domain_split_dir, fname)) as f:
                    raw = json.load(f)
                for sentence in raw.get("sentences", []):
                    text = sentence.get("text", "")
                    words = text.split()
                    for event in sentence.get("events", []):
                        subtype = event.get("subtype", "O")
                        anchor_pos = event.get("trigger_pos", 0)
                        label_id = LABEL2ID.get(subtype, 0)
                        splits[split_name][domain].append({
                            "sent": text,
                            "anc_pos": anchor_pos,
                            "label": label_id,
                        })

                    if not sentence.get("events"):
                        for i, word in enumerate(words):
                            splits[split_name][domain].append({
                                "sent": text,
                                "anc_pos": i,
                                "label": 0,
                            })

    for split_name, domain_data in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(dict(domain_data), f, indent=2)
        total = sum(len(v) for v in domain_data.values())
        print(f"Wrote {out_path}: {total} examples across {len(domain_data)} domains")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Raw ACE-05 directory")
    parser.add_argument("--output_dir", default="data/ace2005/", help="Output directory")
    args = parser.parse_args()
    preprocess_ace05(args.input_dir, args.output_dir)
