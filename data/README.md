# Data

This directory holds the processed datasets used by MSP-DA. Raw data must be obtained separately due to licensing restrictions.

## ACE-05 Event Detection

**License:** LDC2006T06 — requires an LDC license to access.

**Domains:** `bn` (broadcast news), `nw` (newswire), `bc` (broadcast conversation), `cts` (conversational telephone speech), `un` (usenet), `wl` (weblog)

**Paper splits (source → target):**
| Source | Target |
|--------|--------|
| bn + nw | bc |
| bn + nw | cts |
| bn + nw | wl |

**Expected directory layout after preprocessing:**
```
data/ace2005/
├── train.json
├── eval.json
└── test.json
```

Each JSON file maps domain keys to lists of `{"sent": str, "anc_pos": int, "label": int}`.

**Preprocessing:**
```bash
python scripts/preprocess_ace05.py \
    --input_dir /path/to/raw/ace2005 \
    --output_dir data/ace2005/
```

## FDU-MTL Sentiment Analysis

**Source:** [FDU-MTL dataset](https://github.com/FrankWork/fudan_mtl_reviews) — publicly available.

**Domains (16):** MR, apparel, baby, books, camera\_photo, dvd, electronics, health\_personal\_care, imdb, kitchen\_housewares, magazines, music, software, sports\_outdoors, toys\_games, video

**Expected directory layout:**
```
data/fdu-mtl/
├── MR.task.train
├── MR.task.test
├── MR.task.unlabel
├── apparel.task.train
├── ...
```

**Download:**
```bash
git clone https://github.com/FrankWork/fudan_mtl_reviews
cp -r fudan_mtl_reviews/data/ data/fdu-mtl/
```
