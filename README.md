# TRSC

Official PyTorch implementation of TRSC for link prediction on directed
dynamic graphs.

TRSC combines causal tensor graph convolution, historical-activity-guided edge
assimilation, and a causal structural-evidence decoder. This repository
provides the complete reproducibility path from a temporal edge CSV file to a
MATLAB dataset, model training, validation, testing, checkpoints, and metrics.

## Reproducibility workflow

```text
temporal edge CSV/TSV
        │
        ▼
  prepare_data.py
        │
        ├── data/<dataset>/<dataset>.mat
        └── data/<dataset>/<dataset>_node_mapping.csv
        │
        ▼
     train.py
        │
        ├── best AP / ROC-AUC checkpoints
        ├── epoch history
        └── test metrics and multi-run summary
```

## Requirements

- Python 3.9 or later
- PyTorch 2.0 or later
- NumPy, pandas, SciPy, and scikit-learn

After downloading or cloning the repository, create an environment and install
the project:

```bash
cd TRSC
python -m venv .venv
python -m pip install -e .
```

Install the appropriate CUDA build of PyTorch first if GPU training is needed.

## Datasets

| Dataset | Source | Time slices | Repository name |
|---|---|---:|---|
| Wiki-Gl | [KONECT](https://konect.cc/networks/wiki_talk_gl/) | 60 | `wiki_gl` |
| Wiki-Eo | [KONECT](https://konect.cc/networks/wiki_talk_eo/) | 60 | `wiki_eo` |
| Digg | [KONECT](https://konect.cc/networks/munmun_digg_reply/) | 50 | `digg` |
| Bitcoin Alpha | [SNAP](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) | 60 | `bitcoin_alpha` |
| Bitcoin OTC | [SNAP](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) | 60 | `bitcoin_otc` |
| DBLP | [KONECT](https://konect.cc/networks/dblp_coauthor/) | 45 | `dblp` |
| LAST-FM | [Last.fm Dataset 1K](https://ocelma.net/MusicRecommendationDataset/lastfm-1K.html) | 53 | `last_fm` |

Dataset files are not committed to Git. Download each dataset from its source
and prepare a temporal edge table with these columns:

```csv
From,To,Value,TimeStamp
1,401,1,0
9,270,8,0
9,969,8,1
53,118,5,1
```

- `From` and `To` are node identifiers and may be integers or strings.
- `Value` is retained as raw edge metadata; TRSC link-prediction labels are
  binary edge-presence values.
- `TimeStamp` may contain integer snapshot IDs, numeric timestamps, or datetime
  strings.

Different column names or zero-based column positions can be supplied on the
command line. CSV, compressed CSV, and TSV files supported by pandas are
accepted.

## Step 1: CSV to MAT

The dataset name selects the number of time slices used in the experiments:

```bash
python prepare_data.py \
  --dataset bitcoin_alpha \
  --input data/raw/soc-sign-bitcoinalpha.csv.gz \
  --output data \
  --has-header false \
  --source-column 0 \
  --target-column 1 \
  --value-column 2 \
  --timestamp-column 3
```

For a CSV with the standard header, the shorter command is enough:

```bash
python prepare_data.py \
  --dataset wiki_gl \
  --input data/raw/wiki_gl.csv \
  --output data
```

For tab-separated data without a header:

```bash
python prepare_data.py \
  --dataset last_fm \
  --input data/raw/lastfm.tsv \
  --output data \
  --delimiter "\t" \
  --has-header false \
  --source-column 0 \
  --target-column 3 \
  --value-column none \
  --timestamp-column 1
```

The converter performs the following operations:

1. reads the temporal edge table and validates the selected columns;
2. maps arbitrary node identifiers to contiguous zero-based indices;
3. maps timestamps to the configured number of snapshots;
4. removes self-edge events and coalesces duplicate edge events;
5. builds binary link labels for every snapshot;
6. constructs graph-convolution inputs with self-loops and
   `D^{-1/2}(A+I)D^{-1/2}` normalization;
7. constructs the training, validation, and testing rolling windows used by
   the model;
8. writes the sparse arrays and node-ID mapping.

Timestamp handling is controlled by `--binning`:

- `auto` uses existing snapshot IDs `0 ... T-1` when present and otherwise
  uses equal-width time intervals;
- `discrete` maps exactly `T` unique timestamp values in chronological order;
- `equal-width` creates equal-duration intervals;
- `equal-count` creates intervals with approximately equal numbers of events.

Use `--make-symmetric true` when the experimental graph should be treated as
undirected. `--edge-life K` keeps an edge active for the current and previous
`K-1` snapshots. Both default to disabled.

The output paths are:

```text
data/<dataset>/<dataset>.mat
data/<dataset>/<dataset>_node_mapping.csv
```

The MAT schema and loader behavior are described in
[`data/README.md`](data/README.md).

## Step 2: Train and evaluate TRSC

Run the full training protocol after creating the MAT file:

```bash
python train.py \
  --dataset bitcoin_alpha \
  --data-dir data \
  --output-dir outputs \
  --cuda true \
  --epochs 250 \
  --patience 25 \
  --num-runs 5
```

The main model options are:

```text
--num-feature 8
--hidden-features 16
--bandwidth 20
--mixing-choice 2
--tgc-dropout 0.75
--activity-window 3
--min-edge-weight 0.25
--decoder-hidden-dim 32
--decoder-dropout 0.1
--lr 0.005
--weight-decay 0.0005
--lam 0.00001
```

The data split and negative samples are fixed by `--data-seed`. Model
initialization varies from `--seed` through `--seed + num_runs - 1`.

For every run, the trainer saves:

```text
outputs/<dataset>/seed_<seed>/best_ap.pt
outputs/<dataset>/seed_<seed>/best_roc_auc.pt
outputs/<dataset>/seed_<seed>/history.json
outputs/<dataset>/seed_<seed>/metrics.json
```

The mean and standard deviation across runs are written to
`outputs/<dataset>/summary.json`.

## Python API

```python
from trsc import TRSC, load_mat_dataset, split_data

data = load_mat_dataset("bitcoin_alpha", data_dir="data", device="cpu")
split = split_data(data.labels, data.time_slices)
train_edges, train_targets = split[0], split[1]

model = TRSC(
    time_slices=len(data.train_adjacency) - 1,
    N=data.num_nodes,
    hidden_features=[16],
    num_feature=8,
    out_features=1,
    bandwidth=20,
)
```

Queries use flattened node indices: the index of node `v` at time `t` is
`t * N + v`. See [`examples/load_preprocessed_data.py`](examples/load_preprocessed_data.py)
for a complete forward/backward example.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover CSV-to-MAT conversion, MAT loading, all three temporal windows,
negative sampling, structural causality, model forward/backward propagation,
checkpoint creation, and test evaluation.

## Repository layout

```text
TRSC/
├── trsc/
│   ├── model.py
│   ├── layers.py
│   ├── structural_features.py
│   ├── preprocessing.py
│   └── data.py
├── data/README.md
├── examples/
├── tests/
├── prepare_data.py
├── train.py
├── pyproject.toml
└── requirements.txt
```
