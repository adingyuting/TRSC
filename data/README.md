# Data preparation and MAT schema

Raw and generated datasets live under this directory and are ignored by Git.

## Source datasets

- [Wiki-Gl](https://konect.cc/networks/wiki_talk_gl/)
- [Wiki-Eo](https://konect.cc/networks/wiki_talk_eo/)
- [Digg](https://konect.cc/networks/munmun_digg_reply/)
- [Bitcoin Alpha](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)
- [Bitcoin OTC](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)
- [DBLP](https://konect.cc/networks/dblp_coauthor/)
- [LAST-FM](https://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)

Place downloaded or normalized edge tables in `data/raw/`. The standard input
schema is:

```csv
From,To,Value,TimeStamp
```

Convert one dataset from the repository root:

```bash
python prepare_data.py --dataset wiki_gl --input data/raw/wiki_gl.csv
```

This creates:

```text
data/wiki_gl/wiki_gl.mat
data/wiki_gl/wiki_gl_node_mapping.csv
```

## MAT arrays

The converter writes:

| Array | Shape | Meaning |
|---|---|---|
| `tensor_idx` | `[E, 3]` | Full binary edge coordinates `(time, source, target)` |
| `A_idx` | `[3, E]` | Full binary edge coordinates used for link labels |
| `A_vals` | `[E]` | Binary positive labels |
| `train_idx`, `train_vals` | sparse coordinates and values | Normalized training adjacency window |
| `val_idx`, `val_vals` | sparse coordinates and values | Normalized validation adjacency window |
| `test_idx`, `test_vals` | sparse coordinates and values | Normalized test adjacency window |
| `num_nodes` | scalar | Number of remapped nodes |
| `time_slices` | scalar | Number of temporal snapshots |

With `V = floor(0.1T)`, `S = floor(0.2T)`, and `R = T - V - S`, the
rolling adjacency windows are:

```text
train = snapshots[0 : R]
val   = snapshots[V : V + R]
test  = snapshots[V + S : T]
```

Each adjacency snapshot receives self-loops and GCN degree normalization
before being stored. Link labels in `A` remain binary.
