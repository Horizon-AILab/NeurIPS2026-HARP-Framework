# HARP Scorer Checkpoint

HARP scorer artifacts follow a receiver-conditioned hidden-state scoring contract.

```text
checkpoint_dir/
|-- best_harp_s_cnn.pt
|-- scorer_metadata.json
|-- metrics.json
`-- test_predictions.jsonl
```

`scorer_metadata.json` carries the scorer identity, hidden tensor shape, and score convention.

```json
{
  "architecture": "HARP_S_CNN",
  "num_layers": 28,
  "max_len": 96,
  "hidden_dim": 3584,
  "num_labels": 2,
  "positive_class": "useful",
  "score_definition": "P(useful)"
}
```

The hidden tensor follows `[num_layers, max_len, hidden_dim]`. The reported score is the useful-class probability.

The utility entry points are:

```text
scripts/extract_harp_hidden_features.py
scripts/train_harp_scorer_from_features.py
scripts/score_harp_edge.py
scripts/rank_harp_edges.py
```
