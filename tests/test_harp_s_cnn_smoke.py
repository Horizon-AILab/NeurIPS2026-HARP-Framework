import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp import HARP_S_CNN, load_harp_scorer, save_harp_checkpoint, score_useful


def test_harp_s_cnn_forward_shape_and_score_range():
    model = HARP_S_CNN(
        embedding_dim=3584,
        num_classes=2,
        num_layers=28,
        kernel_sizes=(3, 4, 5),
        num_filters=4,
        dropout=0.1,
        deep_conv_layers=1,
        fc_hidden_dim=16,
    )
    x = torch.randn(2, 28, 96, 3584)
    logits = model(x)
    score = score_useful(logits)
    assert list(logits.shape) == [2, 2]
    assert bool(torch.all(score >= 0.0))
    assert bool(torch.all(score <= 1.0))


def test_harp_scorer_checkpoint_loader():
    metadata = {
        "architecture": "HARP_S_CNN",
        "hidden_dim": 8,
        "num_layers": 3,
        "max_len": 5,
        "num_classes": 2,
        "num_labels": 2,
        "kernel_sizes": [2, 3],
        "num_filters": 2,
        "dropout": 0.1,
        "deep_conv_layers": 1,
        "fc_hidden_dim": 4,
        "positive_class": "useful",
        "score_definition": "P(useful)",
    }
    model = HARP_S_CNN(
        embedding_dim=8,
        num_classes=2,
        num_layers=3,
        kernel_sizes=(2, 3),
        num_filters=2,
        dropout=0.1,
        deep_conv_layers=1,
        fc_hidden_dim=4,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        save_harp_checkpoint(model, tmp_path, metadata)
        loaded, loaded_metadata = load_harp_scorer(tmp_path)
        logits = loaded(torch.randn(1, 3, 5, 8))
        score = score_useful(logits)
    assert loaded_metadata["architecture"] == "HARP_S_CNN"
    assert list(logits.shape) == [1, 2]
    assert 0.0 <= float(score[0]) <= 1.0


if __name__ == "__main__":
    test_harp_s_cnn_forward_shape_and_score_range()
    test_harp_scorer_checkpoint_loader()
