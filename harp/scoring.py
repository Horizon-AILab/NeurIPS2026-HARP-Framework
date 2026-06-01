import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from .models import HARP_S_CNN

SCORER_FILENAME = "best_harp_s_cnn.pt"
LEGACY_SCORER_FILENAME = "best_deeptextcnn.pt"
METADATA_FILENAME = "scorer_metadata.json"


def score_useful(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)[..., 1]


def normalize_hidden(hidden: torch.Tensor, max_len: int) -> torch.Tensor:
    if hidden.dim() == 4 and hidden.size(0) == 1:
        hidden = hidden.squeeze(0)
    layers, seq_len, hidden_dim = hidden.shape
    if seq_len > max_len:
        return hidden[:, :max_len, :].contiguous()
    if seq_len < max_len:
        pad = torch.zeros((layers, max_len - seq_len, hidden_dim), dtype=hidden.dtype, device=hidden.device)
        return torch.cat([hidden, pad], dim=1).contiguous()
    return hidden.contiguous()


def find_scorer_checkpoint(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    preferred = path / SCORER_FILENAME
    if preferred.exists():
        return preferred
    legacy = path / LEGACY_SCORER_FILENAME
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"missing scorer checkpoint under {path}")


def load_metadata(path: Path) -> Dict[str, Any]:
    metadata_path = Path(path)
    if metadata_path.is_dir():
        metadata_path = metadata_path / METADATA_FILENAME
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_harp_scorer(metadata: Dict[str, Any], checkpoint_path: Optional[Path] = None) -> HARP_S_CNN:
    legacy = bool(metadata.get("legacy_deeptextcnn", False))
    if checkpoint_path is not None and Path(checkpoint_path).name == LEGACY_SCORER_FILENAME:
        legacy = metadata.get("architecture") not in {"HARP_S_CNN", "harp_s_cnn"}
    hidden_dim = int(metadata.get("hidden_dim", metadata.get("embedding_dim")))
    num_layers = int(metadata["num_layers"])
    kernel_sizes = tuple(int(x) for x in metadata.get("kernel_sizes", [3, 4, 5]))
    num_filters = int(metadata.get("num_filters", 32 if legacy else 256))
    dropout = float(metadata.get("dropout", 0.2 if legacy else 0.5))
    deep_conv_layers = int(metadata.get("deep_conv_layers", 0 if legacy else 2))
    fc_hidden_dim = metadata.get("fc_hidden_dim", None if legacy else 512)
    if fc_hidden_dim is not None:
        fc_hidden_dim = int(fc_hidden_dim)
    return HARP_S_CNN(
        embedding_dim=hidden_dim,
        num_classes=int(metadata.get("num_classes", metadata.get("num_labels", 2))),
        num_layers=num_layers,
        kernel_sizes=kernel_sizes,
        num_filters=num_filters,
        dropout=dropout,
        deep_conv_layers=deep_conv_layers,
        fc_hidden_dim=fc_hidden_dim,
    )


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj
    if isinstance(obj, dict) and "state_dict" in obj:
        return {"model_state_dict": obj["state_dict"]}
    return {"model_state_dict": obj}


def load_harp_scorer(path: Path, device: Optional[torch.device] = None) -> Tuple[HARP_S_CNN, Dict[str, Any]]:
    path = Path(path)
    checkpoint_path = find_scorer_checkpoint(path)
    metadata_path = path / METADATA_FILENAME if path.is_dir() else checkpoint_path.with_name(METADATA_FILENAME)
    metadata = load_metadata(metadata_path)
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_harp_scorer(metadata, checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model.to(device)
    model.eval()
    metadata = dict(metadata)
    metadata.setdefault("architecture", "HARP_S_CNN")
    metadata.setdefault("score_definition", "P(useful)")
    metadata["checkpoint_path"] = str(checkpoint_path)
    metadata["metadata_path"] = str(metadata_path)
    return model, metadata


def save_harp_checkpoint(model: HARP_S_CNN, output_dir: Path, metadata: Dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(metadata)
    metadata.setdefault("architecture", "HARP_S_CNN")
    metadata.setdefault("score_definition", "P(useful)")
    metadata.setdefault("positive_class", "useful")
    metadata_path = output_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint_path = output_dir / SCORER_FILENAME
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, checkpoint_path)
    return checkpoint_path
