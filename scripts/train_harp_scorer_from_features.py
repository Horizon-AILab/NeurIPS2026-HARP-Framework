import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp import HARP_S_CNN, normalize_hidden, save_harp_checkpoint, score_useful
from harp.io import read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HARP_S_CNN from features")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", default="")
    parser.add_argument("--test_manifest", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--kernel_sizes", default="3,4,5")
    parser.add_argument("--num_filters", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--deep_conv_layers", type=int, default=2)
    parser.add_argument("--fc_hidden_dim", type=int, default=512)
    return parser.parse_args()


def torch_load(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_feature_path(row: Dict, manifest_path: Path) -> Path:
    path = Path(row["feature_path"])
    return path if path.is_absolute() else manifest_path.parent / path


class FeatureDataset(Dataset):
    def __init__(self, manifest_path: str, max_len: int = 0) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = [row for row in read_jsonl(self.manifest_path) if "label" in row]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        hidden = torch_load(resolve_feature_path(row, self.manifest_path)).float()
        max_len = self.max_len or int(row.get("max_len", hidden.shape[1]))
        hidden = normalize_hidden(hidden, max_len)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return hidden, label


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_kernel_sizes(value: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def infer_shape(dataset: FeatureDataset) -> Tuple[int, int, int]:
    hidden, _ = dataset[0]
    return int(hidden.shape[0]), int(hidden.shape[1]), int(hidden.shape[2])


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    for hidden, label in loader:
        hidden = hidden.to(device)
        label = label.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(hidden)
        loss = criterion(logits, label)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(label.numel())
        pred = torch.argmax(logits.detach(), dim=-1)
        correct += int((pred == label).sum().detach().cpu())
        total += int(label.numel())
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "count": total,
    }


def predict(model: nn.Module, dataset: FeatureDataset, device: torch.device) -> List[Dict]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rows = []
    model.eval()
    with torch.no_grad():
        for row, (hidden, label) in zip(dataset.rows, loader):
            logits = model(hidden.to(device))
            score = float(score_useful(logits)[0].detach().cpu())
            rows.append({
                "edge_id": row.get("edge_id", ""),
                "label": int(label.item()),
                "harp_score": score,
                "prediction": int(score >= 0.5),
            })
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    train_dataset = FeatureDataset(args.train_manifest, args.max_len)
    val_dataset = FeatureDataset(args.val_manifest, args.max_len) if args.val_manifest else train_dataset
    test_dataset = FeatureDataset(args.test_manifest, args.max_len) if args.test_manifest else None
    num_layers, max_len, hidden_dim = infer_shape(train_dataset)
    kernel_sizes = parse_kernel_sizes(args.kernel_sizes)
    fc_hidden_dim = None if args.fc_hidden_dim < 0 else args.fc_hidden_dim
    model = HARP_S_CNN(
        embedding_dim=hidden_dim,
        num_classes=2,
        num_layers=num_layers,
        kernel_sizes=kernel_sizes,
        num_filters=args.num_filters,
        dropout=args.dropout,
        deep_conv_layers=args.deep_conv_layers,
        fc_hidden_dim=fc_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    history = []
    best_state = None
    best_loss = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if best_loss is None or val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(history[-1], ensure_ascii=False))
    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "architecture": "HARP_S_CNN",
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "max_len": max_len,
        "num_classes": 2,
        "num_labels": 2,
        "kernel_sizes": list(kernel_sizes),
        "num_filters": args.num_filters,
        "dropout": args.dropout,
        "deep_conv_layers": args.deep_conv_layers,
        "fc_hidden_dim": fc_hidden_dim,
        "positive_class": "useful",
        "score_definition": "P(useful)",
    }
    output_dir = Path(args.output_dir)
    checkpoint_path = save_harp_checkpoint(model, output_dir, metadata)
    metrics = {"history": history, "checkpoint_path": str(checkpoint_path)}
    if test_dataset is not None:
        predictions = predict(model, test_dataset, device)
        write_jsonl(output_dir / "test_predictions.jsonl", predictions)
        metrics["test"] = {
            "count": len(predictions),
            "accuracy": sum(int(row["label"] == row["prediction"]) for row in predictions) / max(len(predictions), 1),
        }
    write_json(output_dir / "metrics.json", metrics)


if __name__ == "__main__":
    main()
