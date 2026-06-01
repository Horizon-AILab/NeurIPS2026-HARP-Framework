import argparse
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp import normalize_hidden
from harp.hidden import edge_fields, edge_id, extract_edge_hidden, load_receiver
from harp.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract HARP hidden features")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="")
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--include_embedding", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return text[:120] or "edge"


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.manifest))
    if args.limit > 0:
        rows = rows[: args.limit]
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_receiver(
        args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map or None,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    manifest_rows = []
    for index, row in enumerate(rows):
        query, candidate_message = edge_fields(row)
        hidden, meta = extract_edge_hidden(
            tokenizer,
            model,
            query,
            candidate_message,
            include_embedding=args.include_embedding,
        )
        feature = normalize_hidden(hidden, args.max_len)
        eid = edge_id(row, index)
        feature_path = feature_dir / f"{safe_name(eid)}.pt"
        torch.save(feature, feature_path)
        out = {
            "edge_id": eid,
            "feature_path": str(feature_path),
            "num_layers": int(feature.shape[0]),
            "max_len": int(feature.shape[1]),
            "hidden_dim": int(feature.shape[2]),
            "raw_seq_len": int(meta["raw_seq_len"]),
            "receiver": row.get("receiver", ""),
            "source": row.get("source", ""),
            "metadata": row.get("metadata", {}),
        }
        if "label" in row:
            out["label"] = int(row["label"])
        manifest_rows.append(out)
        print(f"{eid}\t{out['num_layers']}x{out['max_len']}x{out['hidden_dim']}")
    write_jsonl(output_dir / "feature_manifest.jsonl", manifest_rows)


if __name__ == "__main__":
    main()
