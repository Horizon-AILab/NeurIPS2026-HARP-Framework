import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp import load_harp_scorer, normalize_hidden, score_useful
from harp.hidden import edge_fields, edge_id, extract_edge_hidden, load_receiver
from harp.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank HARP edges")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="")
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--include_embedding", action="store_true")
    parser.add_argument("--max_len", type=int, default=0)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.manifest))
    scorer_device = resolve_device(args.device)
    scorer, metadata = load_harp_scorer(Path(args.checkpoint_dir), scorer_device)
    tokenizer, model = load_receiver(
        args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map or None,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    max_len = args.max_len or int(metadata.get("max_len", 96))
    scored = []
    for index, row in enumerate(rows):
        query, candidate_message = edge_fields(row)
        hidden, hidden_meta = extract_edge_hidden(
            tokenizer,
            model,
            query,
            candidate_message,
            include_embedding=args.include_embedding,
        )
        x = normalize_hidden(hidden, max_len).unsqueeze(0).to(scorer_device)
        if scorer_device.type == "cpu":
            x = x.float()
        with torch.no_grad():
            logits = scorer(x)
            harp_score = float(score_useful(logits)[0].detach().cpu())
        out = {
            "edge_id": edge_id(row, index),
            "receiver": row.get("receiver", ""),
            "source": row.get("source", ""),
            "harp_score": harp_score,
            "score_definition": metadata.get("score_definition", "P(useful)"),
            "metadata": row.get("metadata", {}),
            **hidden_meta,
        }
        scored.append(out)
    ranked = sorted(scored, key=lambda item: item["harp_score"], reverse=True)
    if args.output:
        write_jsonl(Path(args.output), ranked)
    for row in ranked:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
