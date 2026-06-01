import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp import load_harp_scorer, normalize_hidden, score_useful
from harp.hidden import extract_edge_hidden, load_receiver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a HARP edge")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--candidate_message", required=True)
    parser.add_argument("--edge_id", default="edge")
    parser.add_argument("--receiver", default="")
    parser.add_argument("--source", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_map", default="")
    parser.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--include_embedding", action="store_true")
    parser.add_argument("--max_len", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
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
    hidden, hidden_meta = extract_edge_hidden(
        tokenizer,
        model,
        args.query,
        args.candidate_message,
        include_embedding=args.include_embedding,
    )
    max_len = args.max_len or int(metadata.get("max_len", 96))
    x = normalize_hidden(hidden, max_len).unsqueeze(0).to(scorer_device)
    if scorer_device.type == "cpu":
        x = x.float()
    with torch.no_grad():
        logits = scorer(x)
        harp_score = float(score_useful(logits)[0].detach().cpu())
    result = {
        "edge_id": args.edge_id,
        "receiver": args.receiver,
        "source": args.source,
        "harp_score": harp_score,
        "score_definition": metadata.get("score_definition", "P(useful)"),
        "decision": "useful" if harp_score >= args.threshold else "not_useful",
        "hidden_shape": [int(x.shape[1]), int(x.shape[2]), int(x.shape[3])],
        "checkpoint_path": metadata.get("checkpoint_path", ""),
        "metadata_path": metadata.get("metadata_path", ""),
        "metadata": {
            "architecture": metadata.get("architecture", "HARP_S_CNN"),
            "num_layers": metadata.get("num_layers"),
            "hidden_dim": metadata.get("hidden_dim"),
            "max_len": max_len,
        },
        "latency_sec": time.perf_counter() - t0,
        **hidden_meta,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
