from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def safe_text(value: Any) -> str:
    return "" if value is None else str(value)


def edge_fields(row: Dict[str, Any]) -> Tuple[str, str]:
    query = safe_text(row.get("query") or row.get("question") or row.get("prompt"))
    candidate = safe_text(
        row.get("candidate_message")
        or row.get("message")
        or row.get("source_message")
        or row.get("source_rationale")
        or row.get("source_answer")
    )
    return query.strip(), candidate.strip()


def resolve_dtype(name: str) -> Any:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def first_model_device(model: Any) -> torch.device:
    return next((p.device for p in model.parameters() if p.device.type != "meta"), torch.device("cpu"))


def load_receiver(
    model_path: str,
    device: str = "auto",
    torch_dtype: str = "auto",
    device_map: Optional[str] = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    kwargs: Dict[str, Any] = {
        "torch_dtype": resolve_dtype(torch_dtype),
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if device_map:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    if not device_map:
        model.to(resolve_device(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


def apply_edge_template(tokenizer: Any, query: str, candidate_message: str) -> Tuple[str, str, int]:
    try:
        prefix_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": query},
                {"role": "assistant", "content": candidate_message},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        response_start = len(tokenizer(prefix_text, add_special_tokens=False).input_ids)
        return prefix_text, full_text, response_start
    except Exception:
        prefix_text = query.strip() + "\n"
        full_text = prefix_text + candidate_message.strip()
        response_start = len(tokenizer(prefix_text, add_special_tokens=False).input_ids)
        return prefix_text, full_text, response_start


def extract_edge_hidden(
    tokenizer: Any,
    model: Any,
    query: str,
    candidate_message: str,
    include_embedding: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    _, full_text, response_start = apply_edge_template(tokenizer, query, candidate_message)
    enc = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
    device = first_model_device(model)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc, output_hidden_states=True)
    states = outputs.hidden_states if include_embedding else outputs.hidden_states[1:]
    input_len = int(enc["input_ids"].shape[1])
    hidden = torch.stack([h[0, response_start:, :].detach().cpu().float() for h in states], dim=0)
    meta = {
        "input_len": input_len,
        "response_start": int(response_start),
        "raw_seq_len": int(hidden.shape[1]),
        "num_layers": int(hidden.shape[0]),
        "hidden_dim": int(hidden.shape[2]),
    }
    return hidden, meta


def edge_id(row: Dict[str, Any], index: int) -> str:
    return safe_text(row.get("edge_id") or f"edge_{index}")
