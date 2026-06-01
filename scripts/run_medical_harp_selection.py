# HARP_Score_Ranking
import atexit
import gc
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harp import LEGACY_SCORER_FILENAME, SCORER_FILENAME, load_harp_scorer, normalize_hidden, score_useful

DATA_DIR = Path(os.getenv("HARP_DATA_DIR", str(REPO_ROOT / "data" / "benchmarks")))
OUTPUT_ROOT = Path(os.getenv("HARP_OUTPUT_ROOT", str(REPO_ROOT / "outputs" / "medical_harp_selection")))
LOCAL_AGENT_OUTPUTS_PATH = Path(
    os.getenv("HARP_AGENT_OUTPUTS_JSONL", str(REPO_ROOT / "data" / "precomputed_medical_agent_outputs.jsonl"))
)

LIMIT_PER_DATASET: Optional[int] = None
SHUFFLE_BEFORE_LIMIT = False
RANDOM_SEED = 42
STORE_PROMPT_IN_DETAIL = True
PROMPT_PREVIEW_CHARS = 600
RESPONSE_PREVIEW_CHARS = 2000
TOKEN_SUMMARY_MODE = "avg"

AGENT_ROLES = ["A1", "A2", "A3", "A4", "S"]
BASE_MODELS = [
    {"role": "A1", "model": "qwen3-4b", "method": "Cached: A1 / qwen3-4b"},
    {"role": "A2", "model": "qwen2.5-7b-instruct", "method": "Cached: A2 / qwen2.5-7b-instruct"},
    {"role": "A3", "model": "llama-3.1-8b-instruct", "method": "Cached: A3 / llama-3.1-8b-instruct"},
    {"role": "A4", "model": "llama-3-8b-instruct", "method": "Cached: A4 / llama-3-8b-instruct"},
    {"role": "S", "model": "qwen3-8b", "method": "Cached: S / qwen3-8b"},
]
AGENT_BY_ROLE = {cfg["role"]: cfg for cfg in BASE_MODELS}

TOPOLOGIES = [
    {
        "name": "HARP-Complete Graph",
        "method": "HARP-Complete Graph: local HARP_Score hidden-state ranking over full DAG",
    },
    {
        "name": "HARP-Layered Graph",
        "method": "HARP-Layered Graph: local HARP_Score hidden-state ranking over A1/A2 -> A3/A4 -> S",
    },
    {
        "name": "HARP-Random Graph",
        "method": "HARP-Random Graph: random 50% candidate DAG then local HARP_Score hidden-state ranking",
    },
]
MODELS = [
    {
        "role": "Five-Agent",
        "model": " | ".join([f"{cfg['role']}:{cfg['model']}" for cfg in BASE_MODELS]),
        "topology": topology["name"],
        "method": topology["method"],
    }
    for topology in TOPOLOGIES
]

DATASET_CANDIDATES = {
    "MedQA": ["RANDOM_FILTERED_MEDQA.JSONL", "filtered_medqa.jsonl", "filtered_medqa(1).jsonl"],
    "MedMCQA": ["RANDOM_FILTERED_MEDMCQA.JSONL", "filtered_medmcqa.jsonl", "filtered_medmcqa(1).jsonl"],
    "PubMedQA": ["RANDOM_FILTERED_PUBMEDQA.JSONL", "filtered_pubmedqa.jsonl", "filtered_pubmedqa(1).jsonl"],
    "MMLU-Medical": [
        "RANDOM_FILTERED_MMLU-MEDICAL.JSONL",
        "filtered_MMLU-Medical.jsonl",
        "filtered_MMLU-Medical(1).jsonl",
    ],
}

HARP_ROOT = Path(os.getenv("HARP_PROJECT_ROOT", str(REPO_ROOT)))
HARP_SCORE_ROOT = Path(os.getenv("HARP_SCORE_ROOT", str(HARP_ROOT / "checkpoints" / "harp_score_modules")))
HARP_LOCAL_MAX_SEQ_LEN = 1536
HARP_RATIONALE_MAX_WORDS = 90
HARP_LOCAL_MODEL_LOAD_RETRIES = 10
HARP_LOCAL_MODEL_LOAD_RETRY_SLEEP = 8.0
HARP_CUDA_MAX_MEMORY_BY_ATTEMPT = [
    "10GiB", "8GiB", "8GiB", "7GiB", "7GiB",
    "7GiB", "7GiB", "7GiB", "7GiB", "7GiB",
]
HARP_TARGET_CONFIG = {
    "A3": {
        "target_tag": "A3-Llama-3.1-8B-Instruct",
        "target_model": "Llama-3.1-8B-Instruct",
        "model_dir": HARP_ROOT / "models" / "Llama-3.1-8B-Instruct",
        "scorer_dir": HARP_SCORE_ROOT / "A2",
    },
    "A4": {
        "target_tag": "A4-Llama-3-8B-Instruct",
        "target_model": "Llama-3-8B-Instruct",
        "model_dir": HARP_ROOT / "models" / "Meta-Llama-3-8B-Instruct",
        "scorer_dir": HARP_SCORE_ROOT / "A3",
    },
    "S": {
        "target_tag": "S-Qwen3-8B",
        "target_model": "Qwen3-8B",
        "model_dir": HARP_ROOT / "models" / "Qwen3-8B",
        "scorer_dir": HARP_SCORE_ROOT / "S",
    },
}

WRITE_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def truncate(text: str, max_chars: int) -> str:
    text = safe_str(text)
    return text if len(text) <= max_chars else text[:max_chars] + " ...[truncated]"


def approx_tokens(text: str) -> int:
    return int(math.ceil(len(text or "") / 4.0))


def resolve_dataset_path(dataset_name: str, candidates: List[str]) -> Path:
    for name in candidates:
        path = DATA_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find dataset file for {dataset_name} under {DATA_DIR}")


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    log(f"[WARN] JSONL parse failed: {path.name}:{line_no}: {exc}")
        return rows
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return list(obj.values())
    raise ValueError(f"Unsupported JSON structure in {path}")


def normalize_options(sample: Dict[str, Any]) -> Dict[str, str]:
    if isinstance(sample.get("options"), dict):
        return {str(k).upper(): safe_str(v) for k, v in sample["options"].items() if safe_str(v)}
    if sample.get("options_json"):
        try:
            obj = json.loads(sample["options_json"])
            if isinstance(obj, dict):
                return {str(k).upper(): safe_str(v) for k, v in obj.items() if safe_str(v)}
        except Exception:
            pass
    keys = [("A", "opa"), ("B", "opb"), ("C", "opc"), ("D", "opd"), ("E", "ope"), ("F", "opf")]
    return {letter: safe_str(sample.get(key)) for letter, key in keys if safe_str(sample.get(key))}


def normalize_gold(sample: Dict[str, Any], options: Dict[str, str]) -> Tuple[Optional[str], str]:
    for key in ["answer_idx", "gold", "answer"]:
        value = sample.get(key)
        if value is None:
            continue
        text = safe_str(value).strip().upper()
        if text in options:
            return text, key
        if text.isdigit():
            idx = int(text)
            letters = sorted(options.keys())
            if 0 <= idx < len(letters):
                return letters[idx], key
            if 1 <= idx <= len(letters):
                return letters[idx - 1], key
    cop = sample.get("cop")
    if cop is not None and safe_str(cop).strip().lstrip("-").isdigit():
        idx = int(cop)
        letters = sorted(options.keys())
        if 0 <= idx < len(letters):
            return letters[idx], "cop"
    return None, ""


def normalize_text_for_match(text: str) -> str:
    text = safe_str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text.strip()


def build_options_text(options: Dict[str, str]) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in sorted(options.items()))


def normalize_pred_to_letter(pred: str, options: Dict[str, str]) -> Optional[str]:
    pred = safe_str(pred).strip()
    if not pred:
        return None
    valid = set(options.keys())
    match = re.match(r"^\s*([A-Fa-f])\b", pred)
    if match:
        letter = match.group(1).upper()
        if letter in valid:
            return letter
    pred_norm = normalize_text_for_match(pred)
    for letter, text in options.items():
        text_norm = normalize_text_for_match(text)
        if pred_norm == text_norm or (text_norm and text_norm in pred_norm):
            return letter
    return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = safe_str(text).strip()
    if not text:
        return None
    text2 = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
    text2 = re.sub(r"```$", "", text2).strip()
    try:
        obj = json.loads(text2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", text2, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_model_answer(raw_text: str, options: Dict[str, str]) -> Tuple[Optional[str], str, str, str]:
    valid = set(options.keys())
    raw = safe_str(raw_text)
    obj = extract_json_object(raw)
    rationale = ""
    answer_value = ""
    if obj:
        for key in ["answer", "answer_idx", "final_answer", "option", "choice"]:
            if key in obj:
                answer_value = safe_str(obj.get(key)).strip()
                break
        for key in ["rationale", "reason", "explanation"]:
            if key in obj:
                rationale = safe_str(obj.get(key)).strip()
                break
        letter = normalize_pred_to_letter(answer_value, options)
        if letter:
            return letter, answer_value, rationale, "json"
    for pattern in [
        r'"answer"\s*:\s*"?\s*([A-Fa-f])\b',
        r"answer\s*:\s*([A-Fa-f])\b",
        r"final answer\s*:?\s*([A-Fa-f])\b",
        r"option\s*([A-Fa-f])\b",
        r"\b([A-Fa-f])\s*[\).]\s*",
    ]:
        match = re.search(pattern, raw, flags=re.I)
        if match:
            letter = match.group(1).upper()
            if letter in valid:
                return letter, letter, rationale, "regex"
    letter = normalize_pred_to_letter(raw, options)
    if letter:
        return letter, raw[:80], rationale, "text_match"
    return None, answer_value, rationale, "unparsed"


def format_agent_outputs(agent_outputs: List[Dict[str, Any]]) -> str:
    if not agent_outputs:
        return "No upstream agent opinion is available."
    chunks = []
    for item in agent_outputs:
        role = safe_str(item.get("role", "Agent"))
        answer = safe_str(item.get("pred") or item.get("answer_text") or "unparsed")
        rationale = safe_str(item.get("rationale", "")).strip() or "No rationale parsed."
        chunks.append(f"{role}: answer={answer}\nrationale={rationale}")
    return "\n\n".join(chunks)


def build_agent_prompt(
    dataset_name: str,
    sample: Dict[str, Any],
    options: Dict[str, str],
    role: str,
    incoming_outputs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, str]], str]:
    question = safe_str(sample.get("question", "")).strip()
    options_text = build_options_text(options)
    valid_letters = ", ".join(sorted(options.keys()))
    advice_text = format_agent_outputs(incoming_outputs)
    system_msg = (
        f"You are medical QA agent {role}. Use only the given question, options, and upstream agent opinions. "
        "Treat other agents' outputs as advice, critically check them, and do not simply imitate their reasoning. "
        "Give one concise answer."
    )
    user_msg = f"""
Dataset: {dataset_name}

Medical question:
{question}

Options:
{options_text}

Other agents' opinions:
{advice_text}

Task:
Select exactly one best option from: {valid_letters}.
Return JSON only, with this exact schema:
{{
  "answer": "one option letter such as A",
  "rationale": "brief rationale within 100 words"
}}
""".strip()
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}], system_msg + "\n\n" + user_msg


def build_decision_prompt(
    dataset_name: str,
    sample: Dict[str, Any],
    options: Dict[str, str],
    incoming_outputs: List[Dict[str, Any]],
    decision_role: str = "S",
) -> Tuple[List[Dict[str, str]], str]:
    question = safe_str(sample.get("question", "")).strip()
    options_text = build_options_text(options)
    valid_letters = ", ".join(sorted(options.keys()))
    advice_text = format_agent_outputs(incoming_outputs)
    system_msg = (
        f"You are agent {decision_role}, the top decision-maker for a medical multiple-choice QA team. "
        "Analyze and summarize other agents' opinions, identify possible mistakes, and make the final decision. "
        "Use only the given question, options, and agent opinions."
    )
    user_msg = f"""
Dataset: {dataset_name}

Medical question:
{question}

Options:
{options_text}

Agents' opinions:
{advice_text}

Task:
Make the final team decision. Select exactly one best option from: {valid_letters}.
Return JSON only, with this exact schema:
{{
  "answer": "one option letter such as A",
  "rationale": "brief decision rationale within 100 words"
}}
""".strip()
    return [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}], system_msg + "\n\n" + user_msg


def load_local_agent_outputs(path: Path) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    if not path.exists():
        log(f"[WARN] Local agent output file not found: {path}")
        return index
    rows = load_json_or_jsonl(path)
    for row in rows:
        dataset = safe_str(row.get("dataset") or row.get("dataset_name"))
        sample_index = safe_str(row.get("sample_index", ""))
        sample_id = safe_str(row.get("sample_id", ""))
        topology = safe_str(row.get("topology") or row.get("topology_name"))
        if isinstance(row.get("agents"), dict):
            for role, agent_payload in row["agents"].items():
                payload = dict(agent_payload)
                payload.setdefault("role", role)
                payload.setdefault("topology", topology)
                payload.setdefault("dataset", dataset)
                payload.setdefault("sample_index", sample_index)
                payload.setdefault("sample_id", sample_id)
                register_output_record(index, payload)
        else:
            register_output_record(index, row)
    return index


def register_output_record(index: Dict[Tuple[str, str, str, str], Dict[str, Any]], row: Dict[str, Any]) -> None:
    dataset = safe_str(row.get("dataset") or row.get("dataset_name"))
    role = safe_str(row.get("role") or row.get("model_role"))
    if not dataset or not role:
        return
    topology = safe_str(row.get("topology") or row.get("topology_name"))
    sample_index = safe_str(row.get("sample_index", ""))
    sample_id = safe_str(row.get("sample_id", ""))
    normalized = dict(row)
    normalized["dataset"] = dataset
    normalized["role"] = role
    normalized["topology"] = topology
    if sample_index:
        index[(dataset, sample_index, topology, role)] = normalized
        index[(dataset, sample_index, "", role)] = normalized
    if sample_id:
        index[(dataset, sample_id, topology, role)] = normalized
        index[(dataset, sample_id, "", role)] = normalized


def find_cached_output(
    output_index: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    dataset_name: str,
    sample_index: int,
    sample_id: str,
    topology_name: str,
    role: str,
) -> Optional[Dict[str, Any]]:
    keys = [
        (dataset_name, str(sample_index), topology_name, role),
        (dataset_name, sample_id, topology_name, role),
        (dataset_name, str(sample_index), "", role),
        (dataset_name, sample_id, "", role),
    ]
    for key in keys:
        if key in output_index:
            return output_index[key]
    return None


def normalize_cached_agent_output(
    row: Optional[Dict[str, Any]],
    role: str,
    model_name: str,
    options: Dict[str, str],
    prompt_text: str,
) -> Dict[str, Any]:
    if row is None:
        raw = ""
        pred = None
        answer_text = ""
        rationale = ""
        parse_status = "missing_local_output"
        latency_sec = 0.0
        prompt_tokens = approx_tokens(prompt_text)
        completion_tokens = 0
    else:
        raw = safe_str(row.get("raw") or row.get("raw_response") or row.get("response") or row.get("model_raw"))
        pred = row.get("pred") or row.get("answer") or row.get("model_answer")
        answer_text = safe_str(row.get("answer_text") or row.get("model_answer_text") or pred or "")
        rationale = safe_str(row.get("rationale") or row.get("model_rationale") or row.get("reason") or "")
        parse_status = safe_str(row.get("parse_status") or "cached")
        if not pred and raw:
            pred, parsed_answer, parsed_rationale, parsed_status = parse_model_answer(raw, options)
            answer_text = answer_text or parsed_answer
            rationale = rationale or parsed_rationale
            parse_status = parsed_status
        pred = normalize_pred_to_letter(safe_str(pred), options) if pred else None
        latency_sec = float(row.get("latency_sec", row.get("latency", 0.0)) or 0.0)
        prompt_tokens = int(row.get("prompt_tokens", approx_tokens(prompt_text)) or 0)
        completion_tokens = int(row.get("completion_tokens", approx_tokens(raw or rationale)) or 0)
    total_tokens = int(prompt_tokens + completion_tokens)
    return {
        "role": role,
        "model_name": model_name,
        "prompt_text": prompt_text,
        "raw": raw,
        "pred": pred,
        "answer_text": answer_text,
        "rationale": rationale,
        "parse_status": parse_status,
        "meta": {
            "latency_sec": latency_sec,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "usage_source": "local_cache_or_estimate",
            "local_attempts": 0,
            "local_error": "" if row is not None else f"missing local output for {role}",
        },
    }


def trim_words(text: str, max_words: int) -> str:
    words = safe_str(text).split()
    return safe_str(text) if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def build_harp_edge_texts(
    dataset_name: str,
    sample: Dict[str, Any],
    options: Dict[str, str],
    source_output: Dict[str, Any],
    target_role: str,
) -> Tuple[str, str]:
    target_cfg = HARP_TARGET_CONFIG[target_role]
    question = safe_str(sample.get("question", "")).strip()
    options_text = build_options_text(options)
    source_answer = safe_str(source_output.get("pred") or source_output.get("answer_text") or "unparsed")
    source_rationale = trim_words(source_output.get("rationale", ""), HARP_RATIONALE_MAX_WORDS)
    user_text = f"""You are target agent {target_cfg['target_tag']} receiving one upstream answer and rationale.

Dataset: {dataset_name}
Target agent: {target_cfg['target_tag']}

Question:
{question}

Options:
{options_text}

The following assistant message is an upstream opinion. It may be high quality or low quality. Read it without knowing which model generated it."""
    assistant_text = f"""Answer: {source_answer}
Rationale: {source_rationale or 'No rationale parsed.'}"""
    return user_text.strip(), assistant_text.strip()


def apply_harp_chat_template(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


class HARPScoreManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_target_role: Optional[str] = None
        self.active_model = None
        self.active_tokenizer = None
        self.scorers: Dict[str, nn.Module] = {}
        self.scorer_metadata: Dict[str, Dict[str, Any]] = {}
        self.failed_target_loads: Dict[str, str] = {}
        self.scorer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def validate_assets(self) -> None:
        for target_role, cfg in HARP_TARGET_CONFIG.items():
            model_dir = Path(cfg["model_dir"])
            scorer_dir = Path(cfg["scorer_dir"])
            required = [
                model_dir / "config.json",
                scorer_dir / "scorer_metadata.json",
            ]
            for path in required:
                if not path.exists():
                    raise FileNotFoundError(f"Missing HARP asset for {target_role}: {path}")
            if not (scorer_dir / SCORER_FILENAME).exists() and not (scorer_dir / LEGACY_SCORER_FILENAME).exists():
                raise FileNotFoundError(f"Missing HARP asset for {target_role}: {scorer_dir / SCORER_FILENAME}")

    def close(self) -> None:
        with self.lock:
            self._unload_active_model()
            self.scorers.clear()
            self.scorer_metadata.clear()
            self.failed_target_loads.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _unload_active_model(self) -> None:
        if self.active_model is not None:
            del self.active_model
        if self.active_tokenizer is not None:
            del self.active_tokenizer
        self.active_model = None
        self.active_tokenizer = None
        self.active_target_role = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _load_target_model_locked(self, target_role: str) -> Tuple[Any, Any]:
        if self.active_target_role == target_role and self.active_model is not None and self.active_tokenizer is not None:
            return self.active_model, self.active_tokenizer
        if target_role in self.failed_target_loads:
            raise RuntimeError(self.failed_target_loads[target_role])
        cfg = HARP_TARGET_CONFIG[target_role]
        model_dir = Path(cfg["model_dir"])
        offload_dir = HARP_ROOT / "checkpoints" / "harp_runtime_offload" / target_role
        offload_dir.mkdir(parents=True, exist_ok=True)
        last_error: Optional[Exception] = None
        for attempt in range(1, HARP_LOCAL_MODEL_LOAD_RETRIES + 1):
            self._unload_active_model()
            model = None
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                if torch.cuda.is_available():
                    dtype = torch.float16
                    cuda_memory = HARP_CUDA_MAX_MEMORY_BY_ATTEMPT[
                        min(attempt - 1, len(HARP_CUDA_MAX_MEMORY_BY_ATTEMPT) - 1)
                    ]
                    kwargs = {
                        "device_map": "auto",
                        "max_memory": {0: cuda_memory, "cpu": "80GiB"},
                        "offload_folder": str(offload_dir),
                    }
                else:
                    dtype = torch.float32
                    kwargs = {}
                log(f"[HARP] loading target model {target_role}: {model_dir} | attempt={attempt}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    local_files_only=True,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    **kwargs,
                )
                model.eval()
                self.active_target_role = target_role
                self.active_model = model
                self.active_tokenizer = tokenizer
                self.failed_target_loads.pop(target_role, None)
                return model, tokenizer
            except Exception as exc:
                last_error = exc
                log(f"[HARP-WARN] target model load failed for {target_role}: {str(exc)[:300]}")
                if model is not None:
                    del model
                if tokenizer is not None:
                    del tokenizer
                self._unload_active_model()
                if attempt < HARP_LOCAL_MODEL_LOAD_RETRIES:
                    time.sleep(HARP_LOCAL_MODEL_LOAD_RETRY_SLEEP)
        failure_text = f"HARP target model load failed for {target_role}. Last error: {last_error}"
        self.failed_target_loads[target_role] = failure_text
        raise RuntimeError(failure_text)

    def _load_scorer_locked(self, target_role: str) -> Tuple[nn.Module, Dict[str, Any]]:
        if target_role in self.scorers:
            return self.scorers[target_role], self.scorer_metadata[target_role]
        scorer_dir = Path(HARP_TARGET_CONFIG[target_role]["scorer_dir"])
        model, metadata = load_harp_scorer(scorer_dir, self.scorer_device)
        self.scorers[target_role] = model
        self.scorer_metadata[target_role] = metadata
        return model, metadata

    def _response_hidden_locked(
        self,
        model: Any,
        tokenizer: Any,
        dataset_name: str,
        sample: Dict[str, Any],
        options: Dict[str, str],
        source_output: Dict[str, Any],
        target_role: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        user_text, assistant_text = build_harp_edge_texts(dataset_name, sample, options, source_output, target_role)
        prefix_text = apply_harp_chat_template(
            tokenizer,
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True,
        )
        full_text = apply_harp_chat_template(
            tokenizer,
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            add_generation_prompt=False,
        )
        if not full_text.startswith(prefix_text):
            raise RuntimeError("HARP chat template prefix mismatch; cannot locate assistant span.")
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        enc = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
        input_len = int(enc["input_ids"].shape[1])
        response_start = len(prefix_ids)
        if input_len > HARP_LOCAL_MAX_SEQ_LEN:
            raise RuntimeError(f"HARP input too long: {input_len} > {HARP_LOCAL_MAX_SEQ_LEN}")
        if response_start >= input_len:
            raise RuntimeError(f"Bad HARP response span: start={response_start}, input_len={input_len}")
        first_device = next(
            (p.device for p in model.parameters() if p.device.type != "meta"),
            torch.device("cpu"),
        )
        enc = {k: v.to(first_device) for k, v in enc.items()}
        with torch.inference_mode():
            outputs = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden = torch.stack(
            [
                h[0, response_start:input_len].detach().to("cpu", dtype=torch.float16)
                for h in outputs.hidden_states[1:]
            ],
            dim=0,
        )
        meta = {
            "input_len": input_len,
            "response_start_idx": response_start,
            "response_len": int(input_len - response_start),
            "hidden_shape": list(hidden.shape),
        }
        return hidden, meta

    @staticmethod
    def _prepare_scorer_input(hidden: torch.Tensor, max_len: int) -> torch.Tensor:
        return normalize_hidden(hidden, max_len).unsqueeze(0)

    def score_edge(
        self,
        dataset_name: str,
        sample: Dict[str, Any],
        options: Dict[str, str],
        source_output: Dict[str, Any],
        edge: Tuple[str, str],
    ) -> Dict[str, Any]:
        source_role, target_role = edge
        if target_role not in HARP_TARGET_CONFIG:
            raise ValueError(f"HARP_Score is not defined for target role {target_role}")
        t0 = time.perf_counter()
        with self.lock:
            try:
                scorer, scorer_meta = self._load_scorer_locked(target_role)
                model, tokenizer = self._load_target_model_locked(target_role)
                hidden, hidden_meta = self._response_hidden_locked(
                    model=model,
                    tokenizer=tokenizer,
                    dataset_name=dataset_name,
                    sample=sample,
                    options=options,
                    source_output=source_output,
                    target_role=target_role,
                )
                max_len = int(scorer_meta.get("max_len", 96))
                x = self._prepare_scorer_input(hidden, max_len).to(self.scorer_device)
                if self.scorer_device.type == "cpu":
                    x = x.float()
                with torch.inference_mode():
                    with torch.autocast(device_type=self.scorer_device.type, enabled=self.scorer_device.type == "cuda"):
                        logits = scorer(x)
                    prob = score_useful(logits)[0].detach().cpu().item()
                latency = time.perf_counter() - t0
                del hidden, x
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {
                    "edge": [source_role, target_role],
                    "score": float(prob),
                    "status": "ok",
                    "error": "",
                    "source_pred": source_output.get("pred"),
                    "source_parse_status": source_output.get("parse_status", ""),
                    "target_role": target_role,
                    "target_model": HARP_TARGET_CONFIG[target_role]["target_model"],
                    "score_method": "HARP_Score",
                    "harp_score_model_type": scorer_meta.get("architecture", scorer_meta.get("scorer_type", "HARP_S_CNN")),
                    "harp_score_checkpoint_path": scorer_meta.get("checkpoint_path", str(Path(HARP_TARGET_CONFIG[target_role]["scorer_dir"]) / SCORER_FILENAME)),
                    "local_latency_sec": latency,
                    **hidden_meta,
                }
            except Exception as exc:
                return {
                    "edge": [source_role, target_role],
                    "score": 0.5,
                    "status": "error",
                    "error": str(exc),
                    "source_pred": source_output.get("pred"),
                    "source_parse_status": source_output.get("parse_status", ""),
                    "target_role": target_role,
                    "target_model": HARP_TARGET_CONFIG[target_role]["target_model"],
                    "score_method": "HARP_Score",
                    "harp_score_model_type": "HARP_S_CNN",
                    "harp_score_checkpoint_path": str(Path(HARP_TARGET_CONFIG[target_role]["scorer_dir"]) / SCORER_FILENAME),
                    "local_latency_sec": time.perf_counter() - t0,
                }


HARP_SCORE_MANAGER = HARPScoreManager()
atexit.register(HARP_SCORE_MANAGER.close)

COMPLETE_DAG_EDGES = [
    (src, dst)
    for src_index, src in enumerate(AGENT_ROLES)
    for dst in AGENT_ROLES[src_index + 1:]
]


def random_graph_edges(dataset_name: str, sample_index: int, sample_id: str, retain_ratio: float = 0.5) -> List[Tuple[str, str]]:
    keep_n = max(1, int(round(len(COMPLETE_DAG_EDGES) * retain_ratio)))
    seed_text = f"{RANDOM_SEED}|{dataset_name}|{sample_index}|{sample_id}|random_graph_50"
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(seed_text))
    rng = random.Random(seed)
    edges = set(rng.sample(COMPLETE_DAG_EDGES, keep_n))
    if not any(dst == "S" for _, dst in edges):
        non_s_edge = next(iter(edges))
        s_edge = rng.choice([edge for edge in COMPLETE_DAG_EDGES if edge[1] == "S"])
        edges.remove(non_s_edge)
        edges.add(s_edge)
    return sorted(edges, key=lambda edge: COMPLETE_DAG_EDGES.index(edge))


def candidate_edges_for_topology(
    topology_name: str,
    dataset_name: str,
    sample_index: int,
    sample_id: str,
) -> List[Tuple[str, str]]:
    if topology_name == "HARP-Complete Graph":
        return list(COMPLETE_DAG_EDGES)
    if topology_name == "HARP-Layered Graph":
        return [("A1", "A3"), ("A2", "A3"), ("A1", "A4"), ("A2", "A4"), ("A3", "S"), ("A4", "S")]
    if topology_name == "HARP-Random Graph":
        return random_graph_edges(dataset_name, sample_index, sample_id, retain_ratio=0.5)
    raise ValueError(f"Unknown topology: {topology_name}")


def rank_scored_edges(scored_edges: List[Dict[str, Any]], target_role: Optional[str] = None) -> List[Dict[str, Any]]:
    candidates = [
        item for item in scored_edges
        if target_role is None or item.get("edge", [None, None])[1] == target_role
    ]
    candidates.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),
            COMPLETE_DAG_EDGES.index(tuple(item["edge"])) if tuple(item["edge"]) in COMPLETE_DAG_EDGES else 999,
        )
    )
    ranked = []
    for rank, item in enumerate(candidates, start=1):
        ranked_item = dict(item)
        ranked_item["rank"] = rank
        ranked.append(ranked_item)
    return ranked


def rank_harp_edges(scored_edges: List[Dict[str, Any]], target_role: str) -> List[Tuple[str, str]]:
    return [tuple(item["edge"]) for item in rank_scored_edges(scored_edges, target_role)]


def make_harp_info(
    topology_name: str,
    candidate_edges: List[Tuple[str, str]],
    ranked_edges: List[Tuple[str, str]],
    scored_edges: List[Dict[str, Any]],
) -> Dict[str, Any]:
    score_errors = [item for item in scored_edges if item.get("status") != "ok"]
    ranked_score_values = rank_scored_edges(scored_edges)
    return {
        "harp_topology": topology_name,
        "candidate_edges": [list(edge) for edge in candidate_edges],
        "ranked_edges": [list(edge) for edge in ranked_edges],
        "harp_score_values": scored_edges,
        "ranked_score_values": ranked_score_values,
        "score_method": "HARP_Score",
        "harp_score_model_type": "HARP_S_CNN",
        "harp_score_checkpoint_paths": {
            role: str(Path(cfg["scorer_dir"]) / SCORER_FILENAME)
            for role, cfg in HARP_TARGET_CONFIG.items()
        },
        "hidden_model_paths": {
            role: str(Path(cfg["model_dir"]))
            for role, cfg in HARP_TARGET_CONFIG.items()
        },
        "local_latency_sec": sum(float(item.get("local_latency_sec", 0.0)) for item in scored_edges),
        "local_status": "ok" if not score_errors else "partial_error",
        "local_error_count": len(score_errors),
        "local_errors": score_errors[:5],
    }


def create_state(
    dataset_name: str,
    dataset_path: Path,
    sample_index: int,
    sample: Dict[str, Any],
    model_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    sample_id = safe_str(sample.get("id", f"{dataset_name}_{sample_index:06d}"))
    options = normalize_options(sample)
    gold, gold_source = normalize_gold(sample, options)
    state = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "sample": sample,
        "model_cfg": model_cfg,
        "topology_name": model_cfg["topology"],
        "options": options,
        "gold": gold,
        "gold_source": gold_source,
        "candidate_edges": [],
        "scored_edges": [],
        "ranked_edges": [],
        "outputs": {},
        "target_candidates": {},
        "fatal_error": "",
        "invalid": False,
    }
    question = safe_str(sample.get("question", "")).strip()
    if not question or not options or gold is None:
        state["invalid"] = True
        state["fatal_error"] = "missing question/options/gold"
        return state
    state["candidate_edges"] = candidate_edges_for_topology(model_cfg["topology"], dataset_name, sample_index, sample_id)
    return state


def incoming_outputs_for_target(state: Dict[str, Any], target_role: str) -> List[Dict[str, Any]]:
    outputs = state.get("outputs", {})
    incoming = []
    for source_role, dst_role in state.get("ranked_edges", []):
        if dst_role == target_role and source_role in outputs:
            incoming.append(outputs[source_role])
    return incoming


def target_candidates(state: Dict[str, Any], target_role: str) -> List[Tuple[str, str]]:
    outputs = state.get("outputs", {})
    candidates = [
        edge for edge in state.get("candidate_edges", [])
        if edge[1] == target_role and edge[0] in outputs
    ]
    if target_role == "S" and state.get("topology_name") == "HARP-Random Graph" and not candidates:
        fallback_source = "A4" if "A4" in outputs else "A3" if "A3" in outputs else ""
        if fallback_source:
            fallback = (fallback_source, "S")
            if fallback not in state["candidate_edges"]:
                state["candidate_edges"].append(fallback)
            candidates = [fallback]
    return candidates


def add_ranked_edges(state: Dict[str, Any], new_edges: List[Tuple[str, str]]) -> None:
    existing = set(state.get("ranked_edges", []))
    for edge in new_edges:
        if edge not in existing:
            state["ranked_edges"].append(edge)
            existing.add(edge)


def hydrate_initial_outputs(
    state: Dict[str, Any],
    output_index: Dict[Tuple[str, str, str, str], Dict[str, Any]],
) -> None:
    for role in AGENT_ROLES:
        messages, prompt_text = (
            build_decision_prompt(state["dataset_name"], state["sample"], state["options"], [], role)
            if role == "S"
            else build_agent_prompt(state["dataset_name"], state["sample"], state["options"], role, [])
        )
        del messages
        row = find_cached_output(
            output_index,
            state["dataset_name"],
            state["sample_index"],
            state["sample_id"],
            state["topology_name"],
            role,
        )
        state["outputs"][role] = normalize_cached_agent_output(
            row,
            role,
            AGENT_BY_ROLE[role]["model"],
            state["options"],
            prompt_text,
        )


def score_and_rank_for_target(state: Dict[str, Any], target_role: str) -> None:
    candidates = target_candidates(state, target_role)
    state.setdefault("target_candidates", {})[target_role] = candidates
    scored = []
    for edge in candidates:
        source_role, _ = edge
        source_output = state["outputs"].get(source_role)
        if source_output is None or source_output.get("parse_status") == "missing_local_output":
            scored.append({
                "edge": [source_role, target_role],
                "score": 0.0,
                "status": "missing_source",
                "error": f"source output {source_role} is not available",
                "local_latency_sec": 0.0,
            })
            continue
        scored.append(
            HARP_SCORE_MANAGER.score_edge(
                dataset_name=state["dataset_name"],
                sample=state["sample"],
                options=state["options"],
                source_output=source_output,
                edge=edge,
            )
        )
    state["scored_edges"].extend(scored)
    if not candidates:
        return
    add_ranked_edges(state, rank_harp_edges(scored, target_role))


def update_prompts_after_ranking(state: Dict[str, Any]) -> None:
    for role in AGENT_ROLES:
        incoming = incoming_outputs_for_target(state, role)
        messages, prompt_text = (
            build_decision_prompt(state["dataset_name"], state["sample"], state["options"], incoming, role)
            if role == "S"
            else build_agent_prompt(state["dataset_name"], state["sample"], state["options"], role, incoming)
        )
        del messages
        output = state["outputs"].get(role)
        if output:
            output["prompt_text"] = prompt_text
            output["meta"]["prompt_tokens"] = approx_tokens(prompt_text)
            output["meta"]["total_tokens"] = output["meta"]["prompt_tokens"] + output["meta"]["completion_tokens"]


def run_local_harp_ranking(
    tasks: List[Tuple[str, int, Dict[str, Any], Dict[str, Any]]],
    dataset_paths: Dict[str, Path],
    output_index: Dict[Tuple[str, str, str, str], Dict[str, Any]],
    detail_jsonl_path: Path,
) -> List[Dict[str, Any]]:
    states = [
        create_state(dataset_name, dataset_paths[dataset_name], idx, sample, model_cfg)
        for dataset_name, idx, sample, model_cfg in tasks
    ]
    for state in states:
        if state.get("invalid"):
            continue
        hydrate_initial_outputs(state, output_index)
        if state["topology_name"] != "HARP-Layered Graph":
            candidates = target_candidates(state, "A2")
            state.setdefault("target_candidates", {})["A2"] = candidates
            add_ranked_edges(state, candidates)
        score_and_rank_for_target(state, "A3")
        score_and_rank_for_target(state, "A4")
        score_and_rank_for_target(state, "S")
        update_prompts_after_ranking(state)
    results = []
    for state in states:
        record = build_record_from_state(state)
        append_jsonl(detail_jsonl_path, record)
        results.append(record)
    return results


def state_prompt_text(state: Dict[str, Any]) -> str:
    outputs = state.get("outputs", {})
    return "\n\n".join(
        [f"{role}\n{outputs[role].get('prompt_text', '')}" for role in AGENT_ROLES if role in outputs]
    )


def state_meta(state: Dict[str, Any], harp_info: Dict[str, Any]) -> Dict[str, Any]:
    outputs = state.get("outputs", {})
    ordered = [outputs[role] for role in AGENT_ROLES if role in outputs]
    metas = [item.get("meta", {}) for item in ordered]
    local_harp_latency = float(harp_info.get("local_latency_sec", 0.0))
    return {
        "latency_sec": sum(float(m.get("latency_sec", 0.0)) for m in metas) + local_harp_latency,
        "prompt_tokens": sum(int(m.get("prompt_tokens", 0)) for m in metas),
        "completion_tokens": sum(int(m.get("completion_tokens", 0)) for m in metas),
        "total_tokens": sum(int(m.get("total_tokens", 0)) for m in metas),
        "usage_source": "local_cache_prompt_estimate_plus_local_harp_latency",
        "local_attempts": 0,
        "local_error": "",
    }


def base_record(state: Dict[str, Any]) -> Dict[str, Any]:
    sample = state["sample"]
    model_cfg = state["model_cfg"]
    return {
        "timestamp": now_str(),
        "block": "Multi-Agent",
        "method": model_cfg["method"],
        "topology": model_cfg["topology"],
        "dataset": state["dataset_name"],
        "dataset_file": str(state["dataset_path"]),
        "sample_index": state["sample_index"],
        "sample_id": state["sample_id"],
        "model_role": model_cfg["role"],
        "model_name": model_cfg["model"],
        "question": safe_str(sample.get("question", "")).strip(),
        "options_json": json.dumps(state["options"], ensure_ascii=False),
        "gold": state["gold"],
        "gold_source": state["gold_source"],
        "subject_name": sample.get("subject_name", sample.get("meta_info", "")),
        "topic_name": sample.get("topic_name", ""),
    }


def build_record_from_state(state: Dict[str, Any]) -> Dict[str, Any]:
    record = base_record(state)
    prompt_text = state_prompt_text(state)
    harp_info = make_harp_info(
        topology_name=state["topology_name"],
        candidate_edges=state.get("candidate_edges", []),
        ranked_edges=state.get("ranked_edges", []),
        scored_edges=state.get("scored_edges", []),
    )
    harp_fields = {
        "topology_edges_json": json.dumps(harp_info.get("ranked_edges", []), ensure_ascii=False),
        "harp_candidate_edges_json": json.dumps(harp_info.get("candidate_edges", []), ensure_ascii=False),
        "harp_score_values_json": json.dumps(harp_info.get("harp_score_values", []), ensure_ascii=False),
        "harp_ranked_score_values_json": json.dumps(harp_info.get("ranked_score_values", []), ensure_ascii=False),
        "harp_ranked_edges_json": json.dumps(harp_info.get("ranked_edges", []), ensure_ascii=False),
        "harp_score_method": harp_info.get("score_method", ""),
        "harp_score_model_type": harp_info.get("harp_score_model_type", ""),
        "harp_score_checkpoint_paths_json": json.dumps(harp_info.get("harp_score_checkpoint_paths", {}), ensure_ascii=False),
        "harp_hidden_model_paths_json": json.dumps(harp_info.get("hidden_model_paths", {}), ensure_ascii=False),
        "harp_local_latency_sec": harp_info.get("local_latency_sec", 0.0),
        "harp_local_status": harp_info.get("local_status", ""),
        "harp_local_error_count": harp_info.get("local_error_count", 0),
        "harp_local_errors_json": json.dumps(harp_info.get("local_errors", []), ensure_ascii=False),
    }
    if state.get("invalid"):
        record.update({
            "status": "skipped_invalid_sample",
            "pred": None,
            "correct": None,
            "raw_response": "",
            "parse_status": "",
            "local_error": state.get("fatal_error", "missing question/options/gold"),
            "latency_sec": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            **harp_fields,
        })
        return record
    final_output = state.get("outputs", {}).get("S")
    meta = state_meta(state, harp_info)
    raw_payload = {
        "topology": state["topology_name"],
        "ranked_edges": harp_info.get("ranked_edges", []),
        "harp": harp_info,
        "agents": {
            role: {
                "model_name": out.get("model_name", ""),
                "pred": out.get("pred"),
                "answer_text": out.get("answer_text"),
                "rationale": out.get("rationale"),
                "parse_status": out.get("parse_status"),
                "raw": out.get("raw"),
                "prompt_text": out.get("prompt_text", ""),
            }
            for role, out in state.get("outputs", {}).items()
        },
    }
    raw = json.dumps(raw_payload, ensure_ascii=False)
    pred = final_output.get("pred") if final_output else None
    status = "ok" if pred is not None else "missing_agent_output"
    parse_status = (
        f"{state['topology_name']}; final_parse={final_output.get('parse_status', '')}"
        if final_output else f"{state['topology_name']}; final_parse=missing"
    )
    record.update({
        "status": status,
        "pred": pred,
        "correct": int(bool(pred == state["gold"])) if pred is not None else 0,
        "model_answer_text": final_output.get("answer_text", "") if final_output else "",
        "model_rationale": final_output.get("rationale", "") if final_output else "",
        "parse_status": parse_status,
        "raw_response": truncate(raw, RESPONSE_PREVIEW_CHARS),
        "raw_response_chars": len(raw),
        "prompt_preview": truncate(prompt_text, PROMPT_PREVIEW_CHARS),
        "prompt_full": prompt_text if STORE_PROMPT_IN_DETAIL else "",
        "prompt_chars": len(prompt_text),
        "local_error": "" if status == "ok" else "missing cached S output",
        **harp_fields,
        **meta,
    })
    return record


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_mean(values: List[float]) -> Optional[float]:
    clean = [value for value in values if value is not None and not pd.isna(value)]
    return float(sum(clean) / len(clean)) if clean else None


def safe_std(values: List[float]) -> Optional[float]:
    clean = [value for value in values if value is not None and not pd.isna(value)]
    if len(clean) < 2:
        return 0.0 if clean else None
    return float(statistics.stdev(clean))


def build_summaries(details_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid_df = details_df[details_df["status"].isin(["ok", "missing_agent_output", "skipped_invalid_sample"])]
    rows_long = []
    for (method, model_role, model_name, dataset), group in valid_df.groupby(
        ["method", "model_role", "model_name", "dataset"],
        dropna=False,
    ):
        n = len(group)
        correct_sum = int(group["correct"].fillna(0).sum())
        rows_long.append({
            "block": "Multi-Agent",
            "method": method,
            "model_role": model_role,
            "model_name": model_name,
            "dataset": dataset,
            "n": n,
            "ok_count": int((group["status"] == "ok").sum()),
            "missing_agent_output_count": int((group["status"] == "missing_agent_output").sum()),
            "skipped_count": int((group["status"] == "skipped_invalid_sample").sum()),
            "correct": correct_sum,
            "accuracy": correct_sum / n if n else None,
            "avg_prompt_tokens": safe_mean(group["prompt_tokens"].astype(float).tolist()),
            "avg_completion_tokens": safe_mean(group["completion_tokens"].astype(float).tolist()),
            "avg_total_tokens": safe_mean(group["total_tokens"].astype(float).tolist()),
            "sum_prompt_tokens": int(group["prompt_tokens"].fillna(0).sum()),
            "sum_completion_tokens": int(group["completion_tokens"].fillna(0).sum()),
            "sum_total_tokens": int(group["total_tokens"].fillna(0).sum()),
            "avg_latency_sec": safe_mean(group["latency_sec"].astype(float).tolist()),
            "std_latency_sec": safe_std(group["latency_sec"].astype(float).tolist()),
            "p50_latency_sec": float(group["latency_sec"].quantile(0.50)) if n else None,
            "p95_latency_sec": float(group["latency_sec"].quantile(0.95)) if n else None,
        })
    summary_long = pd.DataFrame(rows_long).sort_values(["method", "dataset"]).reset_index(drop=True)
    dataset_order = ["MedQA", "MedMCQA", "PubMedQA", "MMLU-Medical"]
    rows_main = []
    for model_cfg in MODELS:
        method = model_cfg["method"]
        row = {"Block": "Multi-Agent", "Method": method}
        accs, toks, lats = [], [], []
        for dataset in dataset_order:
            sub = summary_long[(summary_long["method"] == method) & (summary_long["dataset"] == dataset)]
            if sub.empty:
                row[f"{dataset} Acc"] = None
                row[f"{dataset} Tok"] = None
                row[f"{dataset} Lat"] = None
                continue
            record = sub.iloc[0]
            token_value = record["avg_total_tokens"] if TOKEN_SUMMARY_MODE == "avg" else record["sum_total_tokens"]
            row[f"{dataset} Acc"] = record["accuracy"]
            row[f"{dataset} Tok"] = token_value
            row[f"{dataset} Lat"] = record["avg_latency_sec"]
            if record["accuracy"] is not None:
                accs.append(float(record["accuracy"]))
            if token_value is not None:
                toks.append(float(token_value))
            if record["avg_latency_sec"] is not None:
                lats.append(float(record["avg_latency_sec"]))
        row["Avg Acc"] = safe_mean(accs)
        row["Avg Tok"] = safe_mean(toks)
        row["Avg Lat"] = safe_mean(lats)
        rows_main.append(row)
    summary_main = pd.DataFrame(rows_main)
    errors = details_df[details_df["status"].isin(["missing_agent_output", "skipped_invalid_sample"])].copy()
    return summary_main, summary_long, errors


def sanitize_excel_value(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", value)
    return value


def sanitize_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.applymap(sanitize_excel_value)


def write_excel(
    output_xlsx: Path,
    details_df: pd.DataFrame,
    summary_main: pd.DataFrame,
    summary_long: pd.DataFrame,
    errors_df: pd.DataFrame,
    config_rows: List[Dict[str, Any]],
) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    details_df = sanitize_dataframe_for_excel(details_df)
    summary_main = sanitize_dataframe_for_excel(summary_main)
    summary_long = sanitize_dataframe_for_excel(summary_long)
    errors_df = sanitize_dataframe_for_excel(errors_df)
    config_df = sanitize_dataframe_for_excel(pd.DataFrame(config_rows))
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        summary_main.to_excel(writer, index=False, sheet_name="summary_main")
        summary_long.to_excel(writer, index=False, sheet_name="summary_long")
        details_df.to_excel(writer, index=False, sheet_name="details")
        errors_df.to_excel(writer, index=False, sheet_name="errors")
        config_df.to_excel(writer, index=False, sheet_name="config")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def main() -> None:
    random.seed(RANDOM_SEED)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"five_agent_harp_score_local_ranking_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_jsonl_path = output_dir / "multi_agent_details.jsonl"
    output_xlsx = output_dir / "five_agent_harp_score_medical_results.xlsx"
    log("=" * 80)
    log("Local medical HARP_Score ranking experiment")
    log(f"DATA_DIR: {DATA_DIR}")
    log(f"LOCAL_AGENT_OUTPUTS_PATH: {LOCAL_AGENT_OUTPUTS_PATH}")
    log(f"OUTPUT_DIR: {output_dir}")
    log("=" * 80)
    HARP_SCORE_MANAGER.validate_assets()
    output_index = load_local_agent_outputs(LOCAL_AGENT_OUTPUTS_PATH)
    dataset_data: Dict[str, List[Dict[str, Any]]] = {}
    dataset_paths: Dict[str, Path] = {}
    for dataset_name, candidates in DATASET_CANDIDATES.items():
        path = resolve_dataset_path(dataset_name, candidates)
        rows = load_json_or_jsonl(path)
        if SHUFFLE_BEFORE_LIMIT:
            random.shuffle(rows)
        if LIMIT_PER_DATASET is not None:
            rows = rows[:LIMIT_PER_DATASET]
        dataset_data[dataset_name] = rows
        dataset_paths[dataset_name] = path
        log(f"[DATA] {dataset_name}: {len(rows)} samples from {path}")
    tasks = []
    for dataset_name, rows in dataset_data.items():
        for model_cfg in MODELS:
            for idx, sample in enumerate(rows):
                tasks.append((dataset_name, idx, sample, model_cfg))
    log(f"[TASKS] total local result rows: {len(tasks)}")
    start = time.perf_counter()
    results = run_local_harp_ranking(tasks, dataset_paths, output_index, detail_jsonl_path)
    log(f"[PROGRESS] local HARP ranking finished | records={len(results)} | elapsed={time.perf_counter() - start:.1f}s")
    all_records = read_jsonl(detail_jsonl_path)
    details_df = pd.DataFrame(all_records)
    summary_main, summary_long, errors_df = build_summaries(details_df)
    config_rows = [
        {"key": "run_id", "value": run_id},
        {"key": "start_time", "value": now_str()},
        {"key": "data_dir", "value": str(DATA_DIR)},
        {"key": "local_agent_outputs_path", "value": str(LOCAL_AGENT_OUTPUTS_PATH)},
        {"key": "output_dir", "value": str(output_dir)},
        {"key": "limit_per_dataset", "value": LIMIT_PER_DATASET},
        {"key": "token_summary_mode", "value": TOKEN_SUMMARY_MODE},
        {"key": "harp_score_root", "value": str(HARP_SCORE_ROOT)},
        {"key": "harp_local_max_seq_len", "value": HARP_LOCAL_MAX_SEQ_LEN},
        {"key": "harp_rationale_max_words", "value": HARP_RATIONALE_MAX_WORDS},
        {"key": "topologies", "value": ", ".join([item["name"] for item in TOPOLOGIES])},
        {"key": "runtime_policy", "value": "No network model calls and no live agent inference. The script reads cached agent outputs and runs only local HARP_Score hidden-state scoring and ranking."},
    ]
    for dataset_name, path in dataset_paths.items():
        config_rows.append({"key": f"dataset_path_{dataset_name}", "value": str(path)})
    for model_cfg in BASE_MODELS:
        config_rows.append({"key": f"model_{model_cfg['role']}", "value": model_cfg["model"]})
    for target_role, cfg in HARP_TARGET_CONFIG.items():
        config_rows.append({"key": f"harp_target_model_{target_role}", "value": str(cfg["model_dir"])})
        config_rows.append({"key": f"harp_score_checkpoint_{target_role}", "value": str(Path(cfg["scorer_dir"]) / SCORER_FILENAME)})
    write_excel(output_xlsx, details_df, summary_main, summary_long, errors_df, config_rows)
    HARP_SCORE_MANAGER.close()
    log("=" * 80)
    log(f"Finished: {now_str()}")
    log(f"Detail JSONL: {detail_jsonl_path}")
    log(f"Excel result: {output_xlsx}")
    log("=" * 80)


if __name__ == "__main__":
    main()
