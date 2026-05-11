
import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class HARP_S_CNN(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        num_layers: int,
        kernel_sizes: Tuple[int, int, int] = (3, 4, 5),
        num_filters: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=num_layers,
                    out_channels=num_filters,
                    kernel_size=(k, embedding_dim),
                )
                for k in kernel_sizes
            ]
        )
        self.deep_convs = nn.ModuleList(
            [nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1) for _ in range(2)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * num_filters, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_outputs: List[torch.Tensor] = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            seq_len = x.size(2)
            if seq_len < k:
                pad_amt = k - seq_len
                x_padded = F.pad(x, (0, 0, 0, pad_amt))
                out = F.relu(conv(x_padded)).squeeze(3)
            else:
                out = F.relu(conv(x)).squeeze(3)
            for deep_conv in self.deep_convs:
                out = F.relu(deep_conv(out))
            pooled = F.adaptive_max_pool1d(out, 1).squeeze(2)
            conv_outputs.append(pooled)
        features = torch.cat(conv_outputs, dim=1)
        features = self.dropout(features)
        return self.fc(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HARP_S_CNN on hidden states from a local language-model backbone"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/path/to/local-model",
        help="Path to the LLM checkpoint",
    )
    parser.add_argument(
        "--pos_train_path",
        type=str,
        default="positive_train.json",
        help="Path to positive-sample JSON file",
    )
    parser.add_argument(
        "--neg_dataset1_path",
        type=str,
        default="negative_train_1.json",
        help="Path to first negative-sample JSON file",
    )
    parser.add_argument(
        "--neg_dataset2_path",
        type=str,
        default="negative_train_2.json",
        help="Path to second negative-sample JSON file",
    )
    parser.add_argument(
        "--local_data_path",
        type=str,
        default="train_data_15000.json",
        help="Cached merged dataset path",
    )
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_every_steps", type=int, default=15000)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/harp_scorer",
        help="Directory for scorer checkpoints and loss plots",
    )
    parser.add_argument("--use_layers", choices=["all", "last"], default="all")
    parser.add_argument("--use_part", choices=["front", "middle", "back", "full"], default="full")
    parser.add_argument("--device_cnn", type=str, default="cuda:0")
    parser.add_argument("--device_llm", type=str, default="cuda:1")
    parser.add_argument(
        "--backbone",
        choices=["chat_template", "instruct_template"],
        default="instruct_template",
        help="Choose prompt template style: chat_template or instruct_template",
    )
    return parser.parse_args()


def load_backbone(model_path: str, device_llm: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": device_llm}
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    return tokenizer, model


def build_dataset(
    pos_path: str,
    neg1_path: str,
    neg2_path: str,
    cache_path: str,
    max_steps: int,
) -> List[Tuple[str, str, int]]:
    if Path(cache_path).exists():
        print("Loading cached dataset ...")
        return json.loads(Path(cache_path).read_text(encoding="utf-8"))
    print("Creating dataset from raw files ...")
    with open(pos_path, "r", encoding="utf-8") as f:
        pos_data = json.load(f)
    with open(neg1_path, "r", encoding="utf-8") as f:
        neg1_data = json.load(f)
    with open(neg2_path, "r", encoding="utf-8") as f:
        neg2_data = json.load(f)

    full: List[Tuple[str, str, int]] = []
    for item in pos_data:
        q = f"{item['instruction']}\nInput:{item['input']}"
        full.append((q, item["output"], 1))
    for item in neg1_data:
        q = f"{item['instruction']}\nInput:{item['input']}"
        full.append((q, item["model_output"], 0))
    for item in neg2_data:
        q = f"{item['instruction']}\nInput:{item['input']}"
        full.append((q, item["model_output"], 0))
    random.shuffle(full)
    subset = full[:max_steps]
    Path(cache_path).write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    return subset


def select_hidden_states(hidden_states: torch.Tensor, cfg: argparse.Namespace, total_layers: int) -> torch.Tensor:
    if cfg.use_layers == "last":
        return hidden_states[-1:].contiguous()
    third = total_layers // 3
    if cfg.use_part == "front":
        return hidden_states[:third]
    if cfg.use_part == "middle":
        return hidden_states[third : 2 * third]
    if cfg.use_part == "back":
        return hidden_states[2 * third :]
    return hidden_states


def save_checkpoint(model: nn.Module, loss_hist: List[float], step: int, cfg: argparse.Namespace):
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = output_dir / f"harp_s_step_{step}_{cfg.use_part}.pth"
    torch.save(model.state_dict(), ckpt_name)
    print(f"[Checkpoint] Saved model to {ckpt_name}")

    loss_path = output_dir / f"loss_history_step_{step}_{cfg.use_part}.txt"
    Path(loss_path).write_text("\n".join(str(l) for l in loss_hist))
    print(f"[Checkpoint] Saved loss history to {loss_path}")

    plt.figure()
    plt.plot(loss_hist)
    plt.title("Loss over training")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig(output_dir / f"loss_plot_step_{step}_{cfg.use_part}.png")
    plt.close()


def build_prompts(question: str, answer: str, backbone: str):
    if backbone == "chat_template":
        user_only = [{"role": "user", "content": question}]
        full_dialog = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return user_only, full_dialog
    else:
        user_only = f"<s> [INST] {question} [/INST]"
        full_dialog = f"{user_only} {answer} </s>"
        return user_only, full_dialog


def train(cfg: argparse.Namespace) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_cnn = torch.device(cfg.device_cnn)
    device_llm = torch.device(cfg.device_llm)

    tokenizer, backbone = load_backbone(cfg.model_path, cfg.device_llm)
    total_layers = backbone.config.num_hidden_layers + 1

    num_layers_selected = 1 if cfg.use_layers == "last" else (
        {
            "front": total_layers // 3,
            "middle": total_layers // 3,
            "back": total_layers - 2 * (total_layers // 3),
            "full": total_layers,
        }[cfg.use_part]
    )
    print(f"Using {num_layers_selected} layers (use_layers={cfg.use_layers}, use_part={cfg.use_part})")
    classifier = HARP_S_CNN(
        embedding_dim=backbone.config.hidden_size,
        num_classes=2,
        num_layers=num_layers_selected,
    ).to(device_cnn)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg.learning_rate)
    scaler = GradScaler()

    dataset = build_dataset(
        cfg.pos_train_path,
        cfg.neg_dataset1_path,
        cfg.neg_dataset2_path,
        cfg.local_data_path,
        cfg.max_steps,
    )

    total_loss, step_loss = 0.0, 0.0
    loss_hist: List[float] = []
    accum_counter, step_counter, save_counter = 0, 0, 0

    for question, answer, label in tqdm(dataset, desc=f"Training up to {cfg.max_steps} steps"):
        if step_counter >= cfg.max_steps:
            break

        user_only, full_dialog = build_prompts(question, answer, cfg.backbone)

        if cfg.backbone == "chat_template":
            text2 = tokenizer.apply_chat_template(
                user_only, tokenize=False, add_generation_prompt=True
            )
            text = tokenizer.apply_chat_template(
                full_dialog, tokenize=False, add_generation_prompt=True
            )
        else:
            text2 = user_only
            text = full_dialog

        inputs2 = tokenizer([text2], return_tensors="pt")
        inputs = tokenizer([text], return_tensors="pt").to(device_llm)
        start_idx = len(inputs2["input_ids"][0])

        with torch.no_grad():
            outputs = backbone(inputs["input_ids"], output_hidden_states=True)
        hidden = torch.stack(outputs.hidden_states, dim=0).squeeze(1)
        hidden = hidden[:, start_idx:]
        hidden = select_hidden_states(hidden, cfg, total_layers).to(device_cnn)
        hidden = hidden.unsqueeze(0)
        labels = torch.tensor([label], device=device_cnn)

        with autocast():
            logits = classifier(hidden)
            loss = criterion(logits, labels) / cfg.grad_accum_steps
        scaler.scale(loss).backward()

        total_loss += loss.item() * cfg.grad_accum_steps
        step_loss += loss.item() * cfg.grad_accum_steps

        accum_counter += 1
        save_counter += 1
        step_counter += 1

        if accum_counter >= cfg.grad_accum_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            accum_counter = 0

        if save_counter % cfg.save_every_steps == 0:
            save_checkpoint(classifier, loss_hist, step_counter, cfg)

        if step_counter % (25 * cfg.grad_accum_steps) == 0:
            avg_step_loss = step_loss / (25 * cfg.grad_accum_steps)
            print(f"Step {step_counter}: avg loss {avg_step_loss:.6f}")
            loss_hist.append(avg_step_loss)
            step_loss = 0.0

    if accum_counter:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / max(step_counter, 1)
    print(f"Finished training after {step_counter} steps | final avg loss {avg_loss:.6f}")

    final_ckpt = output_dir / f"harp_s_final_{cfg.use_part}.pth"
    torch.save(classifier.state_dict(), final_ckpt)
    print(f"Saved final model to {final_ckpt}")

    (output_dir / f"loss_history_final_{cfg.use_part}.txt").write_text("\n".join(str(l) for l in loss_hist))
    plt.figure()
    plt.plot(loss_hist)
    plt.title("Final loss trajectory")
    plt.xlabel("Checkpoint index")
    plt.ylabel("Loss")
    plt.savefig(output_dir / f"loss_plot_final_{cfg.use_part}.png")
    plt.close()


def main():
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
