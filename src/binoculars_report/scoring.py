"""Binoculars scoring (Hans et al., ICML 2024, arXiv:2401.12070) with two tokenizer-compatible models: a base model, the observer, and its instruct fine-tune, the performer.

score = log PPL(performer) / log X-PPL(observer, performer). Both are computed per token in nats, so any span can be re-scored without re-running the models. Lower = more predictable = more likely machine-generated. Roles and math follow the reference implementation (https://github.com/ahans30/Binoculars).

Ported and adapted from clarity (https://github.com/roowus/clarity, MIT License, Copyright (c) 2026 roowus): `clarity.models` and `clarity.scoring`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModeConfig:
    observer: str  # base model
    performer: str  # instruct fine-tune of the observer; also writes the local explanations
    threshold_low: float  # below: likely AI
    threshold_high: float  # above: likely human
    max_tokens: int  # scored per document; longer texts are truncated
    dtype: torch.dtype | None  # None: float16 on GPU, float32 on CPU


MODES = {
    # The paper's pair and thresholds (low-FPR at 0.01% and best F1), selected at bfloat16 on texts of up to 512 tokens; 2048 tokens is Falcon's context window.
    "falcon": ModeConfig("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct", 0.8536, 0.9015, 2048, torch.bfloat16),
    # clarity's default pair and its provisional calibration: a 5% human false-positive rate at the low threshold.
    "fast": ModeConfig("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", 0.905, 1.11, 4096, None),
}
DEFAULT_MODE = "falcon"


@dataclass
class TokenScores:
    """Per-token quantities for positions 1..n-1 of the encoding (position 0 has no prediction)."""

    log_ppl: torch.Tensor  # -log p_performer(x_i | x_<i)
    log_xppl: torch.Tensor  # cross-entropy of the observer's next-token distribution under the performer's log-probs
    token_ids: torch.Tensor
    offsets: list[tuple[int, int]]  # char offsets into the original text, per token


class ModelPair:
    """Loads a mode's observer and performer once and scores any number of texts."""

    def __init__(self, mode: str = DEFAULT_MODE) -> None:
        self.config = MODES[mode]
        self.device = pick_device()
        # Like the reference implementation, put the performer on a second GPU when there is one.
        self.performer_device = "cuda:1" if self.device == "cuda" and torch.cuda.device_count() > 1 else self.device
        # Both models share the vocabulary; only the instruct model's tokenizer has the chat template the local explainer needs.
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.performer)
        self.observer = load_model(self.config.observer, self.device, self.config.dtype)
        self.performer = load_model(self.config.performer, self.performer_device, self.config.dtype)

    @torch.inference_mode()
    def score_text(self, text: str) -> tuple[TokenScores, torch.Tensor, bool]:
        """Run both models over the text once; returns the scores, the observer's logits and whether the text was truncated."""
        encoding = self.tokenizer(
            text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=self.config.max_tokens
        )
        input_ids = encoding.input_ids
        observer_logits = self.observer(input_ids.to(self.device)).logits[0]
        performer_logits = self.performer(input_ids.to(self.performer_device)).logits[0]
        offsets = [tuple(o) for o in encoding.offset_mapping[0].tolist()]
        scores = token_scores(observer_logits.cpu(), performer_logits.cpu(), input_ids[0], offsets)
        return scores, observer_logits, input_ids.shape[1] >= self.config.max_tokens


def token_scores(
    observer_logits: torch.Tensor,
    performer_logits: torch.Tensor,
    input_ids: torch.Tensor,
    offsets: list[tuple[int, int]],
) -> TokenScores:
    """Per-token log-perplexity and log-cross-perplexity; logits at position i predict token i + 1."""
    performer_log_probs = F.log_softmax(performer_logits[:-1].float(), dim=-1)
    observer_probs = F.softmax(observer_logits[:-1].float(), dim=-1)
    targets = input_ids[1:]
    return TokenScores(
        log_ppl=-performer_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1),
        log_xppl=-(observer_probs * performer_log_probs).sum(dim=-1),
        token_ids=targets,
        offsets=offsets[1:],
    )


def binoculars_score(scores: TokenScores, start: int = 0, end: int | None = None) -> float:
    """Score over scored-token positions [start, end): the ratio of mean log-perplexity to mean log-cross-perplexity."""
    log_ppl = scores.log_ppl[start:end]
    if log_ppl.numel() == 0:
        raise ValueError("empty token span")
    xppl_mean = scores.log_xppl[start:end].mean().item()
    if xppl_mean == 0.0:
        raise ValueError("degenerate cross-perplexity (identical constant distributions)")
    return log_ppl.mean().item() / xppl_mean


def mean_log_ppl(scores: TokenScores, start: int = 0, end: int | None = None) -> float:
    span = scores.log_ppl[start:end]
    if span.numel() == 0:
        raise ValueError("empty token span")
    return span.mean().item()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str, dtype: torch.dtype | None) -> torch.nn.Module:
    if dtype is None:
        dtype = torch.float16 if device != "cpu" else torch.float32
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
