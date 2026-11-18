"""Turn a flagged sentence's measured evidence into a short plain-language explanation with a local instruct model.

The model only rewords the evidence it is given; it does not judge the text. Its output is a plausible reading of the measurements, not a measurement.

Explainers (`--explainer`): `local` reuses the scoring pair's in-process instruct model (Falcon-7B-Instruct, or Qwen2.5-1.5B-Instruct in fast mode); `ollama:<model>` uses any chat model served by a local Ollama; `none` disables explanations.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    import torch

    from .analysis import DocumentResult, SentenceResult
    from .scoring import ModelPair

LOCAL_EXPLAINER = "local"
NO_EXPLAINER = "none"
OLLAMA_PREFIX = "ollama:"
OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_SECONDS = 600  # the first request also loads the model
OLLAMA_LOAD_REFRESH_SECONDS = 0.5
MAX_NEW_TOKENS = 160
# Falcon's chat format has no end-of-turn token: the model goes on to write the next user turn.
STOP_STRING = "\nUser:"
LABEL_TEXT = {"ai": "likely AI-generated", "uncertain": "uncertain, leaning AI-generated"}
SYSTEM_PROMPT = (
    "You explain the output of an AI-text detector to the author of a scientific paper. "
    "You receive one sentence the detector flagged and the measurements behind the flag. "
    "Answer in exactly two lines. "
    "Line 1: one or two sentences explaining why the detector finds this sentence predictable, quoting only phrases and numbers from the measurements. "
    "Never mention tone, style, jargon, creativity or anything not in the measurements, and never claim who wrote the sentence. "
    "Line 2: start with 'Suggestion:' and propose a concrete replacement for the strongest phrase that keeps the sentence's meaning."
)
# One worked example: small instruct models follow a demonstration far more reliably than instructions alone.
EXAMPLE_EVIDENCE = "\n".join([
    'Sentence: "This approach plays a crucial role in improving the overall performance of the system."',
    "Detector verdict: likely AI-generated.",
    'Phrases that make it most predictable, with their share of the machine-like signal: "plays a crucial role in" (48%), "the overall performance of" (22%).',
    "A language model, writing on its own, would have chosen 12 of 15 words (80%; whole document: 41%).",
    'Longest stretch the model would have written word for word: "plays a crucial role in improving the overall performance".',
    'Where the author departed from the model\'s first choice: wrote "approach" where the model expected "method".',
])
EXAMPLE_ANSWER = (
    'A language model would have written 80% of this sentence by itself (41% for the document), and the stock phrases '
    '"plays a crucial role in" and "the overall performance of" cause 70% of the signal; only "approach" departs from its first choice.\n'
    'Suggestion: replace "plays a crucial role in improving the overall performance of" with the specific effect, e.g. "halves the response time of".'
)


def create_explainer(spec: str, scorer: ModelPair, ollama_url: str = OLLAMA_DEFAULT_URL) -> Explainer | None:
    if spec == NO_EXPLAINER:
        return None
    if spec == LOCAL_EXPLAINER:
        return LocalExplainer(scorer.performer, scorer.tokenizer, scorer.config.performer)
    if spec.startswith(OLLAMA_PREFIX) and spec.removeprefix(OLLAMA_PREFIX):
        return OllamaExplainer(spec.removeprefix(OLLAMA_PREFIX), ollama_url)
    raise ValueError(f"unknown explainer {spec!r}; use {LOCAL_EXPLAINER}, {NO_EXPLAINER} or {OLLAMA_PREFIX}<model>")


def is_valid_explainer(spec: str) -> bool:
    return spec in (LOCAL_EXPLAINER, NO_EXPLAINER) or (spec.startswith(OLLAMA_PREFIX) and bool(spec.removeprefix(OLLAMA_PREFIX)))


class Explainer:
    model_name: str

    def explain(self, sentence: SentenceResult, document: DocumentResult) -> str:
        return self.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE_EVIDENCE},
            {"role": "assistant", "content": EXAMPLE_ANSWER},
            {"role": "user", "content": evidence_prompt(sentence, document)},
        ]).strip()

    def chat(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class LocalExplainer(Explainer):
    """Runs the scoring pair's instruct model in-process; torch is imported lazily so the CLI starts fast."""

    def __init__(self, model: torch.nn.Module, tokenizer, model_name: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

    def chat(self, messages: list[dict[str, str]]) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None, top_p=None, top_k=None,
                stop_strings=[STOP_STRING], tokenizer=self.tokenizer, pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return answer.split(STOP_STRING)[0]


class OllamaExplainer(Explainer):
    """Calls a chat model served by a local Ollama (https://ollama.com) over its HTTP API."""

    def __init__(self, model: str, url: str = OLLAMA_DEFAULT_URL) -> None:
        self.model = model
        self.model_name = f"{model} via Ollama"
        self.url = url.rstrip("/")
        try:
            info = self._post("/api/show", {"model": model})
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}\nIs Ollama running, and is the model pulled (`ollama pull {model}`)?") from exc
        # Thinking only adds latency when rewording given evidence; turn it off where the model supports it.
        self.thinks = "thinking" in info.get("capabilities", [])
        self._load()

    def _load(self) -> None:
        """Load the model into Ollama's memory up front; Ollama reports no load progress, so show the elapsed time."""
        errors: list[RuntimeError] = []

        def load() -> None:
            try:
                # An empty prompt makes Ollama load the model and return without generating.
                self._post("/api/generate", {"model": self.model, "prompt": "", "stream": False})
            except RuntimeError as exc:
                errors.append(exc)

        # A daemon thread, so Ctrl+C is not blocked by a load in progress.
        loader = threading.Thread(target=load, daemon=True)
        loader.start()
        with tqdm(desc=f"Loading {self.model_name}", bar_format="{desc}: {elapsed}{postfix}") as bar:
            while loader.is_alive():
                loader.join(OLLAMA_LOAD_REFRESH_SECONDS)
                bar.refresh()
            if errors:
                raise errors[0]
            if (gpu_share := self._gpu_share()) is not None:
                bar.set_postfix_str(f"{gpu_share:.0%} on GPU")

    def _gpu_share(self) -> float | None:
        """Share of the loaded model held in GPU memory; the rest runs on CPU."""
        tagged = self.model if ":" in self.model else f"{self.model}:latest"
        for running in self._get("/api/ps").get("models", []):
            if running.get("name") in (self.model, tagged) and running.get("size"):
                return running.get("size_vram", 0) / running["size"]
        return None

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_predict": MAX_NEW_TOKENS},
        }
        if self.thinks:
            payload["think"] = False
        return self._post("/api/chat", payload)["message"]["content"]

    def _post(self, path: str, payload: dict) -> dict:
        return self._request(path, json.dumps(payload).encode())

    def _get(self, path: str) -> dict:
        return self._request(path, None)

    def _request(self, path: str, data: bytes | None) -> dict:
        request = urllib.request.Request(f"{self.url}{path}", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Ollama {path} failed ({exc.code}): {exc.read().decode(errors='replace')}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach Ollama at {self.url}: {exc.reason}") from exc


def evidence_prompt(sentence: SentenceResult, document: DocumentResult) -> str:
    agreement = sentence.agreement
    lines = [
        f'Sentence: "{" ".join(sentence.text.split())}"',
        f"Detector verdict: {LABEL_TEXT[sentence.label]}.",
    ]
    if sentence.key_phrases:
        phrases = ", ".join(f'"{p.text}" ({p.share:.0%})' for p in sentence.key_phrases)
        lines.append(f"Phrases that make it most predictable, with their share of the machine-like signal: {phrases}.")
    if agreement.n_words:
        lines.append(
            f"A language model, writing on its own, would have chosen {agreement.matched_words} of its {agreement.n_words} words "
            f"({agreement.matched_words / agreement.n_words:.0%}; whole document: {document.agreement_rate:.0%})."
        )
    if agreement.longest_run:
        lines.append(f'Longest stretch the model would have written word for word: "{agreement.longest_run}".')
    if agreement.deviations:
        deviations = "; ".join(f'wrote "{d.written}" where the model expected "{d.expected[0].text}"' for d in agreement.deviations)
        lines.append(f"Where the author departed from the model's first choice: {deviations}.")
    lines += [f"Other signal: {s.detail}" for s in sentence.signals if s.kind != "low_binoculars"]
    return "\n".join(lines)
