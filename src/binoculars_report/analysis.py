"""Score documents with Binoculars and build a per-sentence report that explains every flag.

The per-sentence verdicts follow clarity (https://github.com/roowus/clarity), and the per-token scores are kept, so every flagged sentence can be explained by:
- its key phrases: the words whose tokens pull the score below the AI threshold,
- the base model's own wording: how many words it would have written itself, and where the text departs from its first choice,
- optionally, a short plain-language explanation written from that evidence by an instruct model (in-process or via Ollama).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tqdm import tqdm

from .evidence import Signal, sentence_signals
from .explainer import LOCAL_EXPLAINER, OLLAMA_DEFAULT_URL, create_explainer
from .scoring import DEFAULT_MODE, ModelPair, TokenScores, binoculars_score, mean_log_ppl
from .sentences import SentenceSpan, blended_span, map_sentences_to_tokens

MIN_RELIABLE_TOKENS = 50  # document verdicts below this are marked unreliable
BLEND_MIN_TOKENS = 30  # sentences shorter than this are scored together with their neighbors
PHRASE_PULL_SHARE = 0.5  # key phrases are built from the fewest words that together cause this share of a sentence's pull toward AI
TOP_PHRASES_PER_SENTENCE = 3
TOP_PHRASES_PER_DOCUMENT = 10
TOP_ALTERNATIVES = 3
TOP_DEVIATIONS = 3
MIN_VERBATIM_RUN = 3  # words
CHOICES_CHUNK_ROWS = 512  # vocabulary-wide rank/top-k is computed in chunks to bound memory
WORD_JOINERS = "-'’"  # kept inside a word when between word characters: "source-code", "model's"


@dataclass
class ScoreComponents:
    """The score is `surprise / expected`; both are means in nats per token over the scored span.

    surprise: how unexpected the actual words are to the performer, the instruct model (mean -log p).
    expected: how unexpected the performer finds what the observer, the base model, would write (cross-perplexity).
    """

    surprise: float
    expected: float


@dataclass
class KeyPhrase:
    """A run of words whose tokens are more predictable than their context justifies, pulling the score toward AI."""

    text: str
    sentence: int  # 1-based
    pull: float  # summed token pull in nats; negative, lower = stronger pull toward AI
    share: float  # fraction of the sentence's total pull toward AI caused by this phrase
    score_without: float | None  # sentence score with the phrase's tokens left out


@dataclass
class Alternative:
    text: str
    probability: float


@dataclass
class Deviation:
    written: str
    expected: list[Alternative]  # the base model's top choices at the token where the written word departs from them


@dataclass
class ModelAgreement:
    """How much of the text is exactly what the base model would have written itself (its first choice, word by word)."""

    matched_words: int
    n_words: int
    longest_run: str | None  # longest stretch of consecutive matched words, if at least MIN_VERBATIM_RUN long
    deviations: list[Deviation]  # departures where the model was most confident in another word


@dataclass
class SentenceResult:
    text: str
    label: str  # "ai" | "uncertain" | "human"
    score: float  # blended score used for the label
    raw_score: float | None  # score of the sentence alone
    blended_with: tuple[int, int] | None  # 1-based first/last sentence whose tokens the blended score covers
    components: ScoreComponents  # over the same span as `score`
    key_phrases: list[KeyPhrase]  # only for flagged (non-human) sentences
    agreement: ModelAgreement
    signals: list[Signal]
    explanation: str | None  # written by the explainer model from the evidence above


@dataclass
class DocumentResult:
    file_name: str
    mode: str
    device: str
    score: float
    label: str
    reliable: bool
    truncated: bool
    n_tokens: int
    threshold_low: float
    threshold_high: float
    components: ScoreComponents
    agreement_rate: float  # share of words the base model would have written itself
    key_phrases: list[KeyPhrase]  # strongest key phrases across flagged sentences
    explanation_model: str | None
    sentences: list[SentenceResult]


@dataclass
class TokenChoices:
    """The base model's view of each scored token, aligned with TokenScores."""

    rank: list[int]  # 0 = the written token was the model's first choice
    top_ids: list[list[int]]
    top_probs: list[list[float]]


@dataclass
class Word:
    start: int  # char offsets into the document
    end: int
    tokens: list[int]  # indices into TokenScores


class Analyzer:
    """Loads the scoring models once and analyzes any number of documents."""

    def __init__(self, mode: str = DEFAULT_MODE, explainer: str = LOCAL_EXPLAINER, ollama_url: str = OLLAMA_DEFAULT_URL) -> None:
        self.mode = mode
        self.scorer = ModelPair(mode)
        self.threshold_low = self.scorer.config.threshold_low
        self.threshold_high = self.scorer.config.threshold_high
        self.explainer = create_explainer(explainer, self.scorer, ollama_url)

    @property
    def device(self) -> str:
        return self.scorer.device

    def analyze(self, file_name: str, text: str) -> DocumentResult:
        scores, observer_logits, truncated = self.scorer.score_text(text)
        # The observer is the base model: its first choices are what a model would write itself.
        choices = token_choices(observer_logits, scores.token_ids)
        del observer_logits
        spans = map_sentences_to_tokens(text, scores.offsets)
        doc_score = binoculars_score(scores)
        doc_components = score_components(scores, 0, None)
        # A span scores below the AI threshold exactly when its token pulls sum below zero.
        pulls = (scores.log_ppl - self.threshold_low * scores.log_xppl).tolist()
        sentence_log_ppls = [mean_log_ppl(scores, s.tok_start, s.tok_end) for s in spans]

        sentences = []
        for i, span in enumerate(spans):
            tok_start, tok_end, blended = blended_span(spans, i, BLEND_MIN_TOKENS)
            score = binoculars_score(scores, tok_start, tok_end)
            components = score_components(scores, tok_start, tok_end)
            label = self.label(score)
            words = split_words(text, scores.offsets, span.tok_start, span.tok_end)
            flagged = label != "human"
            sentences.append(
                SentenceResult(
                    text=span.text,
                    label=label,
                    score=score,
                    raw_score=binoculars_score(scores, span.tok_start, span.tok_end),
                    blended_with=sentence_range(spans, tok_start, tok_end) if blended else None,
                    components=components,
                    key_phrases=key_phrases(text, words, pulls, scores, tok_start, tok_end, i + 1) if flagged else [],
                    agreement=model_agreement(text, words, choices, self.scorer.tokenizer),
                    signals=self.signals(
                        span, score, components, doc_score, doc_components, sentence_log_ppls[max(0, i - 2) : i + 3],
                    ) if flagged else [],
                    explanation=None,
                )
            )

        n_words = sum(s.agreement.n_words for s in sentences)
        matched_words = sum(s.agreement.matched_words for s in sentences)
        result = DocumentResult(
            file_name=file_name,
            mode=self.mode,
            device=self.device,
            score=doc_score,
            label=self.label(doc_score),
            reliable=len(pulls) >= MIN_RELIABLE_TOKENS,
            truncated=truncated,
            n_tokens=len(pulls),
            threshold_low=self.threshold_low,
            threshold_high=self.threshold_high,
            components=doc_components,
            agreement_rate=matched_words / n_words if n_words else 0.0,
            key_phrases=sorted((p for s in sentences for p in s.key_phrases), key=lambda p: p.pull)[:TOP_PHRASES_PER_DOCUMENT],
            explanation_model=self.explainer.model_name if self.explainer else None,
            sentences=sentences,
        )
        if self.explainer:
            flagged = [s for s in sentences if s.label != "human"]
            for sentence in tqdm(flagged, desc=f"Explaining with {self.explainer.model_name}", unit="sentence", leave=False):
                sentence.explanation = self.explainer.explain(sentence, result)
        return result

    def label(self, score: float) -> str:
        if score < self.threshold_low:
            return "ai"
        if score > self.threshold_high:
            return "human"
        return "uncertain"

    def signals(
        self,
        span: SentenceSpan,
        score: float,
        components: ScoreComponents,
        doc_score: float,
        doc_components: ScoreComponents,
        neighbor_log_ppls: list[float],
    ) -> list[Signal]:
        signals = []
        if score < self.threshold_low:
            detail = explain_low_score(score, components, doc_score, doc_components, self.threshold_low)
            signals.append(Signal(kind="low_binoculars", detail=detail))
        return signals + sentence_signals(span.text, neighbor_log_ppls)


def score_components(scores: TokenScores, start: int, end: int | None) -> ScoreComponents:
    return ScoreComponents(
        surprise=scores.log_ppl[start:end].mean().item(),
        expected=scores.log_xppl[start:end].mean().item(),
    )


def sentence_range(spans: list[SentenceSpan], tok_start: int, tok_end: int) -> tuple[int, int]:
    first = next(i for i, s in enumerate(spans) if s.tok_start == tok_start)
    last = next(i for i, s in enumerate(spans) if s.tok_end == tok_end)
    return first + 1, last + 1


@torch.inference_mode()
def token_choices(logits: torch.Tensor, targets: torch.Tensor) -> TokenChoices:
    """Rank of each written token and the model's top alternatives; logits at position i predict token i + 1."""
    logits = logits[:-1]
    targets = targets.to(logits.device)
    ranks, top_ids, top_probs = [], [], []
    for start in range(0, len(targets), CHOICES_CHUNK_ROWS):
        chunk = logits[start : start + CHOICES_CHUNK_ROWS].float()
        written = chunk.gather(1, targets[start : start + len(chunk)].unsqueeze(1))
        ranks.append((chunk > written).sum(dim=1).cpu())
        top = chunk.softmax(dim=1).topk(TOP_ALTERNATIVES, dim=1)
        top_ids.append(top.indices.cpu())
        top_probs.append(top.values.cpu())
    return TokenChoices(
        rank=torch.cat(ranks).tolist(),
        top_ids=torch.cat(top_ids).tolist(),
        top_probs=torch.cat(top_probs).tolist(),
    )


def split_words(text: str, offsets: list[tuple[int, int]], start: int, end: int) -> list[Word]:
    """Group the tokens in [start, end) into words; whitespace-only tokens belong to no word."""
    words: list[Word] = []
    for index in range(start, end):
        char_start, char_end = offsets[index]
        piece = text[char_start:char_end]
        if not piece.strip():
            continue
        first = char_start + len(piece) - len(piece.lstrip())
        last = char_start + len(piece.rstrip())
        previous = words[-1] if words else None
        if previous and previous.end == first and continues_word(text, first):
            previous.tokens.append(index)
            previous.end = last
        else:
            words.append(Word(first, last, [index]))
    return words


def key_phrases(
    text: str,
    words: list[Word],
    pulls: list[float],
    scores: TokenScores,
    score_start: int,
    score_end: int,
    sentence: int,
) -> list[KeyPhrase]:
    """The phrases that pull the sentence score toward AI most, strongest first."""
    word_pulls = [sum(pulls[t] for t in w.tokens) for w in words]
    total_pull = -sum(p for p in word_pulls if p < 0)
    if total_pull <= 0:
        return []
    selected: list[int] = []
    covered = 0.0
    for i in sorted(range(len(words)), key=word_pulls.__getitem__):
        if word_pulls[i] >= 0 or covered >= PHRASE_PULL_SHARE * total_pull:
            break
        selected.append(i)
        covered -= word_pulls[i]

    # Merge selected words into phrases, bridging a single-word gap when that word pulls toward AI too.
    runs: list[list[int]] = []
    for i in sorted(selected):
        gap = i - runs[-1][-1] if runs else None
        if gap == 1 or (gap == 2 and word_pulls[i - 1] < 0):
            runs[-1].extend(range(runs[-1][-1] + 1, i + 1))
        else:
            runs.append([i])

    phrases = []
    for run in runs:
        pull = sum(word_pulls[i] for i in run)
        phrases.append(
            KeyPhrase(
                text=span_text(text, words[run[0]], words[run[-1]]),
                sentence=sentence,
                pull=pull,
                share=-pull / total_pull,
                score_without=score_without(scores, score_start, score_end, [t for i in run for t in words[i].tokens]),
            )
        )
    return sorted(phrases, key=lambda p: p.pull)[:TOP_PHRASES_PER_SENTENCE]


def score_without(scores: TokenScores, start: int, end: int, tokens: list[int]) -> float | None:
    keep = torch.ones(end - start, dtype=torch.bool)
    keep[[t - start for t in tokens]] = False
    if not keep.any():
        return None
    expected = scores.log_xppl[start:end][keep].sum().item()
    if expected <= 0:
        return None
    return scores.log_ppl[start:end][keep].sum().item() / expected


def model_agreement(text: str, words: list[Word], choices: TokenChoices, tokenizer) -> ModelAgreement:
    words = [w for w in words if any(c.isalnum() for c in text[w.start : w.end])]
    matched = [all(choices.rank[t] == 0 for t in w.tokens) for w in words]

    run_start, best_start, best_length = None, 0, 0
    for i, is_matched in enumerate([*matched, False]):
        if is_matched and run_start is None:
            run_start = i
        elif not is_matched and run_start is not None:
            if i - run_start > best_length:
                best_start, best_length = run_start, i - run_start
            run_start = None
    longest_run = (
        span_text(text, words[best_start], words[best_start + best_length - 1])
        if best_length >= MIN_VERBATIM_RUN else None
    )

    departures = []
    for word, is_matched in zip(words, matched):
        if is_matched:
            continue
        token = next(t for t in word.tokens if choices.rank[t] > 0)
        expected = [
            Alternative(text=alternative, probability=probability)
            for token_id, probability in zip(choices.top_ids[token], choices.top_probs[token])
            if (alternative := tokenizer.decode([token_id]).strip())
        ]
        if expected:
            departures.append((choices.top_probs[token][0], Deviation(written=text[word.start : word.end], expected=expected)))
    departures.sort(key=lambda d: d[0], reverse=True)

    return ModelAgreement(
        matched_words=sum(matched),
        n_words=len(words),
        longest_run=longest_run,
        deviations=[d for _, d in departures[:TOP_DEVIATIONS]],
    )


def span_text(text: str, first: Word, last: Word) -> str:
    return " ".join(text[first.start : last.end].split())


def continues_word(text: str, i: int) -> bool:
    """Whether the characters at i - 1 and i belong to one word, e.g. "piv|otal" or "source|-code"."""
    before, after = text[i - 1], text[i]
    if after in WORD_JOINERS:
        return is_word_char(before) and i + 1 < len(text) and is_word_char(text[i + 1])
    if before in WORD_JOINERS:
        return i >= 2 and is_word_char(text[i - 2]) and is_word_char(after)
    return is_word_char(before) and is_word_char(after)


def is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def explain_low_score(
    score: float,
    components: ScoreComponents,
    doc_score: float,
    doc_components: ScoreComponents,
    threshold: float,
) -> str:
    below = (threshold - score) / threshold * 100
    summary = (
        f"Score {score:.3f} = surprise {components.surprise:.2f} ÷ expected {components.expected:.2f} nats/token, "
        f"{below:.0f}% below the AI threshold {threshold:.3f}."
    )
    # Compare both parts with the document average to tell which one pulls the score down.
    lower_surprise = math.log(doc_components.surprise / components.surprise)
    higher_expected = math.log(components.expected / doc_components.expected)
    if max(lower_surprise, higher_expected) <= 0:
        return f"{summary} Its profile matches the document as a whole (document score {doc_score:.3f})."
    if lower_surprise >= higher_expected:
        pct = (1 - components.surprise / doc_components.surprise) * 100
        return (
            f"{summary} Driven by the wording itself: the instruct model is {pct:.0f}% less surprised by it "
            f"than by the document on average ({doc_components.surprise:.2f})."
        )
    pct = (components.expected / doc_components.expected - 1) * 100
    return (
        f"{summary} Driven by the context: a model would expect {pct:.0f}% more varied wording here than on average "
        f"({doc_components.expected:.2f}), but the actual wording is not correspondingly surprising."
    )
