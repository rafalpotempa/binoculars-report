"""Sentence segmentation, mapping of sentences to scored-token spans, and blending of short sentences with their neighbors.

Ported from clarity (https://github.com/roowus/clarity, MIT License, Copyright (c) 2026 roowus): `clarity.sentences` and `clarity.detector._blended_span`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pysbd

SEGMENTER = pysbd.Segmenter(language="en", clean=False, char_span=True)


@dataclass
class SentenceSpan:
    text: str
    char_start: int
    char_end: int
    tok_start: int  # index into the TokenScores arrays
    tok_end: int  # exclusive

    @property
    def n_tokens(self) -> int:
        return self.tok_end - self.tok_start


def map_sentences_to_tokens(text: str, offsets: list[tuple[int, int]]) -> list[SentenceSpan]:
    """Assign each scored token to the sentence containing its start offset; sentences without tokens are dropped."""
    spans: list[SentenceSpan] = []
    token = 0
    for sentence_text, char_start, char_end in split_sentences(text):
        # Skip tokens that start before this sentence (whitespace, overlap with the previous one).
        while token < len(offsets) and offsets[token][0] < char_start:
            token += 1
        start = token
        while token < len(offsets) and offsets[token][0] < char_end:
            token += 1
        if token > start:
            spans.append(SentenceSpan(sentence_text, char_start, char_end, start, token))
    return spans


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """(sentence text, char start, char end) for every non-blank sentence."""
    return [(s.sent, s.start, s.end) for s in SEGMENTER.segment(text) if s.sent.strip()]


def blended_span(spans: list[SentenceSpan], i: int, min_tokens: int) -> tuple[int, int, bool]:
    """Widen sentence i's token span symmetrically over its neighbors until it has at least min_tokens tokens.

    Returns (tok_start, tok_end, blended).
    """
    start, end = spans[i].tok_start, spans[i].tok_end
    if end - start >= min_tokens:
        return start, end, False
    low, high = i, i
    while end - start < min_tokens and (low > 0 or high < len(spans) - 1):
        if low > 0:
            low -= 1
            start = spans[low].tok_start
        if end - start < min_tokens and high < len(spans) - 1:
            high += 1
            end = spans[high].tok_end
    return start, end, True
