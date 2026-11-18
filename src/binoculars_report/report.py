"""Render a DocumentResult as an HTML (default) or Markdown report, and a terminal verdict line.

Both formats share one layout: verdict, heatmap, a table of the flagged (AI and uncertain) sentences, and per-sentence evidence at the end.
Colors and words follow clarity's web UI and CLI (https://github.com/roowus/clarity).
"""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import DocumentResult, SentenceResult

LABEL_COLORS = {
    "ai": ("#DC2626", "#FFFFFF"),
    "uncertain": ("#FDE047", "#713F12"),
    "human": ("#DCFCE7", "#14532D"),
}
LABEL_EMOJI = {"ai": "🟥", "uncertain": "🟨", "human": "🟩"}
LABEL_WORD = {"ai": "likely AI-generated", "uncertain": "uncertain", "human": "likely human"}
LABEL_ANSI = {"ai": "\033[1;97;41m", "uncertain": "\033[30;43m", "human": "\033[32m"}
ANSI_RESET = "\033[0m"
TITLE = "Binoculars report"
SIGNAL_TITLE = {"low_binoculars": "Why the score is low", "low_burstiness": "Low burstiness", "ai_idiom": "AI idiom"}
NUMBER_HELP = "Sentence number in the document. 4≈3–4: a short sentence scored together with its neighbors 3–4."
LABEL_HELP = "Label, the sentence score, and the label boundaries: AI below the low threshold, human above the high one."
HEATMAP_HELP = "Hover a sentence for its number, label and score; click a flagged one to jump to its row in the table."
COMPONENTS_LEGEND = (
    "Score = Surprise ÷ Expected (nats/token). Surprise: how unexpected the actual words are to the instruct model. "
    "Expected: how unexpected the instruct model finds the words its base model would write here. "
    "A low score means the text is as predictable as model-written text."
)
EVIDENCE_LEGEND = (
    "Key phrases: the words that pull the score below the AI threshold, i.e. are more predictable than their context justifies; "
    "the percentage is their share of the sentence's pull toward AI. "
    "Model's own wording: words the base model would have written itself as its first choice. "
    "Departs from the model at: where the text differs from the model's confident first choice, with its top alternatives."
)
DISCLAIMER = (
    "No detector is proof. Scores are calibrated probabilities, not verdicts; "
    "false positives disproportionately affect non-native writers. Always review the evidence."
)
SUGGESTION_PREFIX = re.compile(r"^[\s*_>#-]*suggestion[\s*_]*:[\s*_]*", re.IGNORECASE)
LINE_PREFIX = re.compile(r"^[\s*_]*line\s*\d[\s*_]*:[\s*_]*", re.IGNORECASE)
MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]|$<>#~])")
HTML_STYLE = """
body { font-family: system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #1F2937; }
h1 { font-size: 1.6rem; } h2 { margin-top: 2rem; border-bottom: 1px solid #E5E7EB; }
.pill { padding: 1px 4px; border-radius: 3px; }
.verdict { border-left: 4px solid #D1D5DB; padding: .25rem 1rem; }
.heatmap span { cursor: help; } .heatmap a { text-decoration: none; }
.heatmap a span:hover, .heatmap span:hover { outline: 2px solid #1F2937; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #E5E7EB; padding: 6px 8px; vertical-align: top; text-align: left; }
th { background: #F9FAFB; } th[title] { cursor: help; text-decoration: underline dotted; }
td.number { text-align: right; white-space: nowrap; } td.label { white-space: nowrap; }
.bounds, .muted { color: #6B7280; font-size: .85em; }
tr:target, section:target { background: #FEF9C3; }
blockquote { border-left: 4px solid #D1D5DB; margin: 0 0 .5rem; padding: .25rem 1rem; color: #374151; }
""".strip()


def render_html(result: DocumentResult) -> str:
    title = html.escape(f"{TITLE} — {result.file_name}")
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{title}</title><style>\n{HTML_STYLE}\n</style></head><body>",
        f"<h1>{title}</h1>",
        '<div class="verdict">',
        f"<p>{pill(LABEL_WORD[result.label], result.label)} {html.escape(verdict_line(result, bold=False))}</p>",
        f"<p class=\"muted\">{html.escape(document_summary(result))}</p>",
    ]
    if caveats := document_caveats(result):
        parts.append(f"<p>⚠ {html.escape('; '.join(caveats))}</p>")
    parts.append("</div>")

    legend = " ".join(pill(LABEL_WORD[label], label) for label in LABEL_COLORS)
    heatmap = " ".join(html_heatmap_span(index, s, result) for index, s in enumerate(result.sentences, start=1))
    parts += ["<h2>Heatmap</h2>", f"<p>{legend} <span class=\"muted\">{HEATMAP_HELP}</span></p>", f'<p class="heatmap">{heatmap}</p>']

    flagged = flagged_sentences(result)
    parts += ["<h2>Flagged sentences</h2>"]
    if flagged:
        parts += [
            f'<p class="muted">{html.escape(explanation_note(result))}</p>',
            "<table><thead><tr>",
            f'<th title="{html.escape(NUMBER_HELP)}">#</th><th>Sentence</th><th title="{html.escape(LABEL_HELP)}">Label</th>'
            "<th>Why flagged</th><th>Suggestion</th>",
            "</tr></thead><tbody>",
        ]
        for index, s in flagged:
            number = sentence_number(index, s)
            parts.append(
                f'<tr id="s{index}">'
                f'<td class="number"><a href="#d{index}" title="Evidence for sentence {number}">{number}</a></td>'
                f"<td>{html.escape(flat(s.text))}</td>"
                f'<td class="label">{pill(s.label, s.label)} {s.score:.3f}<br>'
                f'<span class="bounds">AI &lt; {result.threshold_low} · human &gt; {result.threshold_high}</span></td>'
                f"<td>{html.escape(why_flagged(s, result))}</td>"
                f"<td>{html.escape(suggestion(s))}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    else:
        parts.append("<p>No sentence is flagged.</p>")

    if flagged:
        parts += [
            "<h2>Details</h2>",
            f'<p class="muted">{html.escape(COMPONENTS_LEGEND)}</p>',
            f'<p class="muted">{html.escape(EVIDENCE_LEGEND)}</p>',
        ]
        for index, s in flagged:
            items = "".join(f"<li><b>{html.escape(name)}:</b> {html.escape(text)}</li>" for name, text in evidence(s, result))
            parts += [
                f'<section id="d{index}">',
                f'<h3><a href="#s{index}">{sentence_number(index, s)}</a> · {pill(LABEL_WORD[s.label], s.label)} · score {s.score:.3f}</h3>',
                f"<blockquote>{html.escape(flat(s.text))}</blockquote>",
                f"<ul>{items}</ul>",
                "</section>",
            ]

    parts += [f'<p class="muted"><i>{html.escape(DISCLAIMER)}</i></p>', "</body></html>", ""]
    return "\n".join(parts)


def render_markdown(result: DocumentResult) -> str:
    lines = [
        f"# {TITLE} — {result.file_name}",
        "",
        f"> {LABEL_EMOJI[result.label]} {verdict_line(result)}",
        ">",
        f"> {document_summary(result)}",
    ]
    if caveats := document_caveats(result):
        lines += [">", f"> ⚠ {'; '.join(caveats)}"]

    legend = " ".join(pill(LABEL_WORD[label], label) for label in LABEL_COLORS)
    heatmap = " ".join(pill(flat(s.text), s.label, heatmap_tooltip(i, s, result)) for i, s in enumerate(result.sentences, start=1))
    lines += ["", "## Heatmap", "", f"<p>{legend}</p>", "", f"<p>{heatmap}</p>", ""]

    flagged = flagged_sentences(result)
    lines += ["## Flagged sentences", ""]
    if flagged:
        lines += [
            f"*{explanation_note(result)}*",
            "",
            "| # | Sentence | Label | Why flagged | Suggestion |",
            "|--:|---|---|---|---|",
        ]
        for index, s in flagged:
            lines.append(
                f"| {sentence_number(index, s)} | {markdown_cell(s.text)} "
                f"| {LABEL_EMOJI[s.label]} {s.label} {s.score:.3f}<br>AI \\< {result.threshold_low} · human \\> {result.threshold_high} "
                f"| {markdown_cell(why_flagged(s, result))} | {markdown_cell(suggestion(s))} |"
            )
        lines += ["", f"*{NUMBER_HELP}*", ""]
        lines += ["## Details", "", f"*{COMPONENTS_LEGEND}*", "", f"*{EVIDENCE_LEGEND}*", ""]
        for index, s in flagged:
            lines += [
                f"### {sentence_number(index, s)} · {LABEL_EMOJI[s.label]} {LABEL_WORD[s.label]} · score {s.score:.3f}",
                "",
                f"> {markdown_cell(s.text)}",
                "",
            ]
            lines += [f"- **{name}:** {markdown_cell(text)}" for name, text in evidence(s, result)]
            lines.append("")
    else:
        lines += ["No sentence is flagged.", ""]

    lines += [f"*{DISCLAIMER}*", ""]
    return "\n".join(lines)


def verdict_line(result: DocumentResult, ansi: bool = False, bold: bool = True) -> str:
    word = LABEL_WORD[result.label]
    if ansi:
        word = f"{LABEL_ANSI[result.label]} {word} {ANSI_RESET}"
    elif bold:
        word = f"**{word}**"
    return (
        f"Document: {word} (score {result.score:.3f}; "
        f"AI < {result.threshold_low}, human > {result.threshold_high}; mode {result.mode})"
    )


def flagged_sentences(result: DocumentResult) -> list[tuple[int, SentenceResult]]:
    return [(index, s) for index, s in enumerate(result.sentences, start=1) if s.label != "human"]


def why_flagged(s: SentenceResult, result: DocumentResult) -> str:
    why, _ = split_explanation(s.explanation)
    if why:
        return why
    if details := [signal.detail for signal in s.signals]:
        return " ".join(details)
    return f"Score {s.score:.3f} lies between the AI ({result.threshold_low}) and human ({result.threshold_high}) thresholds."


def suggestion(s: SentenceResult) -> str:
    _, text = split_explanation(s.explanation)
    return text or "—"


def split_explanation(explanation: str | None) -> tuple[str, str]:
    """Split the explainer's answer into its reasoning and its 'Suggestion:' line."""
    why, suggested = [], []
    for line in (explanation or "").splitlines():
        if not line.strip():
            continue
        if SUGGESTION_PREFIX.match(line):
            suggested.append(SUGGESTION_PREFIX.sub("", line).strip())
        else:
            (suggested if suggested else why).append(LINE_PREFIX.sub("", line).strip())
    return " ".join(why), " ".join(suggested)


def evidence(s: SentenceResult, result: DocumentResult) -> list[tuple[str, str]]:
    score = f"{s.score:.3f} = surprise {s.components.surprise:.2f} ÷ expected {s.components.expected:.2f}"
    if s.blended_with and s.raw_score is not None:
        score += f" over sentences {s.blended_with[0]}–{s.blended_with[1]}; the sentence alone scores {s.raw_score:.3f}"
    items = [("Score", score)]
    if s.key_phrases:
        items.append(("Key phrases", "; ".join(
            f"«{flat(p.text)}» {p.share:.0%}" + (f" (score without it {p.score_without:.3f})" if p.score_without is not None else "")
            for p in s.key_phrases
        )))
    agreement = s.agreement
    if agreement.n_words:
        own = (
            f"{agreement.matched_words} of {agreement.n_words} words are what the base model would have written itself "
            f"({agreement.matched_words / agreement.n_words:.0%}; document {result.agreement_rate:.0%})"
        )
        if agreement.longest_run:
            own += f"; longest such run: «{flat(agreement.longest_run)}»"
        items.append(("Model's own wording", own))
    if agreement.deviations:
        items.append(("Departs from the model at", "; ".join(
            f"«{flat(d.written)}» (expected " + ", ".join(f"«{flat(a.text)}» {a.probability:.0%}" for a in d.expected) + ")"
            for d in agreement.deviations
        )))
    items += [(SIGNAL_TITLE.get(signal.kind, signal.kind.replace("_", " ").capitalize()), flat(signal.detail)) for signal in s.signals]
    return items


def document_summary(result: DocumentResult) -> str:
    return (
        f"Surprise {result.components.surprise:.2f} ÷ expected {result.components.expected:.2f}; "
        f"{result.n_tokens} tokens; device {result.device}. "
        f"The base model would have written {result.agreement_rate:.0%} of the words itself."
    )


def document_caveats(result: DocumentResult) -> list[str]:
    caveats = []
    if not result.reliable:
        caveats.append(f"only {result.n_tokens} tokens — too short for a reliable verdict")
    if result.truncated:
        caveats.append("text was truncated at the model's scoring window")
    return caveats


def explanation_note(result: DocumentResult) -> str:
    if result.explanation_model:
        return (
            f"Why flagged and Suggestion are written by {result.explanation_model} from the measured evidence in Details: "
            "a plausible reading, not a measurement."
        )
    return "Why flagged is the measured evidence; run with an explainer for plain-language reasons and suggestions."


def html_heatmap_span(index: int, s: SentenceResult, result: DocumentResult) -> str:
    span = pill(flat(s.text), s.label, heatmap_tooltip(index, s, result))
    return f'<a href="#s{index}">{span}</a>' if s.label != "human" else span


def heatmap_tooltip(index: int, s: SentenceResult, result: DocumentResult) -> str:
    tooltip = f"#{sentence_number(index, s)} · {LABEL_WORD[s.label]} · score {s.score:.3f}"
    if s.label != "human":
        tooltip += f"\nWhy flagged: {why_flagged(s, result)}"
        if (text := suggestion(s)) != "—":
            tooltip += f"\nSuggestion: {text}"
    return tooltip


def pill(text: str, label: str, tooltip: str | None = None) -> str:
    background, foreground = LABEL_COLORS[label]
    title = f' title="{html.escape(tooltip)}"' if tooltip else ""
    return (
        f'<span class="pill"{title} style="background:{background};color:{foreground};padding:1px 3px;border-radius:3px">'
        f"{html.escape(text)}</span>"
    )


def sentence_number(index: int, s: SentenceResult) -> str:
    return f"{index}≈{s.blended_with[0]}–{s.blended_with[1]}" if s.blended_with else str(index)


def flat(text: str) -> str:
    return " ".join(text.split())


def markdown_cell(text: str) -> str:
    return MARKDOWN_SPECIAL.sub(r"\\\1", flat(text))
