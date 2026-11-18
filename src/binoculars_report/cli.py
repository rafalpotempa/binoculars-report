"""CLI: `binoculars-report <directory>` analyzes every matching file and writes a report per file."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .explainer import LOCAL_EXPLAINER, OLLAMA_DEFAULT_URL, is_valid_explainer
from .report import render_html, render_markdown, verdict_line

if TYPE_CHECKING:
    from .analysis import Analyzer

DEFAULT_EXTENSIONS = [".tex", ".md", ".txt"]
DEFAULT_OUTPUT_DIR_NAME = ".binoculars_report"
MODES = ["falcon", "fast"]  # scoring.MODES; repeated here so `--help` does not import torch
RENDERERS = {"html": render_html, "md": render_markdown}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = sorted(
        path for path in args.directory.iterdir()
        if path.is_file() and path.suffix.lower() in args.ext
    )
    if not files:
        print(f"No files with extensions {sorted(args.ext)} in {args.directory}")
        return 1
    output_dir: Path = args.output_dir or args.directory / DEFAULT_OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Deferred: importing torch/transformers is slow, and `--help` should not pay for it.
        import torch

        from .analysis import Analyzer
        from .installation.gpu_hint import gpu_hint

        print(f"Loading {args.mode} models (first run downloads weights) ...", flush=True)
        try:
            analyzer = Analyzer(args.mode, explainer=args.explainer, ollama_url=args.ollama_url)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            if isinstance(exc, torch.OutOfMemoryError) and args.mode != "fast":
                print("The models do not fit in GPU memory; --mode fast needs about 6 GB.", file=sys.stderr)
            return 1
        print(f"Models ready on {analyzer.device}.", flush=True)
        if hint := gpu_hint(analyzer.device):
            print(hint, file=sys.stderr, flush=True)
        for path in files:
            analyze_and_save(analyzer, path, output_dir, report_format=args.format, save_json=args.json)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="binoculars-report",
        description="Analyze each text file in a directory with Binoculars and write a colored HTML or Markdown report per file.",
    )
    parser.add_argument("directory", type=Path, help="directory with files to analyze (not recursive)")
    parser.add_argument("--ext", nargs="+", default=DEFAULT_EXTENSIONS, help="file extensions to analyze")
    parser.add_argument(
        "--mode", choices=MODES, default=MODES[0],
        help="falcon: Falcon-7B pair, the paper's models and thresholds; fast: Qwen2.5-1.5B pair, clarity's (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"where to write reports (default: <directory>/{DEFAULT_OUTPUT_DIR_NAME})",
    )
    parser.add_argument("--format", choices=list(RENDERERS), default="html", help="report format (default: %(default)s)")
    parser.add_argument("--json", action="store_true", help="also save the full result as <filename>.json")
    parser.add_argument(
        "--explainer", default=LOCAL_EXPLAINER, metavar="{local,none,ollama:<model>}",
        help="model that writes a plain-language explanation of each flagged sentence: local (the scoring pair's instruct model, "
        "in-process), none, or ollama:<model> for any model served by a local Ollama, e.g. ollama:gemma4 (default: %(default)s)",
    )
    parser.add_argument("--ollama-url", default=OLLAMA_DEFAULT_URL, help="Ollama server URL (default: %(default)s)")
    args = parser.parse_args(argv)
    if not is_valid_explainer(args.explainer):
        parser.error(f"invalid --explainer {args.explainer!r}; use local, none or ollama:<model>")
    args.ext = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in args.ext}
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")
    return args


def analyze_and_save(analyzer: Analyzer, path: Path, output_dir: Path, report_format: str, save_json: bool) -> None:
    print(f"Analyzing {path.name} ...", flush=True)
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("file is empty")
        result = analyzer.analyze(path.name, text)
    except Exception as exc:  # noqa: BLE001 — keep processing remaining files
        print(f"  failed: {exc}", file=sys.stderr)
        return
    saved = []
    if save_json:
        json_path = output_dir / f"{path.stem}.json"
        json_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
        saved.append(json_path.name)
    report_path = output_dir / f"{path.stem}.{report_format}"
    report_path.write_text(RENDERERS[report_format](result), encoding="utf-8")
    saved.append(report_path.name)
    print(f"  {verdict_line(result, ansi=True)}")
    print(f"  saved {', '.join(saved)} to {output_dir}")
