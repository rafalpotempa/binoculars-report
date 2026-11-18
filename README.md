# binoculars-report

Batch AI-text detection reports with [Binoculars](https://arxiv.org/abs/2401.12070) (Hans et al., ICML 2024). Every text file in a directory is scored in-process (no server) and gets a colored report: an HTML page by default (`--format md` for Markdown). The report shows the document verdict, a heatmap of all sentences (hover a sentence for its label, score, reason and suggestion), a table of the flagged (AI and uncertain) sentences with their label, score, label boundaries, why they are flagged and a rewording suggestion, and the full evidence at the end. Every flagged sentence is explained by:

- **key phrases**: the words whose tokens pull the score below the AI threshold, i.e. are more predictable than their context justifies. Each token's pull is `surprise − threshold × expected`; a sentence is flagged exactly when its pulls sum below zero. Each phrase shows its share of that pull and the score the sentence would get without it,
- **the model's own wording**: how many words the base model would have written itself as its first choice (compared with the document), the longest such run, and where the text departs from the model's confident first choice, with its top alternatives,
- **an explanation**: one or two plain-language sentences and a rewording suggestion, written from the evidence above by an instruct model: by default the scoring pair's own instruct model (`falcon-7b-instruct`, or `Qwen2.5-1.5B-Instruct` in `fast` mode, so nothing extra is loaded), or any Ollama model (`--explainer ollama:<model>`). It is a plausible reading of the measurements, not a measurement,
- **the two parts of its score**: `score = surprise ÷ expected`, so you can tell whether a sentence is flagged because its wording is very predictable or because a model would expect more variety there,
- other evidence signals (low burstiness, AI idioms).

## Modes

Binoculars scores a text with two models that share a tokenizer: a base model (the observer) and its instruct fine-tune (the performer). The score is the performer's perplexity divided by the cross-perplexity between the two; low means the text is as predictable as model-written text.

| Mode | Models | Thresholds (AI < low, human > high) | Memory |
|---|---|---|---|
| `falcon` (default) | `tiiuae/falcon-7b` + `tiiuae/falcon-7b-instruct`, bfloat16, up to 2048 tokens per document | 0.8536 / 0.9015: the paper's low-false-positive and best-accuracy thresholds | ~28 GB (GPU, or RAM on CPU) |
| `fast` | `Qwen/Qwen2.5-1.5B` + `Qwen/Qwen2.5-1.5B-Instruct`, up to 4096 tokens per document | 0.905 / 1.11: clarity's provisional calibration | ~6 GB |

The paper's thresholds were selected on texts of up to 512 tokens, so per-sentence labels are an indication rather than a calibrated verdict in both modes. With two GPUs, the performer is placed on the second one, as in the [reference implementation](https://github.com/ahans30/Binoculars).

## Credits

Inspired by [clarity](https://github.com/roowus/clarity), an evidence-first AI-text detector by roowus. Its scoring, sentence-splitting, sentence-blending and evidence-signal code is ported into this project (MIT License; see [LICENSE](LICENSE)), and the `fast` mode uses its default models and calibration.

## Install

```powershell
uv tool install --torch-backend auto git+https://github.com/rafalpotempa/binoculars-report.git
# or, from a local checkout:
uv tool install --torch-backend auto .
```

This installs the `binoculars-report` command (Python 3.11–3.13).

## Dependencies

Every dependency, including transitive ones, is pinned to an exact version in `requirements/requirements.txt`, which `pyproject.toml` reads at build time, so a new (possibly compromised) release is never picked up automatically. The file is generated from the direct dependencies in `requirements/requirements.in`:

```powershell
uv pip compile requirements/requirements.in -o requirements/requirements.txt
# to bump versions, add --upgrade (or --upgrade-package <name>) and review the diff
```

Compile options live in `[tool.uv.pip]` in `pyproject.toml`: one file for all platforms (`universal`), and torch's Linux CUDA runtime packages (`nvidia-*`, `cuda-*`, `triton`) left out, because torch pins them exactly itself and they differ between the CUDA builds `--torch-backend auto` may choose. The pins carry no file hashes; package metadata cannot express them.

`--torch-backend auto` makes uv detect the installed GPU driver and pick the matching torch build (CUDA, ROCm or Intel XPU), falling back to CPU-only when there is no GPU. It is a uv preview feature and needs a recent uv (`uv self update`). Without it, torch comes from PyPI: CPU-only on Windows, CUDA-enabled on Linux, MPS on macOS. The device in use is shown at startup (`Models ready on cuda`); when an NVIDIA GPU is present but torch is a CPU-only build, the tool prints the exact reinstall command.

## Usage

```powershell
binoculars-report <directory> [--ext .tex .md .txt] [--mode falcon|fast] [--output-dir DIR] [--format html|md] [--json] [--explainer local|none|ollama:<model>] [--ollama-url URL]
```

| Option | Default | Description |
|---|---|---|
| `directory` | | Directory with files to analyze (not recursive). |
| `--ext` | `.tex .md .txt` | File extensions to analyze. |
| `--mode` | `falcon` | `falcon`: the paper's Falcon-7B pair and thresholds; `fast`: clarity's Qwen2.5-1.5B pair and calibration (less memory, weaker). See [Modes](#modes). |
| `--output-dir` | `<directory>/.binoculars_report` | Where reports are written. |
| `--format` | `html` | Report format: `html` (open in a browser; hover the heatmap for details) or `md` (Markdown). |
| `--json` | off | Also save the full result as `<filename>.json`. |
| `--explainer` | `local` | Model that writes the plain-language explanation of each flagged sentence. `local`: the scoring pair's in-process instruct model (`falcon-7b-instruct`, or `Qwen2.5-1.5B-Instruct` in `fast` mode; no extra download). `ollama:<model>`: any chat model served by a local [Ollama](https://ollama.com), e.g. `ollama:gemma4`; larger, newer models stick to the evidence much better. `none`: no explanations. |
| `--ollama-url` | `http://localhost:11434` | Ollama server URL. |

Reports are written as `<filename>.html` (or `<filename>.md` with `--format md`; open it with Markdown Preview, `Ctrl+Shift+V` in VS Code, to see the colored heatmap).

Models are loaded once and reused for all files. The first run downloads the model weights from Hugging Face.
