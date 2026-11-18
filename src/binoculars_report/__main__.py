"""Entry point for `python -m binoculars_report`, e.g. to run without installing (`uv run python -m binoculars_report <directory>`).

The installed `binoculars-report` command does not use this file; it calls `binoculars_report.cli:main` directly (see `[project.scripts]` in pyproject.toml).
"""

import sys

from .cli import main

sys.exit(main())
