"""Fill the {{PLACEHOLDERS}} in the notebook's Markdown cells with the values a run produced.

The notebook writes one data-driven sentence per placeholder to results/narrative.json
while it runs. This script substitutes them into the Markdown cells, leaves every code
cell and output untouched, and writes <name>_FINAL.ipynb (or edits in place).

Usage:
    python scripts/fill_report.py notebooks/Assignment3_Hyperparameter_Tuning_Anush_Goel.ipynb
    python scripts/fill_report.py NOTEBOOK --narrative results/narrative.json --inplace
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "Assignment3_Hyperparameter_Tuning_Anush_Goel.ipynb"
PATTERN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


def fill_notebook(nb: dict, narrative: dict) -> tuple[dict, int, set[str]]:
    """Return the filled notebook, the number of substitutions and the placeholders left unfilled."""
    missing: set[str] = set()
    filled = 0

    def substitute(match: re.Match) -> str:
        nonlocal filled
        key = match.group(1)
        if key in narrative:
            filled += 1
            return str(narrative[key])
        missing.add(key)
        return match.group(0)

    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        cell["source"] = PATTERN.sub(substitute, text).splitlines(keepends=True)
    return nb, filled, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", nargs="?", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--narrative", type=Path, default=None, help="defaults to ./results, then <repo>/results")
    parser.add_argument("--inplace", action="store_true", help="overwrite the notebook instead of writing *_FINAL.ipynb")
    args = parser.parse_args(argv)

    if not args.notebook.exists():
        print(f"Notebook not found: {args.notebook}", file=sys.stderr)
        return 2
    candidates = [args.narrative] if args.narrative else [Path("results/narrative.json"), ROOT / "results" / "narrative.json"]
    narrative_path = next((p for p in candidates if p and p.exists()), None)
    if narrative_path is None:
        print("results/narrative.json not found. Run the notebook first and unzip results.zip in the repo root.", file=sys.stderr)
        return 2

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    narrative = json.loads(narrative_path.read_text(encoding="utf-8"))
    nb, filled, missing = fill_notebook(nb, narrative)

    out = args.notebook if args.inplace else args.notebook.with_name(args.notebook.stem + "_FINAL.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Filled {filled} placeholders -> {out}")
    if missing:
        print("Not filled (the cell that creates them may not have run): " + ", ".join(sorted(missing)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
