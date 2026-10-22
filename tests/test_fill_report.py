"""The notebook's written analysis must be fully fillable from results/narrative.json."""
import json
import shutil
from pathlib import Path

import pytest

import fill_report

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "results"
NOTEBOOK = ROOT / "notebooks" / "Assignment3_Hyperparameter_Tuning_Anush_Goel.ipynb"


def markdown_text(nb: dict) -> str:
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def code_sources(nb: dict) -> list:
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_every_placeholder_has_a_value():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    tokens = set(fill_report.PATTERN.findall(markdown_text(nb)))
    if not tokens:
        pytest.skip("notebook has already been filled")
    narrative = json.loads((FIXTURE / "narrative.json").read_text(encoding="utf-8"))
    assert not tokens - set(narrative), f"placeholders without a value: {sorted(tokens - set(narrative))}"


def test_fill_writes_final_copy_and_leaves_code_untouched(tmp_path):
    src = tmp_path / "study.ipynb"
    shutil.copy(NOTEBOOK, src)
    assert fill_report.main([str(src), "--narrative", str(FIXTURE / "narrative.json")]) == 0
    filled = json.loads((tmp_path / "study_FINAL.ipynb").read_text(encoding="utf-8"))
    original = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert not fill_report.PATTERN.search(markdown_text(filled))
    assert code_sources(filled) == code_sources(original)
    assert len(filled["cells"]) == len(original["cells"])


def test_missing_narrative_is_a_clear_error(tmp_path, capsys):
    src = tmp_path / "study.ipynb"
    shutil.copy(NOTEBOOK, src)
    assert fill_report.main([str(src), "--narrative", str(tmp_path / "nope.json")]) == 2
    assert "narrative.json not found" in capsys.readouterr().err
