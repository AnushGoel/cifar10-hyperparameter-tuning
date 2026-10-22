"""Headless smoke test: every dashboard page renders without an exception."""
from pathlib import Path

import pytest

testing = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "results"
APP = str(ROOT / "streamlit_app.py")
PAGES = ["Overview", "1. Data", "2. Baseline", "3. Batch size", "4. Batch normalization", "5–6. Search lab",
         "7. Final model", "Training explorer", "Insights and recommendations"]


def test_every_page_renders(monkeypatch):
    monkeypatch.setenv("RESULTS_DIR", str(FIXTURE))
    at = testing.AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    for page in PAGES:
        at.radio(key="nav").set_value(page).run()
        assert not at.exception, f"{page}: {at.exception}"


def test_empty_state_explains_what_to_do(monkeypatch, tmp_path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    at = testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("tuning lab" in t.value.lower() for t in at.title)
