"""REPORT.md and the README results block must render completely from the run logs."""
import shutil
from pathlib import Path

import build_report

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "results"


def make_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "reports").mkdir(parents=True)
    shutil.copy(ROOT / "reports" / "REPORT.template.md", root / "reports")
    shutil.copy(ROOT / "README.md", root)
    shutil.copy(ROOT / "CITATION.cff", root)
    return root


def test_report_renders_with_no_missing_placeholders(tmp_path):
    root = make_repo(tmp_path)
    assert build_report.build(FIXTURE, root, repo="someone/cifar10-hyperparameter-tuning") == set()
    report = (root / "REPORT.md").read_text(encoding="utf-8")
    assert not build_report.TOKEN.search(report)
    for heading in ("## Abstract", "## 3. Methods", "## 4. Results", "## 5. Discussion", "## References", "## Appendix A. Every run"):
        assert heading in report
    assert "| Batch size |" in report and "| Trial |" in report


def test_readme_block_figures_and_repo_slug(tmp_path):
    root = make_repo(tmp_path)
    build_report.build(FIXTURE, root, repo="someone/cifar10-hyperparameter-tuning")
    readme = (root / "README.md").read_text(encoding="utf-8")
    block = readme.split(build_report.START)[1].split(build_report.END)[0]
    assert "Final selected model" in block and "Headline findings" in block
    assert "YOUR-USERNAME" not in readme
    assert "someone/cifar10-hyperparameter-tuning" in (root / "CITATION.cff").read_text(encoding="utf-8")
    assert (root / "docs" / "figures" / "figure_13.png").exists()


def test_rebuilding_is_idempotent(tmp_path):
    root = make_repo(tmp_path)
    build_report.build(FIXTURE, root)
    first = (root / "README.md").read_text(encoding="utf-8")
    build_report.build(FIXTURE, root)
    assert (root / "README.md").read_text(encoding="utf-8") == first


def test_md_table_escapes_pipes():
    table = build_report.md_table([{"A": "x|y", "B": 1}])
    assert "x\\|y" in table and table.count("\n") == 2


def test_missing_results_is_a_clear_error(tmp_path):
    assert build_report.main(["--results", str(tmp_path), "--root", str(make_repo(tmp_path))]) == 2
