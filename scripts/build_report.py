"""Build the GitHub research report from the notebook's exported results.

Reads results/ (written by the notebook) and:
  1. copies every figure to docs/figures/,
  2. renders reports/REPORT.template.md into REPORT.md, with tables generated from the run logs,
  3. rewrites the block between the RESULTS markers in README.md with the headline numbers,
  4. optionally replaces the YOUR-USERNAME placeholder in README.md and CITATION.cff (--repo).

Usage:
    python scripts/build_report.py
    python scripts/build_report.py --repo your-name/cifar10-hyperparameter-tuning
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
REPO_PLACEHOLDER = "YOUR-USERNAME/cifar10-hyperparameter-tuning"
METHODS = {"random": "Random search", "bayes": "Bayesian optimization (TPE)"}


# ------------------------------------------------------------------ formatting
def pct(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}%"


def md_table(rows: list[dict], align: str | None = None) -> str:
    """Render a list of dicts as a GitHub Markdown table (no tabulate dependency)."""
    if not rows:
        return "_No data available._"
    cols = list(rows[0])
    align = align or ("l" + "r" * (len(cols) - 1))
    rule = {"l": ":---", "r": "---:", "c": ":---:"}

    def cell(v) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(rule[a] for a in align[: len(cols)].ljust(len(cols), "r")) + " |"]
    lines += ["| " + " | ".join(cell(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def hp_text(row) -> str:
    return f"lr {row['learning_rate']:g}, batch {int(row['batch_size'])}, {int(row['layers'])}×{int(row['units'])}, dropout {row['dropout']:g}"


# ------------------------------------------------------------------ loading
def load(results: Path) -> dict:
    needed = ["runs.json", "final.json", "narrative.json", "search_trials.csv"]
    missing = [f for f in needed if not (results / f).exists()]
    if missing:
        raise FileNotFoundError(f"{results} is missing {', '.join(missing)}. Run the notebook and unzip results.zip first.")
    runs = {r["name"]: r for r in json.loads((results / "runs.json").read_text(encoding="utf-8"))}
    return {"runs": runs,
            "final": json.loads((results / "final.json").read_text(encoding="utf-8")),
            "narrative": json.loads((results / "narrative.json").read_text(encoding="utf-8")),
            "trials": pd.read_csv(results / "search_trials.csv")}


def key_runs(data: dict) -> dict:
    runs, final, trials = data["runs"], data["final"], data["trials"]
    sel = final["selected_batch_size"]
    best = {m: trials[trials.method == m].sort_values("best_val_acc", ascending=False).iloc[0]["name"]
            for m in METHODS if (trials.method == m).any()}
    return {"base": runs["baseline"], "bs": {32: runs.get("bs32"), 128: runs.get("baseline"), 512: runs.get("bs512")},
            "sel": sel, "nobn": {32: runs.get("bs32"), 128: runs.get("baseline"), 512: runs.get("bs512")}.get(sel),
            "bn": runs.get(f"bn_bs{sel}"), "ext_plain": runs.get(f"nobn_lr005_bs{sel}"), "ext_bn": runs.get(f"bn_lr005_bs{sel}"),
            "rs_best": runs.get(best.get("random")), "bo_best": runs.get(best.get("bayes")), "final": runs[final["name"]]}


# ------------------------------------------------------------------ tables
def table_batch(k: dict) -> str:
    rows = []
    for bs, r in k["bs"].items():
        if r is None:
            continue
        rows.append({"Batch size": bs, "Updates / epoch": f"{r['updates_per_epoch']:,}", "Best val. accuracy": pct(r["best_val_acc"]),
                     "Best val. loss": f"{r['best_val_loss']:.4f}", "Epoch of best accuracy": r["best_epoch"],
                     "Updates to best": f"{r['best_epoch'] * r['updates_per_epoch']:,}", "Training time (s)": f"{r['total_time']:.0f}",
                     "Val. jitter (pp)": f"{100 * r['jitter']:.2f}", "Selected": "✓" if bs == k["sel"] else ""})
    return md_table(rows)


def table_bn(k: dict) -> str:
    rows = []
    for label, r in (("Without batch norm", k["nobn"]), ("With batch norm", k["bn"])):
        if r is None:
            continue
        rows.append({"Model": label, "Best val. accuracy": pct(r["best_val_acc"]), "Best val. loss": f"{r['best_val_loss']:.4f}",
                     "Epoch of best accuracy": r["best_epoch"], "Val. accuracy after epoch 1": pct(r["history"]["val_accuracy"][0]),
                     "Val. jitter (pp)": f"{100 * r['jitter']:.2f}", "Train–val gap at epoch 30 (pp)": f"{100 * r['gap_at_end']:.1f}",
                     "Median epoch time (s)": f"{np.median(r['epoch_times']):.2f}", "Parameters": f"{r['params']:,}"})
    return md_table(rows)


def table_trials(trials: pd.DataFrame, method: str) -> str:
    t = trials[trials.method == method].sort_values("trial")
    rows = []
    for _, r in t.iterrows():
        row = {"Trial": int(r.trial), "Learning rate": f"{r.learning_rate:g}", "Batch size": int(r.batch_size), "Layers": int(r.layers),
               "Hidden units": int(r.units), "Dropout": f"{r.dropout:g}", "Best val. accuracy": pct(r.best_val_acc),
               "Best epoch": int(r.best_epoch), "Status": r.status, "Key observation": r.observation}
        if method == "bayes":
            row = {"Phase": r.phase, **row}
        rows.append(row)
    return md_table(rows, align="l" * (2 if method == "bayes" else 1) + "r" * 8 + "ll")


def table_search_cmp(trials: pd.DataFrame, band: float) -> str:
    strong = trials.best_val_acc.max() - band
    rows = []
    for m, label in METHODS.items():
        t = trials[trials.method == m].sort_values("trial")
        if t.empty:
            continue
        hit = t[t.best_val_acc >= strong]
        rows.append({"Method": label, "Best val. accuracy": pct(t.best_val_acc.max()),
                     "Best trial": int(t.loc[t.best_val_acc.idxmax(), "trial"]),
                     "First trial within 1 pp of overall best": int(hit.trial.iloc[0]) if len(hit) else "never",
                     "Median": pct(t.best_val_acc.median()), "Std. dev. (pp)": f"{100 * t.best_val_acc.std(ddof=1):.2f}",
                     "Failed / unstable": f"{(t.status == 'failed').sum()} / {(t.status == 'unstable').sum()}",
                     "Epochs trained": int(t.epochs_run.sum()), "Time (min)": f"{t.total_time.sum() / 60:.1f}"})
    return md_table(rows)


def table_summary(k: dict, final: dict, trials: pd.DataFrame) -> str:
    def obs(run):
        hit = trials[trials.name == run["name"]]
        return hit.observation.iloc[0] if len(hit) else ""
    rows = [("Baseline (batch 128, no BN)", k["base"], f"Peaks at epoch {k['base']['best_epoch']}, then overfits"),
            (f"Best batch size ({k['sel']}, no BN)", k["nobn"], "Chosen by the noise-band rule"),
            (f"Batch normalization (batch {k['sel']})", k["bn"], "Dense → BatchNorm → ReLU, all else fixed"),
            ("Best random-search trial", k["rs_best"], obs(k["rs_best"]) if k["rs_best"] else ""),
            ("Best Bayesian (TPE) trial", k["bo_best"], obs(k["bo_best"]) if k["bo_best"] else ""),
            (f"**Final selected model** (`{final['name']}`)", k["final"], "One-standard-error rule on validation")]
    out = []
    for label, run, note in rows:
        if run is None:
            continue
        is_final = label.startswith("**Final")
        out.append({"Configuration": label, "Best val. accuracy": pct(run["best_val_acc"]), "Best epoch": run["best_epoch"],
                    "Parameters": f"{run['params']:,}", "Key observation": note,
                    "Test accuracy": f"**{pct(final['test_acc'])}**" if is_final else "—"})
    return md_table(out, align="lrrrlr")


def table_final(final: dict) -> str:
    cfg = final["config"]
    lo, hi = final["ci"]
    rows = [("Hidden layers", len(cfg["hidden"])), ("Units per layer", ", ".join(map(str, cfg["hidden"]))),
            ("Batch normalization", "yes" if cfg["batch_norm"] else "no"), ("Dropout", f"{cfg['dropout']:g}"),
            ("Learning rate (Adam)", f"{(cfg['learning_rate'] or 0.001):g}"), ("Batch size", cfg["batch_size"]),
            ("Epochs", f"weights from epoch {final['best_epoch']} of {final['epochs_run']}"), ("Trainable parameters", f"{final['params']:,}"),
            ("Best validation accuracy", pct(final["best_val_acc"])), ("Test accuracy", f"{pct(final['test_acc'])} (95% CI {pct(lo)} to {pct(hi)})"),
            ("Test loss", f"{final['test_loss']:.4f}"), ("Expected calibration error", f"{final['ece']:.3f}")]
    return md_table([{"Setting": a, "Value": b} for a, b in rows], align="ll")


def table_all_runs(runs: dict) -> str:
    rows = [{"Run": f"`{r['name']}`", "Stage": r["group"], "Best val. accuracy": pct(r["best_val_acc"]), "Best epoch": r["best_epoch"],
             "Epochs run": r["epochs_run"], "Parameters": f"{r['params']:,}", "Status": r["status"], "Time (s)": f"{r['total_time']:.0f}"}
            for r in sorted(runs.values(), key=lambda r: -r["best_val_acc"])]
    return md_table(rows, align="llrrrrlr")


def table_per_class(final: dict) -> str:
    order = np.argsort(final["per_class"])
    return md_table([{"Class": final["class_names"][i], "Test accuracy": pct(final["per_class"][i])} for i in order], align="lr")


# ------------------------------------------------------------------ assembly
def tokens(data: dict) -> dict:
    k, final, trials, runs = key_runs(data), data["final"], data["trials"], data["runs"]
    lo, hi = final["ci"]
    versions = final.get("versions", {})
    software = ", ".join(f"{name} {v}" for name, v in versions.items()) or "see notebook header"
    built = {
        "TABLE_BATCH": table_batch(k), "TABLE_BN": table_bn(k), "TABLE_RANDOM": table_trials(trials, "random"),
        "TABLE_BAYES": table_trials(trials, "bayes"), "TABLE_SEARCH_CMP": table_search_cmp(trials, final["noise_band"]),
        "TABLE_SUMMARY": table_summary(k, final, trials), "TABLE_FINAL": table_final(final), "TABLE_ALL_RUNS": table_all_runs(runs),
        "TABLE_PER_CLASS": table_per_class(final),
        "REPORT_DATE": final.get("date", "n.d."), "HARDWARE": final.get("hardware", "unknown hardware"), "SOFTWARE": software,
        "N_RUNS": len(runs), "TOTAL_MINUTES": f"{final.get('total_train_minutes', sum(r['total_time'] for r in runs.values()) / 60):.0f}",
        "FINAL_NAME": final["name"], "FINAL_DESCRIPTION": final["description"], "FINAL_PARAMS": f"{final['params']:,}",
        "FINAL_ECE_VALUE": f"{final['ece']:.3f}", "CI_SHORT": f"{pct(lo)} to {pct(hi)}", "SELECTED_BS": k["sel"],
        "RS_BEST_ACC": pct(k["rs_best"]["best_val_acc"]) if k["rs_best"] else "n/a",
        "BO_BEST_ACC": pct(k["bo_best"]["best_val_acc"]) if k["bo_best"] else "n/a",
        "GAIN_PP": f"{100 * (final['best_val_acc'] - k['base']['best_val_acc']):+.2f} pp",
    }
    return {**built, **{key: str(v) for key, v in data["narrative"].items()}}


def render(template: str, values: dict) -> tuple[str, set[str]]:
    missing: set[str] = set()

    def sub(m: re.Match) -> str:
        if m.group(1) in values:
            return str(values[m.group(1)])
        missing.add(m.group(1))
        return m.group(0)

    return TOKEN.sub(sub, template), missing


def readme_block(values: dict, data: dict) -> str:
    k, final = key_runs(data), data["final"]
    lines = [
        START,
        '<p align="center"><img src="docs/figures/figure_13.png" width="880" alt="Best validation accuracy at each tuning stage"></p>',
        "",
        "| Model | Validation accuracy | Test accuracy |",
        "| :--- | ---: | ---: |",
        f"| Fixed baseline (512-256-128, Adam, batch 128) | {pct(k['base']['best_val_acc'])} | — |",
        f"| Best random-search trial | {values['RS_BEST_ACC']} | — |",
        f"| Best Bayesian (TPE) trial | {values['BO_BEST_ACC']} | — |",
        f"| **Final selected model** | **{pct(final['best_val_acc'])}** | **{pct(final['test_acc'])}** (95% CI {values['CI_SHORT']}) |",
        "",
        f"**Final configuration:** {final['description']}; weights from epoch {final['best_epoch']}, "
        f"{final['params']:,} parameters, test loss {final['test_loss']:.4f}, calibration error {final['ece']:.3f}.",
        "",
        "**Headline findings**",
        "",
    ]
    for key in ("SUMMARY_GAIN_SENTENCE", "BS_RANK_SENTENCE", "BN_VERDICT_SENTENCE", "CMP_VERDICT_SENTENCE", "SUMMARY_PEAK_SENTENCE"):
        if values.get(key):
            lines.append(f"- {values[key]}")
    lines += ["", f"<sub>Generated by <code>scripts/build_report.py</code> on {values['REPORT_DATE']} from {values['N_RUNS']} runs "
              f"({values['TOTAL_MINUTES']} minutes of training on {values['HARDWARE']}).</sub>", END]
    return "\n".join(lines)


def build(results: Path, root: Path, repo: str | None = None) -> set[str]:
    data = load(results)
    values = tokens(data)

    fig_src, fig_dst = results / "figures", root / "docs" / "figures"
    fig_dst.mkdir(parents=True, exist_ok=True)
    copied = [shutil.copy2(p, fig_dst / p.name) for p in sorted(fig_src.glob("*.png"))] if fig_src.exists() else []

    template = (root / "reports" / "REPORT.template.md").read_text(encoding="utf-8")
    report, missing = render(template, values)
    (root / "REPORT.md").write_text(report, encoding="utf-8")

    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    if START in readme and END in readme:
        pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), flags=re.S)
        readme = pattern.sub(lambda _: readme_block(values, data), readme)
    if repo:
        readme = readme.replace(REPO_PLACEHOLDER, repo)
        cff = root / "CITATION.cff"
        if cff.exists():
            cff.write_text(cff.read_text(encoding="utf-8").replace(REPO_PLACEHOLDER, repo), encoding="utf-8")
    readme_path.write_text(readme, encoding="utf-8")

    print(f"REPORT.md written ({len(report.splitlines())} lines); README results block updated; {len(copied)} figures -> {fig_dst}")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root to write into")
    parser.add_argument("--repo", default=None, help="GitHub owner/name, replaces the YOUR-USERNAME placeholder")
    args = parser.parse_args(argv)
    try:
        missing = build(args.results, args.root, args.repo)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 2
    if missing:
        print("Unfilled report placeholders: " + ", ".join(sorted(missing)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
