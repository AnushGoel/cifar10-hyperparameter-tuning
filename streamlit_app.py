"""CIFAR-10 tuning lab: interactive companion to Assignment 3 (Hyperparameter Tuning).

Author: Anush Goel (NUID 002570173)

Run the notebook first. It writes a results/ folder (and results.zip). Unzip it in the
repository root and start the dashboard with:

    streamlit run streamlit_app.py        (or: make app)

Set the RESULTS_DIR environment variable to read results from somewhere else.
"""
import html
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="CIFAR-10 tuning lab | Anush Goel", page_icon="🧪", layout="wide")

RESULTS = Path(os.environ.get("RESULTS_DIR", Path(__file__).parent / "results"))
N_VAL = 5_000
LR_VALUES = [0.0001, 0.0005, 0.001, 0.005]
SPACE = {"learning_rate": LR_VALUES, "batch_size": [32, 64, 128, 256], "layers": [2, 3, 4],
         "units": [128, 256, 512], "dropout": [0.0, 0.2, 0.4]}
HP_LABELS = {"learning_rate": "Learning rate", "batch_size": "Batch size", "layers": "Hidden layers",
             "units": "Units per layer", "dropout": "Dropout"}
C = {"baseline": "#7A8494", "bs32": "#9CC3E6", "bs64": "#6FA6D8", "bs128": "#4C8BC9", "bs256": "#2F6DAE",
     "bs512": "#1E4E8C", "bn": "#4F46B8", "random": "#B83280", "bayes": "#0F7C80", "final": "#C8961B",
     "train": "#9AA3B0", "ink": "#17212B"}
HIT_RGB, MISS_RGB = (15, 124, 128), (184, 50, 128)
METHOD_NAMES = {"random": "Random search", "bayes": "Bayesian (TPE)"}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
html, body, [class*="st-"], .stMarkdown, .stText { font-family: 'Archivo', system-ui, -apple-system, 'Segoe UI', sans-serif; }
h1 { font-weight: 800 !important; letter-spacing: -0.02em; line-height: 1.08 !important; }
h2, h3 { font-weight: 700 !important; letter-spacing: -0.01em; }
.lede { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.22rem; line-height: 1.55; max-width: 68ch; margin: 0.2rem 0 1.2rem 0; }
.narrative { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.04rem; line-height: 1.68; max-width: 74ch; }
.finding { font-family: 'Source Serif 4', Georgia, serif; font-size: 1.02rem; line-height: 1.6; max-width: 74ch;
           border-left: 3px solid #0F7C80; padding: 0.15rem 0 0.15rem 0.9rem; margin: 0 0 0.8rem 0; }
.chip { display: inline-block; width: 0.75rem; height: 0.75rem; border-radius: 2px; margin: 0 0.3rem 0 0.6rem; vertical-align: -1px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def rich(text):
    """Escape a narrative sentence for HTML and turn `code` spans into <code>."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(str(text)))


def prose(text, cls="narrative"):
    st.markdown(f"<div class='{cls}'>{rich(text)}</div>", unsafe_allow_html=True)


def say(n, *keys, cls="narrative"):
    parts = [n[k] for k in keys if n.get(k)]
    if parts:
        prose(" ".join(parts), cls)


def _render(fn, obj, **kw):
    """Stretch to the column width on both new (width="stretch") and older (use_container_width) Streamlit."""
    for extra in ({"width": "stretch"}, {"use_container_width": True}):
        try:
            return fn(obj, **kw, **extra)
        except Exception:
            continue
    return fn(obj, **kw)


def show(fig, key=None):
    _render(st.plotly_chart, fig, config={"displaylogo": False}, key=key)


def table(df):
    _render(st.dataframe, df, hide_index=True)


def picture(img, caption=None):
    _render(st.image, img, caption=caption)


def style(fig, height=420, title=None):
    layout = dict(height=height, margin=dict(l=10, r=10, t=58 if title else 26, b=10),
                  font=dict(family="Archivo, system-ui, sans-serif", size=13),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    if title:
        layout["title"] = dict(text=title, x=0, xanchor="left", font=dict(size=16))
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(127,127,127,0.18)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(127,127,127,0.18)", zeroline=False)
    return fig


def sampling_half_width(acc, n=N_VAL):
    return 1.96 * np.sqrt(acc * (1 - acc) / n)


def color_for(run):
    name, group = run["name"], run.get("group", "")
    if name.startswith("random"):
        return C["random"]
    if name.startswith("bayes"):
        return C["bayes"]
    if name.startswith("seed"):
        return C["final"]
    if run["config"]["batch_norm"]:
        return C["bn"]
    if group.startswith("Step 2"):
        return C["baseline"]
    return C.get(f"bs{run['config']['batch_size']}", C["baseline"])


def clock(run, axis):
    ep = np.arange(1, run["epochs_run"] + 1)
    if axis == "Weight updates":
        return ep * run["updates_per_epoch"]
    if axis == "Seconds":
        return np.cumsum(run["epoch_times"])
    return ep


def contact_sheet(images, correct, idx, cols=16, scale=3, frame=3, gap=4):
    """Grid of upscaled images, each framed teal (correct) or magenta (wrong), on a transparent background."""
    rows = int(np.ceil(len(idx) / cols))
    tile = 32 * scale
    cell = tile + 2 * frame + gap
    sheet = np.zeros((rows * cell, cols * cell, 4), dtype=np.uint8)
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        y, x = r * cell, c * cell
        sheet[y:y + tile + 2 * frame, x:x + tile + 2 * frame, :3] = HIT_RGB if correct[i] else MISS_RGB
        sheet[y:y + tile + 2 * frame, x:x + tile + 2 * frame, 3] = 255
        sheet[y + frame:y + frame + tile, x + frame:x + frame + tile, :3] = images[i].repeat(scale, 0).repeat(scale, 1)
    return sheet


@st.cache_data(show_spinner="Loading experiment results…")
def load_results(folder):
    path = Path(folder)
    needed = ["runs.json", "final.json", "narrative.json", "search_trials.csv"]
    if not all((path / f).exists() for f in needed):
        return None
    runs = json.loads((path / "runs.json").read_text(encoding="utf-8"))
    out = {"runs": {r["name"]: r for r in runs},
           "final": json.loads((path / "final.json").read_text(encoding="utf-8")),
           "narrative": json.loads((path / "narrative.json").read_text(encoding="utf-8")),
           "trials": pd.read_csv(path / "search_trials.csv"),
           "counts": pd.read_csv(path / "class_counts.csv", index_col=0) if (path / "class_counts.csv").exists() else None,
           "preds": None, "samples": None}
    if (path / "test_predictions.npz").exists():
        with np.load(path / "test_predictions.npz") as z:
            out["preds"] = {k: z[k] for k in z.files}
    if (path / "data_samples.npz").exists():
        with np.load(path / "data_samples.npz") as z:
            out["samples"] = z["grid"]
    return out


@st.cache_resource(show_spinner="Loading the final model…")
def load_final_model(cfg_json, weights_path):
    from tensorflow import keras
    cfg = json.loads(cfg_json)
    inputs = keras.Input(shape=(3072,), name="pixels")
    x = inputs
    for i, units in enumerate(cfg["hidden"], start=1):
        if cfg["batch_norm"]:
            x = keras.layers.Dense(units, use_bias=False, name=f"dense_{i}")(x)
            x = keras.layers.BatchNormalization(name=f"batchnorm_{i}")(x)
            x = keras.layers.Activation("relu", name=f"relu_{i}")(x)
        else:
            x = keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        if cfg["dropout"] > 0:
            x = keras.layers.Dropout(cfg["dropout"], name=f"dropout_{i}")(x)
    outputs = keras.layers.Dense(10, activation="softmax", name="class_probs")(x)
    model = keras.Model(inputs, outputs)
    model.load_weights(weights_path)
    return model


def key_runs(R):
    runs, f = R["runs"], R["final"]
    sel = f["selected_batch_size"]
    bs_runs = {32: runs.get("bs32"), 128: runs.get("baseline"), 512: runs.get("bs512")}
    trials = R["trials"]
    best = {m: trials[trials.method == m].sort_values("best_val_acc", ascending=False).iloc[0]["name"]
            for m in ("random", "bayes") if (trials.method == m).any()}
    return dict(base=runs["baseline"], bs=bs_runs, sel=sel, nobn=bs_runs.get(sel), bn=runs.get(f"bn_bs{sel}"),
                ext_plain=runs.get(f"nobn_lr005_bs{sel}"), ext_bn=runs.get(f"bn_lr005_bs{sel}"),
                rs_best=runs.get(best.get("random")), bo_best=runs.get(best.get("bayes")), final=runs[f["name"]])


# ---------------------------------------------------------------- pages
def page_overview(R):
    f, n, K = R["final"], R["narrative"], key_runs(R)
    st.title("Tuning a fully connected network on CIFAR-10")
    lo, hi = f["ci"]
    st.markdown(f"<div class='lede'>Validation accuracy went from {pct(K['base']['best_val_acc'])} for the fixed baseline to "
                f"{pct(f['best_val_acc'])} for the selected model, which then scored {pct(f['test_acc'])} on the untouched "
                f"test set (95% CI {pct(lo)} to {pct(hi)}).</div>", unsafe_allow_html=True)

    if R["preds"] is not None:
        p = R["preds"]
        correct = p["probs"].argmax(1) == p["y_true"]
        if "sheet_seed" not in st.session_state:
            st.session_state.sheet_seed = 7
        idx = np.random.default_rng(st.session_state.sheet_seed).choice(len(correct), 64, replace=False)
        picture(contact_sheet(p["images"], correct, idx))
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"<span class='chip' style='background:#0F7C80'></span>classified correctly by the final model"
                    f"<span class='chip' style='background:#B83280'></span>misclassified. "
                    f"{int(correct[idx].sum())} of these 64 are correct.", unsafe_allow_html=True)
        if c2.button("Draw another 64"):
            st.session_state.sheet_seed += 1
            st.rerun()

    st.subheader("The tuning journey")
    stages = [("Baseline", K["base"], C["baseline"]), (f"Best batch size ({K['sel']})", K["nobn"], C.get(f"bs{K['sel']}")),
              (f"Batch normalization (batch {K['sel']})", K["bn"], C["bn"]), ("Best random-search trial", K["rs_best"], C["random"]),
              ("Best Bayesian trial", K["bo_best"], C["bayes"]), (f"Final model ({f['name']})", K["final"], C["final"])]
    stages = [s for s in stages if s[1] is not None]
    fig = go.Figure()
    for label, run, color in stages:
        acc = run["best_val_acc"]
        fig.add_trace(go.Scatter(x=[100 * acc], y=[label], mode="markers+text", text=[pct(acc)], textposition="middle right",
                                 marker=dict(size=13, color=color), error_x=dict(type="data", array=[100 * sampling_half_width(acc)], color=color),
                                 name=label, showlegend=False,
                                 hovertemplate=f"{label}<br>best validation accuracy %{{x:.2f}}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[100 * f["test_acc"]], y=[stages[-1][0]], mode="markers", name="Final model on the test set",
                             marker=dict(symbol="diamond", size=12, color=C["ink"]),
                             error_x=dict(type="data", symmetric=False, array=[100 * (hi - f["test_acc"])],
                                          arrayminus=[100 * (f["test_acc"] - lo)], color=C["ink"])))
    fig.add_vline(x=100 * K["base"]["best_val_acc"], line_dash="dot", line_color=C["baseline"])
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Best validation accuracy (%) with 95% sampling interval", ticksuffix="%")
    show(style(fig, 380))

    st.subheader("What the experiments showed")
    for k in ["SUMMARY_GAIN_SENTENCE", "SUMMARY_LEVER_SENTENCE", "SUMMARY_PEAK_SENTENCE", "BS_SELECT_SENTENCE",
              "BN_VERDICT_SENTENCE", "CMP_VERDICT_SENTENCE", "FINAL_GAP_SENTENCE"]:
        if n.get(k):
            prose(n[k], "finding")


def page_data(R):
    st.title("Step 1. Data")
    prose("CIFAR-10 has 60,000 colour images of 32 × 32 pixels in ten balanced classes. Each image is flattened into 3,072 "
          "values scaled to [0, 1]. A stratified, seeded split gives 45,000 training and 5,000 validation images; the 10,000 "
          "test images are used once, at the very end.")
    if R["samples"] is not None:
        grid = R["samples"]
        rows, cols = grid.shape[:2]
        scale, gap = 3, 4
        cell = 32 * scale + gap
        canvas = np.full((rows * cell, cols * cell, 3), 255, dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                canvas[r * cell:r * cell + 96, c * cell:c * cell + 96] = grid[r, c].repeat(scale, 0).repeat(scale, 1)
        fig = px.imshow(canvas)
        names = R["final"]["class_names"]
        fig.update_xaxes(tickvals=[c * cell + 48 for c in range(cols)], ticktext=names, side="top", showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        fig.update_traces(hoverinfo="skip", hovertemplate=None)
        show(style(fig, 470, "Six random training images per class"))
    if R["counts"] is not None:
        share = 100 * R["counts"] / R["counts"].sum()
        long = share.reset_index(names="class").melt(id_vars="class", var_name="split", value_name="share")
        fig = px.bar(long, x="class", y="share", color="split", barmode="group",
                     color_discrete_map={"train": C["bs128"], "validation": C["final"], "test": C["baseline"]})
        fig.add_hline(y=10, line_dash="dash", line_color=C["ink"])
        fig.update_yaxes(title="Share of split (%)", range=[0, 12.5])
        fig.update_xaxes(title=None)
        show(style(fig, 360, "Each class makes up 10% of every split"))
        table(R["counts"].reset_index(names="class"))


def page_baseline(R):
    K, n = key_runs(R), R["narrative"]
    b = K["base"]
    h = b["history"]
    ep = np.arange(1, b["epochs_run"] + 1)
    st.title("Step 2. Baseline")
    st.caption("512 → 256 → 128 → 10, ReLU, softmax, Adam defaults, batch 128, 30 epochs, no dropout or batch normalization.")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"), horizontal_spacing=0.08)
    fig.add_trace(go.Scatter(x=ep, y=h["loss"], name="Training", line=dict(color=C["train"], dash="dash", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=ep, y=h["val_loss"], name="Validation", line=dict(color=C["bs128"], width=3)), 1, 1)
    fig.add_trace(go.Scatter(x=ep, y=h["accuracy"], showlegend=False, line=dict(color=C["train"], dash="dash", width=2)), 1, 2)
    fig.add_trace(go.Scatter(x=ep, y=h["val_accuracy"], name="Generalization gap", fill="tonexty",
                             fillcolor="rgba(76,139,201,0.12)", line=dict(color=C["bs128"], width=3)), 1, 2)
    fig.add_trace(go.Scatter(x=[b["best_loss_epoch"]], y=[b["best_val_loss"]], mode="markers", showlegend=False,
                             marker=dict(size=12, color=C["final"]), hovertemplate="lowest validation loss %{y:.3f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=[b["best_epoch"]], y=[b["best_val_acc"]], mode="markers", showlegend=False,
                             marker=dict(size=12, color=C["final"]), hovertemplate="best validation accuracy %{y:.2%}<extra></extra>"), 1, 2)
    fig.add_vrect(x0=b["best_loss_epoch"], x1=ep[-1], fillcolor="rgba(184,50,128,0.06)", line_width=0, row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_xaxes(title="Epoch")
    show(style(fig, 430))
    say(n, "BASE_TRAIN_SENTENCE", "BASE_ACC_SENTENCE")
    say(n, "BASE_TREND_SENTENCE", "BASE_GAP_SENTENCE")
    say(n, "BASE_STABILITY_SENTENCE")


def page_batch(R):
    K, n = key_runs(R), R["narrative"]
    runs = {bs: r for bs, r in K["bs"].items() if r is not None}
    st.title("Step 3. Batch size")
    c1, c2 = st.columns([1, 1])
    axis = c1.radio("Measure progress in", ["Epochs", "Weight updates", "Seconds"], horizontal=True)
    metric = c2.selectbox("Curve", ["val_loss", "val_accuracy", "loss", "accuracy"],
                          format_func=lambda m: {"val_loss": "Validation loss", "val_accuracy": "Validation accuracy",
                                                 "loss": "Training loss", "accuracy": "Training accuracy"}[m])
    fig = go.Figure()
    for bs, r in runs.items():
        fig.add_trace(go.Scatter(x=clock(r, axis), y=r["history"][metric], name=f"batch {bs}", mode="lines+markers",
                                 marker=dict(size=4), line=dict(color=C[f"bs{bs}"], width=3)))
    fig.update_xaxes(title=axis, type="log" if axis == "Weight updates" else "linear")
    if "accuracy" in metric:
        fig.update_yaxes(tickformat=".0%")
    show(style(fig, 430))
    st.caption("Switching the clock is the point: batch 32 makes 1,407 updates per epoch and batch 512 only 88, so epoch counts alone mislead.")

    rows = [{"Batch size": bs, "Updates per epoch": r["updates_per_epoch"], "Best val accuracy": pct(r["best_val_acc"]),
             "Best val loss": f"{r['best_val_loss']:.4f}", "Epoch of best val accuracy": r["best_epoch"],
             "Updates to best": f"{r['best_epoch'] * r['updates_per_epoch']:,}", "Training time (s)": f"{r['total_time']:.0f}",
             "Val-accuracy jitter (pp)": f"{100 * r['jitter']:.2f}"} for bs, r in runs.items()]
    table(pd.DataFrame(rows))

    fig = go.Figure()
    for bs, r in runs.items():
        fig.add_trace(go.Scatter(x=[r["total_time"]], y=[100 * r["best_val_acc"]], mode="markers+text", text=[f"batch {bs}"],
                                 textposition="top center", name=f"batch {bs}", showlegend=False,
                                 marker=dict(size=10 + 6 * np.log10(r["updates_per_epoch"]), color=C[f"bs{bs}"]),
                                 error_y=dict(type="data", array=[100 * sampling_half_width(r["best_val_acc"])], color=C[f"bs{bs}"])))
    fig.update_xaxes(title="Total training time (s)")
    fig.update_yaxes(title="Best validation accuracy (%)", ticksuffix="%")
    show(style(fig, 360, "What each batch size buys per second (bars = 95% sampling interval)"))
    say(n, "BS_RANK_SENTENCE", "BS_SPREAD_SENTENCE")
    say(n, "BS_EPOCH_SENTENCE")
    say(n, "BS_LOSS_SENTENCE", "BS_STABILITY_SENTENCE", "BS_TIME_SENTENCE")
    say(n, "BS_SELECT_SENTENCE", cls="finding")


def page_bn(R):
    K, n = key_runs(R), R["narrative"]
    st.title("Step 4. Batch normalization")
    if K["bn"] is None or K["nobn"] is None:
        st.info("The batch-normalization runs are missing from results/. Rerun Step 4 of the notebook and export again.")
        return
    pairs = [("Without batch norm", K["nobn"], C["baseline"]), ("With batch norm", K["bn"], C["bn"])]
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.08,
                        subplot_titles=("Loss (dashed = training)", "Accuracy (dashed = training)",
                                        "Generalization gap (pp)", "Epoch-to-epoch change in validation accuracy (pp)"))
    for label, r, color in pairs:
        h, ep = r["history"], np.arange(1, r["epochs_run"] + 1)
        fig.add_trace(go.Scatter(x=ep, y=h["loss"], line=dict(color=color, dash="dash", width=1.8), showlegend=False), 1, 1)
        fig.add_trace(go.Scatter(x=ep, y=h["val_loss"], name=label, line=dict(color=color, width=3)), 1, 1)
        fig.add_trace(go.Scatter(x=ep, y=h["accuracy"], line=dict(color=color, dash="dash", width=1.8), showlegend=False), 1, 2)
        fig.add_trace(go.Scatter(x=ep, y=h["val_accuracy"], line=dict(color=color, width=3), showlegend=False), 1, 2)
        gap = 100 * (np.array(h["accuracy"]) - np.array(h["val_accuracy"]))
        fig.add_trace(go.Scatter(x=ep, y=gap, line=dict(color=color, width=3), showlegend=False), 2, 1)
        fig.add_trace(go.Bar(x=ep[1:], y=100 * np.diff(h["val_accuracy"]), marker_color=color, opacity=0.75, showlegend=False), 2, 2)
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    show(style(fig, 640))
    say(n, "BN_ACC_SENTENCE", "BN_LOSS_SENTENCE")
    say(n, "BN_SPEED_SENTENCE", "BN_STABILITY_SENTENCE")
    say(n, "BN_GAP_SENTENCE", "BN_TIME_SENTENCE")
    say(n, "BN_VERDICT_SENTENCE", cls="finding")

    if K["ext_plain"] is not None and K["ext_bn"] is not None:
        st.subheader("Extension: a learning rate five times higher")
        fig = go.Figure()
        for label, r, color, dash in [("No BN, lr 0.001", K["nobn"], C["baseline"], "solid"), ("BN, lr 0.001", K["bn"], C["bn"], "solid"),
                                      ("No BN, lr 0.005", K["ext_plain"], C["baseline"], "dot"), ("BN, lr 0.005", K["ext_bn"], C["bn"], "dot")]:
            fig.add_trace(go.Scatter(x=np.arange(1, r["epochs_run"] + 1), y=r["history"]["val_accuracy"], name=label,
                                     line=dict(color=color, dash=dash, width=3)))
        fig.update_yaxes(tickformat=".0%", title="Validation accuracy")
        fig.update_xaxes(title="Epoch")
        show(style(fig, 380))
        say(n, "EXT_SENTENCE")


def page_search(R):
    n, trials, runs = R["narrative"], R["trials"].copy(), R["runs"]
    st.title("Steps 5 and 6. Search lab")
    st.caption("Eight random-search trials against eight Bayesian (TPE) trials over the same 432-point space, with identical training protocols.")
    choice = st.radio("Show", ["Both methods", "Random search", "Bayesian (TPE)"], horizontal=True)
    keep = {"Both methods": ["random", "bayes"], "Random search": ["random"], "Bayesian (TPE)": ["bayes"]}[choice]
    view = trials[trials.method.isin(keep)]

    fig = go.Figure()
    best_overall, band = trials.best_val_acc.max(), R["final"]["noise_band"]
    fig.add_hrect(y0=100 * (best_overall - band), y1=100 * best_overall, fillcolor="rgba(200,150,27,0.15)", line_width=0)
    for m in keep:
        t = trials[trials.method == m].sort_values("trial")
        fig.add_trace(go.Scatter(x=t.trial, y=100 * t.best_val_acc.cummax(), name=f"{METHOD_NAMES[m]}: best so far",
                                 line=dict(color=C[m], width=3, shape="hv")))
        fig.add_trace(go.Scatter(x=t.trial, y=100 * t.best_val_acc, mode="markers", name=f"{METHOD_NAMES[m]}: each trial",
                                 marker=dict(color=C[m], size=10, opacity=0.6, symbol=[{"failed": "x", "unstable": "diamond"}.get(s, "circle") for s in t.status]),
                                 customdata=t[["learning_rate", "batch_size", "layers", "units", "dropout", "status"]].values,
                                 hovertemplate="trial %{x}: %{y:.2f}%<br>lr %{customdata[0]}, batch %{customdata[1]}, "
                                               "%{customdata[2]}×%{customdata[3]}, dropout %{customdata[4]}<br>%{customdata[5]}<extra></extra>"))
    fig.update_xaxes(title="Trial", dtick=1)
    fig.update_yaxes(title="Best validation accuracy (%)", ticksuffix="%")
    show(style(fig, 400, "Search progress (shaded band = within 1 pp of the best result from either method)"))

    lr_idx = {v: i for i, v in enumerate(LR_VALUES)}
    dims = [dict(label="Learning rate", values=view.learning_rate.map(lr_idx), tickvals=[0, 1, 2, 3], ticktext=["1e-4", "5e-4", "1e-3", "5e-3"], range=[0, 3]),
            dict(label="Batch size", values=view.batch_size.map({v: i for i, v in enumerate(SPACE["batch_size"])}), tickvals=[0, 1, 2, 3],
                 ticktext=[str(v) for v in SPACE["batch_size"]], range=[0, 3]),
            dict(label="Hidden layers", values=view.layers, tickvals=[2, 3, 4], range=[2, 4]),
            dict(label="Units per layer", values=view.units.map({v: i for i, v in enumerate(SPACE["units"])}), tickvals=[0, 1, 2],
                 ticktext=[str(v) for v in SPACE["units"]], range=[0, 2]),
            dict(label="Dropout", values=view.dropout, tickvals=[0, 0.2, 0.4], range=[0, 0.4]),
            dict(label="Method", values=view.method.map({"random": 0, "bayes": 1}), tickvals=[0, 1], ticktext=["Random", "TPE"], range=[0, 1]),
            dict(label="Best val acc (%)", values=100 * view.best_val_acc)]
    fig = go.Figure(go.Parcoords(line=dict(color=100 * view.best_val_acc, colorscale="Teal", showscale=True,
                                           colorbar=dict(title="Val acc %")), dimensions=dims))
    show(style(fig, 420, "Every trial as one line through the search space (drag along an axis to filter)"))

    c1, c2 = st.columns(2)
    hp = c1.selectbox("Hyperparameter effect", list(SPACE), format_func=HP_LABELS.get)
    fig = px.strip(view.assign(Method=view.method.map(METHOD_NAMES), acc=100 * view.best_val_acc), x=hp, y="acc", color="Method",
                   color_discrete_map={METHOD_NAMES["random"]: C["random"], METHOD_NAMES["bayes"]: C["bayes"]},
                   hover_data=["trial", "status"], stripmode="overlay")
    means = view.groupby(hp)["best_val_acc"].mean().reset_index()
    fig.add_trace(go.Scatter(x=means[hp], y=100 * means.best_val_acc, mode="markers", name="Group mean",
                             marker=dict(symbol="line-ew-open", size=28, line=dict(width=3, color=C["ink"]))))
    fig.update_traces(marker_size=12, selector=dict(type="box"))
    fig.update_xaxes(type="category", title=HP_LABELS[hp])
    fig.update_yaxes(title="Best validation accuracy (%)", ticksuffix="%")
    with c1:
        show(style(fig, 380))
    xk = c2.selectbox("Coverage map: x axis", list(SPACE), index=0, format_func=HP_LABELS.get)
    yk = c2.selectbox("Coverage map: y axis", list(SPACE), index=1, format_func=HP_LABELS.get)
    jitter = view.method.map({"random": -0.12, "bayes": 0.12})
    fig = go.Figure()
    for m in keep:
        t = view[view.method == m]
        fig.add_trace(go.Scatter(x=t[xk].map({v: i for i, v in enumerate(SPACE[xk])}) + jitter[t.index],
                                 y=t[yk].map({v: i for i, v in enumerate(SPACE[yk])}), mode="markers", name=METHOD_NAMES[m],
                                 marker=dict(color=C[m], opacity=0.65, size=12 + 60 * (t.best_val_acc - trials.best_val_acc.min())
                                             / max(trials.best_val_acc.max() - trials.best_val_acc.min(), 1e-9)),
                                 text=[f"trial {a}: {100 * b:.2f}%" for a, b in zip(t.trial, t.best_val_acc)], hoverinfo="text"))
    fig.update_xaxes(tickvals=list(range(len(SPACE[xk]))), ticktext=[f"{v:g}" for v in SPACE[xk]], title=HP_LABELS[xk], range=[-0.6, len(SPACE[xk]) - 0.4])
    fig.update_yaxes(tickvals=list(range(len(SPACE[yk]))), ticktext=[f"{v:g}" for v in SPACE[yk]], title=HP_LABELS[yk], range=[-0.6, len(SPACE[yk]) - 0.4])
    with c2:
        show(style(fig, 380))

    shown = view.assign(**{"Best val acc": (100 * view.best_val_acc).map("{:.2f}%".format), "Method": view.method.map(METHOD_NAMES),
                           "Time (s)": view.total_time.round(0).astype(int)})
    table(shown[["Method", "trial", "phase", "learning_rate", "batch_size", "layers", "units", "dropout", "Best val acc",
                 "best_epoch", "epochs_run", "status", "observation", "Time (s)"]].rename(columns={
        "trial": "Trial", "phase": "Phase", "learning_rate": "Learning rate", "batch_size": "Batch size", "layers": "Layers",
        "units": "Units", "dropout": "Dropout", "best_epoch": "Best epoch", "epochs_run": "Epochs run", "status": "Status",
        "observation": "Key observation"}))

    options = list(view.name)
    picked = st.multiselect("Overlay learning curves", options, default=options[:3],
                            format_func=lambda s: f"{METHOD_NAMES[s.split('_')[0]]} trial {s.split('_t')[1]}")
    if picked:
        fig = go.Figure()
        for name in picked:
            r = runs[name]
            fig.add_trace(go.Scatter(x=np.arange(1, r["epochs_run"] + 1), y=r["history"]["val_accuracy"], name=name,
                                     line=dict(width=3, color=color_for(r))))
        fig.update_yaxes(tickformat=".0%", title="Validation accuracy")
        fig.update_xaxes(title="Epoch")
        show(style(fig, 360))

    say(n, "RS_BEST_SENTENCE", "RS_SPREAD_SENTENCE")
    say(n, "RS_EFFECT_SENTENCE", "RS_UNSTABLE_SENTENCE")
    say(n, "BO_BEST_SENTENCE", "BO_PHASE_SENTENCE", "BO_UNSTABLE_SENTENCE")
    say(n, "CMP_ACC_SENTENCE", "CMP_TRIALS_SENTENCE", "CMP_STABILITY_SENTENCE", "CMP_COMPUTE_SENTENCE")
    say(n, "CMP_VERDICT_SENTENCE", cls="finding")


def page_final(R):
    f, n = R["final"], R["narrative"]
    names = f["class_names"]
    st.title("Step 7. Final model")
    prose(f["description"].capitalize() + f". Weights from epoch {f['best_epoch']} of {f['epochs_run']}; {f['params']:,} parameters.")
    m = st.columns(4)
    m[0].metric("Best validation accuracy", pct(f["best_val_acc"]))
    m[1].metric("Test accuracy", pct(f["test_acc"]), f"{100 * (f['test_acc'] - f['best_val_acc']):+.2f} pp vs validation")
    m[2].metric("Test loss", f"{f['test_loss']:.4f}")
    m[3].metric("Calibration error (ECE)", f"{f['ece']:.3f}")
    say(n, "FINAL_SELECTION_SENTENCE", "FINAL_GAP_SENTENCE")

    cm = np.array(f["confusion"])
    cmn = cm / cm.sum(1, keepdims=True)
    c1, c2 = st.columns([1.15, 1])
    fig = px.imshow(cmn, x=names, y=names, color_continuous_scale="Blues", zmin=0, zmax=1, text_auto=".2f", aspect="auto")
    fig.update_xaxes(title="Predicted class", side="bottom")
    fig.update_yaxes(title="True class")
    fig.update_traces(hovertemplate="true %{y}, predicted %{x}: %{z:.1%}<extra></extra>")
    with c1:
        show(style(fig, 520, "Confusion matrix (rows sum to 1)"))
    pc = pd.DataFrame({"class": names, "acc": f["per_class"]}).sort_values("acc")
    fig = go.Figure(go.Bar(x=100 * pc.acc, y=pc["class"], orientation="h",
                           marker_color=[C["bayes"] if a >= f["test_acc"] else C["random"] for a in pc.acc],
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.add_vline(x=100 * f["test_acc"], line_dash="dash", line_color=C["ink"])
    fig.update_xaxes(title="Test accuracy (%)", ticksuffix="%")
    with c2:
        show(style(fig, 520, "Per-class accuracy (magenta = below overall)"))
    say(n, "FINAL_CLASS_SENTENCE", "FINAL_CONFUSED_SENTENCE")

    c1, c2 = st.columns(2)
    cal = pd.DataFrame(f["calibration"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=(cal.lo + cal.hi) / 2, y=cal.accuracy, width=cal.hi - cal.lo, name="Observed accuracy", marker_color=C["bs128"],
                         customdata=cal["count"], hovertemplate="confidence %{x:.2f}: accuracy %{y:.2f} (%{customdata} images)<extra></extra>"))
    fig.add_trace(go.Scatter(x=(cal.lo + cal.hi) / 2, y=cal.confidence, mode="markers", name="Mean confidence", marker=dict(color=C["random"], size=9)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color=C["ink"], dash="dash", width=1)))
    fig.update_xaxes(range=[0, 1], title="Predicted confidence")
    fig.update_yaxes(range=[0, 1], title="Accuracy")
    with c1:
        show(style(fig, 400, f"Reliability diagram (ECE = {f['ece']:.3f})"))
    fig = go.Figure(go.Histogram(x=100 * np.array(f["bootstrap"]), nbinsx=40, marker_color=C["final"]))
    for v in f["ci"]:
        fig.add_vline(x=100 * v, line_dash="dash", line_color=C["ink"])
    fig.add_vline(x=100 * f["best_val_acc"], line_color=C["bn"], line_width=3, annotation_text="best validation", annotation_position="top")
    fig.update_xaxes(title="Test accuracy (%)", ticksuffix="%")
    fig.update_yaxes(title="Bootstrap resamples")
    with c2:
        show(style(fig, 400, "2,000 bootstrap resamples of test accuracy"))
    say(n, "FINAL_CALIB_SENTENCE")

    if R["preds"] is not None:
        st.subheader("Look through the test predictions")
        p = R["preds"]
        pred, conf = p["probs"].argmax(1), p["probs"].max(1)
        c1, c2, c3 = st.columns(3)
        cls = c1.selectbox("True class", ["All classes"] + names)
        outcome = c2.selectbox("Outcome", ["Misclassified", "Correct", "All"])
        order = c3.selectbox("Order", ["Most confident first", "Least confident first", "Test-set order"])
        mask = np.ones(len(pred), bool)
        if cls != "All classes":
            mask &= p["y_true"] == names.index(cls)
        if outcome != "All":
            mask &= (pred == p["y_true"]) == (outcome == "Correct")
        idx = np.where(mask)[0]
        if order != "Test-set order":
            idx = idx[np.argsort(-conf[idx] if order.startswith("Most") else conf[idx])]
        if len(idx) == 0:
            st.info("No test images match these filters. Choose another class or set the outcome to All.")
        else:
            st.caption(f"{len(idx):,} matching images. The first 32 are shown below; pick any one to inspect it.")
            picture(contact_sheet(p["images"], pred == p["y_true"], idx[:32], cols=16, scale=2, frame=2, gap=3))
            k = st.number_input("Image to inspect (position in the filtered list)", min_value=1, max_value=int(len(idx)), value=1, step=1)
            i = int(idx[int(k) - 1])
            a, b = st.columns([1, 2])
            with a:
                picture(p["images"][i].repeat(8, 0).repeat(8, 1),
                        caption=f"Test image {i}: true {names[p['y_true'][i]]}, predicted {names[pred[i]]}")
            colors = [C["bayes"] if j == p["y_true"][i] else (C["random"] if j == pred[i] else C["train"]) for j in range(10)]
            fig = go.Figure(go.Bar(x=100 * p["probs"][i], y=names, orientation="h", marker_color=colors,
                                   hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="Predicted probability (%)", range=[0, 100])
            with b:
                show(style(fig, 330, "Class probabilities (teal = true class, magenta = wrong prediction)"))

    with st.expander("Classify your own image with the final model"):
        weights = RESULTS / "final_model.weights.h5"
        try:
            import tensorflow  # noqa: F401
            has_tf = True
        except ImportError:
            has_tf = False
        if not has_tf:
            st.caption("This needs TensorFlow (pip install tensorflow). Everything else in the dashboard works without it.")
        elif not weights.exists():
            st.caption("results/final_model.weights.h5 is missing. Rerun the export cell in the notebook.")
        else:
            upload = st.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png"])
            if upload is not None:
                from PIL import Image
                img = Image.open(upload).convert("RGB").resize((32, 32))
                x = np.asarray(img, dtype="float32").reshape(1, -1) / 255.0
                probs = load_final_model(json.dumps(f["config"]), str(weights)).predict(x, verbose=0)[0]
                a, b = st.columns([1, 2])
                with a:
                    picture(np.asarray(img).repeat(8, 0).repeat(8, 1), caption="What the model sees (32 × 32)")
                fig = go.Figure(go.Bar(x=100 * probs, y=names, orientation="h",
                                       marker_color=[C["bayes"] if j == probs.argmax() else C["train"] for j in range(10)]))
                fig.update_yaxes(autorange="reversed")
                fig.update_xaxes(title="Predicted probability (%)", range=[0, 100])
                with b:
                    show(style(fig, 330, f"Prediction: {names[int(probs.argmax())]} ({probs.max():.0%})"))


def page_explorer(R):
    runs = R["runs"]
    st.title("Training explorer")
    st.caption("Overlay any runs from the notebook and compare them on the clock and metric of your choice.")
    names = list(runs)
    K = key_runs(R)
    default = [r["name"] for r in (K["base"], K["bn"], K["rs_best"], K["bo_best"]) if r is not None]
    picked = st.multiselect("Runs", names, default=default)
    c1, c2, c3 = st.columns(3)
    metric = c1.selectbox("Metric", ["val_accuracy", "val_loss", "accuracy", "loss", "gap"],
                          format_func=lambda m: {"val_accuracy": "Validation accuracy", "val_loss": "Validation loss", "accuracy": "Training accuracy",
                                                 "loss": "Training loss", "gap": "Train minus validation accuracy"}[m])
    axis = c2.radio("Clock", ["Epochs", "Weight updates", "Seconds"], horizontal=True)
    smooth = c3.slider("Smoothing (exponential moving average)", 0.0, 0.9, 0.0, 0.1)
    fig = go.Figure()
    for name in picked:
        r = runs[name]
        h = r["history"]
        y = np.array(h["accuracy"]) - np.array(h["val_accuracy"]) if metric == "gap" else np.array(h[metric])
        if smooth > 0:
            y = pd.Series(y).ewm(alpha=1 - smooth).mean().values
        fig.add_trace(go.Scatter(x=clock(r, axis), y=y, name=name, line=dict(width=2.6, color=color_for(r))))
    fig.update_xaxes(title=axis, type="log" if axis == "Weight updates" else "linear")
    if metric in ("val_accuracy", "accuracy", "gap"):
        fig.update_yaxes(tickformat=".0%")
    show(style(fig, 480))
    if picked:
        table(pd.DataFrame([{"Run": nm, "Stage": runs[nm]["group"], "Best val accuracy": pct(runs[nm]["best_val_acc"]),
                             "Best epoch": runs[nm]["best_epoch"], "Epochs run": runs[nm]["epochs_run"],
                             "Parameters": f"{runs[nm]['params']:,}", "Status": runs[nm]["status"],
                             "Time (s)": f"{runs[nm]['total_time']:.0f}"} for nm in picked]))


def page_insights(R):
    n = R["narrative"]
    st.title("Insights and recommendations")
    st.subheader("How batch size, batch normalization and the searched hyperparameters shaped training")
    say(n, "REF_BS_SENTENCE")
    say(n, "REF_BN_SENTENCE", "REF_HP_SENTENCE")
    say(n, "REF_GEN_SENTENCE")
    st.subheader("Random search versus Bayesian optimization")
    say(n, "REF_SEARCH_SENTENCE")
    prose("Random search needs no warm-up and is hard to beat when only a few hyperparameters matter. TPE spent three of its eight "
          "trials sampling at random before modelling anything, so it had five guided trials to make up ground. Left at Optuna's "
          "default warm-up of ten trials, both searches would have been random.")
    st.subheader("The final model")
    say(n, "REF_FINAL_SENTENCE", "SEED_SENTENCE")
    st.subheader("What I would do next")
    for text in [n.get("REC_CONFIG_SENTENCE", ""),
                 "Add a learning-rate schedule such as cosine annealing, so a higher starting rate can be used without late-training noise.",
                 "Use multi-fidelity search (Hyperband or ASHA) to try many configurations for a few epochs and train only the promising ones in full.",
                 "Rerun the top configurations over several seeds before declaring a winner; single-seed gaps under about 1 pp are ties.",
                 "For real accuracy gains, switch to convolutional layers with simple augmentation (random crops and flips). "
                 "Tuning gets the most out of an architecture; it cannot make up for the wrong one."]:
        if text:
            prose(text, "finding")


def page_empty():
    st.title("CIFAR-10 tuning lab")
    st.markdown("The dashboard reads the experiment results that the notebook exports. To fill it:")
    st.markdown("1. Run every cell of `notebooks/Assignment3_Hyperparameter_Tuning_Anush_Goel.ipynb` (a GPU runtime is recommended).\n"
                "2. The export cell writes a `results/` folder and downloads `results.zip`.\n"
                "3. Unzip it in the repository root so that `results/` sits next to `streamlit_app.py`, then reload this page.")
    st.caption(f"Looking for results in: {RESULTS}")


# ---------------------------------------------------------------- layout
PAGES = {"Overview": page_overview, "1. Data": page_data, "2. Baseline": page_baseline, "3. Batch size": page_batch,
         "4. Batch normalization": page_bn, "5–6. Search lab": page_search, "7. Final model": page_final,
         "Training explorer": page_explorer, "Insights and recommendations": page_insights}

with st.sidebar:
    st.markdown("### CIFAR-10 tuning lab")
    st.markdown("Anush Goel  \nNUID 002570173  \nAssignment 3: Hyperparameter tuning")
    results = load_results(str(RESULTS))
    page = st.radio("Section", list(PAGES), key="nav", label_visibility="collapsed") if results else None
    if results:
        st.caption(f"{len(results['runs'])} runs, trained on {results['final'].get('hardware', 'unknown hardware')}")

if results is None:
    page_empty()
else:
    PAGES[page](results)
