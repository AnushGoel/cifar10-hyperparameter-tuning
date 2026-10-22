"""Finalize the executed notebook for submission.

1. Revises a few passages of static (hand-written) Markdown so the interpretation matches
   what the actual run found. The original prose anticipated textbook behaviour; this run
   showed batch normalization hurting at batch size 512, Bayesian optimization winning with
   a random warm-up trial, and dropout as the decisive hyperparameter.
2. Fills every {{PLACEHOLDER}} from results/narrative.json (see fill_report.py).

Only Markdown cells change; code cells and outputs are left exactly as executed.

Usage:
    python scripts/finalize_notebook.py
    python scripts/finalize_notebook.py notebooks/Assignment3_Hyperparameter_Tuning_Anush_Goel.ipynb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fill_report import DEFAULT_NOTEBOOK, ROOT, fill_notebook  # noqa: E402

REVISIONS = [
    # ---- Step 2: baseline -------------------------------------------------------------
    ("{{BASE_GAP_SENTENCE}} After the first few epochs,",
     "{{BASE_GAP_SENTENCE}} (The negative starting value is an artefact of how Keras reports training accuracy: it is averaged over "
     "the whole epoch while the weights are still improving, whereas validation accuracy is measured once, at the end of the epoch.) "
     "After the first few epochs,"),
    ("It would be wrong to say the baseline underfits the *training* data; it clearly does not.",
     "It would be wrong to say the baseline underfits the *training* data: training accuracy was still climbing at epoch 30, long "
     "after validation accuracy had peaked."),

    # ---- Step 3: batch size -----------------------------------------------------------
    ("The extra gradient noise of small batches is often described as an implicit regularizer, and very large batches have been linked "
     "to sharp minima that generalize worse (Keskar et al., 2017). Later work showed that much of that large-batch gap comes from taking "
     "fewer updates rather than from batch size itself, and that it narrows when the number of updates or the learning rate is adjusted "
     "(Goyal et al., 2017; Hoffer et al., 2017). With three settings and one seed each, my results cannot separate those explanations, "
     "so I read them as descriptive rather than causal.",
     "This runs against the common intuition that the extra gradient noise of small batches acts as an implicit regularizer and that "
     "large batches generalize worse (Keskar et al., 2017). The likely explanation is the fixed budget. Thirty epochs at batch 512 is "
     "only 2,640 weight updates in total, fewer than batch 32 makes in its first two epochs, so the large-batch network was still close "
     "to its best point when training stopped and had little opportunity to overfit; in effect, it received a form of early stopping "
     "for free. That reading fits the finding that much of the large-batch gap comes from the number of updates rather than from batch "
     "size itself (Goyal et al., 2017; Hoffer et al., 2017). With three settings and one seed each, I treat it as descriptive rather "
     "than causal, and Steps 5 and 6 show the trend reversing once batch normalization and dropout are added."),

    # ---- Step 4: batch normalization --------------------------------------------------
    ("**Validation performance.** {{BN_ACC_SENTENCE}} {{BN_LOSS_SENTENCE}}\n",
     "**Validation performance.** {{BN_ACC_SENTENCE}} {{BN_LOSS_SENTENCE}} This is the opposite of what batch normalization is usually "
     "expected to do, so it deserves a closer look rather than a quick verdict.\n"),
    ("Normalizing each layer's pre-activations keeps their scale steady while earlier layers change, so later layers are not chasing a "
     "moving target. Ioffe and Szegedy (2015) framed this as reducing internal covariate shift; Santurkar et al. (2018) later showed that "
     "the larger effect is a smoother loss landscape with more predictable gradients, which is what lets the optimizer make faster early "
     "progress.",
     "Batch normalization is normally credited with faster early progress, first through reduced internal covariate shift (Ioffe & "
     "Szegedy, 2015) and, on later analysis, through a smoother loss landscape (Santurkar et al., 2018). On the *training* set that is "
     "exactly what happened: after six epochs the normalized network had reached 67% training accuracy against 46% for the plain one. "
     "The problem lies in what the validation curve measures. During training, batch normalization uses the statistics of each "
     "mini-batch, but at evaluation time it uses running averages that Keras updates with a momentum of 0.99, so they reflect roughly "
     "the last 100 updates. At batch size 512 an epoch has only 88 updates, so after the first epoch about 0.99⁸⁸ ≈ 41% of those "
     "averages still came from their initial values, and later they always described weights from about an epoch earlier. The low "
     "validation accuracy after epoch 1 is consistent with that lag."),
    ("One subtlety matters when reading these curves. During training, batch normalization uses the statistics of the current "
     "mini-batch, but at validation time it uses running averages accumulated over training. When those averages lag behind rapidly "
     "changing weights, the validation curve can wobble even while the training loss is smooth.",
     "The same lag explains the jagged validation curve in Figure 5: the weights moved quickly while the statistics used to evaluate "
     "them trailed behind, so validation accuracy jumped around even though training loss fell smoothly."),
    ("The noise in batch statistics gives batch normalization a mild regularizing effect, but that effect is small next to dropout, "
     "and on its own it does not stop a large fully connected network from memorizing the training set.",
     "This is the larger effect. By making optimization easier, batch normalization let a 1.7-million-parameter network with no other "
     "regularization memorize its training set almost completely: training loss fell to about 0.15 while validation loss climbed above "
     "4, the signature of a model that has become confidently wrong. The mild regularizing noise of batch statistics is nowhere near "
     "enough to counter that, least of all with large batches, where those statistics are least noisy."),
    ("Because the search space in Steps 5 and 6 is defined for a batch-normalized network, this run is also the natural reference point "
     "for those searches.",
     "I read this as a verdict on batch normalization *in this setting* (large batches, no dropout), not on batch normalization in "
     "general. The searches in Steps 5 and 6 test the other half of the picture: every model there is batch-normalized, and the best "
     "ones pair it with dropout and small batches, whose many updates per epoch keep the running statistics fresh."),

    # ---- Step 5: random search --------------------------------------------------------
    ("so if it does not top the list here, the small sample is the more likely explanation than a genuinely flat response.",
     "and it does top the list here, but only by a hair over dropout, which turns out to be the more decisive hyperparameter once both "
     "searches are pooled in Step 6."),
    ("In this space, the usual causes of instability are the largest learning rate (0.005) combined with small batches, where each noisy "
     "step is also a large one. The usual causes of weak but stable runs are very small learning rates (the network simply has not "
     "converged within 30 epochs) and narrow networks with heavy dropout, which leave too little effective capacity.",
     "Two of the four flagged trials used the largest learning rate (0.005), where each noisy step is also a large one, and the best "
     "random trial was one of them, so its score comes with a caveat. The weakest runs share a different trait: none used dropout "
     "above 0.2. The smallest learning rate (0.0001), which might be expected to converge too slowly within 30 epochs, instead produced "
     "one of the best and steadiest trials, improving until almost the end of the epoch budget."),

    # ---- Step 6: Bayesian optimization ------------------------------------------------
    ("With three random warm-up trials, Bayesian optimization cannot do anything clever until trial 4, so any advantage it has can only "
     "show up in the second half of the budget.",
     "That detail matters: trial 2 was one of the three random warm-up trials, drawn before TPE had modelled anything. Bayesian "
     "optimization's winning configuration therefore came from a lucky random draw, not from its surrogate model."),
    ("A tighter spread is what one would expect from a method that exploits what it has learned: once TPE has seen which region works, "
     "it keeps proposing nearby configurations, whereas random search keeps sampling the whole space, including its bad corners. "
     "Figure 9's coverage maps make this visible.",
     "The TPE-guided trials did learn from the warm-up: four of the five used dropout of at least 0.2, three used dropout 0.4, and three "
     "used batch size 32, the settings behind the warm-up winner. That exploitation is why no Bayesian trial was unstable. The wider "
     "spread of Bayesian scores comes mostly from its three dropout-free trials (two of them in the warm-up), and Figure 9's coverage "
     "maps show the guided trials clustering around the winning region.\n\n"
     "**What mattered across all sixteen trials.** The strongest pattern in the whole search is dropout. Every one of the five "
     "dropout-free trials scored below every one of the eleven trials that used dropout (best without dropout 45.32%, worst with it "
     "45.60%), and mean best validation accuracy rose from 43.0% at dropout 0 to 47.5% at 0.2 and 50.4% at 0.4. Batch size also showed "
     "a clear trend, in the opposite direction to Step 3: with batch normalization and dropout, mean accuracy fell steadily from 49.3% "
     "at batch 32 to 44.7% at batch 256. Both patterns fit the Step 4 diagnosis: a batch-normalized network needs explicit "
     "regularization, and it is better served by small batches, whose frequent updates keep its running statistics current."),
    ("I did not expect a decisive winner. Bayesian optimization's advantage comes from learning a model of the response surface, and five "
     "guided trials is very little data to learn from;",
     "The honest reading is more modest than the headline. The decisive configuration was found by a random warm-up trial and none of "
     "the model-guided trials improved on it, so this comparison says more about the luck of individual draws than about the value of "
     "a surrogate model. What TPE did show is the beginning of its advantage: its guided trials steered away from the unstable corners "
     "of the space. That advantage comes from learning a model of the response surface, and five guided trials is very little data to "
     "learn from;"),

    # ---- Step 7: final model ----------------------------------------------------------
    ("{{FINAL_CONFUSED_SENTENCE}} Confusions of this kind are typical",
     "{{FINAL_CONFUSED_SENTENCE}} The dog row is striking: more dog images were labelled as cats (327) than as dogs (314). Confusions of "
     "this kind are typical"),
    ("Modern neural networks are often miscalibrated (Guo et al., 2017), and this matters in practice: if the model's probabilities were "
     "used to decide when to defer to a human, a prediction made with 70% confidence should be right about 70% of the time. Temperature "
     "scaling fitted on the validation set would be the natural, inexpensive next step.",
     "That is unusually good. Modern neural networks are often overconfident, and Guo et al. (2017) found that miscalibration tends to "
     "grow with model capacity and to shrink with stronger regularization; a mid-sized network trained with dropout of 0.4 sits at the "
     "well-calibrated end of that spectrum. In practice, the model's probabilities can be taken roughly at face value, and post-hoc "
     "temperature scaling would change very little."),

    # ---- Step 8: summary --------------------------------------------------------------
    ("{{SUMMARY_LEVER_SENTENCE}} Two broader findings stand out.",
     "{{SUMMARY_LEVER_SENTENCE}} Measured against the best plain network (49.78% at batch 512) rather than the weakened batch-norm "
     "reference, the search added 3.36 pp. Two broader findings stand out."),

    # ---- Recommendations --------------------------------------------------------------
    ("I would keep batch normalization, moderate dropout and best-epoch checkpointing, and add",
     "I would keep batch normalization only together with dropout and small batches, widen the dropout range beyond 0.4 (accuracy was "
     "still rising at the top of the range), allow more epochs for small learning rates (the final model peaked at epoch 27 of 30), keep "
     "best-epoch checkpointing, and add"),

    # ---- Final reflection -------------------------------------------------------------
    ("{{REF_BN_SENTENCE}} This is consistent with the account in Santurkar et al. (2018): by keeping each layer's inputs on a stable "
     "scale, batch normalization makes the loss surface easier to descend, especially early in training.",
     "{{REF_BN_SENTENCE}} That result looks like a contradiction of the literature (Ioffe & Szegedy, 2015; Santurkar et al., 2018) but "
     "is better read as an interaction. Batch normalization did make the network easier to optimize, since it fitted 95% of its "
     "training set, but at batch size 512 its running statistics trailed about an epoch behind the weights and nothing restrained "
     "memorization, so the validation curve was both lower and far noisier. In the searches, where batch normalization was combined "
     "with smaller batches and dropout, it was part of every configuration, including the best ones."),
    ("which amounts to training an implicit ensemble of thinned networks (Srivastava et al., 2014).",
     "which amounts to training an implicit ensemble of thinned networks (Srivastava et al., 2014). In this study it was also the most "
     "decisive hyperparameter: every dropout-free search trial scored below every trial that used dropout."),
    ("The result fits what each method is designed to do.",
     "On paper Bayesian optimization won, but its winning configuration came from its second trial, one of the three random warm-up "
     "draws, so the win belongs to sampling luck rather than to the surrogate model. That fits what each method is designed to do."),
    ("With five guided trials, TPE had enough data to start steering but not enough to pull decisively ahead.",
     "With five guided trials, TPE had enough data to steer away from unstable settings (none of its trials was unstable, against four "
     "for random search) but not enough to find anything better than its lucky warm-up draw."),
    ("and its test score was measured exactly once, with an honest uncertainty interval around it.",
     "and its test score was measured exactly once, with an honest uncertainty interval around it. The test score landed within 0.1 pp "
     "of the validation score, so selection does not appear to have overfit the validation set, although the seed check suggests that "
     "this configuration's typical validation accuracy is nearer 52% than the 53.14% it was selected on."),
]


def revise(nb: dict) -> tuple[int, list[str]]:
    """Apply each revision once to the Markdown cells. Returns (applied, not_found)."""
    applied, not_found = 0, []
    md_cells = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    for old, new in REVISIONS:
        for cell in md_cells:
            text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if old in text:
                cell["source"] = text.replace(old, new, 1).splitlines(keepends=True)
                applied += 1
                break
        else:
            not_found.append(old[:70] + "…")
    return applied, not_found


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    src = Path(argv[0]) if argv else DEFAULT_NOTEBOOK
    narrative_path = next(p for p in (Path("results/narrative.json"), ROOT / "results" / "narrative.json") if p.exists())
    nb = json.loads(src.read_text(encoding="utf-8"))

    applied, not_found = revise(nb)
    nb, filled, missing = fill_notebook(nb, json.loads(narrative_path.read_text(encoding="utf-8")))

    out = src.with_name(src.stem.removesuffix("_FINAL") + "_FINAL.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Revised {applied} of {len(REVISIONS)} passages, filled {filled} placeholders -> {out}")
    for snippet in not_found:
        print("  passage not found (already revised or edited by hand):", snippet)
    if missing:
        print("  unfilled placeholders:", ", ".join(sorted(missing)))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
