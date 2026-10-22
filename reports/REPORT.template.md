# Hyperparameter Tuning of a Fully Connected Network on CIFAR-10

### A controlled study of batch size, batch normalization and search strategy under a fixed trial budget

**Anush Goel** (NUID 002570173)
Assignment 3: Hyperparameter Tuning

<sub>This report was generated on {{REPORT_DATE}} by `scripts/build_report.py`, using the experiment logs that `notebooks/Assignment3_Hyperparameter_Tuning_Anush_Goel.ipynb` exported. Every number below comes from those logs. The notebook holds the code, the raw outputs and a step-by-step commentary; this report presents the same study in research form.</sub>

---

## Abstract

Hyperparameter tuning is often reported as a single winning configuration with no indication of how much of the improvement is real. I tuned a fully connected network on CIFAR-10 through one-variable-at-a-time experiments followed by two equal-budget searches, making every decision on a stratified 5,000-image validation set with an explicit noise band and paired McNemar tests, and touching the test set exactly once. The fixed baseline reached {{BASE_BEST_VAL_ACC}} validation accuracy before overfitting. Under a fixed 30-epoch budget, larger batches trained eight times faster and generalized slightly better. Batch normalization on its own let the network memorize its training set and cut validation accuracy by more than 9 percentage points, yet every strong configuration in the searches used it together with dropout, which proved the most decisive hyperparameter: all five dropout-free trials scored below all eleven trials with dropout. Bayesian optimization (TPE) beat random search, {{BO_BEST_ACC}} against {{RS_BEST_ACC}}, but its winning configuration came from a random warm-up trial, not from the model. A one-standard-error rule selected `{{FINAL_NAME}}` ({{FINAL_DESCRIPTION}}), which scored {{FINAL_VAL_ACC}} on validation and {{FINAL_TEST_ACC}} on the held-out test set (95% bootstrap CI {{CI_SHORT}}) and was well calibrated (ECE {{FINAL_ECE_VALUE}}). The main lessons are methodological: compare batch sizes on every clock, judge batch normalization together with its regularization and batch size, check whether a Bayesian tuner's best result actually came from its model, and treat differences inside the sampling noise of the validation set as ties.

---

## 1. Introduction

A neural network's hyperparameters decide how quickly it learns, how stable that learning is and how well the result transfers to new data. Tuning them is expensive, and with a small budget it is easy to mistake noise for progress: a configuration can look 0.5 points better simply because of the random seed or the particular validation images. This study treats tuning as a small controlled experiment rather than a leaderboard, and addresses four research questions:

- **RQ1.** With everything else fixed, how does batch size affect convergence, stability, computational cost and validation performance?
- **RQ2.** Does batch normalization improve training stability, convergence speed or validation performance at the selected batch size?
- **RQ3.** Under an identical budget of eight trials, how do random search and Bayesian optimization compare in best result, speed of discovery, stability and cost?
- **RQ4.** Which configuration should be selected, and how well does it generalize to data never used for any decision?

The emphasis on controlled comparison follows the brief. The model family (a dense network on raw pixels) is deliberately simple, so the ceiling on accuracy is low, and the interest lies in how reliably each tuning decision can be justified.

**Contributions beyond the brief.** (1) A pre-specified 1-percentage-point noise band and paired McNemar tests for every key comparison. (2) Batch-size curves on three clocks: epochs, weight updates and seconds. (3) A Bayesian search configured so that it is actually model-guided within eight trials. (4) A one-standard-error selection rule, a single test pass with a bootstrap interval, a calibration analysis and a seed-robustness check. (5) A reproducible pipeline that regenerates this report and an interactive dashboard from the run logs.

## 2. Background

**Batch size and gradient noise.** Mini-batch gradients are noisy estimates of the full-data gradient. At a fixed learning rate, the scale of that noise grows roughly with the ratio of learning rate to batch size (Smith et al., 2018), so small batches take many noisy steps per epoch and large batches take a few smooth ones. Very large batches have been associated with sharp minima and weaker generalization (Keskar et al., 2017), but much of that gap closes when the number of updates or the learning rate is adjusted (Goyal et al., 2017; Hoffer et al., 2017), and small batches often train most reliably (Masters & Luschi, 2018).

**Batch normalization.** Batch normalization standardizes each layer's pre-activations over the mini-batch and then applies a learned scale and shift (Ioffe & Szegedy, 2015). It was proposed to reduce internal covariate shift. Later analysis attributes its benefit mainly to a smoother optimization landscape with more predictable gradients (Santurkar et al., 2018).

**Hyperparameter search.** Random search outperforms grid search when only a few hyperparameters matter, because it tries each dimension at many distinct values (Bergstra & Bengio, 2012). Bayesian optimization instead fits a surrogate of the objective and proposes points that balance exploration against exploitation (Shahriari et al., 2016; Snoek et al., 2012). The Tree-structured Parzen Estimator (TPE) models the densities of "good" and "rest" configurations rather than the objective itself (Bergstra et al., 2011; Watanabe, 2023) and is Optuna's default sampler (Akiba et al., 2019).

**Comparing models fairly.** Accuracy on a finite evaluation set is a noisy statistic. When two classifiers are scored on the same examples, McNemar's test on the discordant pairs is the recommended way to compare them (Dietterich, 1998). Seeds, data order and hyperparameter choices add further variance that single-run comparisons ignore (Bouthillier et al., 2021). Repeatedly selecting on a validation set biases its score upward, which is why the test set must be kept separate (Raschka, 2018).

## 3. Methods

### 3.1 Data

CIFAR-10 contains 50,000 training and 10,000 test images of 32 × 32 × 3 pixels in ten balanced classes (Krizhevsky, 2009). Each image was flattened to 3,072 features and divided by 255. A stratified split with a fixed seed separated 45,000 training images from 5,000 validation images (500 per class). Labels were kept as integers for sparse categorical cross-entropy.

### 3.2 Models

| Variant | Architecture | Regularization | Optimizer |
| :--- | :--- | :--- | :--- |
| Baseline (Step 2) | 3,072 → 512 → 256 → 128 → 10, ReLU, softmax | none | Adam defaults (lr 0.001) |
| Batch-norm variant (Step 4) | Same, each hidden block Dense (no bias) → BatchNorm → ReLU | none | Adam defaults |
| Search model (Steps 5–6) | 2 to 4 equal-width blocks Dense → BatchNorm → ReLU → Dropout | dropout 0 to 0.4 | Adam, searched lr |

The Dense layers before batch normalization omit their bias, because the learned shift β makes it redundant (Ioffe & Szegedy, 2015).

### 3.3 Training protocol

Every run used the same seed, set before the weights were created, with TensorFlow op determinism enabled. Each run had a cap of 30 epochs and kept the weights from its epoch of highest validation accuracy. Steps 2 to 4 always trained for all 30 epochs, so that their curves show overfitting in full. Search trials added early stopping on validation accuracy with a patience of 8 epochs, identical for every trial. Batch size 128 in Step 3 reused the baseline run, because the configuration is identical.

### 3.4 Search strategies

Both searches covered the same 432-point grid: learning rate {0.0001, 0.0005, 0.001, 0.005}, batch size {32, 64, 128, 256}, hidden layers {2, 3, 4}, units per layer {128, 256, 512} and dropout {0, 0.2, 0.4}. Random search drew eight distinct configurations with a seeded generator. Bayesian optimization used Optuna's TPE sampler with two deliberate changes from its defaults:

- **A three-trial random warm-up instead of ten.** Optuna's default warm-up (Optuna Contributors, n.d.), like KerasTuner's default of three random points per hyperparameter (Keras Team, n.d.), exceeds an eight-trial budget and would make the "Bayesian" search entirely random.
- **Ordinal encoding.** Each hyperparameter was proposed as an integer index into its value list, so TPE can exploit the ordering of the values while proposing exactly the same values as random search.

### 3.5 Decision rules and statistics

All rules were written into the code before any experiment ran.

| Decision | Rule |
| :--- | :--- |
| Is a difference real? | The standard error of one validation accuracy at *n* = 5,000 is at most 0.71 pp, so a difference between two models has a standard error of about 1.0 pp (the **noise band**). Key comparisons also use an exact paired McNemar test (Dietterich, 1998). |
| Failed trial | NaN loss, or best validation accuracy below 20% (twice chance). |
| Unstable trial | Mean epoch-to-epoch swing in validation accuracy above 2 pp after epoch 5, or any single-epoch drop of 5 pp or more. |
| Batch size for Step 4 | The highest best validation accuracy, unless another setting is inside the noise band of it, in which case the fastest wins. |
| Final model | Among non-failed runs inside the noise band of the top score, keep the stable ones and choose the fewest parameters, breaking ties by lower validation loss; if none is stable, keep the top run. This is a version of the one-standard-error rule (Hastie et al., 2009). |
| Test evaluation | One prediction pass. Loss, accuracy, a 2,000-sample bootstrap 95% CI, per-class accuracy, the confusion matrix and the expected calibration error with 15 bins (Guo et al., 2017) all come from that single pass. |

### 3.6 Implementation

The experiments ran on {{HARDWARE}} with {{SOFTWARE}}. In total, {{N_RUNS}} training runs took about {{TOTAL_MINUTES}} minutes.

## 4. Results

### 4.1 Baseline behavior

![Figure 2](docs/figures/figure_02.png)
*Figure 2. Baseline loss and accuracy. The shaded region marks epochs after the validation-loss minimum.*

{{BASE_TRAIN_SENTENCE}} {{BASE_ACC_SENTENCE}} {{BASE_TREND_SENTENCE}} {{BASE_GAP_SENTENCE}} The negative starting gap is an artefact of how Keras reports training accuracy: it is averaged over the epoch while the weights are still improving, whereas validation accuracy is measured at the end of it. {{BASE_STABILITY_SENTENCE}}

The baseline therefore fits its training data easily but generalizes poorly. A dense layer has no notion of spatial locality, so it must learn the same feature separately at every position. Convolutional layers avoid this through parameter sharing (Goodfellow et al., 2016), which caps what any amount of tuning can achieve with this architecture.

### 4.2 RQ1: Batch size

**Table 1.** Batch-size comparison (30 epochs each, Adam defaults).

{{TABLE_BATCH}}

![Figure 3](docs/figures/figure_03.png)
*Figure 3. Validation loss for each batch size against epochs, cumulative weight updates and wall-clock time.*

{{BS_RANK_SENTENCE}} {{BS_SPREAD_SENTENCE}} {{BS_EPOCH_SENTENCE}} {{BS_LOSS_SENTENCE}} {{BS_STABILITY_SENTENCE}} {{BS_TIME_SENTENCE}}

![Figure 4](docs/figures/figure_04.png)
*Figure 4. Training and validation accuracy by batch size (left), and best validation accuracy against training time (right).*

{{BS_SELECT_SENTENCE}}

The large batch's small advantage runs against the intuition that small-batch gradient noise regularizes and large batches generalize worse (Keskar et al., 2017). The fixed budget is the likely explanation: 30 epochs at batch 512 is only 2,640 updates, fewer than batch 32 makes in its first two epochs, so the large-batch network stopped close to its best point and had little chance to overfit. This is consistent with the view that much of the large-batch gap is about the number of updates rather than batch size itself (Goyal et al., 2017; Hoffer et al., 2017).

### 4.3 RQ2: Batch normalization

**Table 2.** Batch normalization at batch size {{SELECTED_BS}}, all other settings fixed.

{{TABLE_BN}}

![Figure 5](docs/figures/figure_05.png)
*Figure 5. With and without batch normalization: loss, accuracy, generalization gap and epoch-to-epoch stability.*

{{BN_ACC_SENTENCE}} {{BN_LOSS_SENTENCE}} {{BN_SPEED_SENTENCE}} {{BN_STABILITY_SENTENCE}} {{BN_GAP_SENTENCE}} {{BN_TIME_SENTENCE}} {{BN_VERDICT_SENTENCE}}

**Why batch normalization hurt here.** Two mechanisms are visible in Figure 5. First, memorization: batch normalization did speed up optimization, as the literature predicts (Ioffe & Szegedy, 2015; Santurkar et al., 2018). After six epochs the normalized network had 67% training accuracy against 46% for the plain one, and it ended near 95%, while its validation loss climbed above 4, the signature of a confidently wrong model. With no dropout, nothing restrained that. Second, stale statistics: at evaluation time batch normalization uses running averages that Keras updates with momentum 0.99, a window of roughly 100 updates. At batch 512 an epoch has only 88 updates, so after epoch 1 about 0.99⁸⁸ ≈ 41% of the averages still reflected their initial values, and thereafter they described weights from about an epoch earlier. This fits both the poor first-epoch validation accuracy and the jagged validation curve. The verdict therefore applies to batch normalization at large batch sizes without dropout; Sections 4.4 and 4.5 show the same layer inside every strong configuration once those conditions change.

**Extension.** {{EXT_SENTENCE}} Because this changes the learning rate as well, it is reported separately (Figure 6).

![Figure 6](docs/figures/figure_06.png)
*Figure 6. Validation accuracy at the default and at a five-times-higher learning rate, with and without batch normalization.*

### 4.4 RQ3a: Random search

**Table 3.** Random search, eight trials under the shared protocol.

{{TABLE_RANDOM}}

{{RS_BEST_SENTENCE}} {{RS_SPREAD_SENTENCE}} {{RS_EFFECT_SENTENCE}} {{RS_UNSTABLE_SENTENCE}} The best random trial was itself flagged as unstable, so its score comes with a caveat, and two of the four flagged trials used the largest learning rate. The smallest learning rate, which might be expected to converge too slowly in 30 epochs, produced one of the best and steadiest trials.

![Figure 7](docs/figures/figure_07.png)
*Figure 7. Each random-search trial's best validation accuracy by hyperparameter value (bars show group means).*

![Figure 8](docs/figures/figure_08.png)
*Figure 8. Learning curves of the eight random-search trials.*

### 4.5 RQ3b: Bayesian optimization and the comparison

**Table 4.** Bayesian optimization (TPE), eight trials: three random warm-up trials, then five model-guided trials.

{{TABLE_BAYES}}

{{BO_BEST_SENTENCE}} {{BO_PHASE_SENTENCE}} {{BO_UNSTABLE_SENTENCE}} {{BO_REPEAT_SENTENCE}}

**Table 5.** The two search methods under the same eight-trial budget.

{{TABLE_SEARCH_CMP}}

![Figure 9](docs/figures/figure_09.png)
*Figure 9. Best result so far per trial (left) and the regions of the search space each method sampled (centre and right).*

{{CMP_ACC_SENTENCE}} {{CMP_TRIALS_SENTENCE}} {{CMP_STABILITY_SENTENCE}} {{CMP_COMPUTE_SENTENCE}}

**Where the Bayesian win came from.** Trial 2, the winning configuration, was one of the three random warm-up trials, drawn before TPE had modelled anything, and none of the five guided trials improved on it. The guided trials did learn from the warm-up: four of the five used dropout of at least 0.2, three used dropout 0.4 and three used batch size 32, the settings behind the warm-up winner. That is why none of them was unstable. The headline comparison therefore reflects the luck of a random draw more than the value of the surrogate.

**What mattered across all sixteen trials.** Dropout dominated. Every one of the five dropout-free trials scored below every one of the eleven trials that used dropout (best without dropout 45.32%, worst with it 45.60%), and mean best validation accuracy rose from 43.0% at dropout 0 to 47.5% at 0.2 and 50.4% at 0.4. Batch size reversed its Step 3 trend: with batch normalization and dropout, mean accuracy fell steadily from 49.3% at batch 32 to 44.7% at batch 256, consistent with small batches keeping batch normalization's running statistics current.

![Figure 10](docs/figures/figure_10.png)
*Figure 10. Two rough views of hyperparameter importance: the pooled spread in group means and Optuna's fANOVA importance.*

### 4.6 RQ4: Final model

**Table 6.** The selected configuration and its held-out results.

{{TABLE_FINAL}}

{{FINAL_SELECTION_SENTENCE}} {{FINAL_GAP_SENTENCE}}

![Figure 11](docs/figures/figure_11.png)
*Figure 11. Test-set confusion matrix, per-class accuracy, reliability diagram and bootstrap distribution of test accuracy.*

{{FINAL_CLASS_SENTENCE}} {{FINAL_CONFUSED_SENTENCE}} More dog images were labelled as cats (327) than as dogs (314), and the other leading confusions pair classes with similar backgrounds and colours, which a dense network on raw pixels relies on heavily. {{FINAL_CALIB_SENTENCE}} This is unusually good calibration. Guo et al. (2017) found that miscalibration tends to grow with model capacity and shrink with stronger regularization; a mid-sized network trained with dropout of 0.4 sits at the well-calibrated end, so its probabilities can be taken roughly at face value.

![Figure 12](docs/figures/figure_12.png)
*Figure 12. The sixteen most confident mistakes on the test set.*

### 4.7 Summary and robustness

**Table 7.** Summary comparison across the tuning stages. Only the final model was evaluated on the test set.

{{TABLE_SUMMARY}}

![Figure 13](docs/figures/figure_13.png)
*Figure 13. Best validation accuracy at each stage with 95% sampling intervals; the diamond shows the final model's test accuracy.*

{{SUMMARY_GAIN_SENTENCE}} {{SUMMARY_LEVER_SENTENCE}} Measured against the best plain network (49.78% at batch 512) rather than the weakened batch-norm reference, the search added 3.36 pp. {{SUMMARY_PEAK_SENTENCE}} {{SUMMARY_RANGE_SENTENCE}}

{{SEED_SENTENCE}}

![Figure 14](docs/figures/figure_14.png)
*Figure 14. Best validation accuracy of the final configuration across four seeds.*

## 5. Discussion

**RQ1: Under a fixed budget, batch size changed both the path and the destination.** {{REF_BS_SENTENCE}} Comparing runs by epoch alone would have suggested that small batches learn faster, but per update the large batch made more progress, and per second it was eight times cheaper. Its small accuracy advantage is best explained by the budget: 2,640 updates left the network near its best point rather than past it. Once batch normalization and dropout were added in the searches the trend reversed, which shows that the right batch size depends on the rest of the configuration.

**RQ2: Batch normalization is only as good as the setting it is used in.** {{REF_BN_SENTENCE}} The normalized network fitted 95% of its training set, so the optimization benefit described by Santurkar et al. (2018) was real, but without other regularization it simply memorized faster, and at 88 updates per epoch its running statistics lagged behind the weights. In the searches, the same layer appeared in every strong configuration once it was paired with dropout and small batches. {{REF_GEN_SENTENCE}} Dropout was the decisive regularizer (Srivastava et al., 2014): all five dropout-free search trials scored below all eleven that used it.

**RQ3: At eight trials, luck and protocol matter more than the search method.** {{REF_SEARCH_SENTENCE}} The headline favours Bayesian optimization, but its best configuration came from a random warm-up draw, and none of the five model-guided trials improved on it. What the guided trials did show was the start of learning: they concentrated on dropout and small batches, and none was unstable, against four unstable random-search trials. Published gains from Bayesian optimization usually appear over dozens to hundreds of evaluations (Shahriari et al., 2016). {{REF_HP_SENTENCE}} The most consequential configuration choice was invisible in the results tables: with Optuna's default warm-up of ten trials, both searches would have been random.

**RQ4: A defensible final model.** {{REF_FINAL_SENTENCE}} It was chosen by a rule fixed before the test set was opened; {{REF_WHY_CLAUSE}}; and its test score was measured exactly once and reported with an uncertainty interval. Test accuracy landed within 0.1 pp of validation, so the selection does not appear to have overfit the validation set. The seed check adds a caution: across four seeds the configuration averaged 52.13%, so the 53.14% it was selected on was about 1 pp lucky, a small instance of the winner's curse that a single-seed study cannot see.

## 6. Limitations and threats to validity

- **One seed per configuration.** Apart from the robustness check, each configuration was trained once, so rankings within the seed-to-seed spread are ties (Bouthillier et al., 2021).
- **Validation reuse.** More than twenty runs, and each run's best epoch, were chosen on the same 5,000 images, which makes the best validation score optimistic.
- **Small budget.** Sixteen trials cover under 4% of a 432-point grid. The search comparison holds at this budget only.
- **Epoch cap and patience.** A 30-epoch cap with a patience of 8 favours configurations that learn quickly, and it may judge very small learning rates before they converge.
- **Timing.** Wall-clock figures depend on the hardware. The ratios between settings transfer to other machines better than the absolute seconds do.

## 7. Recommendations and future work

1. {{REC_CONFIG_SENTENCE}} Use batch normalization only together with dropout and small batches.
2. Add a learning-rate schedule such as cosine annealing (Loshchilov & Hutter, 2017) and search the learning rate on a continuous log scale.
3. With a larger budget, use multi-fidelity search such as Hyperband (Li et al., 2018), and rerun the top configurations over several seeds before declaring a winner.
4. Widen the dropout range beyond 0.4 and allow more epochs for small learning rates: accuracy was still rising at the top of the dropout range, and the final model peaked at epoch 27 of 30.
5. For substantive accuracy gains, change the architecture: convolutional layers and simple data augmentation address the bottleneck identified in Section 4.1 (Goodfellow et al., 2016).

## 8. Reproducibility statement

All code, settings and decision rules live in the notebook, and every number in this report is regenerated from its exported logs by `make report`. Seeds, data split, epoch caps, patience, thresholds and the search space are set in one configuration cell. Op determinism is enabled, so rerunning on the same hardware and software versions should reproduce the results; across different GPUs or library versions, small numerical differences are expected. The full log of every run is listed in Appendix A.

## 9. Conclusion

On a deliberately limited architecture, careful tuning moved validation accuracy by {{GAIN_PP}}, and the evidence for each step is stated with its uncertainty. The more durable results are about method rather than about this model. Measure batch size on every clock, configure Bayesian optimization so that it actually models something within the budget, define "better" relative to the noise of the evaluation set, and spend the test set only once.

---

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 2623–2631). Association for Computing Machinery. https://doi.org/10.1145/3292500.3330701

Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. In *Advances in Neural Information Processing Systems 24* (pp. 2546–2554). Curran Associates.

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305. https://www.jmlr.org/papers/v13/bergstra12a.html

Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Mohammadi Sepahvand, N., Raff, E., Madan, K., Voleti, V., Ebrahimi Kahou, S., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems, 3*, 747–769. https://arxiv.org/abs/2103.03098

Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation, 10*(7), 1895–1923. https://doi.org/10.1162/089976698300017197

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. https://www.deeplearningbook.org

Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., & He, K. (2017). *Accurate, large minibatch SGD: Training ImageNet in 1 hour* (arXiv:1706.02677). arXiv. https://doi.org/10.48550/arXiv.1706.02677

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 1321–1330). PMLR.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7

Hoffer, E., Hubara, I., & Soudry, D. (2017). Train longer, generalize better: Closing the generalization gap in large batch training of neural networks. In *Advances in Neural Information Processing Systems 30* (pp. 1731–1741). Curran Associates.

Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *Proceedings of the 32nd International Conference on Machine Learning* (pp. 448–456). PMLR.

Keras Team. (n.d.). *BayesianOptimization tuner* [Software documentation]. KerasTuner. Retrieved October 20, 2026, from https://keras.io/keras_tuner/api/tuners/bayesian/

Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017). On large-batch training for deep learning: Generalization gap and sharp minima. In *5th International Conference on Learning Representations*. https://arxiv.org/abs/1609.04836

Krizhevsky, A. (2009). *Learning multiple layers of features from tiny images* [Technical report]. University of Toronto. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. *Journal of Machine Learning Research, 18*(185), 1–52. https://jmlr.org/papers/v18/16-558.html

Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. In *5th International Conference on Learning Representations*. https://arxiv.org/abs/1608.03983

Masters, D., & Luschi, C. (2018). *Revisiting small batch training for deep neural networks* (arXiv:1804.07612). arXiv. https://doi.org/10.48550/arXiv.1804.07612

Optuna Contributors. (n.d.). *optuna.samplers.TPESampler* [Software documentation]. Optuna. Retrieved October 20, 2026, from https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html

Raschka, S. (2018). *Model evaluation, model selection, and algorithm selection in machine learning* (arXiv:1811.12808). arXiv. https://doi.org/10.48550/arXiv.1811.12808

Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How does batch normalization help optimization? In *Advances in Neural Information Processing Systems 31* (pp. 2483–2493). Curran Associates.

Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). Taking the human out of the loop: A review of Bayesian optimization. *Proceedings of the IEEE, 104*(1), 148–175. https://doi.org/10.1109/JPROC.2015.2494218

Smith, S. L., Kindermans, P.-J., Ying, C., & Le, Q. V. (2018). Don't decay the learning rate, increase the batch size. In *6th International Conference on Learning Representations*. https://arxiv.org/abs/1711.00489

Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. In *Advances in Neural Information Processing Systems 25* (pp. 2951–2959). Curran Associates.

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html

Watanabe, S. (2023). *Tree-structured Parzen estimator: Understanding its algorithm components and their roles for better empirical performance* (arXiv:2304.11127). arXiv. https://doi.org/10.48550/arXiv.2304.11127

---

## Appendix A. Every run

{{TABLE_ALL_RUNS}}

## Appendix B. Per-class test accuracy of the final model

{{TABLE_PER_CLASS}}
