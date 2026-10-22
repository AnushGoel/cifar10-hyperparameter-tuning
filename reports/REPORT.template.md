# Hyperparameter Tuning of a Fully Connected Network on CIFAR-10

### A controlled study of batch size, batch normalization and search strategy under a fixed trial budget

**Anush Goel** (NUID 002570173)
Assignment 3: Hyperparameter Tuning

<sub>This report was generated on {{REPORT_DATE}} by `scripts/build_report.py`, using the experiment logs that `notebooks/Assignment3_Hyperparameter_Tuning_Anush_Goel.ipynb` exported. Every number below comes from those logs. The notebook holds the code, the raw outputs and a step-by-step commentary; this report presents the same study in research form.</sub>

---

## Abstract

Hyperparameter tuning is often reported as a single winning configuration with no indication of how much of the improvement is real. I tuned a fully connected network on CIFAR-10 through a sequence of one-variable-at-a-time experiments followed by two equal-budget searches. Every decision was made on a stratified 5,000-image validation set, using an explicit noise band and paired McNemar tests, and the test set was touched exactly once. The fixed baseline reached {{BASE_BEST_VAL_ACC}} validation accuracy and began to overfit after its best epoch. {{BN_VERDICT_SENTENCE}} Eight trials of random search reached {{RS_BEST_ACC}} and eight trials of Bayesian optimization with a Tree-structured Parzen Estimator reached {{BO_BEST_ACC}}. {{CMP_VERDICT_SENTENCE}} A one-standard-error selection rule chose `{{FINAL_NAME}}` ({{FINAL_DESCRIPTION}}). It scored {{FINAL_VAL_ACC}} on validation and {{FINAL_TEST_ACC}} on the held-out test set (95% bootstrap CI {{CI_SHORT}}). The study's main lessons are methodological: compare batch sizes by updates and wall-clock time as well as by epochs, check that a "Bayesian" tuner is not secretly random under a small budget, and treat differences inside the sampling noise of the validation set as ties.

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

{{BASE_TRAIN_SENTENCE}} {{BASE_ACC_SENTENCE}} {{BASE_TREND_SENTENCE}} {{BASE_GAP_SENTENCE}} {{BASE_STABILITY_SENTENCE}}

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

### 4.3 RQ2: Batch normalization

**Table 2.** Batch normalization at batch size {{SELECTED_BS}}, all other settings fixed.

{{TABLE_BN}}

![Figure 5](docs/figures/figure_05.png)
*Figure 5. With and without batch normalization: loss, accuracy, generalization gap and epoch-to-epoch stability.*

{{BN_ACC_SENTENCE}} {{BN_LOSS_SENTENCE}} {{BN_SPEED_SENTENCE}} {{BN_STABILITY_SENTENCE}} {{BN_GAP_SENTENCE}} {{BN_TIME_SENTENCE}} {{BN_VERDICT_SENTENCE}}

**Extension.** {{EXT_SENTENCE}} Because this changes the learning rate as well, it is reported separately (Figure 6).

![Figure 6](docs/figures/figure_06.png)
*Figure 6. Validation accuracy at the default and at a five-times-higher learning rate, with and without batch normalization.*

### 4.4 RQ3a: Random search

**Table 3.** Random search, eight trials under the shared protocol.

{{TABLE_RANDOM}}

{{RS_BEST_SENTENCE}} {{RS_SPREAD_SENTENCE}} {{RS_EFFECT_SENTENCE}} {{RS_UNSTABLE_SENTENCE}}

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

![Figure 10](docs/figures/figure_10.png)
*Figure 10. Two rough views of hyperparameter importance: the pooled spread in group means and Optuna's fANOVA importance.*

### 4.6 RQ4: Final model

**Table 6.** The selected configuration and its held-out results.

{{TABLE_FINAL}}

{{FINAL_SELECTION_SENTENCE}} {{FINAL_GAP_SENTENCE}}

![Figure 11](docs/figures/figure_11.png)
*Figure 11. Test-set confusion matrix, per-class accuracy, reliability diagram and bootstrap distribution of test accuracy.*

{{FINAL_CLASS_SENTENCE}} {{FINAL_CONFUSED_SENTENCE}} {{FINAL_CALIB_SENTENCE}}

![Figure 12](docs/figures/figure_12.png)
*Figure 12. The sixteen most confident mistakes on the test set.*

### 4.7 Summary and robustness

**Table 7.** Summary comparison across the tuning stages. Only the final model was evaluated on the test set.

{{TABLE_SUMMARY}}

![Figure 13](docs/figures/figure_13.png)
*Figure 13. Best validation accuracy at each stage with 95% sampling intervals; the diamond shows the final model's test accuracy.*

{{SUMMARY_GAIN_SENTENCE}} {{SUMMARY_LEVER_SENTENCE}} {{SUMMARY_PEAK_SENTENCE}} {{SUMMARY_RANGE_SENTENCE}}

{{SEED_SENTENCE}}

![Figure 14](docs/figures/figure_14.png)
*Figure 14. Best validation accuracy of the final configuration across four seeds.*

## 5. Discussion

**RQ1: Batch size changes the path more than the destination.** {{REF_BS_SENTENCE}} Comparing runs by epoch alone would have suggested that small batches learn faster, but per update the large batch made more progress, and per second it was much cheaper. For a fixed learning rate, the practical question is which setting delivers a given validation accuracy for the least compute, and only the wall-clock view answers it.

**RQ2: Batch normalization mainly helps optimization.** {{REF_BN_SENTENCE}} This is consistent with the smoother-landscape account (Santurkar et al., 2018). Batch normalization did not, on its own, prevent the network from memorizing the training set; that requires an explicit regularizer such as dropout (Srivastava et al., 2014). {{REF_GEN_SENTENCE}}

**RQ3: At eight trials, the protocol matters more than the search method.** {{REF_SEARCH_SENTENCE}} With three warm-up trials, TPE had five model-guided proposals in which to make up ground, which is very little data for a surrogate model; published gains from Bayesian optimization usually appear over dozens to hundreds of evaluations (Shahriari et al., 2016). {{REF_HP_SENTENCE}} The most consequential configuration choice in the whole study was invisible in the results tables: with Optuna's default warm-up, the comparison would have been random search against random search.

**RQ4: A defensible final model.** {{REF_FINAL_SENTENCE}} It was chosen by a rule fixed before the test set was opened; {{REF_WHY_CLAUSE}}; and its test score was measured exactly once and reported with an uncertainty interval. The gap between validation and test accuracy is the expected cost of selecting on the validation set, and it is the main reason the test set was kept separate.

## 6. Limitations and threats to validity

- **One seed per configuration.** Apart from the robustness check, each configuration was trained once, so rankings within the seed-to-seed spread are ties (Bouthillier et al., 2021).
- **Validation reuse.** More than twenty runs, and each run's best epoch, were chosen on the same 5,000 images, which makes the best validation score optimistic.
- **Small budget.** Sixteen trials cover under 4% of a 432-point grid. The search comparison holds at this budget only.
- **Epoch cap and patience.** A 30-epoch cap with a patience of 8 favours configurations that learn quickly, and it may judge very small learning rates before they converge.
- **Timing.** Wall-clock figures depend on the hardware. The ratios between settings transfer to other machines better than the absolute seconds do.

## 7. Recommendations and future work

1. {{REC_CONFIG_SENTENCE}}
2. Add a learning-rate schedule such as cosine annealing (Loshchilov & Hutter, 2017) and search the learning rate on a continuous log scale.
3. With a larger budget, use multi-fidelity search such as Hyperband (Li et al., 2018), and rerun the top configurations over several seeds before declaring a winner.
4. Apply post-hoc temperature scaling, fitted on the validation set, if the model's probabilities are to be used for decisions (Guo et al., 2017).
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
