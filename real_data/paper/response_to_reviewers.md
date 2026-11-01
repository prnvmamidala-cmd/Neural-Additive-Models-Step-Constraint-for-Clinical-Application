# Response to Reviewers (camera-ready, IEEE BigData 2026)

We thank the reviewers for their careful comments. Below we map each comment to the change we made.

| Comment | Change |
|---|---|
| R1: circular synthetic evaluation | Synthetic data removed. All results now come from three real public cohorts: NHANES 2017 to March 2020 (n=6979), PhysioNet 2012 (n=11520), eICU demo (n=1929). |
| R2: test on real clinical data | Same as above. Two ICU mortality cohorts plus NHANES. |
| R2: simple binned Shock Index baseline | Added binned logistic regression (one-hot SI bands + linear age) and binned LR + cubic age spline (5 knots). Both reached AUROC similar to the step NAM (Table I, paired differences in Sec. III-A). We now state that the step pathway is equivalent to one-hot discretization and frame our contribution as a transparent implementation of guideline discretization inside a NAM, not as a predictive gain. |
| R2: shifted thresholds / smooth risk | Added a cutoff sensitivity analysis with (0.6, 0.9) and (0.8, 1.1) on the same splits (Table III). We describe it as cutoff sensitivity only; smooth-risk scenarios remain listed as a limitation and future work. |
| R2: validation vs. test split | Clarified in Sec. II-D: for each seed 0 to 4, one stratified 80/20 split; all models are trained on the same 80% and scored once on the same 20% held-out test set. There is no validation set because nothing was selected on held-out data (fixed 150 epochs, fixed hyperparameters, no early stopping). "Validation BCE" was renamed test BCE, and the separate seed-42 run was dropped so every number uses seeds 0 to 4. |
| R2: monotonicity guarantee | Added: "The proposed architecture guarantees piecewise-constant Shock Index contributions at predefined thresholds, but does not independently guarantee monotonic risk contributions, clinical validity, or patient safety." Monotonic levels are reported as observed (15/15 runs at 0.7/1.0), and Table III shows cases where they did not hold under shifted cutoffs. |
| R2: clinical safety claims | Removed "clinical safety rules", "enforce guideline alignment", and "without losing predictive performance"; replaced with "similar observed discrimination" and paired 95% CIs. Conclusion now states the results are retrospective and not evidence of bedside readiness. |
| Clinical interpretation / outcomes | Abstract and conclusion now describe retrospective risk classification: prevalent self-reported CVD history (NHANES, not early warning) and in-hospital mortality (ICU cohorts). |
| Justification of 0.7 / 1.0 cutoffs | Added a sentence citing Allgower and Burri [1] and Koch et al. [2], noting these are clinically motivated literature cutoffs and not universal hospital cutoffs. |
| Table rendering | Table I rebuilt (models as rows, cohorts as columns) and every page checked visually. |
| Reproducibility | Single script (`code/run_camera_ready.py`) regenerates n, outcome, splits, all model metrics, paired differences, and sensitivity results; `code/make_figures.py` regenerates all figures. Absolute paths removed. The step and vanilla results reproduce our previously reported seed 0 to 4 numbers exactly. |
