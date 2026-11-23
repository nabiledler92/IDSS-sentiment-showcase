# Extended 20-result comparison

Reproduces and extends the model comparison in `Kopi_Kenangan_Sentiment_Model.ipynb`
to cover all 20 model-condition results:

- (a) 6 classifiers, baseline (no SMOTE, default hyperparameters)
- (b) 6 classifiers, SMOTE only (no tuning)
- (c) 6 classifiers, SMOTE + GridSearchCV
- (d) Random Forest + XGBoost, soft-voting ensemble ("bagging-style")
- (e) Random Forest + XGBoost, stacking ensemble ("boosting-style")

## Why split into 5 scripts

Each stage saves its state to `checkpoints/*.pkl` so it can be run independently
and resumed. `stage3_tuned.py` (GridSearchCV) is the slowest step, so it also
checkpoints per-classifier — if it's interrupted partway through, rerunning it
skips classifiers already completed.

## How to run

From this folder, in order:

```
python stage1_prep.py                  # data prep, TF-IDF, split, SMOTE -> checkpoints/prep.pkl
python stage2_baseline_and_smoteonly.py # conditions (a) and (b)         -> checkpoints/stage2_results.pkl
python stage3_tuned.py                  # condition (c), all 6 models    -> checkpoints/stage3_tuned_partial.pkl
python stage4_ensembles.py              # conditions (d) and (e)         -> checkpoints/stage4_ensembles.pkl
python stage5_finalize.py               # combines everything, writes final outputs
python stage6_export_deployed.py        # re-exports the app's model files from condition (b)
```

`stage3_tuned.py` also accepts an optional classifier name to run just one at a
time, e.g. `python stage3_tuned.py "Random Forest"`.

## Outputs

`stage5_finalize.py` writes the final deliverables to the project's `outputs/` folder:

- `outputs/tables/extended_20_results/master_20_results.csv` — all 20 rows (Accuracy, Precision, Recall, F1-macro, F1-weighted, timings, best params)
- `outputs/tables/extended_20_results/mcnemar_20_results.csv` — McNemar's test, tuned Logistic Regression vs. both ensembles
- `outputs/tables/extended_20_results/summary.json` — best-overall model/condition, key comparison numbers
- `outputs/figures/full_20_results_comparison.png` — grouped bar chart, accuracy + weighted F1 across all 20 results

`stage6_export_deployed.py` refits the six condition (b) models (SMOTE only) and
writes what the Streamlit app loads: the files in `models/`,
`outputs/tables/model_results_all.json` (read by the Model Info tab) and
`outputs/figures/confusion_matrix_smote_only_lr.png`.

## Key finding

Condition (b), SMOTE-only Logistic Regression (accuracy 0.9292, weighted F1 0.9258),
beat condition (c), SMOTE + GridSearchCV Logistic Regression (accuracy 0.9250,
weighted F1 0.9170), on every metric. It is the model shipped in
`models/best_model.pkl`. Both ensemble combinations (d, e) underperformed tuned
Logistic Regression, and McNemar's test confirms the gap (p = 0.039 against the
voting ensemble, p = 0.021 against stacking).

All results and model files were produced with the versions pinned in
`requirements.txt` (scikit-learn 1.9.0, XGBoost 3.3.0).

## Notes

- `stage1_prep.py` expects the review dataset at `data/KopiKenangan_appreviews.xlsx`
  (sheet `kopken_dataset`). The dataset is not included in this repository; see
  the main README's "Data" section.
- Requires the packages in the root `requirements.txt`.
- Random state is fixed at 42 throughout, so results are reproducible. Small
  numeric differences can appear with library versions other than the pinned ones.
