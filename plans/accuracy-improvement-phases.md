## Accuracy Improvement Plan

### Phase 1
- Add a true holdout split that stays disjoint from validation.
- Remove target-shaping leakage by fitting target clipping on training data only.
- Align serving with training by predicting from the latest completed feature row and matching anchor price logic.
- Add regression tests for split disjointness, fold-local target shaping, and train/serve alignment helpers.

### Phase 2
- Refit selected winners on the full pre-live training window.
- Make the served ensemble/stacked artifact subject to champion governance.
- Replace random global candidate capping with stratified budgeting.
- Add regression tests around refit artifacts and budgeting behavior.

### Phase 3
- Add richer context features beyond single-series OHLCV transforms.
- Make calibration and bias correction regime-aware.
- Add tests for feature availability and regime-bucket calibration behavior.

### Phase 4
- Tighten model-family search after the evaluation contract is fixed.
- Add/adjust model families and search strategy with targeted regression checks.
