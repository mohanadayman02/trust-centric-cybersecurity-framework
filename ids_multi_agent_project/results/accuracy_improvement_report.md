# Accuracy Improvement Report

- Original best baseline accuracy: 0.9565
- Tuned best baseline accuracy: 0.9992
- Best trust accuracy before accuracy-focused tuning: 0.9992
- Best final trust accuracy: 0.9993
- Doctor target accuracy: 0.9784
- Oracle accuracy upper bound: 1.0000
- +2 percentage points theoretically possible vs original baseline: True
- +2 percentage points theoretically possible vs tuned baseline: False
- Target reached vs original baseline: True
- Target reached vs tuned baseline: False

## Interpretation
- The target was reached relative to the original baseline from the historical four-model setup.
- The target was not reached relative to the tuned baseline; the tuned base models are already near the oracle ceiling.
- Best final trust method: stacking_meta_trust
- Best cybersecurity tradeoff: stacking_meta_trust