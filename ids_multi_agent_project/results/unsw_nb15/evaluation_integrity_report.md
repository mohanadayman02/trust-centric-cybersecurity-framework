# Integrity Report: UNSW-NB15

## Dataset Size
- Training samples: 140272
- Validation samples: 35069
- Test samples: 82332
- Total: 257673

## Feature Information
- Total features: 194
- Categorical features: 3 (proto, service, state)
- Numeric features: 39 (dur, spkts, dpkts, sbytes, dbytes...)

## Class Distribution
- Training: Normal=44800 (31.9%), Attack=95472 (68.1%)
- Validation: Normal=11200 (31.9%), Attack=23869 (68.1%)
- Test: Normal=37000 (44.9%), Attack=45332 (55.1%)

## Leakage Prevention
- Dropped columns: None
- Preprocessing fit: Training data only
- Validation transformed: After training fit
- Test transformed: After training fit
- Test labels used for: Final evaluation only

## Validation Discipline
- ✓ Threshold tuning: Validation data only (no test labels)
- ✓ Feature selection: Training data only
- ✓ Hard-case identification: Validation data only
- ✓ Preprocessing: Fit on training data only
- ✓ Trust selector tuning: Validation data only
- ✓ Test labels: Reserved for final metrics calculation

## Data Quality
- Missing values in training features: 0
- Missing values in validation features: 0
- Missing values in test features: 0
- Binary labels verified: True

## Preprocessing Pipeline
- Numeric scaler: StandardScaler (fitted on train, applied to val/test)
- Categorical encoder: OneHotEncoder (fitted on train, applied to val/test)
- Imputation: Median for numeric, 'unknown' for categorical
- Feature alignment: Train/val/test have identical feature columns after preprocessing
