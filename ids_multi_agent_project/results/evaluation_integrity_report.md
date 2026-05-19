# Integrity Report: NSL-KDD

## Dataset Size
- Training samples: 83168
- Validation samples: 20793
- Test samples: 44556
- Total: 148517

## Feature Information
- Total features: 122
- Categorical features: 3 (protocol_type, service, flag)
- Numeric features: 39 (duration, src_bytes, dst_bytes, land, wrong_fragment...)

## Class Distribution
- Training: Normal=43149 (51.9%), Attack=40019 (48.1%)
- Validation: Normal=10788 (51.9%), Attack=10005 (48.1%)
- Test: Normal=23117 (51.9%), Attack=21439 (48.1%)

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
