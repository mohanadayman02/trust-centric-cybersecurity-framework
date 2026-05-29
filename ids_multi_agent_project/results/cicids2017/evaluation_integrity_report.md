# Integrity Report: CICIDS2017

## Dataset Size
- Training samples: 100000
- Validation samples: 25000
- Test samples: 50000
- Total: 175000

## Feature Information
- Total features: 70
- Categorical features: 0 ()
- Numeric features: 70 (ACK Flag Count, Active Max, Active Mean, Active Min, Active Std...)

## Class Distribution
- Training: Normal=80300 (80.3%), Attack=19700 (19.7%)
- Validation: Normal=20075 (80.3%), Attack=4925 (19.7%)
- Test: Normal=40150 (80.3%), Attack=9850 (19.7%)

## Leakage Prevention
- Dropped columns: Bwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd PSH Flags, Bwd URG Flags, Fwd Avg Bulk Rate, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, label
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
