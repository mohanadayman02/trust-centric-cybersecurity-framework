# Integrity Report: ToN-IoT

## Dataset Size
- Training samples: 168834
- Validation samples: 42209
- Test samples: 211043
- Total: 422086

## Feature Information
- Total features: 1678
- Categorical features: 26 (conn_state, dns_AA, dns_RA, dns_RD, dns_query...)
- Numeric features: 16 (dns_qclass, dns_qtype, dns_rcode, dst_bytes, dst_ip_bytes...)

## Class Distribution
- Training: Normal=40000 (23.7%), Attack=128834 (76.3%)
- Validation: Normal=10000 (23.7%), Attack=32209 (76.3%)
- Test: Normal=50000 (23.7%), Attack=161043 (76.3%)

## Leakage Prevention
- Dropped columns: label, type
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
