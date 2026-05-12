# Leakage Validation Report

Scope: canonical feature-frame construction, duplicate/conflict cleaning before split, train/test exact feature-row overlap checks, suspicious column-name audit, and optional group-identifier overlap checks.

## UNSW-NB15

### Canonical Dataset
- Raw row count: 257673
- Final canonical row count before deduplication: 257673
- Identifier columns dropped: id
- Target-proxy columns dropped: attack_cat
- Canonical feature columns finalized (n=42): dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports

### Deduplication
- Rows before deduplication: 257673
- Rows after deduplication: 153270
- Total removed: 104403 (40.5176%)
- Exact duplicate records removed (features + label): 103575
- Conflicting duplicate feature patterns removed: 414 patterns / 828 rows
- Final row count after canonical cleaning: 153270

### Duplicate Row Check
- Post-fix full dataset exact duplicates: 0 / 153270 (0.0000%)
- Post-fix train/test overlap (exact feature rows): unique_shared=0, test_rows=0 (0.0000%), train_rows=0 (0.0000%)

### Label-Proxy / Leakage Column Audit
- Existing drop policy (pipeline.data_loader.POTENTIAL_LEAKAGE_COLUMNS): attack_cat, attack_category, binary_label, class, label_binary, target
- Raw suspicious columns matched by drop policy (excluding label): attack_cat
- Dropped identifier columns from features: id
- Suspicious name matches that remain in features: none

### Group-Leakage Check
- Candidate group/session identifier columns: none detected
- No group/session overlap check applied (no candidate identifier column).

### Final Verdict
- No obvious exact feature-row leakage after canonical cleaning and split; Identifier columns removed from model features: id

