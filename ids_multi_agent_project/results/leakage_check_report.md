# Leakage Validation Report

Scope: canonical feature-frame construction, duplicate/conflict cleaning before split, train/test exact feature-row overlap checks, suspicious column-name audit, and optional group-identifier overlap checks.

## NSL-KDD

### Canonical Dataset
- Raw row count: 148517
- Final canonical row count before deduplication: 148517
- Identifier columns dropped: none
- Target-proxy columns dropped: none
- Canonical feature columns finalized (n=42): duration, protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment, urgent, hot, num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, num_root, num_file_creations, num_shells, num_access_files, num_outbound_cmds, is_host_login, is_guest_login, count, srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate, difficulty_level

### Deduplication
- Rows before deduplication: 148517
- Rows after deduplication: 147888
- Total removed: 629 (0.4235%)
- Exact duplicate records removed (features + label): 629
- Conflicting duplicate feature patterns removed: 0 patterns / 0 rows
- Final row count after canonical cleaning: 147888

### Duplicate Row Check
- Post-fix full dataset exact duplicates: 0 / 147888 (0.0000%)
- Post-fix train/test overlap (exact feature rows): unique_shared=0, test_rows=0 (0.0000%), train_rows=0 (0.0000%)

### Label-Proxy / Leakage Column Audit
- Existing drop policy (pipeline.data_loader.POTENTIAL_LEAKAGE_COLUMNS): attack_cat, attack_category, binary_label, class, label_binary, target
- Raw suspicious columns matched by drop policy (excluding label): none
- Dropped identifier columns from features: none
- Suspicious name matches that remain in features: none

### Group-Leakage Check
- Candidate group/session identifier columns: none detected
- No group/session overlap check applied (no candidate identifier column).

### Final Verdict
- No obvious exact feature-row leakage after canonical cleaning and split

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

## CICIDS2017

### Canonical Dataset
- Raw row count: 2830743
- Final canonical row count before deduplication: 2830743
- Identifier columns dropped: none
- Target-proxy columns dropped: none
- Canonical feature columns finalized (n=78): Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets, Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std, Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std, Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min, Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length, Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance, FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length.1, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate, Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes, Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward, Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max, Idle Min

### Deduplication
- Rows before deduplication: 2830743
- Rows after deduplication: 2520966
- Total removed: 309777 (10.9433%)
- Exact duplicate records removed (features + label): 308381
- Conflicting duplicate feature patterns removed: 698 patterns / 1396 rows
- Final row count after canonical cleaning: 2520966

### Duplicate Row Check
- Post-fix full dataset exact duplicates: 0 / 2520966 (0.0000%)
- Post-fix train/test overlap (exact feature rows): unique_shared=0, test_rows=0 (0.0000%), train_rows=0 (0.0000%)

### Label-Proxy / Leakage Column Audit
- Existing drop policy (pipeline.data_loader.POTENTIAL_LEAKAGE_COLUMNS): attack_cat, attack_category, binary_label, class, label_binary, target
- Raw suspicious columns matched by drop policy (excluding label): none
- Dropped identifier columns from features: none
- Suspicious name matches that remain in features: none

### Group-Leakage Check
- Candidate group/session identifier columns: none detected
- No group/session overlap check applied (no candidate identifier column).

### Final Verdict
- No obvious exact feature-row leakage after canonical cleaning and split

