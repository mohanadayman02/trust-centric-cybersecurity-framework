# Final IDS Evaluation Report

## Dataset Overview
- Experiment: `ids_two_agents_validation`
- Generated at: `2026-04-21T17:08:12`
- Datasets evaluated: UNSW-NB15
- Label convention: normal=0, attack=1.
- Two-agent design: both agents operate on the same processed intrusion dataset rows.
- Role differentiation is analytical (not separate raw sensor streams).
- BehavioralAnalysisAgent backbone: RandomForestClassifier.
- TrafficAnalysisAgent backbone: SVM (SVC, RBF kernel).
- TrafficAnalysisAgent preprocessing uses train-only fitted imputation/scaling for leakage-safe numerical stability.

## Train/Test Split Summary
- Split policy: train=70%, test=30%, stratified by label, random_state=42.
- UNSW-NB15: train(normal=59716, attack=47573), balanced_train(normal=47573, attack=47573), test(normal=25592, attack=20389), method=random_undersample_train_only

## Individual Agent Results
### UNSW-NB15
- BehavioralAnalysisAgent: CV Acc=0.8953, CV Prec=0.8555, CV Rec=0.9513, CV F1=0.9008, Test Acc=0.8854, Test Prec=0.8148, Test Rec=0.9597, Test F1=0.8814, TP=19568, TN=21145, FP=4447, FN=821, FPR=0.1738, FNR=0.0403, TPR=0.9597, TNR=0.8262, Specificity=0.8262, BalancedAcc=0.8930
- TrafficAnalysisAgent: CV Acc=0.8858, CV Prec=0.8473, CV Rec=0.9411, CV F1=0.8918, Test Acc=0.8806, Test Prec=0.8189, Test Rec=0.9382, Test F1=0.8745, TP=19128, TN=21361, FP=4231, FN=1261, FPR=0.1653, FNR=0.0618, TPR=0.9382, TNR=0.8347, Specificity=0.8347, BalancedAcc=0.8864

## Agreement/Disagreement Summary
- UNSW-NB15: agreement_rate=0.9354, disagreement_rate=0.0646, behavioral_wins_on_disagreement=0.2850, traffic_wins_on_disagreement=0.7150, contested_case_rate=0.0396, threshold=0.10

## Conflict Resolution Summary
- Protocol: agreement -> agreed label; disagreement -> trust-based winner; small trust gap (< trust threshold) -> trust_contested with trust-based fallback label.

## Final Resolved Multi-Agent Results
- UNSW-NB15: Acc=0.8909, Prec=0.8358, Rec=0.9383, F1=0.8841, TP=19131, TN=21833, FP=3759, FN=1258, FPR=0.1469, FNR=0.0617, BalancedAcc=0.8957

## Trust Layer
- Trust design overview: static trust-aware prioritization over the existing two-agent system.
- Trust formula per agent/sample: trust_score = w1*global_reliability + w2*confidence + w3*disagreement_reliability.
- Weights: w1=0.50, w2=0.30, w3=0.20; trust_gap_threshold=0.05.
- Note: trust is used for decision prioritization only and does not replace either classifier backbone.
### UNSW-NB15
- Global trust values: behavioral=0.8901, traffic=0.8834
- Disagreement trust values: behavioral=0.5377, traffic=0.4623
- Trust-based final performance: accuracy=0.8909, precision=0.8358, recall=0.9383, f1=0.8841
- Interpretation: contested cases=1821, resolution relies on trust prioritization during disagreements.

## Ollama Status Summary
- Ollama reasoning enabled: True
- Ollama model: `llama3.1:8b`
- Reasoning success rows: 0
- Reasoning non-success rows: 50

## Output Files
- `experiment_results.csv` -> Per-agent and final multi-agent metrics.
- `sample_level_predictions.csv` -> Sample-level labels/confidence and final resolved output.
- `agent_agreement.csv` -> Dataset-level agreement/disagreement and final decision summary metrics.
- `trust_summary.csv` -> Per-dataset trust values, trust weights, and trust-based final metrics.
- `agent_interactions.csv` -> Per-sample interaction records with disagreement resolution fields.
- `trust_interactions.csv` -> Per-sample trust-layer interactions and trust-based winner decisions.
- `class_balance_summary.csv` -> Per-dataset class balance before/after train-only balancing.
- `agent_reasoning_outputs.csv` -> Role-aligned agent reasoning outputs (Ollama/fallback).
- `leakage_check_report.md` -> Per-dataset duplicate/overlap and leakage-indicator validation report.
- `final_report.md` -> Within-dataset multi-agent report with conflict-resolution summary.
