# Final IDS Evaluation Report

## Dataset Overview
- Experiment: `ids_two_agents_validation`
- Generated at: `2026-04-21T17:08:12`
- Datasets evaluated: UNSW-NB15
- Label convention: normal=0, attack=1.
 - Two-source design: independent base models (decision sources) operate on the same processed intrusion dataset rows.
- Role differentiation is analytical (not separate raw sensor streams).
- BehavioralAnalysisAgent backbone: RandomForestClassifier.
- TrafficAnalysisAgent backbone: SVM (SVC, RBF kernel).
- TrafficAnalysisAgent preprocessing uses train-only fitted imputation/scaling for leakage-safe numerical stability.

## Train/Test Split Summary
- Split policy: train=70%, test=30%, stratified by label, random_state=42.
- UNSW-NB15: train(normal=59716, attack=47573), balanced_train(normal=47573, attack=47573), test(normal=25592, attack=20389), method=random_undersample_train_only

<<<<<<< HEAD
## Individual Model (Decision Source) Results
### CICIDS2017
- BehavioralAnalysisAgent: CV Acc=0.9959, CV Prec=0.9989, CV Rec=0.9930, CV F1=0.9959, Test Acc=0.9977, Test Prec=0.9941, Test Rec=0.9925, Test F1=0.9933, TP=126593, TN=627989, FP=747, FN=961, FPR=0.0012, FNR=0.0075, TPR=0.9925, TNR=0.9988, Specificity=0.9988, BalancedAcc=0.9956
- TrafficAnalysisAgent: CV Acc=0.9890, CV Prec=0.9936, CV Rec=0.9844, CV F1=0.9890, Test Acc=0.9925, Test Prec=0.9707, Test Rec=0.9852, Test F1=0.9779, TP=125666, TN=624946, FP=3790, FN=1888, FPR=0.0060, FNR=0.0148, TPR=0.9852, TNR=0.9940, Specificity=0.9940, BalancedAcc=0.9896
### NSL-KDD
- BehavioralAnalysisAgent: CV Acc=0.9957, CV Prec=0.9931, CV Rec=0.9981, CV F1=0.9956, Test Acc=0.9968, Test Prec=0.9943, Test Rec=0.9989, Test F1=0.9966, TP=21254, TN=22969, FP=121, FN=23, FPR=0.0052, FNR=0.0011, TPR=0.9989, TNR=0.9948, Specificity=0.9948, BalancedAcc=0.9968
- TrafficAnalysisAgent: CV Acc=0.9959, CV Prec=0.9923, CV Rec=0.9991, CV F1=0.9957, Test Acc=0.9964, Test Prec=0.9937, Test Rec=0.9988, Test F1=0.9962, TP=21251, TN=22955, FP=135, FN=26, FPR=0.0058, FNR=0.0012, TPR=0.9988, TNR=0.9942, Specificity=0.9942, BalancedAcc=0.9965
=======
## Individual Agent Results
>>>>>>> 4b6c430aee6a4dac207a8c70191bb7a525caf013
### UNSW-NB15
- BehavioralAnalysisAgent: CV Acc=0.8953, CV Prec=0.8555, CV Rec=0.9513, CV F1=0.9008, Test Acc=0.8854, Test Prec=0.8148, Test Rec=0.9597, Test F1=0.8814, TP=19568, TN=21145, FP=4447, FN=821, FPR=0.1738, FNR=0.0403, TPR=0.9597, TNR=0.8262, Specificity=0.8262, BalancedAcc=0.8930
- TrafficAnalysisAgent: CV Acc=0.8858, CV Prec=0.8473, CV Rec=0.9411, CV F1=0.8918, Test Acc=0.8806, Test Prec=0.8189, Test Rec=0.9382, Test F1=0.8745, TP=19128, TN=21361, FP=4231, FN=1261, FPR=0.1653, FNR=0.0618, TPR=0.9382, TNR=0.8347, Specificity=0.8347, BalancedAcc=0.8864

## Agreement/Disagreement Summary
- UNSW-NB15: agreement_rate=0.9354, disagreement_rate=0.0646, behavioral_wins_on_disagreement=0.2850, traffic_wins_on_disagreement=0.7150, contested_case_rate=0.0396, threshold=0.10

## Conflict Resolution Summary
- Protocol: agreement -> agreed label; disagreement -> trust-based winner; small trust gap (< trust threshold) -> trust_contested with trust-based fallback label.

<<<<<<< HEAD
## Final Resolved Trust-Based Results
- NSL-KDD: Acc=0.9970, Prec=0.9953, Rec=0.9984, F1=0.9969, TP=21244, TN=22989, FP=101, FN=33, FPR=0.0044, FNR=0.0016, BalancedAcc=0.9970
- UNSW-NB15: Acc=0.8902, Prec=0.8345, Rec=0.9386, F1=0.8835, TP=19137, TN=21796, FP=3796, FN=1252, FPR=0.1483, FNR=0.0614, BalancedAcc=0.8951
- CICIDS2017: Acc=0.9977, Prec=0.9941, Rec=0.9925, F1=0.9933, TP=126593, TN=627989, FP=747, FN=961, FPR=0.0012, FNR=0.0075, BalancedAcc=0.9956
=======
## Final Resolved Multi-Agent Results
- UNSW-NB15: Acc=0.8909, Prec=0.8358, Rec=0.9383, F1=0.8841, TP=19131, TN=21833, FP=3759, FN=1258, FPR=0.1469, FNR=0.0617, BalancedAcc=0.8957
>>>>>>> 4b6c430aee6a4dac207a8c70191bb7a525caf013

## Trust Layer
- Trust design overview: static trust-aware prioritization over independent decision sources.
- Trust formula per model/sample: trust_score = w1*global_reliability + w2*confidence + w3*disagreement_reliability.
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
