# Four-Agent Trust-Centric Ablation Report

## Framework
- This run uses exactly four specialized cybersecurity decision sources coordinated by a trust-centric coordination layer.
- Decision source 1: General Traffic Agent (balanced full-feature behavior).
- Decision source 2: Attack Recall Agent (attack-sensitive threshold with validation-only tuning).
- Decision source 3: Normal Behavior Agent (specificity-oriented threshold with validation-only tuning).
- Decision source 4: Hard-Case Agent (trained with extra emphasis on validation hard cases).

## Best Results
- Best single agent: Hard-Case Agent with accuracy=0.9716, F1=0.9322, recall=0.9911, FNR=0.0089.
- Best trust-based final decision: Trust Agent Selector with accuracy=0.9875, F1=0.9686, recall=0.9805, FNR=0.0195.

## Improvement
- Accuracy improvement vs best single agent: 0.0159
- F1 improvement vs best single agent: 0.0365
- Recall improvement vs best single agent: -0.0106
- FNR reduction vs best single agent: -0.0106

## Agent Balance Checks
- Best single agent accuracy: 0.9716
- Worst single agent accuracy: 0.8668
- Accuracy spread between best and worst agents: 0.1048
- Trust accuracy: 0.9875
- Trust improvement over best agent: 0.0159

## Oracle Gap Analysis
- Oracle accuracy among the four specialized cybersecurity decision sources: 0.9979
- Maximum possible improvement over the best single agent: 0.0263
- +2 percentage-point improvement is theoretically possible: True
- +2 percentage-point improvement was achieved: False

## Oracle Breakdown
- Samples all agents correct: 39309
- Samples all agents wrong: 107
- Samples only one agent correct: 1312
- Samples where Hard-Case Agent is wrong but another agent is correct: 1314
- Samples where trust selector failed despite at least one correct agent: 519
- Trust selector missed-opportunity count: 519
- Honest interpretation: +2 points was possible in theory but was not reached by the current trust-centric coordination layer.