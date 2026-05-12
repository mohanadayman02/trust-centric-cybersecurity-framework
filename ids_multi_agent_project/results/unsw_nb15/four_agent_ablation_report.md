# Four-Agent Trust-Centric Ablation Report

## Framework
- This run uses exactly four specialized cybersecurity decision sources coordinated by a trust-centric coordination layer.
- Decision source 1: General Traffic Agent (balanced full-feature behavior).
- Decision source 2: Attack Recall Agent (attack-sensitive threshold with validation-only tuning).
- Decision source 3: Normal Behavior Agent (specificity-oriented threshold with validation-only tuning).
- Decision source 4: Hard-Case Agent (trained with extra emphasis on validation hard cases).

## Best Results
- Best single agent: Hard-Case Agent with accuracy=0.9212, F1=0.9301, recall=0.9515, FNR=0.0485.
- Best trust-based final decision: Majority Voting with accuracy=0.8383, F1=0.8696, recall=0.9795, FNR=0.0205.

## Improvement
- Accuracy improvement vs best single agent: -0.0830
- F1 improvement vs best single agent: -0.0605
- Recall improvement vs best single agent: 0.0280
- FNR reduction vs best single agent: 0.0280

## Agent Balance Checks
- Best single agent accuracy: 0.9212
- Worst single agent accuracy: 0.7615
- Accuracy spread between best and worst agents: 0.1598
- Trust accuracy: 0.8383
- Trust improvement over best agent: -0.0830

## Oracle Gap Analysis
- Oracle accuracy among the four specialized cybersecurity decision sources: 0.9584
- Maximum possible improvement over the best single agent: 0.0372
- +2 percentage-point improvement is theoretically possible: True
- +2 percentage-point improvement was achieved: False

## Oracle Breakdown
- Samples all agents correct: 57207
- Samples all agents wrong: 3421
- Samples only one agent correct: 9013
- Samples where Hard-Case Agent is wrong but another agent is correct: 3063
- Samples where trust selector failed despite at least one correct agent: 11602
- Trust selector missed-opportunity count: 11602
- Honest interpretation: +2 points was possible in theory but was not reached by the current trust-centric coordination layer.