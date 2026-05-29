# Four-Agent Trust-Centric Ablation Report

## Framework
- This run uses exactly four specialized cybersecurity decision sources coordinated by a trust-centric coordination layer.
- Decision source 1: General Traffic Agent (balanced full-feature behavior).
- Decision source 2: Attack Recall Agent (attack-sensitive threshold with validation-only tuning).
- Decision source 3: Normal Behavior Agent (specificity-oriented threshold with validation-only tuning).
- Decision source 4: Hard-Case Agent (trained with extra emphasis on validation hard cases).

## Best Results
- Best single agent: Hard-Case Agent with accuracy=0.9682, F1=0.9248, recall=0.9915, FNR=0.0085.
- Best trust-based final decision: Trust Agent Selector with accuracy=0.9884, F1=0.9707, recall=0.9798, FNR=0.0202.

## Improvement
- Accuracy improvement vs best single agent: 0.0201
- F1 improvement vs best single agent: 0.0459
- Recall improvement vs best single agent: -0.0117
- FNR reduction vs best single agent: -0.0117

## Agent Balance Checks
- Best single agent accuracy: 0.9682
- Worst single agent accuracy: 0.8668
- Accuracy spread between best and worst agents: 0.1014
- Trust accuracy: 0.9884
- Trust improvement over best agent: 0.0201

## Oracle Gap Analysis
- Oracle accuracy among the four specialized cybersecurity decision sources: 0.9980
- Maximum possible improvement over the best single agent: 0.0298
- +2 percentage-point improvement is theoretically possible: True
- +2 percentage-point improvement was achieved: True

## Oracle Breakdown
- Samples all agents correct: 39129
- Samples all agents wrong: 100
- Samples only one agent correct: 1327
- Samples where Hard-Case Agent is wrong but another agent is correct: 1488
- Samples where trust selector failed despite at least one correct agent: 482
- Trust selector missed-opportunity count: 482
- The trust-centric coordination layer achieved the +2 point target without using extra expanded models.