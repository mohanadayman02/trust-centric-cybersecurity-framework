# Four-Agent Trust-Centric Ablation Report

## Framework
- This run uses exactly four specialized cybersecurity decision sources coordinated by a trust-centric coordination layer.
- Decision source 1: General Traffic Agent (balanced full-feature behavior).
- Decision source 2: Attack Recall Agent (attack-sensitive threshold with validation-only tuning).
- Decision source 3: Normal Behavior Agent (specificity-oriented threshold with validation-only tuning).
- Decision source 4: Hard-Case Agent (trained with extra emphasis on validation hard cases).

## Best Results
- Best single agent: General Traffic Agent with accuracy=0.9508, F1=0.9484, recall=0.9396, FNR=0.0604.
- Best trust-based final decision: Trust Agent Selector with accuracy=0.9745, F1=0.9738, recall=0.9824, FNR=0.0176.

## Improvement
- Accuracy improvement vs best single agent: 0.0238
- F1 improvement vs best single agent: 0.0254
- Recall improvement vs best single agent: 0.0428
- FNR reduction vs best single agent: 0.0428

## Agent Balance Checks
- Best single agent accuracy: 0.9508
- Worst single agent accuracy: 0.8362
- Accuracy spread between best and worst agents: 0.1146
- Trust accuracy: 0.9745
- Trust improvement over best agent: 0.0238

## Oracle Gap Analysis
- Oracle accuracy among the four specialized cybersecurity decision sources: 0.9933
- Maximum possible improvement over the best single agent: 0.0425
- +2 percentage-point improvement is theoretically possible: True
- +2 percentage-point improvement was achieved: True

## Oracle Breakdown
- Samples all agents correct: 32260
- Samples all agents wrong: 298
- Samples only one agent correct: 1693
- Samples where Hard-Case Agent is wrong but another agent is correct: 7000
- Samples where trust selector failed despite at least one correct agent: 836
- Trust selector missed-opportunity count: 836
- The trust-centric coordination layer achieved the +2 point target without using extra expanded models.