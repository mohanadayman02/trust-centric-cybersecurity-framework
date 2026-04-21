# IDS Multi-Agent Project (Stage 1)

## Purpose
This project provides a clean, modular baseline for a classical machine-learning intrusion detection experiment with **two IDS agents** across **multiple datasets**.

Current scope (stage 1):
- Multi-dataset support from YAML config
- Preprocessing pipeline (missing values, encoding, scaling)
- Two independent IDS agents
- K-fold cross-validation on training split
- Held-out test set evaluation
- CSV results export

Not included yet:
- Trust scoring
- Agent fusion/ensemble logic
- Malicious/adversarial agent simulation

## Folder Structure
```text
ids_multi_agent_project/
├── config/
│   └── experiment.yml
├── data/
├── models/
│   ├── agent_factory.py
│   └── __init__.py
├── pipeline/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── __init__.py
├── results/
├── main.py
├── requirements.txt
└── README.md
```

## Installation
From `ids_multi_agent_project/`:

```bash
pip install -r requirements.txt
```

## Dataset Placement
Put your CSV files inside the `data/` folder and make sure their names/paths match `config/experiment.yml`.

Default expected files:
- `data/nsl_kdd.csv`
- `data/unsw_nb15.csv`

Each dataset must include the configured label column (default: `label`).

## Run Experiment
From `ids_multi_agent_project/`:

```bash
python main.py
```

The script will:
- Read `config/experiment.yml`
- Loop over all datasets
- Train and evaluate the two configured IDS agents
- Print metrics and confusion matrix per dataset/agent
- Save results CSV in `results/`

## Notes
- Assumption: dataset paths in YAML are relative to project root (as in provided starter config).
- To add more datasets or change models/hyperparameters, update `config/experiment.yml` only.
