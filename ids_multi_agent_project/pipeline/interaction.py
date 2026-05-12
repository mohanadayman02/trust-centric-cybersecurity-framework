"""Interaction and trust-aware conflict-resolution utilities for two-model (decision-source) decisions.

Terminology updated to reflect trust-centric decision layer as the system-level coordinator.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def _safe_metric(metrics: Dict[str, Any] | None, key: str, default: float) -> float:
    if metrics is None:
        return float(default)
    value = metrics.get(key, default)
    if value is None or pd.isna(value):
        return float(default)
    return float(value)


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _global_reliability(metrics: Dict[str, Any] | None) -> float:
    """Compute normalized global trust from per-agent evaluation metrics."""
    f1 = _clip01(_safe_metric(metrics, "test_f1", 0.5))
    balanced_accuracy = _clip01(_safe_metric(metrics, "balanced_accuracy", 0.5))
    fpr = _clip01(_safe_metric(metrics, "fpr", 0.5))
    fnr = _clip01(_safe_metric(metrics, "fnr", 0.5))
    return float((f1 + balanced_accuracy + (1.0 - fpr) + (1.0 - fnr)) / 4.0)


def resolve_agent_interactions(
    dataset_name: str,
    sample_indices: Iterable[Any],
    true_labels: Iterable[int],
    behavioral_decisions: List[Dict[str, Any]],
    traffic_decisions: List[Dict[str, Any]],
    behavioral_metrics: Dict[str, Any] | None = None,
    traffic_metrics: Dict[str, Any] | None = None,
    trust_weight_global: float = 0.5,
    trust_weight_confidence: float = 0.3,
    trust_weight_disagreement: float = 0.2,
    trust_gap_threshold: float = 0.05,
    disagreement_confidence_threshold: float = 0.10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resolve two-agent outputs into trust-aware final multi-agent decisions."""
    idx_list = list(sample_indices)
    y_true = [int(v) for v in true_labels]

    if len(behavioral_decisions) != len(traffic_decisions):
        raise ValueError("behavioral_decisions and traffic_decisions must have equal length.")
    if len(behavioral_decisions) != len(idx_list):
        raise ValueError("Decision lists must match sample count.")

    behavioral_global_trust = _global_reliability(behavioral_metrics)
    traffic_global_trust = _global_reliability(traffic_metrics)

    total_disagreements = 0
    behavioral_disagreement_wins = 0
    traffic_disagreement_wins = 0
    for true_label, b_decision, t_decision in zip(y_true, behavioral_decisions, traffic_decisions):
        b_label = int(b_decision["predicted_label"])
        t_label = int(t_decision["predicted_label"])
        if b_label == t_label:
            continue
        total_disagreements += 1
        if b_label == int(true_label):
            behavioral_disagreement_wins += 1
        elif t_label == int(true_label):
            traffic_disagreement_wins += 1

    if total_disagreements == 0:
        behavioral_disagreement_trust = 0.5
        traffic_disagreement_trust = 0.5
    else:
        behavioral_disagreement_trust = float(behavioral_disagreement_wins / total_disagreements)
        traffic_disagreement_trust = float(traffic_disagreement_wins / total_disagreements)

    rows: List[Dict[str, Any]] = []
    agreement_count = 0
    disagreement_count = 0
    behavioral_wins = 0
    traffic_wins = 0
    contested_count = 0

    for idx, true_label, b_decision, t_decision in zip(
        idx_list, y_true, behavioral_decisions, traffic_decisions
    ):
        b_label = int(b_decision["predicted_label"])
        t_label = int(t_decision["predicted_label"])
        b_conf = float(b_decision["confidence"])
        t_conf = float(t_decision["confidence"])
        b_prob_attack = float(b_decision["probability_attack"])
        t_prob_attack = float(t_decision["probability_attack"])
        b_prob_normal = float(b_decision["probability_normal"])
        t_prob_normal = float(t_decision["probability_normal"])

        agreement = bool(b_label == t_label)
        confidence_gap = float(abs(b_conf - t_conf))

        behavioral_final_trust = float(
            trust_weight_global * behavioral_global_trust
            + trust_weight_confidence * b_conf
            + trust_weight_disagreement * behavioral_disagreement_trust
        )
        traffic_final_trust = float(
            trust_weight_global * traffic_global_trust
            + trust_weight_confidence * t_conf
            + trust_weight_disagreement * traffic_disagreement_trust
        )
        trust_gap = float(abs(behavioral_final_trust - traffic_final_trust))
        behavioral_stronger = behavioral_final_trust >= traffic_final_trust

        if agreement:
            resolution_type = "agreement"
            final_label = b_label
            trust_winner = "none"
            final_confidence = float(max(b_conf, t_conf))
            if b_conf >= t_conf:
                final_probability_attack = b_prob_attack
                final_probability_normal = b_prob_normal
            else:
                final_probability_attack = t_prob_attack
                final_probability_normal = t_prob_normal
            agreement_count += 1
        else:
            disagreement_count += 1
            if behavioral_stronger:
                trust_winner = str(b_decision["agent_name"])
                final_label = b_label
                final_confidence = b_conf
                final_probability_attack = b_prob_attack
                final_probability_normal = b_prob_normal
                behavioral_wins += 1
            else:
                trust_winner = str(t_decision["agent_name"])
                final_label = t_label
                final_confidence = t_conf
                final_probability_attack = t_prob_attack
                final_probability_normal = t_prob_normal
                traffic_wins += 1

            if trust_gap < float(trust_gap_threshold):
                resolution_type = "trust_contested"
                contested_count += 1
            else:
                resolution_type = "trust_resolution"

        rows.append(
            {
                "dataset": dataset_name,
                "sample_index": idx,
                "true_label": int(true_label),
                "behavioral_label": b_label,
                "behavioral_confidence": b_conf,
                "behavioral_reasoning": str(b_decision.get("reasoning", "")),
                "behavioral_stance": str(b_decision.get("stance", "")),
                "traffic_label": t_label,
                "traffic_confidence": t_conf,
                "traffic_reasoning": str(t_decision.get("reasoning", "")),
                "traffic_stance": str(t_decision.get("stance", "")),
                "agreement": agreement,
                "confidence_gap": confidence_gap,
                "winner_agent": trust_winner,
                "trust_winner": trust_winner,
                "behavioral_global_trust": float(behavioral_global_trust),
                "traffic_global_trust": float(traffic_global_trust),
                "behavioral_disagreement_trust": float(behavioral_disagreement_trust),
                "traffic_disagreement_trust": float(traffic_disagreement_trust),
                "behavioral_final_trust": float(behavioral_final_trust),
                "traffic_final_trust": float(traffic_final_trust),
                "trust_gap": trust_gap,
                "resolution_type": resolution_type,
                "final_label": int(final_label),
                "final_confidence": float(final_confidence),
                "final_probability_normal": float(final_probability_normal),
                "final_probability_attack": float(final_probability_attack),
                "final_correct": bool(int(final_label) == int(true_label)),
            }
        )

    total = len(rows)
    summary = {
        "dataset": dataset_name,
        "total_samples": int(total),
        "agreement_count": int(agreement_count),
        "disagreement_count": int(disagreement_count),
        "agreement_rate": float(agreement_count / total) if total else 0.0,
        "disagreement_rate": float(disagreement_count / total) if total else 0.0,
        "behavioral_wins_on_disagreement_count": int(behavioral_wins),
        "traffic_wins_on_disagreement_count": int(traffic_wins),
        "behavioral_wins_on_disagreement": (
            float(behavioral_wins / disagreement_count) if disagreement_count else 0.0
        ),
        "traffic_wins_on_disagreement": (
            float(traffic_wins / disagreement_count) if disagreement_count else 0.0
        ),
        "contested_case_count": int(contested_count),
        "contested_case_rate": float(contested_count / total) if total else 0.0,
        "behavioral_global_reliability": float(behavioral_global_trust),
        "traffic_global_reliability": float(traffic_global_trust),
        "behavioral_disagreement_trust": float(behavioral_disagreement_trust),
        "traffic_disagreement_trust": float(traffic_disagreement_trust),
        "trust_weight_global": float(trust_weight_global),
        "trust_weight_confidence": float(trust_weight_confidence),
        "trust_weight_disagreement": float(trust_weight_disagreement),
        "trust_gap_threshold": float(trust_gap_threshold),
        "disagreement_confidence_threshold": float(disagreement_confidence_threshold),
    }

    return pd.DataFrame(rows), summary
