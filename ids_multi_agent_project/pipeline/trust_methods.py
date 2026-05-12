from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def _normalize_trust_scores(scores: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([scores[k] for k in scores], dtype=float)
    if np.all(np.isnan(vals)) or np.sum(vals) <= 0.0:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    vals = np.nan_to_num(vals, nan=0.0)
    s = float(np.sum(vals))
    if s <= 0.0:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: float(v) / s for k, v in zip(scores.keys(), vals)}


def _get_metric_value(metrics: Dict[str, float], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return float(default)


def _resolve_attack_label(attack_label: object) -> int:
    if attack_label is None:
        return 1
    if isinstance(attack_label, str) and attack_label.strip().lower() == "auto":
        return 1
    try:
        return int(attack_label)
    except Exception:  # pylint: disable=broad-except
        return 1


def build_trust_diagnostics_rows(
    method_name: str,
    validation_metrics: Dict[str, Dict[str, float]],
    trust_scores: Dict[str, float],
    selected_model: Optional[str] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_name, metrics in validation_metrics.items():
        rows.append(
            {
                "method": method_name,
                "model": model_name,
                "trust_score": float(trust_scores.get(model_name, 0.0)),
                "accuracy": _get_metric_value(metrics, "test_accuracy", "accuracy"),
                "precision": _get_metric_value(metrics, "test_precision", "precision"),
                "recall": _get_metric_value(metrics, "test_recall", "recall"),
                "f1": _get_metric_value(metrics, "test_f1", "f1"),
                "fpr": _get_metric_value(metrics, "fpr"),
                "fnr": _get_metric_value(metrics, "fnr"),
                "specificity": _get_metric_value(metrics, "specificity"),
                "selected_as_primary_model": bool(selected_model is not None and model_name == selected_model),
            }
        )
    return rows


def majority_voting(predictions: Dict[str, np.ndarray], tie_breaker: Optional[str] = None) -> Dict:
    names = list(predictions.keys())
    stacked = np.vstack([predictions[n] for n in names])
    votes = np.sum(stacked == 1, axis=0)
    threshold = int(np.floor(len(names) / 2)) + 1
    preds = np.where(votes >= threshold, 1, 0)
    return {"predictions": preds, "meta": {"method": "majority_voting", "trust_scores": {name: 1.0 for name in names}}}


def accuracy_based_trust(
    predictions: Dict[str, np.ndarray],
    validation_metrics: Dict[str, Dict[str, float]],
) -> Dict:
    scores = {name: float(validation_metrics.get(name, {}).get("test_accuracy", 0.0)) for name in predictions}
    weights = _normalize_trust_scores(scores)
    names = list(predictions.keys())
    stacked = np.vstack([predictions[n] for n in names])
    weighted = np.zeros(stacked.shape[1], dtype=float)
    for i, n in enumerate(names):
        weighted += weights[n] * stacked[i]
    preds = np.where(weighted >= 0.5, 1, 0)
    return {"predictions": preds, "meta": {"method": "accuracy_based_trust", "weights": weights, "trust_scores": weights}}


def f1_based_trust(predictions: Dict[str, np.ndarray], validation_metrics: Dict[str, Dict[str, float]]) -> Dict:
    scores = {name: float(validation_metrics.get(name, {}).get("test_f1", 0.0)) for name in predictions}
    weights = _normalize_trust_scores(scores)
    names = list(predictions.keys())
    stacked = np.vstack([predictions[n] for n in names])
    weighted = np.zeros(stacked.shape[1], dtype=float)
    for i, n in enumerate(names):
        weighted += weights[n] * stacked[i]
    preds = np.where(weighted >= 0.5, 1, 0)
    return {"predictions": preds, "meta": {"method": "f1_based_trust", "weights": weights, "trust_scores": weights}}


def error_aware_trust(
    predictions: Dict[str, np.ndarray],
    validation_metrics: Dict[str, Dict[str, float]],
    alpha: float = 0.3,
    beta: float = 0.7,
) -> Dict:
    scores = {}
    for name in predictions:
        m = validation_metrics.get(name, {})
        acc = float(m.get("test_accuracy", 0.0))
        fpr = float(m.get("fpr", 0.0))
        fnr = float(m.get("fnr", 0.0))
        trust = acc - alpha * fpr - beta * fnr
        scores[name] = max(0.0, min(1.0, trust))

    weights = _normalize_trust_scores(scores)
    names = list(predictions.keys())
    stacked = np.vstack([predictions[n] for n in names])
    weighted = np.zeros(stacked.shape[1], dtype=float)
    for i, n in enumerate(names):
        weighted += weights[n] * stacked[i]
    preds = np.where(weighted >= 0.5, 1, 0)
    return {"predictions": preds, "meta": {"method": "error_aware_trust", "weights": weights, "trust_scores": weights}}


def confidence_based_trust(
    predictions: Dict[str, np.ndarray],
    probabilities: Optional[Dict[str, np.ndarray]],
    validation_metrics: Dict[str, Dict[str, float]],
) -> Dict:
    names = list(predictions.keys())
    n_samples = next(iter(predictions.values())).shape[0]
    weighted = np.zeros(n_samples, dtype=float)
    weights = {n: float(validation_metrics.get(n, {}).get("test_f1", 0.0)) for n in names}
    weights = _normalize_trust_scores(weights)
    for n in names:
        conf = None
        if probabilities and n in probabilities and probabilities[n] is not None:
            prob = probabilities[n]
            if prob.ndim == 2 and prob.shape[1] >= 2:
                conf = prob[:, 1]
            else:
                conf = np.asarray(prob).ravel()
        if conf is None:
            conf = np.ones(n_samples, dtype=float)
        weighted += weights[n] * conf * predictions[n]
    preds = np.where(weighted >= 0.5, 1, 0)
    return {"predictions": preds, "meta": {"method": "confidence_based_trust", "weights": weights, "trust_scores": weights}}


def class_specific_trust(
    predictions: Dict[str, np.ndarray],
    validation_metrics: Dict[str, Dict[str, float]],
    per_class_f1: Optional[Dict[str, Dict[int, float]]] = None,
) -> Dict:
    names = list(predictions.keys())
    n_samples = next(iter(predictions.values())).shape[0]
    weighted_votes = {0: np.zeros(n_samples, dtype=float), 1: np.zeros(n_samples, dtype=float)}
    for n in names:
        preds = predictions[n]
        if per_class_f1 and n in per_class_f1:
            class_scores = per_class_f1[n]
        else:
            class_scores = {0: validation_metrics.get(n, {}).get("test_f1", 0.0), 1: validation_metrics.get(n, {}).get("test_f1", 0.0)}
        for cls in [0, 1]:
            mask = preds == cls
            weighted_votes[cls][mask] += float(class_scores.get(cls, 0.0))
    preds = np.where(weighted_votes[1] >= weighted_votes[0], 1, 0)
    trust_scores = _normalize_trust_scores({name: float(validation_metrics.get(name, {}).get("test_f1", 0.0)) for name in names})
    return {"predictions": preds, "meta": {"method": "class_specific_trust", "trust_scores": trust_scores}}


def dynamic_trust(
    predictions: Dict[str, np.ndarray],
    validation_metrics: Dict[str, Dict[str, float]],
    lambda_value: float = 0.8,
    batch_size: int = 50,
) -> Dict:
    names = list(predictions.keys())
    n = next(iter(predictions.values())).shape[0]
    trusts = {name: float(validation_metrics.get(name, {}).get("test_f1", 0.0)) for name in names}
    preds_matrix = np.vstack([predictions[n] for n in names])
    final_preds = np.zeros(n, dtype=int)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch = preds_matrix[:, start:end]
        pseudo = (np.sum(batch == 1, axis=0) >= (len(names) // 2 + 1)).astype(int)
        for i, name in enumerate(names):
            acc = float(np.mean(batch[i] == pseudo))
            trusts[name] = lambda_value * trusts[name] + (1 - lambda_value) * acc
        weights = _normalize_trust_scores(trusts)
        weighted = np.zeros(end - start, dtype=float)
        for i, name in enumerate(names):
            weighted += weights[name] * batch[i]
        final_preds[start:end] = np.where(weighted >= 0.5, 1, 0)
    final_weights = _normalize_trust_scores(trusts)
    return {"predictions": final_preds, "meta": {"method": "dynamic_trust", "final_trusts": trusts, "trust_scores": final_weights}}


def hybrid_trust(
    predictions: Dict[str, np.ndarray],
    probabilities: Optional[Dict[str, np.ndarray]],
    validation_metrics: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    if weights is None:
        weights = {"w_f1": 0.30, "w_recall": 0.25, "w_conf": 0.20, "w_specificity": 0.15, "w_fnr": 0.10}
    names = list(predictions.keys())
    model_trust = {}
    for n in names:
        m = validation_metrics.get(n, {})
        f1 = float(m.get("test_f1", 0.0))
        recall = float(m.get("test_recall", 0.0))
        spec = float(m.get("specificity", 0.0))
        fnr = float(m.get("fnr", 0.0))
        score = (
            weights["w_f1"] * f1
            + weights["w_recall"] * recall
            + weights["w_specificity"] * spec
            + weights["w_fnr"] * (1.0 - fnr)
        )
        model_trust[n] = max(0.0, min(1.0, score))
    norm = _normalize_trust_scores(model_trust)
    n_samples = next(iter(predictions.values())).shape[0]
    weighted = np.zeros(n_samples, dtype=float)
    for n in names:
        conf = None
        if probabilities and n in probabilities and probabilities[n] is not None:
            prob = probabilities[n]
            if prob.ndim == 2 and prob.shape[1] >= 2:
                conf = prob[:, 1]
            else:
                conf = np.asarray(prob).ravel()
        if conf is None:
            conf = np.ones(n_samples, dtype=float)
        weighted += norm[n] * conf * predictions[n]
    preds = np.where(weighted >= 0.5, 1, 0)
    return {"predictions": preds, "meta": {"method": "hybrid_trust", "weights": norm, "trust_scores": norm}}


def attack_recall_trust(
    predictions: Dict[str, np.ndarray],
    validation_metrics: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    if weights is None:
        weights = {"w_recall": 0.40, "w_low_fnr": 0.30, "w_f1": 0.20, "w_precision": 0.10}
    scores = {}
    for name in predictions:
        m = validation_metrics.get(name, {})
        recall = _get_metric_value(m, "test_recall", "recall")
        fnr = _get_metric_value(m, "fnr")
        f1 = _get_metric_value(m, "test_f1", "f1")
        precision = _get_metric_value(m, "test_precision", "precision")
        trust = (
            weights["w_recall"] * recall
            + weights["w_low_fnr"] * (1.0 - fnr)
            + weights["w_f1"] * f1
            + weights["w_precision"] * precision
        )
        scores[name] = max(0.0, min(1.0, trust))

    weights_norm = _normalize_trust_scores(scores)
    names = list(predictions.keys())
    stacked = np.vstack([predictions[n] for n in names])
    weighted = np.zeros(stacked.shape[1], dtype=float)
    for i, n in enumerate(names):
        weighted += weights_norm[n] * stacked[i]
    preds = np.where(weighted >= 0.5, 1, 0)
    return {"predictions": preds, "meta": {"method": "attack_recall_trust", "trust_scores": scores, "weights": weights_norm}}


def fnr_penalty_trust(
    predictions: Dict[str, np.ndarray],
    validation_metrics: Dict[str, Dict[str, float]],
    beta: float = 0.8,
    alpha: float = 0.2,
) -> Dict:
    scores = {}
    for name in predictions:
        m = validation_metrics.get(name, {})
        f1 = _get_metric_value(m, "test_f1", "f1")
        fnr = _get_metric_value(m, "fnr")
        specificity = _get_metric_value(m, "specificity")
        trust = f1 - beta * fnr + alpha * specificity
        scores[name] = max(0.0, min(1.0, trust))

    weights_norm = _normalize_trust_scores(scores)
    names = list(predictions.keys())
    stacked = np.vstack([predictions[n] for n in names])
    weighted = np.zeros(stacked.shape[1], dtype=float)
    for i, n in enumerate(names):
        weighted += weights_norm[n] * stacked[i]
    preds = np.where(weighted >= 0.5, 1, 0)
    return {"predictions": preds, "meta": {"method": "fnr_penalty_trust", "trust_scores": scores, "weights": weights_norm}}


def attack_override_trust(
    predictions: Dict[str, np.ndarray],
    probabilities: Optional[Dict[str, np.ndarray]],
    validation_metrics: Dict[str, Dict[str, float]],
    attack_label: object = "auto",
    attack_confidence_threshold: float = 0.65,
    min_attack_model_trust: float = 0.80,
) -> Dict:
    attack_value = _resolve_attack_label(attack_label)
    names = list(predictions.keys())
    n_samples = next(iter(predictions.values())).shape[0]
    trust_scores = {name: _get_metric_value(validation_metrics.get(name, {}), "test_f1", "f1") for name in names}
    weights = _normalize_trust_scores(trust_scores)
    final_preds = np.zeros(n_samples, dtype=int)

    for sample_idx in range(n_samples):
        attack_candidates: List[str] = []
        for name in names:
            if int(predictions[name][sample_idx]) != attack_value:
                continue
            if trust_scores[name] < float(min_attack_model_trust):
                continue
            confidence = 1.0
            if probabilities and name in probabilities and probabilities[name] is not None:
                prob = probabilities[name]
                if prob.ndim == 2 and prob.shape[1] > attack_value:
                    confidence = float(prob[sample_idx, attack_value])
                else:
                    confidence = float(np.asarray(prob).ravel()[sample_idx])
            if confidence >= float(attack_confidence_threshold):
                attack_candidates.append(name)

        if attack_candidates:
            final_preds[sample_idx] = attack_value
            continue

        weighted_attack_score = 0.0
        for name in names:
            weighted_attack_score += weights[name] * float(predictions[name][sample_idx] == attack_value)
        final_preds[sample_idx] = attack_value if weighted_attack_score >= 0.5 else (1 - attack_value)

    selected_model = max(trust_scores.items(), key=lambda item: item[1])[0] if trust_scores else None
    return {
        "predictions": final_preds,
        "meta": {
            "method": "attack_override_trust",
            "trust_scores": trust_scores,
            "weights": weights,
            "selected_model": selected_model,
        },
    }


def best_safe_model_selector(
    predictions: Dict[str, np.ndarray],
    validation_metrics: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    if weights is None:
        weights = {"w_recall": 0.45, "w_f1": 0.30, "w_low_fnr": 0.20, "w_accuracy": 0.05}

    safety_scores: Dict[str, float] = {}
    for name in predictions:
        m = validation_metrics.get(name, {})
        recall = _get_metric_value(m, "test_recall", "recall")
        f1 = _get_metric_value(m, "test_f1", "f1")
        fnr = _get_metric_value(m, "fnr")
        accuracy = _get_metric_value(m, "test_accuracy", "accuracy")
        safety_score = (
            weights["w_recall"] * recall
            + weights["w_f1"] * f1
            + weights["w_low_fnr"] * (1.0 - fnr)
            + weights["w_accuracy"] * accuracy
        )
        safety_scores[name] = max(0.0, min(1.0, safety_score))

    selected_model = max(safety_scores.items(), key=lambda item: item[1])[0]
    final_preds = np.asarray(predictions[selected_model], dtype=int)
    trust_scores = {name: (1.0 if name == selected_model else 0.0) for name in predictions}
    return {
        "predictions": final_preds,
        "meta": {
            "method": "best_safe_model_selector",
            "trust_scores": safety_scores,
            "selected_model": selected_model,
            "weights": trust_scores,
        },
    }


def best_accuracy_selector(
    predictions: Dict[str, np.ndarray],
    validation_metrics: Dict[str, Dict[str, float]],
) -> Dict:
    scores = {
        name: _get_metric_value(metrics, "test_accuracy", "accuracy")
        for name, metrics in validation_metrics.items()
        if name in predictions
    }
    if not scores:
        scores = {name: 0.0 for name in predictions}
    selected_model = max(scores.items(), key=lambda item: item[1])[0]
    final_preds = np.asarray(predictions[selected_model], dtype=int)
    trust_scores = {name: float(scores.get(name, 0.0)) for name in predictions}
    return {
        "predictions": final_preds,
        "meta": {
            "method": "best_accuracy_selector",
            "selected_model": selected_model,
            "trust_scores": trust_scores,
            "weights": {name: (1.0 if name == selected_model else 0.0) for name in predictions},
        },
    }


def confidence_margin_trust(
    predictions: Dict[str, np.ndarray],
    probabilities: Optional[Dict[str, np.ndarray]],
    validation_metrics: Dict[str, Dict[str, float]],
) -> Dict:
    names = list(predictions.keys())
    n_samples = next(iter(predictions.values())).shape[0]
    model_scores = {
        name: _get_metric_value(validation_metrics.get(name, {}), "test_accuracy", "accuracy") for name in names
    }
    trust_scores = _normalize_trust_scores(model_scores)
    class_scores = np.zeros(n_samples, dtype=float)
    for name in names:
        conf = np.ones(n_samples, dtype=float)
        margin = np.zeros(n_samples, dtype=float)
        if probabilities and name in probabilities and probabilities[name] is not None:
            prob = np.asarray(probabilities[name], dtype=float)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                top_idx = np.argmax(prob, axis=1)
                top_prob = np.max(prob, axis=1)
                second_prob = np.partition(prob, -2, axis=1)[:, -2] if prob.shape[1] > 1 else np.zeros(n_samples)
                conf = top_prob
                margin = np.maximum(top_prob - second_prob, 0.0)
            else:
                conf = prob.ravel()
                margin = np.zeros_like(conf)
        score = model_scores[name] * conf * (1.0 + margin)
        class_scores += trust_scores[name] * score * predictions[name]
    preds = np.where(class_scores >= 0.5, 1, 0)
    return {
        "predictions": preds,
        "meta": {"method": "confidence_margin_trust", "trust_scores": trust_scores, "weights": trust_scores},
    }


def stacking_meta_trust(
    validation_meta_features: np.ndarray,
    validation_labels: np.ndarray,
    test_meta_features: np.ndarray,
    *,
    validation_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Dict:
    meta_model = LogisticRegression(max_iter=max_iter, solver="liblinear", random_state=random_state)
    meta_model.fit(np.asarray(validation_meta_features, dtype=float), np.asarray(validation_labels, dtype=int))
    preds = meta_model.predict(np.asarray(test_meta_features, dtype=float))
    trust_scores = {}
    if validation_metrics:
        trust_scores = {
            name: _get_metric_value(metrics, "test_accuracy", "accuracy")
            for name, metrics in validation_metrics.items()
        }
    return {
        "predictions": np.asarray(preds, dtype=int),
        "meta": {
            "method": "stacking_meta_trust",
            "trust_scores": trust_scores,
            "selected_model": "stacking_meta_model",
            "weights": trust_scores,
        },
    }


def local_accuracy_trust(
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_predictions: Dict[str, np.ndarray],
    test_features: np.ndarray,
    test_predictions: Dict[str, np.ndarray],
    *,
    candidate_k_values: Sequence[int] = (15, 25, 50),
) -> Dict:
    x_val = np.asarray(validation_features, dtype=float)
    x_test = np.asarray(test_features, dtype=float)
    y_val = np.asarray(validation_labels, dtype=int)
    names = list(validation_predictions.keys())

    validation_correctness = {
        name: np.asarray(validation_predictions[name], dtype=int) == y_val for name in names
    }

    def _predict_with_k(
        k_value: int,
        query_features: np.ndarray,
        query_predictions: Dict[str, np.ndarray],
        exclude_self: bool = False,
    ) -> np.ndarray:
        k_value = max(1, min(int(k_value), len(x_val)))
        neighbor_count = min(len(x_val), k_value + 1 if exclude_self else k_value)
        neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
        neighbors.fit(x_val)
        _, indices = neighbors.kneighbors(query_features)

        preds = np.zeros(query_features.shape[0], dtype=int)
        for row_idx, neighbor_indices in enumerate(indices):
            if exclude_self:
                neighbor_indices = [idx for idx in neighbor_indices if idx != row_idx][:k_value]
            local_scores = {}
            for name in names:
                local_scores[name] = float(np.mean(validation_correctness[name][neighbor_indices])) if len(neighbor_indices) > 0 else 0.0
            weighted_attack = 0.0
            total = 0.0
            for name in names:
                total += local_scores[name]
                weighted_attack += local_scores[name] * float(query_predictions[name][row_idx])
            if total <= 0.0:
                weighted_attack = float(sum(np.asarray(query_predictions[name], dtype=int).mean() for name in names) / len(names))
            else:
                weighted_attack /= total
            preds[row_idx] = 1 if weighted_attack >= 0.5 else 0
        return preds

    best_k = int(candidate_k_values[0])
    best_score = -1.0
    for k_value in candidate_k_values:
        candidate_preds = _predict_with_k(int(k_value), x_val, validation_predictions, exclude_self=True)
        candidate_acc = float(np.mean(candidate_preds == y_val))
        if candidate_acc > best_score:
            best_score = candidate_acc
            best_k = int(k_value)

    test_preds = _predict_with_k(best_k, x_test, test_predictions, exclude_self=False)
    trust_scores = {
        name: float(np.mean(validation_correctness[name])) for name in names
    }
    trust_scores = _normalize_trust_scores(trust_scores)
    return {
        "predictions": np.asarray(test_preds, dtype=int),
        "meta": {
            "method": "local_accuracy_trust",
            "selected_k": best_k,
            "trust_scores": trust_scores,
            "weights": trust_scores,
        },
    }


def _attack_probability_and_confidence(
    prediction: np.ndarray,
    probability: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=int)
    if probability is None:
        p_attack = pred.astype(float)
    else:
        prob = np.asarray(probability, dtype=float)
        if prob.ndim == 2 and prob.shape[1] >= 2:
            p_attack = prob[:, 1]
        else:
            p_attack = prob.ravel()
    p_attack = np.clip(p_attack, 0.0, 1.0)
    confidence = np.maximum(p_attack, 1.0 - p_attack)
    margin = np.abs(p_attack - 0.5) * 2.0
    return p_attack, confidence, margin


def trust_agent_selector(
    predictions: Dict[str, np.ndarray],
    probabilities: Optional[Dict[str, np.ndarray]],
    validation_metrics: Dict[str, Dict[str, float]],
    roles: Dict[str, str],
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_predictions: Dict[str, np.ndarray],
    test_features: np.ndarray,
    *,
    neighbor_k: int = 25,
    validation_role_weight: float = 0.25,
    confidence_weight: float = 0.20,
    margin_weight: float = 0.15,
    local_accuracy_weight: float = 0.10,
    disagreement_bonus: float = 0.10,
    attack_role_bonus: float = 0.08,
    normal_role_bonus: float = 0.08,
    attack_confidence_threshold: float = 0.60,
    normal_confidence_threshold: float = 0.65,
) -> Dict:
    names = list(predictions.keys())
    if not names:
        return {"predictions": np.array([], dtype=int), "meta": {"method": "trust_agent_selector"}}

    x_val = np.asarray(validation_features, dtype=float)
    y_val = np.asarray(validation_labels, dtype=int)
    x_test = np.asarray(test_features, dtype=float)

    k = max(1, min(int(neighbor_k), len(x_val)))
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(x_val)
    _, neighbor_indices = knn.kneighbors(x_test)

    local_accuracy = {}
    confidence_arrays = {}
    margin_arrays = {}
    for name in names:
        val_pred = np.asarray(validation_predictions[name], dtype=int)
        correct = val_pred == y_val
        local_accuracy[name] = np.array([float(np.mean(correct[idxs])) for idxs in neighbor_indices], dtype=float)
        p_attack, confidence, margin = _attack_probability_and_confidence(
            np.asarray(predictions[name], dtype=int),
            probabilities.get(name) if probabilities else None,
        )
        confidence_arrays[name] = confidence
        margin_arrays[name] = margin

    selected_agents: List[str] = []
    final_preds = np.zeros(x_test.shape[0], dtype=int)
    for sample_idx in range(x_test.shape[0]):
        sample_predictions = {name: int(np.asarray(predictions[name], dtype=int)[sample_idx]) for name in names}
        disagreement = len(set(sample_predictions.values())) > 1
        best_name = names[0]
        best_score = -1.0
        for name in names:
            metrics = validation_metrics.get(name, {})
            role = roles.get(name, "general")
            val_f1 = _get_metric_value(metrics, "test_f1", "f1")
            role_metric = val_f1
            if role == "attack_recall":
                role_metric = _get_metric_value(metrics, "test_recall", "recall")
            elif role == "normal_behavior":
                role_metric = _get_metric_value(metrics, "specificity")
            specialization_bonus = 0.0
            if disagreement and role == "hard_case":
                specialization_bonus += float(disagreement_bonus)
            if role == "attack_recall":
                if sample_predictions[name] == 1 and float(confidence_arrays[name][sample_idx]) >= float(attack_confidence_threshold):
                    specialization_bonus += float(attack_role_bonus)
            if role == "normal_behavior":
                if sample_predictions[name] == 0 and float(confidence_arrays[name][sample_idx]) >= float(normal_confidence_threshold):
                    specialization_bonus += float(normal_role_bonus)
            score = (
                0.30 * val_f1
                + float(validation_role_weight) * role_metric
                + float(confidence_weight) * float(confidence_arrays[name][sample_idx])
                + float(margin_weight) * float(margin_arrays[name][sample_idx])
                + float(local_accuracy_weight) * float(local_accuracy[name][sample_idx])
                + specialization_bonus
            )
            if score > best_score:
                best_score = score
                best_name = name
        selected_agents.append(best_name)
        final_preds[sample_idx] = int(np.asarray(predictions[best_name], dtype=int)[sample_idx])

    trust_scores = {
        name: float(_get_metric_value(validation_metrics.get(name, {}), "test_f1", "f1")) for name in names
    }
    return {
        "predictions": final_preds,
        "meta": {
            "method": "trust_agent_selector",
            "selected_agents": selected_agents,
            "trust_scores": trust_scores,
            "weights": _normalize_trust_scores(trust_scores),
        },
    }


def role_aware_trust_voting(
    predictions: Dict[str, np.ndarray],
    probabilities: Optional[Dict[str, np.ndarray]],
    validation_metrics: Dict[str, Dict[str, float]],
    roles: Dict[str, str],
    *,
    attack_threshold: float = 0.60,
    normal_threshold: float = 0.65,
    attack_bonus: float = 0.20,
    normal_bonus: float = 0.20,
    hard_case_disagreement_bonus: float = 0.25,
) -> Dict:
    names = list(predictions.keys())
    if not names:
        return {"predictions": np.array([], dtype=int), "meta": {"method": "role_aware_trust_voting"}}

    n_samples = next(iter(predictions.values())).shape[0]
    base_scores = {
        name: float(_get_metric_value(validation_metrics.get(name, {}), "test_f1", "f1")) for name in names
    }
    base_weights = _normalize_trust_scores(base_scores)

    role_to_name = {role: name for name, role in roles.items()}
    attack_agent = role_to_name.get("attack_recall")
    normal_agent = role_to_name.get("normal_behavior")
    hard_agent = role_to_name.get("hard_case")
    general_agent = role_to_name.get("general")

    final_preds = np.zeros(n_samples, dtype=int)
    for sample_idx in range(n_samples):
        sample_weights = dict(base_weights)
        sample_preds = {name: int(np.asarray(predictions[name], dtype=int)[sample_idx]) for name in names}
        disagreement = len(set(sample_preds.values())) > 1
        if disagreement and hard_agent in sample_weights:
            sample_weights[hard_agent] += float(hard_case_disagreement_bonus)

        if attack_agent is not None:
            attack_prob, attack_conf, _ = _attack_probability_and_confidence(
                np.asarray(predictions[attack_agent], dtype=int),
                probabilities.get(attack_agent) if probabilities else None,
            )
            if sample_preds[attack_agent] == 1 and float(attack_conf[sample_idx]) >= float(attack_threshold):
                sample_weights[attack_agent] += float(attack_bonus)

        if normal_agent is not None:
            normal_prob, normal_conf, _ = _attack_probability_and_confidence(
                np.asarray(predictions[normal_agent], dtype=int),
                probabilities.get(normal_agent) if probabilities else None,
            )
            majority_normal = sum(1 for value in sample_preds.values() if value == 0) >= (len(names) // 2 + 1)
            if majority_normal and sample_preds[normal_agent] == 0 and float(normal_conf[sample_idx]) >= float(normal_threshold):
                sample_weights[normal_agent] += float(normal_bonus)

        if general_agent is not None and not disagreement:
            sample_weights[general_agent] += 0.05

        norm_weights = _normalize_trust_scores(sample_weights)
        weighted_attack_vote = sum(norm_weights[name] * float(sample_preds[name] == 1) for name in names)
        final_preds[sample_idx] = 1 if weighted_attack_vote >= 0.5 else 0

    return {
        "predictions": final_preds,
        "meta": {
            "method": "role_aware_trust_voting",
            "trust_scores": base_scores,
            "weights": base_weights,
        },
    }
