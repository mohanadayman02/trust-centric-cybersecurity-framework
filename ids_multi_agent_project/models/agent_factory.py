"""Factory module for creating IDS model wrappers (decision sources).

This module preserves backward-compatible names (agent_factory) but exposes
model/decision-source creation for the trust-centric framework.
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from models.detection_agents import (
    AutoencoderAgent,
    BehavioralAnalysisAgent,
    KNNAgent,
    MLPAgent,
    TrafficAnalysisAgent,
    normalize_traffic_svm_params,
)


def create_agent(model_name: str, params: Dict[str, Any] | None = None):
    """Create and return an sklearn classifier based on the given model name."""
    supported_models = {
        "Autoencoder": AutoencoderAgent,
        "AutoencoderAgent": AutoencoderAgent,
        "MLP": MLPClassifier,
        "MLPAgent": MLPClassifier,
        "KNN": KNeighborsClassifier,
        "KNNAgent": KNeighborsClassifier,
        "RandomForest": RandomForestClassifier,
        "SVC": SVC,
        "SVM": SVC,
        "DecisionTree": DecisionTreeClassifier,
    }

    if model_name not in supported_models:
        supported = ", ".join(sorted(supported_models.keys()))
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models: {supported}"
        )

    model_class = supported_models[model_name]
    model_params = dict(params or {})

    if model_name in {"SVC", "SVM"}:
        model_params = normalize_traffic_svm_params(model_params)

    if model_name in {"MLP", "MLPAgent"}:
        model_params.setdefault("random_state", 42)
        model_params.setdefault("max_iter", 300)
        model_params.setdefault("hidden_layer_sizes", (64, 32))

    if model_name in {"KNN", "KNNAgent"}:
        model_params.setdefault("n_neighbors", 5)
        model_params.setdefault("weights", "distance")

    return model_class(**model_params)


def create_detection_agent(
    model_name: str,
    params: Dict[str, Any] | None = None,
    reasoning_config: Dict[str, Any] | None = None,
):
    """Create standardized role-based detection-agent wrappers for inference outputs."""
    model_params = dict(params or {})

    detection_agents = {
        "Autoencoder": AutoencoderAgent,
        "AutoencoderAgent": AutoencoderAgent,
        "MLP": MLPAgent,
        "MLPAgent": MLPAgent,
        "KNN": KNNAgent,
        "KNNAgent": KNNAgent,
        "RandomForest": BehavioralAnalysisAgent,
        "SVC": TrafficAnalysisAgent,
        "SVM": TrafficAnalysisAgent,
        "BehavioralAnalysisAgent": BehavioralAnalysisAgent,
        "TrafficAnalysisAgent": TrafficAnalysisAgent,
    }

    if model_name not in detection_agents:
        supported = ", ".join(sorted(detection_agents.keys()))
        raise ValueError(
            f"Unsupported detection agent '{model_name}'. Supported models: {supported}"
        )

    if model_name in {"Autoencoder", "AutoencoderAgent"}:
        return detection_agents[model_name](model_params)

    return detection_agents[model_name](model_params, reasoning_config=reasoning_config)
