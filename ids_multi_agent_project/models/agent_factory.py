"""Factory module for creating IDS agent models."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from models.detection_agents import (
    BehavioralAnalysisAgent,
    TrafficAnalysisAgent,
    normalize_traffic_svm_params,
)


def create_agent(model_name: str, params: Dict[str, Any] | None = None):
    """Create and return an sklearn classifier based on the given model name."""
    supported_models = {
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

    return model_class(**model_params)


def create_detection_agent(
    model_name: str,
    params: Dict[str, Any] | None = None,
    reasoning_config: Dict[str, Any] | None = None,
):
    """Create standardized role-based detection-agent wrappers for inference outputs."""
    model_params = dict(params or {})

    detection_agents = {
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

    return detection_agents[model_name](model_params, reasoning_config=reasoning_config)
