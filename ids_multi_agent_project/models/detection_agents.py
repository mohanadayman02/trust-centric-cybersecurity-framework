"""Standardized detection-model wrappers for IDS models (decision sources).

Deprecated: the project previously used the term 'agent' in some modules. These wrappers
remain for compatibility but are referred to as models/decision sources in the trust-centric
framework.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC


def normalize_traffic_svm_params(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Normalize and validate SVC params for TrafficAnalysisAgent."""
    model_params = dict(params or {})
    model_params["kernel"] = str(model_params.get("kernel", "rbf")).lower()
    model_params.setdefault("C", 1.0)
    model_params.setdefault("gamma", "scale")
    model_params["probability"] = bool(model_params.get("probability", True))
    model_params.setdefault("random_state", 42)
    return model_params


@dataclass
class AgentDecision:
    """Structured per-sample decision emitted by a detection agent."""

    agent_name: str
    backbone_model: str
    role: str
    predicted_label: int
    predicted_label_name: str
    confidence: float
    probability_normal: float
    probability_attack: float
    reasoning: str
    stance: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseDetectionAgent:
    """Base wrapper exposing a unified fit/predict interface."""

    def __init__(
        self,
        agent_name: str,
        role: str,
        backbone_model: str,
        model: Any,
        reasoning_config: Dict[str, Any] | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.role = role
        self.backbone_model = backbone_model
        self.model = model

        config = dict(reasoning_config or {})
        self.ollama_enabled = bool(config.get("ollama_enabled", False))
        self.ollama_model_name = str(config.get("ollama_model_name", "llama3.1:8b"))
        self.ollama_base_url = str(config.get("ollama_base_url", "http://localhost:11434"))
        self.ollama_timeout_seconds = int(config.get("ollama_timeout_seconds", 15))

    def fit(self, x_train, y_train) -> "BaseDetectionAgent":
        """Fit underlying sklearn model."""
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x) -> Dict[str, Any]:
        """Return standardized prediction payload."""
        y_pred = self.model.predict(x)
        y_prob = self._predict_attack_probability(x)

        return {
            "agent_name": self.agent_name,
            "role": self.role,
            "backbone_model": self.backbone_model,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "supports_proba": y_prob is not None,
        }

    def predict_decisions(self, x, sample_indices: Iterable[Any]) -> List[Dict[str, Any]]:
        """Return structured decision objects for each input sample."""
        prediction_output = self.predict(x)
        y_pred = np.asarray(prediction_output["y_pred"])
        y_prob = prediction_output.get("y_prob")

        if y_prob is None:
            prob_attack = np.where(y_pred == 1, 1.0, 0.0).astype(float)
        else:
            prob_attack = np.asarray(y_prob, dtype=float)

        sample_list = list(sample_indices)
        rows: List[Dict[str, Any]] = []
        for idx, pred, p_attack in zip(sample_list, y_pred, prob_attack):
            p_attack_val = float(np.clip(p_attack, 0.0, 1.0))
            p_normal_val = float(1.0 - p_attack_val)
            confidence = float(max(p_normal_val, p_attack_val))
            predicted_label = int(pred)
            predicted_label_name = "attack" if predicted_label == 1 else "normal"
            decision = AgentDecision(
                agent_name=self.agent_name,
                backbone_model=self.backbone_model,
                role=self.role,
                predicted_label=predicted_label,
                predicted_label_name=predicted_label_name,
                confidence=confidence,
                probability_normal=p_normal_val,
                probability_attack=p_attack_val,
                reasoning=self._default_reasoning_summary(predicted_label, confidence),
                stance=self._stance_from_prediction(predicted_label, confidence),
            )
            row = decision.to_dict()
            row["sample_index"] = idx
            rows.append(row)
        return rows

    @staticmethod
    def _extract_json_dict(text: str) -> Dict[str, Any]:
        """Parse a JSON dict from text, allowing small wrapper noise."""
        stripped = (text or "").strip()
        if not stripped:
            raise ValueError("Empty reasoning response")

        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        left = stripped.find("{")
        right = stripped.rfind("}")
        if left == -1 or right == -1 or left >= right:
            raise ValueError("No JSON object found in reasoning response")

        payload = json.loads(stripped[left : right + 1])
        if not isinstance(payload, dict):
            raise ValueError("Reasoning response is not a JSON object")
        return payload

    def _call_ollama_reasoning(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama and return parsed JSON reasoning payload."""
        endpoint = self.ollama_base_url.rstrip("/") + "/api/generate"
        request_payload = {
            "model": self.ollama_model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        data = json.dumps(request_payload).encode("utf-8")
        req = urllib_request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=self.ollama_timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        model_output = str(body.get("response", ""))
        return self._extract_json_dict(model_output)

    def _reasoning_style(self) -> str:
        """Return concise reasoning instruction for this role."""
        return "Use role-aligned IDS reasoning grounded in model probabilities."

    def _default_reasoning_summary(self, predicted_label: int, confidence: float) -> str:
        """Return deterministic fallback reasoning when Ollama is skipped/unavailable."""
        label_name = "attack" if predicted_label == 1 else "normal"
        if predicted_label == 1:
            return (
                f"{self.agent_name} identifies an attack-like profile from the processed features "
                f"(confidence={confidence:.3f}, label={label_name})."
            )
        return (
            f"{self.agent_name} identifies a normal-like profile from the processed features "
            f"(confidence={confidence:.3f}, label={label_name})."
        )

    @staticmethod
    def _stance_from_prediction(predicted_label: int, confidence: float) -> str:
        """Map prediction + confidence to a compact stance label."""
        if predicted_label == 1 and confidence >= 0.85:
            return "strong_attack"
        if predicted_label == 1 and confidence >= 0.65:
            return "attack_leaning"
        if predicted_label == 1:
            return "weak_attack"
        if confidence >= 0.85:
            return "strong_normal"
        if confidence >= 0.65:
            return "normal_leaning"
        return "weak_normal"

    def _predict_attack_probability(self, x):
        """Return probability of class 1 (attack) when available."""
        if not hasattr(self.model, "predict_proba"):
            return None

        proba = self.model.predict_proba(x)
        classes = list(self.model.classes_)

        if 1 in classes:
            attack_col_index = classes.index(1)
            return proba[:, attack_col_index]

        return None

    def _build_reasoning_prompt(
        self,
        dataset_name: str,
        prediction: int,
        probability_attack: float | None,
        agreement_flag: str,
        agent_metrics: Dict[str, Any] | None,
    ) -> str:
        """Build deterministic compact reasoning prompt."""
        metrics = dict(agent_metrics or {})
        probability_text = (
            f"{float(probability_attack):.6f}"
            if probability_attack is not None and np.isfinite(probability_attack)
            else "nan"
        )
        metric_f1 = metrics.get("test_f1")
        metric_fpr = metrics.get("fpr")
        metric_fnr = metrics.get("fnr")

        return (
            "Return ONLY valid JSON with keys: reasoning_summary, confidence_band, "
            "risk_note, recommended_attention.\n"
            "Use short thesis-friendly text and deterministic style.\n"
            f"agent_name={self.agent_name}; role={self.role}; backbone={self.backbone_model}; "
            f"dataset={dataset_name}; prediction={prediction}; "
            f"probability_attack={probability_text}; agreement={agreement_flag}; "
            f"test_f1={metric_f1}; fpr={metric_fpr}; fnr={metric_fnr}.\n"
            f"Role guidance: {self._reasoning_style()}\n"
            "Do not claim extra sensors, payload inspection, or capabilities not present in tabular features.\n"
            "confidence_band must be one of: low, medium, high.\n"
            "recommended_attention must be one of: low, medium, high."
        )

    def reason_about_prediction(
        self,
        dataset_name: str,
        prediction: int,
        probability_attack: float | None,
        agreement_flag: str = "unknown",
        agent_metrics: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Generate structured reasoning for one prediction."""
        fallback_conf = float(max(1.0 - (probability_attack or 0.0), probability_attack or 0.0))
        base = {
            "reasoning_summary": self._default_reasoning_summary(prediction, fallback_conf),
            "confidence_band": "medium",
            "risk_note": "",
            "recommended_attention": "medium",
            "reasoning_status": "reasoning_skipped",
            "ollama_model": self.ollama_model_name,
        }

        if not self.ollama_enabled:
            base["reasoning_summary"] = (
                f"{base['reasoning_summary']} Ollama reasoning disabled by configuration."
            )
            return base

        prompt = self._build_reasoning_prompt(
            dataset_name=dataset_name,
            prediction=prediction,
            probability_attack=probability_attack,
            agreement_flag=agreement_flag,
            agent_metrics=agent_metrics,
        )
        try:
            payload = self._call_ollama_reasoning(prompt)
        except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, OSError):
            base["reasoning_status"] = "ollama_unavailable"
            base["reasoning_summary"] = (
                f"{base['reasoning_summary']} Ollama endpoint unavailable; numeric prediction kept."
            )
            return base
        except Exception:  # pylint: disable=broad-except
            base["reasoning_status"] = "parsing_failed"
            base["reasoning_summary"] = (
                f"{base['reasoning_summary']} Ollama response parsing failed; numeric prediction kept."
            )
            return base

        base.update(
            {
                "reasoning_summary": str(
                    payload.get("reasoning_summary", "Structured reasoning generated.")
                ),
                "confidence_band": str(payload.get("confidence_band", "medium")).lower(),
                "risk_note": str(payload.get("risk_note", "")),
                "recommended_attention": str(
                    payload.get("recommended_attention", "medium")
                ).lower(),
                "reasoning_status": "success",
            }
        )
        return base

    def generate_reasoning(
        self,
        dataset_name: str,
        sample_indices: Iterable[Any],
        predictions: Iterable[Any],
        probabilities: Iterable[Any] | None,
        agreement_flags: Iterable[Any] | None = None,
        agent_metrics: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate reasoning rows for a list of samples."""
        indices_list = list(sample_indices)
        predictions_list = list(predictions)
        if probabilities is None:
            probabilities_list = [np.nan] * len(indices_list)
        else:
            probabilities_list = list(probabilities)
        if agreement_flags is None:
            agreement_list = ["unknown"] * len(indices_list)
        else:
            agreement_list = list(agreement_flags)

        rows: List[Dict[str, Any]] = []
        for idx, pred, prob, agree in zip(
            indices_list, predictions_list, probabilities_list, agreement_list
        ):
            prob_value = float(prob) if prob is not None and np.isfinite(prob) else None
            reasoning = self.reason_about_prediction(
                dataset_name=dataset_name,
                prediction=int(pred),
                probability_attack=prob_value,
                agreement_flag=str(agree),
                agent_metrics=agent_metrics,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "agent": self.agent_name,
                    "role": self.role,
                    "backbone_model": self.backbone_model,
                    "sample_index": idx,
                    "prediction": int(pred),
                    "probability_attack": prob_value,
                    "reasoning_summary": reasoning["reasoning_summary"],
                    "confidence_band": reasoning["confidence_band"],
                    "risk_note": reasoning["risk_note"],
                    "recommended_attention": reasoning["recommended_attention"],
                    "reasoning_status": reasoning["reasoning_status"],
                    "ollama_model": reasoning["ollama_model"],
                }
            )
        return rows


class BehavioralAnalysisAgent(BaseDetectionAgent):
    """Behavior-oriented IDS agent backed by RandomForestClassifier."""

    def __init__(
        self,
        params: Dict[str, Any] | None = None,
        reasoning_config: Dict[str, Any] | None = None,
    ) -> None:
        model_params = dict(params or {})
        model_params.setdefault("random_state", 42)
        super().__init__(
            agent_name="BehavioralAnalysisAgent",
            role="behavioral_analysis",
            backbone_model="RandomForestClassifier",
            model=RandomForestClassifier(**model_params),
            reasoning_config=reasoning_config,
        )

    def _reasoning_style(self) -> str:
        return (
            "Emphasize suspicious behavioral patterns, unusual feature combinations, "
            "and anomaly-like structure in the processed tabular feature space."
        )


class TrafficAnalysisAgent(BaseDetectionAgent):
    """Traffic-evidence IDS agent backed by SVC."""

    def __init__(
        self,
        params: Dict[str, Any] | None = None,
        reasoning_config: Dict[str, Any] | None = None,
    ) -> None:
        model_params = normalize_traffic_svm_params(params)
        super().__init__(
            agent_name="TrafficAnalysisAgent",
            role="traffic_analysis",
            backbone_model="SVC",
            model=SVC(**model_params),
            reasoning_config=reasoning_config,
        )

    def _reasoning_style(self) -> str:
        return (
            "Emphasize direct traffic-feature evidence and margin-based separation behavior "
            "from an SVM decision boundary in the processed tabular feature space."
        )


# Backward-compatible aliases
RandomForestAgent = BehavioralAnalysisAgent
SVMAgent = TrafficAnalysisAgent
LogisticRegressionAgent = TrafficAnalysisAgent


class AutoencoderAgent:
    """Autoencoder-like feature reducer implemented with a shallow MLP regressor."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        model_params = dict(params or {})
        self.encoding_dim = int(model_params.get("encoding_dim", 16))
        self.activation = str(model_params.get("activation", "tanh"))
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.encoding_dim,),
            activation=self.activation,
            solver=str(model_params.get("solver", "adam")),
            alpha=float(model_params.get("alpha", 0.001)),
            learning_rate_init=float(model_params.get("learning_rate_init", 0.0003)),
            max_iter=int(model_params.get("max_iter", 80)),
            random_state=int(model_params.get("random_state", 42)),
            early_stopping=bool(model_params.get("early_stopping", True)),
        )

    @staticmethod
    def _activate(values: np.ndarray, activation: str) -> np.ndarray:
        name = str(activation).lower()
        if name == "identity":
            return values
        if name == "tanh":
            return np.tanh(values)
        if name == "logistic":
            return 1.0 / (1.0 + np.exp(-values))
        # Default to ReLU to mirror sklearn MLP behavior.
        return np.maximum(values, 0.0)

    def fit(self, x_train, y_train=None) -> "AutoencoderAgent":
        x_arr = np.asarray(x_train, dtype=np.float64)
        x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=0.0, neginf=0.0)
        self.model.fit(x_arr, x_arr)
        return self

    def transform(self, x):
        x_arr = np.asarray(x, dtype=np.float64)
        x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=0.0, neginf=0.0)
        hidden_linear = np.dot(x_arr, self.model.coefs_[0]) + self.model.intercepts_[0]
        hidden_linear = np.clip(hidden_linear, -1e6, 1e6)
        encoded = self._activate(hidden_linear, self.model.activation)
        encoded = np.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(encoded, dtype=np.float64)

    def fit_transform(self, x_train):
        self.fit(x_train)
        return self.transform(x_train)


class MLPAgent(BaseDetectionAgent):
    """MLP classifier agent for encoded feature-space intrusion detection."""

    def __init__(
        self,
        params: Dict[str, Any] | None = None,
        reasoning_config: Dict[str, Any] | None = None,
    ) -> None:
        model_params = dict(params or {})
        model_params.setdefault("random_state", 42)
        model_params.setdefault("max_iter", 300)
        model_params.setdefault("hidden_layer_sizes", (64, 32))
        model_params.setdefault("activation", "tanh")
        model_params.setdefault("alpha", 0.001)
        model_params.setdefault("learning_rate_init", 0.0003)
        model_params.setdefault("early_stopping", True)
        super().__init__(
            agent_name="MLPAgent",
            role="mlp_classification",
            backbone_model="MLPClassifier",
            model=MLPClassifier(**model_params),
            reasoning_config=reasoning_config,
        )


class KNNAgent(BaseDetectionAgent):
    """KNN classifier agent for encoded feature-space intrusion detection."""

    def __init__(
        self,
        params: Dict[str, Any] | None = None,
        reasoning_config: Dict[str, Any] | None = None,
    ) -> None:
        model_params = dict(params or {})
        model_params.setdefault("n_neighbors", 5)
        model_params.setdefault("weights", "distance")
        super().__init__(
            agent_name="KNNAgent",
            role="knn_classification",
            backbone_model="KNeighborsClassifier",
            model=KNeighborsClassifier(**model_params),
            reasoning_config=reasoning_config,
        )
