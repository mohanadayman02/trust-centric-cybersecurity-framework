from pathlib import Path

import pandas as pd

from pipeline.ai_trust_auditor import AITrustConfig, select_method_with_ai
from pipeline.poisoning import run_poisoned_agent_experiments
from utils.thesis_table_export import export_thesis_tables


def _dummy_method_metrics():
    return {
        "Majority Vote": {"Accuracy": 0.90, "F1": 0.88, "FPR": 0.05, "FNR": 0.10, "Precision": 0.9, "Recall": 0.9, "Balanced Accuracy": 0.9},
        "Global Trust Voting": {"Accuracy": 0.91, "F1": 0.89, "FPR": 0.06, "FNR": 0.09, "Precision": 0.9, "Recall": 0.91, "Balanced Accuracy": 0.91},
        "Dynamic Trust": {"Accuracy": 0.93, "F1": 0.91, "FPR": 0.07, "FNR": 0.06, "Precision": 0.92, "Recall": 0.92, "Balanced Accuracy": 0.92},
        "Trust Agent Selector": {"Accuracy": 0.92, "F1": 0.90, "FPR": 0.05, "FNR": 0.08, "Precision": 0.91, "Recall": 0.91, "Balanced Accuracy": 0.91},
    }


def test_provider_unavailable_triggers_fallback(tmp_path):
    cfg = AITrustConfig(enabled=True, provider="nonexistent", fallback_method="Dynamic Trust")
    out = select_method_with_ai(
        dataset="NSL-KDD",
        scenario="Poisoned",
        poisoned_agent="General Traffic Agent",
        poison_mode="flip",
        poison_rate=0.3,
        allowed_methods=list(_dummy_method_metrics().keys()),
        method_metrics=_dummy_method_metrics(),
        method_predictions={"Dynamic Trust": 1},
        agreement_level=0.5,
        suspected_poisoned_agent="General Traffic Agent",
        config=cfg,
        cache_file=tmp_path / "cache.jsonl",
    )
    assert out["fallback_used"] is True
    assert out["selected_method"] == "Dynamic Trust"


def test_invalid_json_triggers_fallback(monkeypatch, tmp_path):
    from pipeline import ai_trust_auditor as mod

    def _bad_call(*args, **kwargs):
        return "not-json"

    monkeypatch.setattr(mod, "_call_provider", _bad_call)
    cfg = AITrustConfig(enabled=True, provider="openai", fallback_method="Dynamic Trust")
    out = select_method_with_ai(
        dataset="NSL-KDD",
        scenario="Clean",
        poisoned_agent="None",
        poison_mode="none",
        poison_rate=0.0,
        allowed_methods=list(_dummy_method_metrics().keys()),
        method_metrics=_dummy_method_metrics(),
        method_predictions={"Dynamic Trust": 1},
        agreement_level=0.5,
        suspected_poisoned_agent=None,
        config=cfg,
        cache_file=tmp_path / "cache.jsonl",
    )
    assert out["fallback_used"] is True


def test_selected_method_must_be_allowed(monkeypatch, tmp_path):
    from pipeline import ai_trust_auditor as mod

    def _bad_choice(*args, **kwargs):
        return '{"selected_method":"Invalid","selected_prediction":1,"suspected_unreliable_agents":[],"confidence":0.5,"reason":"x"}'

    monkeypatch.setattr(mod, "_call_provider", _bad_choice)
    cfg = AITrustConfig(enabled=True, provider="openai", fallback_method="Dynamic Trust")
    out = select_method_with_ai(
        dataset="NSL-KDD",
        scenario="Clean",
        poisoned_agent="None",
        poison_mode="none",
        poison_rate=0.0,
        allowed_methods=list(_dummy_method_metrics().keys()),
        method_metrics=_dummy_method_metrics(),
        method_predictions={"Dynamic Trust": 1},
        agreement_level=0.5,
        suspected_poisoned_agent=None,
        config=cfg,
        cache_file=tmp_path / "cache.jsonl",
    )
    assert out["fallback_used"] is True


def test_context_excludes_raw_features(monkeypatch, tmp_path):
    from pipeline import ai_trust_auditor as mod

    captured = {}

    def _capture(provider, *, model, timeout, system_prompt, user_payload):
        captured["payload"] = user_payload
        return '{"selected_method":"Dynamic Trust","selected_prediction":1,"suspected_unreliable_agents":[],"confidence":0.8,"reason":"ok"}'

    monkeypatch.setattr(mod, "_call_provider", _capture)
    cfg = AITrustConfig(enabled=True, provider="openai", fallback_method="Dynamic Trust")
    select_method_with_ai(
        dataset="NSL-KDD",
        scenario="Clean",
        poisoned_agent="None",
        poison_mode="none",
        poison_rate=0.0,
        allowed_methods=list(_dummy_method_metrics().keys()),
        method_metrics=_dummy_method_metrics(),
        method_predictions={"Dynamic Trust": 1},
        agreement_level=0.5,
        suspected_poisoned_agent=None,
        config=cfg,
        cache_file=tmp_path / "cache.jsonl",
    )
    payload = captured["payload"]
    assert "raw_features" not in payload
    assert "method_metrics" in payload


def test_cache_reuse_avoids_second_provider_call(monkeypatch, tmp_path):
    from pipeline import ai_trust_auditor as mod

    calls = {"n": 0}

    def _good_call(*args, **kwargs):
        calls["n"] += 1
        return '{"selected_method":"Dynamic Trust","selected_prediction":1,"suspected_unreliable_agents":[],"confidence":0.9,"reason":"ok"}'

    monkeypatch.setattr(mod, "_call_provider", _good_call)
    cfg = AITrustConfig(enabled=True, provider="openai", fallback_method="Dynamic Trust")
    kwargs = dict(
        dataset="NSL-KDD",
        scenario="Clean",
        poisoned_agent="None",
        poison_mode="none",
        poison_rate=0.0,
        allowed_methods=list(_dummy_method_metrics().keys()),
        method_metrics=_dummy_method_metrics(),
        method_predictions={"Dynamic Trust": 1},
        agreement_level=0.5,
        suspected_poisoned_agent=None,
        config=cfg,
        cache_file=tmp_path / "cache.jsonl",
    )
    out1 = select_method_with_ai(**kwargs)
    out2 = select_method_with_ai(**kwargs)
    assert calls["n"] == 1
    assert out1["selected_method"] == out2["selected_method"]


def test_ollama_provider_parses_json_with_think_tags(monkeypatch, tmp_path):
    from pipeline import ai_trust_auditor as mod

    def _ollama_like(*args, **kwargs):
        return "<think>internal reasoning</think>{\"selected_method\":\"Dynamic Trust\",\"selected_prediction\":1,\"suspected_unreliable_agents\":[\"Attack Recall Agent\"],\"confidence\":0.77,\"reason\":\"low fnr\"}"

    monkeypatch.setattr(mod, "_call_provider", _ollama_like)
    cfg = AITrustConfig(enabled=True, provider="ollama", model="deepseek-r1:14b", fallback_method="Dynamic Trust")
    out = select_method_with_ai(
        dataset="NSL-KDD",
        scenario="Poisoned",
        poisoned_agent="General Traffic Agent",
        poison_mode="flip",
        poison_rate=0.3,
        allowed_methods=list(_dummy_method_metrics().keys()),
        method_metrics=_dummy_method_metrics(),
        method_predictions={"Dynamic Trust": 1},
        agreement_level=0.5,
        suspected_poisoned_agent="General Traffic Agent",
        config=cfg,
        cache_file=tmp_path / "cache.jsonl",
    )
    assert out["fallback_used"] is False
    assert out["selected_method"] == "Dynamic Trust"
    assert out["suspected_unreliable_agents"] == ["Attack Recall Agent"]


def test_ai_row_only_when_enabled():
    import numpy as np

    model_preds = {
        "General Traffic Agent": np.array([0, 1, 0, 1], dtype=int),
        "Attack Recall Agent": np.array([1, 1, 0, 1], dtype=int),
        "Normal Behavior Agent": np.array([0, 0, 0, 1], dtype=int),
        "Hard-Case Agent": np.array([1, 1, 1, 1], dtype=int),
    }
    model_probs = {k: np.array([[0.3, 0.7]] * 4) for k in model_preds}
    metrics = {k: {"test_f1": 0.8, "test_accuracy": 0.8, "test_recall": 0.8, "specificity": 0.8, "fnr": 0.2} for k in model_preds}
    roles = {
        "General Traffic Agent": "general",
        "Attack Recall Agent": "attack_recall",
        "Normal Behavior Agent": "normal_behavior",
        "Hard-Case Agent": "hard_case",
    }
    x = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0, 1, 0, 1], dtype=int)
    val_preds = {k: v.copy() for k, v in model_preds.items()}
    selector_params = {
        "neighbor_k": 2,
        "validation_role_weight": 0.25,
        "confidence_weight": 0.2,
        "margin_weight": 0.15,
        "local_accuracy_weight": 0.1,
        "disagreement_bonus": 0.1,
        "attack_role_bonus": 0.08,
        "normal_role_bonus": 0.08,
        "attack_confidence_threshold": 0.6,
        "normal_confidence_threshold": 0.65,
    }

    out_disabled = run_poisoned_agent_experiments(
        dataset_name="DummyDS",
        y_test=y,
        model_preds=model_preds,
        model_probs=model_probs,
        validation_model_metrics=metrics,
        roles=roles,
        x_val_full=x,
        y_val=y,
        validation_predictions=val_preds,
        x_test_full=x,
        role_aware_cfg={"attack_threshold": 0.6, "normal_threshold": 0.65},
        selector_params=selector_params,
        return_artifacts=False,
    )
    assert "AI Trust Auditor" not in set(out_disabled["Evaluation Type"]) 


def test_thesis_export_includes_ai_when_present(tmp_path):
    # Smoke test: exporter still runs and can include AI if present in source files.
    # Here we only verify no regression in export flow.
    results_dir = Path(__file__).resolve().parent.parent / "results"
    written, _, _ = export_thesis_tables(results_dir)
    assert any(p.name.endswith("_clean_trust_methods.csv") for p in written)
