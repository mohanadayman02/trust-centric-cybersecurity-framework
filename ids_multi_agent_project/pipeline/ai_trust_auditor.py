"""Optional AI-assisted trust auditor for method-level scenario selection.

This module never receives raw dataset feature vectors.
It only consumes structured method-level outputs/metrics and poisoning context.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request, error


DEFAULT_SYSTEM_PROMPT = (
    "You are an AI trust auditor for an intrusion detection ensemble. "
    "You do not classify raw network traffic. "
    "You only evaluate structured agent/method outputs. "
    "Return valid JSON only."
)


@dataclass
class AITrustConfig:
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-5.4-mini"
    sample_limit: int = 200
    timeout: float = 8.0
    fallback_method: str = "Dynamic Trust"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def build_ai_trust_config(options: Optional[Dict[str, Any]] = None) -> AITrustConfig:
    opts = options or {}
    enabled = bool(opts.get("enable_ai_trust", _env_bool("AI_TRUST_ENABLED", False)))
    provider = str(opts.get("ai_trust_provider") or os.getenv("AI_TRUST_PROVIDER", "openai"))
    model = str(opts.get("ai_trust_model") or os.getenv("AI_TRUST_MODEL", "gpt-5.4-mini"))
    sample_limit = int(opts.get("ai_trust_sample_limit") or os.getenv("AI_TRUST_SAMPLE_LIMIT", 200))
    timeout = float(opts.get("ai_trust_timeout") or os.getenv("AI_TRUST_TIMEOUT", 8.0))
    fallback_method = str(opts.get("ai_trust_fallback_method") or os.getenv("AI_TRUST_FALLBACK_METHOD", "Dynamic Trust"))
    return AITrustConfig(
        enabled=enabled,
        provider=provider,
        model=model,
        sample_limit=sample_limit,
        timeout=timeout,
        fallback_method=fallback_method,
    )


def _stable_key(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_cache(cache_file: Path) -> Dict[str, Dict[str, Any]]:
    if not cache_file.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for line in cache_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = str(row.get("key", ""))
            if key:
                out[key] = row.get("value", {})
        except Exception:
            continue
    return out


def _append_cache(cache_file: Path, key: str, value: Dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=True) + "\n")


def _extract_json_object(text: str) -> Dict[str, Any]:
    candidate = str(text or "").strip()
    # Some reasoning models prepend hidden reasoning sections in plain text.
    candidate = re.sub(r"<think>.*?</think>", "", candidate, flags=re.DOTALL | re.IGNORECASE).strip()
    if not candidate:
        raise ValueError("Empty AI response")
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found")
    return json.loads(candidate[start : end + 1])


def _call_openai_provider(*, model: str, timeout: float, system_prompt: str, user_payload: Dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    body = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(user_payload, ensure_ascii=True)}],
            },
        ],
    }

    req = request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"OpenAI HTTP error: {detail}") from exc

    payload = json.loads(raw)
    out_text = payload.get("output_text")
    if out_text:
        return str(out_text)

    chunks: List[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            txt = content.get("text")
            if txt:
                chunks.append(str(txt))
    return "\n".join(chunks).strip()


def _call_provider(provider: str, *, model: str, timeout: float, system_prompt: str, user_payload: Dict[str, Any]) -> str:
    if provider.lower() == "openai":
        return _call_openai_provider(model=model, timeout=timeout, system_prompt=system_prompt, user_payload=user_payload)
    if provider.lower() == "ollama":
        body = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ],
        }
        req = request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"Ollama call failed: {exc}") from exc
        payload = json.loads(raw)
        message = payload.get("message", {}) or {}
        content = message.get("content", "")
        return str(content)
    raise RuntimeError(f"Unsupported AI trust provider: {provider}")


def _fallback_decision(
    *,
    fallback_method: str,
    allowed_methods: List[str],
    method_metrics: Dict[str, Dict[str, float]],
    reason: str,
    status: str,
) -> Dict[str, Any]:
    chosen = fallback_method if fallback_method in allowed_methods else (allowed_methods[0] if allowed_methods else "")
    m = method_metrics.get(chosen, {})
    return {
        "selected_method": chosen,
        "selected_prediction": None,
        "suspected_unreliable_agents": [],
        "confidence": 0.0,
        "reason": reason,
        "status": status,
        "fallback_used": True,
        "selected_accuracy": m.get("Accuracy"),
        "selected_f1": m.get("F1"),
        "selected_fpr": m.get("FPR"),
        "selected_fnr": m.get("FNR"),
    }


def select_method_with_ai(
    *,
    dataset: str,
    scenario: str,
    poisoned_agent: str,
    poison_mode: str,
    poison_rate: float,
    allowed_methods: List[str],
    method_metrics: Dict[str, Dict[str, float]],
    method_predictions: Optional[Dict[str, int]],
    agreement_level: Optional[float],
    suspected_poisoned_agent: Optional[str],
    config: AITrustConfig,
    cache_file: Path,
) -> Dict[str, Any]:
    print(f"[AI_TRUST_START] dataset={dataset} scenario={scenario} poisoned_agent={poisoned_agent}")

    if not config.enabled:
        return _fallback_decision(
            fallback_method=config.fallback_method,
            allowed_methods=allowed_methods,
            method_metrics=method_metrics,
            reason="AI trust disabled",
            status="disabled",
        )

    context = {
        "dataset": dataset,
        "scenario": scenario,
        "poisoned_agent": poisoned_agent,
        "poison_mode": poison_mode,
        "poison_rate": poison_rate,
        "allowed_methods": allowed_methods,
        "method_metrics": method_metrics,
        "method_predictions": method_predictions or {},
        "agreement_level": agreement_level,
        "suspected_poisoned_agent": suspected_poisoned_agent,
        "instruction": (
            "Select the safest method emphasizing low FNR, high F1, and controlled FPR. "
            "Return JSON with selected_method, selected_prediction (0/1 or null), "
            "suspected_unreliable_agents, confidence (0..1), reason."
        ),
    }
    print("[AI_TRUST_CONTEXT_BUILT] keys=" + ",".join(sorted(context.keys())))

    cache_key = _stable_key(context)
    cached = _load_cache(cache_file)
    if cache_key in cached:
        cached_value = dict(cached[cache_key])
        cached_value.setdefault("status", "cached")
        cached_value.setdefault("fallback_used", False)
        m = method_metrics.get(cached_value.get("selected_method", ""), {})
        cached_value.setdefault("selected_accuracy", m.get("Accuracy"))
        cached_value.setdefault("selected_f1", m.get("F1"))
        cached_value.setdefault("selected_fpr", m.get("FPR"))
        cached_value.setdefault("selected_fnr", m.get("FNR"))
        print(f"[AI_TRUST_DONE] dataset={dataset} scenario={scenario} status=cached")
        return cached_value

    if not allowed_methods:
        return _fallback_decision(
            fallback_method=config.fallback_method,
            allowed_methods=allowed_methods,
            method_metrics=method_metrics,
            reason="No allowed methods provided",
            status="fallback_no_methods",
        )

    try:
        print(f"[AI_TRUST_PROVIDER_CALL] provider={config.provider} model={config.model}")
        raw = _call_provider(
            config.provider,
            model=config.model,
            timeout=config.timeout,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            user_payload=context,
        )
        parsed = _extract_json_object(raw)
        selected_method = str(parsed.get("selected_method", "")).strip()
        if selected_method not in allowed_methods:
            raise ValueError(f"Invalid selected_method: {selected_method}")

        selected_prediction = parsed.get("selected_prediction", None)
        if selected_prediction not in (0, 1, None):
            selected_prediction = None

        decision = {
            "selected_method": selected_method,
            "selected_prediction": selected_prediction,
            "suspected_unreliable_agents": parsed.get("suspected_unreliable_agents", []),
            "confidence": float(parsed.get("confidence", 0.0)),
            "reason": str(parsed.get("reason", ""))[:240],
            "status": "ok",
            "fallback_used": False,
        }
        m = method_metrics.get(selected_method, {})
        decision["selected_accuracy"] = m.get("Accuracy")
        decision["selected_f1"] = m.get("F1")
        decision["selected_fpr"] = m.get("FPR")
        decision["selected_fnr"] = m.get("FNR")

        _append_cache(cache_file, cache_key, decision)
        print(f"[AI_TRUST_RESULT] method={selected_method} confidence={decision['confidence']:.4f}")
        print(f"[AI_TRUST_DONE] dataset={dataset} scenario={scenario} status=ok")
        return decision
    except Exception as exc:
        print(f"[AI_TRUST_FALLBACK] reason={exc}")
        decision = _fallback_decision(
            fallback_method=config.fallback_method,
            allowed_methods=allowed_methods,
            method_metrics=method_metrics,
            reason=f"AI fallback: {exc}",
            status="fallback",
        )
        print(f"[AI_TRUST_DONE] dataset={dataset} scenario={scenario} status=fallback")
        return decision
