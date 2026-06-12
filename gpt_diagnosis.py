"""GPT diagnosis wrapper for the Dewu Decision Content Agent MVP.

Python remains responsible for deterministic metric calculation. This module
only converts computed summaries into a business diagnosis.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


DEFAULT_MODEL = "gpt-5.4-mini"
FALLBACK_MODELS = ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-4.1-mini", "gpt-4o-mini"]


def _get_streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st

        value = st.secrets.get(name)
        return str(value) if value else None
    except Exception:
        return None


def get_openai_api_key() -> str | None:
    """Read API key from Streamlit secrets first, then environment variables."""

    return _get_streamlit_secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def has_openai_api_key() -> bool:
    return bool(get_openai_api_key())


def _round_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _json_friendly_table(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        small = value.head(20).copy()
        numeric_cols = small.select_dtypes(include=["number"]).columns
        small[numeric_cols] = small[numeric_cols].round(4)
        return small.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return {_round_value(k): _round_value(v) for k, v in value.head(20).to_dict().items()}
    if isinstance(value, dict):
        return {str(k): _json_friendly_table(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_friendly_table(item) for item in value[:20]]
    return _round_value(value)


def _summarize_result_tables(result_tables: dict) -> dict:
    return {str(name): _json_friendly_table(table) for name, table in result_tables.items()}


def _fallback_response(
    *,
    model: str,
    rule_based_conclusion: str,
    rule_based_suggestions: list,
    fallback_reason: str,
) -> dict:
    return {
        "enabled": False,
        "model": model,
        "diagnosis": rule_based_conclusion,
        "key_findings": [
            "当前展示的是 Python 规则版诊断，未使用 GPT 生成动态解释。",
            "所有指标和表格均来自本地模拟数据与确定性计算。",
        ],
        "next_steps": list(rule_based_suggestions or [])[:3],
        "risk_notes": [
            "本项目使用模拟数据，不能代表得物真实业务表现。",
            "当前结论用于 Demo 展示，相关性不能直接解释为因果关系。",
            "正式上线前需要用真实业务数据和 A/B 实验验证诊断有效性。",
        ],
        "fallback_reason": fallback_reason,
    }


def _candidate_models() -> list[str]:
    configured = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    candidates = [configured] + FALLBACK_MODELS
    deduped: list[str] = []
    for model in candidates:
        if model and model not in deduped:
            deduped.append(model)
    return deduped


def _safe_parse_json(text: str, rule_based_suggestions: list) -> dict:
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("GPT output is not a JSON object")
        return parsed
    except Exception:
        return {
            "diagnosis": text,
            "key_findings": ["GPT 返回内容不是严格 JSON，已保留原始文本作为诊断。"],
            "next_steps": list(rule_based_suggestions or [])[:3],
            "risk_notes": [
                "GPT 输出解析失败，页面已自动 fallback，未影响 Python 指标计算。",
                "本项目使用模拟数据，不能代表真实业务数据。",
            ],
        }


def generate_gpt_diagnosis(
    user_question: str,
    route_result: dict,
    analysis_name: str,
    analysis_steps: list,
    result_tables: dict,
    rule_based_conclusion: str,
    rule_based_suggestions: list,
) -> dict:
    """Generate a GPT diagnosis from Python-computed analysis summaries."""

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    api_key = get_openai_api_key()
    if not api_key:
        return _fallback_response(
            model=model,
            rule_based_conclusion=rule_based_conclusion,
            rule_based_suggestions=rule_based_suggestions,
            fallback_reason="未检测到 OPENAI_API_KEY，使用规则版诊断。",
        )

    try:
        from openai import OpenAI
    except Exception as exc:
        return _fallback_response(
            model=model,
            rule_based_conclusion=rule_based_conclusion,
            rule_based_suggestions=rule_based_suggestions,
            fallback_reason=f"OpenAI SDK 不可用：{exc}",
        )

    payload = {
        "user_question": user_question,
        "route_result": route_result,
        "analysis_name": analysis_name,
        "analysis_steps": analysis_steps,
        "python_result_tables_summary": _summarize_result_tables(result_tables),
        "rule_based_conclusion": rule_based_conclusion,
        "rule_based_suggestions": rule_based_suggestions,
    }

    system_prompt = (
        "你是一个 AI 数据产品中的分析解释模块。你不能编造数据，不能声称使用了未提供的数据，"
        "不能把相关性说成因果。你的任务是基于 Python 已经计算好的结果，给得物内容运营/"
        "品类运营提供动态诊断。GPT 不参与指标计算，只解释 Python 计算结果。"
    )
    user_prompt = (
        "请基于以下结构化输入生成诊断。必须返回 JSON，不要返回 Markdown。\n"
        "JSON 字段：diagnosis（150-250 字中文诊断）、key_findings（3 条以内）、"
        "next_steps（3 条以内）、risk_notes（3 条以内）。\n\n"
        + json.dumps(payload, ensure_ascii=False, default=str)
    )

    try:
        client = OpenAI(api_key=api_key)
    except Exception as exc:
        return _fallback_response(
            model=model,
            rule_based_conclusion=rule_based_conclusion,
            rule_based_suggestions=rule_based_suggestions,
            fallback_reason=f"OpenAI client 初始化失败，已回退规则版诊断：{exc}",
        )

    errors: list[str] = []
    for candidate_model in _candidate_models():
        try:
            response = client.responses.create(
                model=candidate_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_text = getattr(response, "output_text", "") or str(response)
            parsed = _safe_parse_json(raw_text, rule_based_suggestions)
            return {
                "enabled": True,
                "model": candidate_model,
                "diagnosis": parsed.get("diagnosis", raw_text),
                "key_findings": parsed.get("key_findings", []),
                "next_steps": parsed.get("next_steps", list(rule_based_suggestions or [])[:3]),
                "risk_notes": parsed.get(
                    "risk_notes",
                    ["本项目使用模拟数据，不能代表得物真实业务表现。"],
                ),
                "fallback_reason": None,
            }
        except Exception as exc:
            errors.append(f"{candidate_model}: {exc}")
            text = str(exc).lower()
            if "model_not_found" in text or "does not exist" in text or "requested model" in text:
                continue
            return _fallback_response(
                model=candidate_model,
                rule_based_conclusion=rule_based_conclusion,
                rule_based_suggestions=rule_based_suggestions,
                fallback_reason=f"GPT 调用失败，已回退规则版诊断：{exc}",
            )

    return _fallback_response(
        model=model,
        rule_based_conclusion=rule_based_conclusion,
        rule_based_suggestions=rule_based_suggestions,
        fallback_reason="所有候选模型均不可用，已回退规则版诊断：" + " | ".join(errors[-3:]),
    )
