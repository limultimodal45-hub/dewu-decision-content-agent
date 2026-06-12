"""Strategy sandbox for creator incentives and user trust.

This module is intentionally deterministic and explainable. It is not an A/B
test, recommendation forecast, or GMV prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dewu_agent import apply_filters


STRATEGIES = ["真实体验型", "模板化真实体验型", "同质化好评型", "穿搭种草型"]

STRATEGY_FROM_CONTENT_TYPE = {
    "真实体验": "真实体验型",
    "价格判断": "真实体验型",
    "对比测评": "真实体验型",
    "疑似软广": "模板化真实体验型",
    "同质化好评": "同质化好评型",
    "穿搭展示": "穿搭种草型",
}

TEXT_SIGNAL_SCORE = {
    "真实体验型": 0.85,
    "模板化真实体验型": 0.78,
    "同质化好评型": 0.25,
    "穿搭种草型": 0.40,
}

PRODUCTION_COST = {
    "真实体验型": 0.22,
    "模板化真实体验型": 0.12,
    "同质化好评型": 0.05,
    "穿搭种草型": 0.14,
}

TEMPLATE_RISK = {
    "真实体验型": 0.10,
    "模板化真实体验型": 0.85,
    "同质化好评型": 0.20,
    "穿搭种草型": 0.15,
}

DEFAULT_GAME_PARAMS = {
    "w_shallow": 0.15,
    "w_text_signal": 0.15,
    "w_product_click": 0.25,
    "w_decision_depth": 0.20,
    "w_conversion": 0.20,
    "w_return_penalty": 0.20,
    "exploration_ratio": 0.10,
    "adaptation_speed": 0.35,
    "template_arbitrage_tendency": 0.35,
    "user_trust_sensitivity": 0.35,
    "rounds": 8,
}

PRESET_PARAMS = {
    "浅层互动优先": {
        **DEFAULT_GAME_PARAMS,
        "w_shallow": 0.45,
        "w_text_signal": 0.20,
        "w_product_click": 0.12,
        "w_decision_depth": 0.08,
        "w_conversion": 0.10,
        "w_return_penalty": 0.08,
        "exploration_ratio": 0.05,
        "template_arbitrage_tendency": 0.45,
    },
    "文本真实感优先": {
        **DEFAULT_GAME_PARAMS,
        "w_shallow": 0.18,
        "w_text_signal": 0.45,
        "w_product_click": 0.20,
        "w_decision_depth": 0.12,
        "w_conversion": 0.12,
        "w_return_penalty": 0.16,
        "template_arbitrage_tendency": 0.65,
    },
    "行为验证优先": {
        **DEFAULT_GAME_PARAMS,
        "w_shallow": 0.08,
        "w_text_signal": 0.18,
        "w_product_click": 0.42,
        "w_decision_depth": 0.38,
        "w_conversion": 0.35,
        "w_return_penalty": 0.42,
        "exploration_ratio": 0.18,
        "template_arbitrage_tendency": 0.18,
        "user_trust_sensitivity": 0.45,
        "adaptation_speed": 0.42,
    },
    "平衡策略": DEFAULT_GAME_PARAMS,
}


def _normalize(values: pd.Series, default: float = 0.5) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if values.dropna().empty:
        return pd.Series(default, index=values.index)
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(default, index=values.index)
    return ((values - min_value) / (max_value - min_value)).fillna(default).clip(0, 1)


def _strategy_for_content_type(content_type: str) -> str:
    return STRATEGY_FROM_CONTENT_TYPE.get(str(content_type), "穿搭种草型")


def _filtered_df(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    filtered = apply_filters(df, filters)
    if isinstance(filtered, tuple):
        return filtered[0].copy()
    if isinstance(filtered, dict):
        candidate = filtered.get("df_filtered")
        if candidate is None:
            candidate = filtered.get("data")
        if isinstance(candidate, pd.DataFrame):
            return candidate.copy()
        return df.copy()
    return filtered.copy()


def prepare_strategy_metrics(df: pd.DataFrame, filters: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    df_filtered = _filtered_df(df, filters)
    df_filtered["strategy"] = df_filtered["content_type"].map(_strategy_for_content_type)

    raw_counts = df_filtered["strategy"].value_counts().reindex(STRATEGIES, fill_value=0).astype(float)
    shares = raw_counts / max(raw_counts.sum(), 1)
    shares = shares.mask(shares == 0, 0.02)
    shares = shares / shares.sum()

    grouped = df_filtered.groupby("strategy")
    metrics = pd.DataFrame(index=STRATEGIES)
    metrics["strategy"] = STRATEGIES
    metrics["initial_share"] = shares.reindex(STRATEGIES).values
    metrics["exposure_raw"] = grouped["exposure"].mean().reindex(STRATEGIES)

    shallow_cols = [col for col in ["like_rate", "favorite_rate", "comment_rate"] if col in df_filtered.columns]
    if shallow_cols:
        df_filtered["_shallow_raw"] = df_filtered[shallow_cols].mean(axis=1)
        metrics["shallow_raw"] = df_filtered.groupby("strategy")["_shallow_raw"].mean().reindex(STRATEGIES)
    else:
        metrics["shallow_raw"] = 0.5

    metrics["product_click_raw"] = grouped["product_card_click_rate"].mean().reindex(STRATEGIES)

    depth_cols = [col for col in ["price_view_rate", "review_view_rate", "product_detail_uv_rate"] if col in df_filtered.columns]
    if depth_cols:
        df_filtered["_depth_raw"] = df_filtered[depth_cols].mean(axis=1)
        metrics["decision_depth_raw"] = df_filtered.groupby("strategy")["_depth_raw"].mean().reindex(STRATEGIES)
    elif "product_detail_uv_rate" in df_filtered.columns:
        metrics["decision_depth_raw"] = grouped["product_detail_uv_rate"].mean().reindex(STRATEGIES)
    else:
        metrics["decision_depth_raw"] = 0.5

    metrics["conversion_raw"] = grouped["purchase_conversion_rate"].mean().reindex(STRATEGIES)
    metrics["return_risk_raw"] = grouped["cancel_return_rate"].mean().reindex(STRATEGIES)

    metrics["exposure_score"] = _normalize(metrics["exposure_raw"])
    metrics["shallow_score"] = _normalize(metrics["shallow_raw"])
    metrics["product_click_score"] = _normalize(metrics["product_click_raw"])
    metrics["decision_depth_score"] = _normalize(metrics["decision_depth_raw"])
    metrics["conversion_score"] = _normalize(metrics["conversion_raw"])
    metrics["return_risk_score"] = _normalize(metrics["return_risk_raw"])
    metrics["text_signal_score"] = metrics["strategy"].map(TEXT_SIGNAL_SCORE)
    metrics["production_cost"] = metrics["strategy"].map(PRODUCTION_COST)
    metrics["template_risk"] = metrics["strategy"].map(TEMPLATE_RISK)

    score_cols = [
        "exposure_score",
        "shallow_score",
        "text_signal_score",
        "product_click_score",
        "decision_depth_score",
        "conversion_score",
        "return_risk_score",
        "production_cost",
        "template_risk",
    ]
    metrics[score_cols] = metrics[score_cols].fillna(0.5)
    return metrics.reset_index(drop=True), shares


def calculate_strategy_payoffs(strategy_metrics: pd.DataFrame, shares: pd.Series, params: dict) -> pd.Series:
    metrics = strategy_metrics.set_index("strategy")
    shares = shares.reindex(STRATEGIES).fillna(0.01)

    behavior_quality = (
        0.35 * metrics["product_click_score"]
        + 0.30 * metrics["decision_depth_score"]
        + 0.25 * metrics["conversion_score"]
        + 0.10 * (1 - metrics["return_risk_score"])
    )

    exploration_bonus = (
        params["exploration_ratio"]
        * metrics["product_click_score"]
        * metrics["decision_depth_score"]
        * (1 - shares)
    )

    trust_gap = (metrics["text_signal_score"] - behavior_quality).clip(lower=0)
    trust_penalty = 0.20 * params["user_trust_sensitivity"] * trust_gap * metrics["template_risk"]
    template_strategy = "模板化真实体验型"
    if template_strategy in trust_penalty.index:
        trust_penalty.loc[template_strategy] = (
            params["user_trust_sensitivity"]
            * params["template_arbitrage_tendency"]
            * trust_gap.loc[template_strategy]
            * (1 + shares.loc[template_strategy])
        )

    payoffs = (
        0.10
        + params["w_shallow"] * metrics["shallow_score"]
        + params["w_text_signal"] * metrics["text_signal_score"]
        + params["w_product_click"] * metrics["product_click_score"]
        + params["w_decision_depth"] * metrics["decision_depth_score"]
        + params["w_conversion"] * metrics["conversion_score"]
        + exploration_bonus
        - params["w_return_penalty"] * metrics["return_risk_score"]
        - 0.15 * metrics["production_cost"]
        - trust_penalty
    )

    min_payoff = payoffs.min()
    if min_payoff < 0.05:
        payoffs = payoffs + (0.05 - min_payoff)
    return payoffs.clip(lower=0.05)


def replicator_update(shares: pd.Series, payoffs: pd.Series, adaptation_speed: float = 0.35) -> pd.Series:
    shares = shares.reindex(STRATEGIES).fillna(0.01)
    payoffs = payoffs.reindex(STRATEGIES).fillna(0.05)
    denominator = float((shares * payoffs).sum())
    if denominator <= 0:
        replicator_share = shares
    else:
        replicator_share = shares * payoffs / denominator
    next_share = shares * (1 - adaptation_speed) + replicator_share * adaptation_speed
    next_share = next_share.clip(lower=0.01)
    return next_share / next_share.sum()


def _weighted(shares: pd.Series, metrics: pd.DataFrame, column: str) -> float:
    values = metrics.set_index("strategy")[column].reindex(STRATEGIES).fillna(0.0)
    return float((shares.reindex(STRATEGIES).fillna(0.0) * values).sum())


def calculate_ecosystem_metrics(shares: pd.Series, strategy_metrics: pd.DataFrame, params: dict) -> dict:
    shares = shares.reindex(STRATEGIES).fillna(0.0)
    metrics = strategy_metrics.set_index("strategy")

    exposure_mismatch_rate = (
        shares["同质化好评型"] * 0.60
        + shares["模板化真实体验型"] * (0.35 + 0.25 * params["template_arbitrage_tendency"])
        + shares["穿搭种草型"] * 0.25
        + shares["真实体验型"] * 0.10
    )
    exposure_mismatch_rate -= 0.08 * params["w_return_penalty"] + 0.06 * params["w_decision_depth"]
    exposure_mismatch_rate = float(np.clip(exposure_mismatch_rate, 0, 1))

    trust_index = 0.0
    for strategy in STRATEGIES:
        trust_index += shares[strategy] * (
            0.30 * metrics.loc[strategy, "product_click_score"]
            + 0.30 * metrics.loc[strategy, "decision_depth_score"]
            + 0.25 * metrics.loc[strategy, "conversion_score"]
            + 0.15 * (1 - metrics.loc[strategy, "return_risk_score"])
        )
    trust_index -= shares["模板化真实体验型"] * params["template_arbitrage_tendency"] * params["user_trust_sensitivity"] * 0.15
    trust_index = float(np.clip(trust_index, 0, 1))

    search_cost_index = float(np.clip(1 - trust_index + exposure_mismatch_rate * 0.20, 0, 1))
    conversion_weighted_score = _weighted(shares, strategy_metrics, "conversion_score")
    decision_depth_weighted_score = _weighted(shares, strategy_metrics, "decision_depth_score")
    platform_payoff = (
        0.35 * trust_index
        + 0.25 * conversion_weighted_score
        + 0.20 * decision_depth_weighted_score
        - 0.20 * exposure_mismatch_rate
    )

    verified_sample_share = (
        shares["真实体验型"] * 0.45
        + shares["穿搭种草型"] * 0.15
        + shares["模板化真实体验型"] * max(0.05, 0.20 - params["template_arbitrage_tendency"] * 0.10)
    )
    buried_decision_share = shares["真实体验型"] * max(0.10, 0.35 - params["exploration_ratio"]) + shares["模板化真实体验型"] * 0.10
    overheated_mismatch_share = exposure_mismatch_rate
    low_priority_share = 1 - verified_sample_share - buried_decision_share - overheated_mismatch_share

    quadrant_values = pd.Series(
        {
            "已验证样板内容": verified_sample_share,
            "被埋没的决策内容": buried_decision_share,
            "虚热内容 / 曝光资源错配": overheated_mismatch_share,
            "低优先级观察内容": low_priority_share,
        }
    ).clip(lower=0)
    quadrant_values = quadrant_values / quadrant_values.sum()

    return {
        "exposure_mismatch_rate": exposure_mismatch_rate,
        "trust_index": trust_index,
        "search_cost_index": search_cost_index,
        "platform_payoff": float(platform_payoff),
        **{f"quadrant__{name}": float(value) for name, value in quadrant_values.items()},
    }


def simulate_content_game(df: pd.DataFrame, filters: dict | None = None, params: dict | None = None) -> dict:
    params = {**DEFAULT_GAME_PARAMS, **(params or {})}
    strategy_metrics, initial_share = prepare_strategy_metrics(df, filters)
    shares = initial_share.copy()

    strategy_rows = []
    payoff_rows = []
    ecosystem_rows = []
    quadrant_rows = []

    for round_idx in range(int(params["rounds"]) + 1):
        payoffs = calculate_strategy_payoffs(strategy_metrics, shares, params)
        ecosystem = calculate_ecosystem_metrics(shares, strategy_metrics, params)

        for strategy in STRATEGIES:
            strategy_rows.append({"round": round_idx, "strategy": strategy, "share": float(shares[strategy])})
            payoff_rows.append({"round": round_idx, "strategy": strategy, "payoff": float(payoffs[strategy]), "share": float(shares[strategy])})

        ecosystem_rows.append(
            {
                "round": round_idx,
                "exposure_mismatch_rate": ecosystem["exposure_mismatch_rate"],
                "search_cost_index": ecosystem["search_cost_index"],
                "trust_index": ecosystem["trust_index"],
                "platform_payoff": ecosystem["platform_payoff"],
            }
        )
        for key, value in ecosystem.items():
            if key.startswith("quadrant__"):
                quadrant_rows.append({"round": round_idx, "quadrant": key.replace("quadrant__", ""), "share": float(value)})

        if round_idx < int(params["rounds"]):
            shares = replicator_update(shares, payoffs, params["adaptation_speed"])

    return {
        "strategy_history": pd.DataFrame(strategy_rows),
        "payoff_history": pd.DataFrame(payoff_rows),
        "ecosystem_history": pd.DataFrame(ecosystem_rows),
        "quadrant_history": pd.DataFrame(quadrant_rows),
        "strategy_metrics": strategy_metrics,
        "params": params,
    }
