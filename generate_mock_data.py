"""Generate mock Dewu sneaker community content data for the MVP."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "dewu_decision_content_mock.csv"
RANDOM_SEED = 42
ROW_COUNT = 500


BRANDS = ["Nike", "Adidas", "New Balance", "Asics", "Puma", "Jordan", "Converse", "Vans"]
CATEGORIES = ["潮鞋"]
CONTENT_TYPES = ["同质化好评", "真实体验", "价格判断", "对比测评", "穿搭展示", "疑似软广"]


TYPE_WEIGHTS = {
    "同质化好评": 0.25,
    "真实体验": 0.22,
    "价格判断": 0.16,
    "对比测评": 0.14,
    "穿搭展示": 0.17,
    "疑似软广": 0.06,
}


TYPE_PROFILES = {
    # High exposure / likes, low decision depth.
    "同质化好评": {
        "exposure": (18000, 8000),
        "like_rate": (0.075, 0.02),
        "favorite_rate": (0.032, 0.01),
        "comment_rate": (0.018, 0.008),
        "cliche": (0.72, 0.12),
        "signals": {
            "negative": 0.08,
            "size": 0.14,
            "price": 0.09,
            "comparison": 0.06,
            "scenario": 0.28,
        },
    },
    # Lower hype, stronger decision value.
    "真实体验": {
        "exposure": (12500, 5200),
        "like_rate": (0.058, 0.018),
        "favorite_rate": (0.041, 0.012),
        "comment_rate": (0.041, 0.012),
        "cliche": (0.24, 0.10),
        "signals": {
            "negative": 0.62,
            "size": 0.78,
            "price": 0.35,
            "comparison": 0.28,
            "scenario": 0.66,
        },
    },
    "价格判断": {
        "exposure": (11000, 4800),
        "like_rate": (0.049, 0.016),
        "favorite_rate": (0.035, 0.011),
        "comment_rate": (0.039, 0.012),
        "cliche": (0.18, 0.08),
        "signals": {
            "negative": 0.32,
            "size": 0.25,
            "price": 0.88,
            "comparison": 0.48,
            "scenario": 0.38,
        },
    },
    "对比测评": {
        "exposure": (11800, 5200),
        "like_rate": (0.053, 0.018),
        "favorite_rate": (0.045, 0.014),
        "comment_rate": (0.044, 0.013),
        "cliche": (0.16, 0.08),
        "signals": {
            "negative": 0.54,
            "size": 0.46,
            "price": 0.50,
            "comparison": 0.92,
            "scenario": 0.50,
        },
    },
    # Good for atmosphere, medium transaction value.
    "穿搭展示": {
        "exposure": (16500, 6800),
        "like_rate": (0.083, 0.024),
        "favorite_rate": (0.050, 0.016),
        "comment_rate": (0.025, 0.009),
        "cliche": (0.42, 0.13),
        "signals": {
            "negative": 0.10,
            "size": 0.20,
            "price": 0.12,
            "comparison": 0.14,
            "scenario": 0.82,
        },
    },
    "疑似软广": {
        "exposure": (21000, 9000),
        "like_rate": (0.068, 0.022),
        "favorite_rate": (0.026, 0.01),
        "comment_rate": (0.014, 0.006),
        "cliche": (0.82, 0.10),
        "signals": {
            "negative": 0.04,
            "size": 0.08,
            "price": 0.06,
            "comparison": 0.05,
            "scenario": 0.18,
        },
    },
}


def clipped_normal(rng: np.random.Generator, mean: float, sd: float, low: float, high: float) -> float:
    return float(np.clip(rng.normal(mean, sd), low, high))


def bool_by_probability(rng: np.random.Generator, probability: float) -> int:
    return int(rng.random() < probability)


def build_row(rng: np.random.Generator, idx: int) -> dict:
    content_type = rng.choice(CONTENT_TYPES, p=[TYPE_WEIGHTS[t] for t in CONTENT_TYPES])
    profile = TYPE_PROFILES[content_type]
    signals = profile["signals"]

    has_negative_detail = bool_by_probability(rng, signals["negative"])
    has_size_feedback = bool_by_probability(rng, signals["size"])
    has_price_judgment = bool_by_probability(rng, signals["price"])
    has_comparison = bool_by_probability(rng, signals["comparison"])
    has_usage_scenario = bool_by_probability(rng, signals["scenario"])
    positive_cliche_ratio = clipped_normal(rng, *profile["cliche"], 0.02, 0.98)

    signal_count = (
        has_negative_detail
        + has_size_feedback
        + has_price_judgment
        + has_comparison
        + has_usage_scenario
    )

    decision_info_score = np.clip(
        34
        + 9.2 * signal_count
        + 6 * has_negative_detail
        + 5 * has_size_feedback
        + 5 * has_price_judgment
        + 5 * has_comparison
        - 23 * positive_cliche_ratio
        + rng.normal(0, 5),
        5,
        98,
    )
    effective_interaction_score = np.clip(
        31
        + 240 * clipped_normal(rng, *profile["comment_rate"], 0.002, 0.09)
        + 18 * has_negative_detail
        + 10 * has_comparison
        - 10 * positive_cliche_ratio
        + rng.normal(0, 6),
        5,
        98,
    )
    decision_path_entry_score = np.clip(
        30
        + 7.5 * signal_count
        + 10 * has_price_judgment
        + 8 * has_comparison
        + 5 * has_size_feedback
        - 18 * positive_cliche_ratio
        + rng.normal(0, 5),
        5,
        98,
    )
    transaction_feedback_score = np.clip(
        36
        + 6 * signal_count
        + 10 * has_size_feedback
        + 9 * has_negative_detail
        + 8 * has_price_judgment
        - 15 * positive_cliche_ratio
        + rng.normal(0, 5),
        5,
        98,
    )
    author_credibility_score = np.clip(
        43
        + 5 * signal_count
        + (8 if content_type in {"真实体验", "对比测评"} else 0)
        - (12 if content_type == "疑似软广" else 0)
        - 7 * positive_cliche_ratio
        + rng.normal(0, 8),
        5,
        98,
    )

    decision_value_score = (
        0.25 * decision_info_score
        + 0.20 * effective_interaction_score
        + 0.25 * decision_path_entry_score
        + 0.20 * transaction_feedback_score
        + 0.10 * author_credibility_score
    )

    exposure = int(max(800, rng.normal(*profile["exposure"])))
    like_rate = clipped_normal(rng, *profile["like_rate"], 0.005, 0.16)
    favorite_rate = clipped_normal(rng, *profile["favorite_rate"], 0.003, 0.11)
    comment_rate = clipped_normal(rng, *profile["comment_rate"], 0.002, 0.10)

    decision_factor = decision_value_score / 100
    product_card_click_rate = np.clip(
        0.012 + 0.058 * decision_factor + 0.012 * has_price_judgment + 0.009 * has_comparison
        - 0.015 * positive_cliche_ratio + rng.normal(0, 0.006),
        0.003,
        0.12,
    )
    product_detail_uv_rate = np.clip(
        product_card_click_rate * rng.normal(0.70, 0.10) + 0.006 * has_size_feedback,
        0.002,
        0.10,
    )
    price_view_rate = np.clip(
        0.010 + 0.052 * decision_factor + 0.030 * has_price_judgment - 0.012 * positive_cliche_ratio
        + rng.normal(0, 0.006),
        0.002,
        0.13,
    )
    review_view_rate = np.clip(
        0.012 + 0.050 * decision_factor + 0.020 * has_negative_detail + 0.014 * has_size_feedback
        - 0.010 * positive_cliche_ratio + rng.normal(0, 0.006),
        0.002,
        0.13,
    )
    effective_comment_rate = np.clip(
        comment_rate * (0.35 + 0.45 * decision_factor + 0.10 * has_negative_detail),
        0.001,
        0.09,
    )
    purchase_conversion_rate = np.clip(
        0.004 + 0.040 * decision_factor + 0.012 * has_price_judgment + 0.007 * has_size_feedback
        - 0.016 * positive_cliche_ratio + rng.normal(0, 0.004),
        0.001,
        0.085,
    )
    cancel_return_rate = np.clip(
        0.105 - 0.045 * decision_factor - 0.014 * has_size_feedback - 0.010 * has_negative_detail
        + 0.020 * positive_cliche_ratio + rng.normal(0, 0.008),
        0.015,
        0.16,
    )

    publish_date = pd.Timestamp("2026-06-11") - pd.Timedelta(days=int(rng.integers(0, 56)))

    return {
        "content_id": f"C{idx:05d}",
        "publish_date": publish_date.date().isoformat(),
        "category": "潮鞋",
        "brand": rng.choice(BRANDS),
        "product_id": f"SKU{rng.integers(1000, 9999)}",
        "content_type": content_type,
        "author_id": f"A{rng.integers(10000, 99999)}",
        "has_negative_detail": has_negative_detail,
        "has_size_feedback": has_size_feedback,
        "has_price_judgment": has_price_judgment,
        "has_comparison": has_comparison,
        "has_usage_scenario": has_usage_scenario,
        "positive_cliche_ratio": round(positive_cliche_ratio, 4),
        "decision_info_score": round(float(decision_info_score), 2),
        "effective_interaction_score": round(float(effective_interaction_score), 2),
        "decision_path_entry_score": round(float(decision_path_entry_score), 2),
        "transaction_feedback_score": round(float(transaction_feedback_score), 2),
        "author_credibility_score": round(float(author_credibility_score), 2),
        "decision_value_score": round(float(decision_value_score), 2),
        "exposure": exposure,
        "like_rate": round(float(like_rate), 4),
        "favorite_rate": round(float(favorite_rate), 4),
        "comment_rate": round(float(comment_rate), 4),
        "effective_comment_rate": round(float(effective_comment_rate), 4),
        "product_card_click_rate": round(float(product_card_click_rate), 4),
        "product_detail_uv_rate": round(float(product_detail_uv_rate), 4),
        "price_view_rate": round(float(price_view_rate), 4),
        "review_view_rate": round(float(review_view_rate), 4),
        "purchase_conversion_rate": round(float(purchase_conversion_rate), 4),
        "cancel_return_rate": round(float(cancel_return_rate), 4),
    }


def generate_mock_data(row_count: int = ROW_COUNT) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = [build_row(rng, idx + 1) for idx in range(row_count)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_mock_data()
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Generated {len(df)} rows -> {OUTPUT_PATH}")
