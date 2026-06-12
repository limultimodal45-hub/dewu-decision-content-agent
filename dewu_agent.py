"""Rule-based DataAgent functions for Dewu Decision Content Agent MVP."""

from __future__ import annotations

from copy import deepcopy

import pandas as pd


CONTENT_TYPES = ["真实体验", "同质化好评", "疑似软广", "价格判断", "对比测评", "穿搭展示"]
TIME_TERMS = ["最近7天", "近7天", "上周", "本周", "本月"]
METRIC_KEYWORDS = ["曝光", "商品点击", "点商品", "商品卡", "详情页", "转化", "退货", "取消", "价格查看", "评论查看"]


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _score_keywords(question: str, keywords: list[str]) -> list[str]:
    normalized = question.lower()
    return [keyword for keyword in keywords if keyword.lower() in normalized]


def extract_filters(question: str, df: pd.DataFrame) -> dict:
    """Extract simple business entities from a natural-language question."""

    text = question or ""
    text_lower = text.lower()

    brand = None
    for candidate in sorted(df["brand"].dropna().unique(), key=len, reverse=True):
        if str(candidate).lower() in text_lower:
            brand = str(candidate)
            break

    category = "潮鞋" if "潮鞋" in text else "潮鞋"

    content_type = None
    for candidate in CONTENT_TYPES:
        if candidate in text:
            content_type = candidate
            break

    time_range = None
    for candidate in TIME_TERMS:
        if candidate in text:
            time_range = candidate
            break

    metric_terms = [keyword for keyword in METRIC_KEYWORDS if keyword in text]

    return {
        "brand": brand,
        "category": category,
        "content_type": content_type,
        "time_range": time_range,
        "metric_terms": metric_terms,
    }


def apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """Apply entity filters and attach a warning in DataFrame attrs if needed."""

    if not filters:
        result = df.copy()
        result.attrs["filter_warning"] = None
        return result

    original_count = len(df)
    filtered = df.copy()
    warnings: list[str] = []

    if filters.get("brand"):
        filtered = filtered[filtered["brand"] == filters["brand"]]
    if filters.get("category"):
        filtered = filtered[filtered["category"] == filters["category"]]
    if filters.get("content_type"):
        filtered = filtered[filtered["content_type"] == filters["content_type"]]

    time_range = filters.get("time_range")
    if time_range in {"最近7天", "近7天"}:
        dates = pd.to_datetime(filtered["publish_date"], errors="coerce")
        max_date = dates.max()
        if pd.notna(max_date):
            filtered = filtered[dates >= max_date - pd.Timedelta(days=7)]
            warnings.append("当前模拟数据使用 publish_date 最大日期往前 7 天做近似时间筛选。")
    elif time_range:
        warnings.append(f"当前模拟数据仅对“最近7天/近7天”做近似时间筛选，未强制筛选“{time_range}”。")

    if filtered.empty:
        fallback = df.copy()
        fallback.attrs["filter_warning"] = "筛选后数据为空，已回退到原始样本进行分析。"
        fallback.attrs["original_row_count"] = original_count
        fallback.attrs["filtered_row_count"] = original_count
        return fallback

    filtered = filtered.copy()
    filtered.attrs["filter_warning"] = "；".join(warnings) if warnings else None
    filtered.attrs["original_row_count"] = original_count
    filtered.attrs["filtered_row_count"] = len(filtered)
    return filtered


def _filter_context(df: pd.DataFrame, filters: dict | None, *, ignore_content_type: bool = False) -> tuple[pd.DataFrame, dict, str | None]:
    used_filters = deepcopy(filters or {})
    warning_parts: list[str] = []
    if ignore_content_type and used_filters.get("content_type"):
        warning_parts.append("对比分析需要保留两组内容类型，已忽略单一 content_type 筛选。")
        used_filters["content_type"] = None
    filtered = apply_filters(df, used_filters)
    if filtered.attrs.get("filter_warning"):
        warning_parts.append(filtered.attrs["filter_warning"])
    warning = "；".join(warning_parts) if warning_parts else None
    return filtered, used_filters, warning


def route_question(question: str, df: pd.DataFrame | None = None) -> dict:
    """Route a natural-language business question to a rule-based analysis intent."""

    normalized = (question or "").strip()
    if not normalized:
        route = {
            "intent": "unknown",
            "intent_name": "未识别问题",
            "confidence": 0.0,
            "reason": "问题为空，无法判断要分析的业务对象和指标链路。",
            "analysis_steps": ["请描述你要分析的内容对象、指标或业务现象。", "也可以点击示例问题快速填入。"],
        }
        if df is not None:
            route["filters"] = extract_filters(question, df)
        return route

    rules = {
        "high_exposure_low_decision": {
            "intent_name": "高曝光低商品决策进入分析",
            "keywords": [
                "高曝光", "热度高", "看的人多", "曝光很多", "没人点商品", "不点商品",
                "没有带来商品点击", "商品点击", "低商品点击", "详情页", "低详情页访问",
                "不带货", "种草无效", "高曝光低转化", "没有带来", "商品卡", "访问",
                "疑似软广", "占了太多曝光", "太多曝光",
            ],
            "reason": "问题指向高曝光内容是否有效进入商品点击和详情页访问链路。",
            "analysis_steps": [
                "识别业务对象：潮鞋社区内容",
                "识别核心问题：高曝光但没有进入商品决策",
                "定义高曝光：exposure 位于前 25%",
                "定义低决策进入：product_card_click_rate 和 product_detail_uv_rate 低于中位数",
                "输出异常内容分布、内容类型特征和样例内容",
            ],
        },
        "real_experience_vs_cliche": {
            "intent_name": "真实体验 vs 同质化好评对比",
            "keywords": [
                "真实体验", "同质化好评", "普通好评", "尺码", "脚感", "缺点", "价格判断",
                "同款对比", "对比测评", "对比", "百搭好看", "闭眼入", "决策型内容",
            ],
            "reason": "问题指向真实体验、尺码、缺点、价格判断或同款对比内容是否比泛好评更有效。",
            "analysis_steps": [
                "识别对比对象：真实体验型内容 vs 同质化好评内容",
                "识别核心假设：真实体验内容是否更能推动商品决策",
                "对比商品卡点击率、评论查看率、价格查看率、下单转化率、取消/退货率",
                "输出均值对比和结论",
            ],
        },
        "conversion_drop_diagnosis": {
            "intent_name": "GMV 或转化下降链路诊断",
            "keywords": [
                "gmv", "下滑", "下降", "转化下降", "劝退", "价格", "尺码", "履约",
                "发货", "评论反馈", "商品页", "详情页承接", "取消", "退货", "转化",
            ],
            "reason": "问题指向交易转化链路，需要判断断点在内容层、商品页承接层还是交易后体验层。",
            "analysis_steps": [
                "识别业务问题：GMV 或交易转化下降",
                "拆解链路：曝光 → 商品卡点击 → 商品详情页访问 → 下单转化 → 取消/退货",
                "判断问题发生在哪一段",
                "输出主要诊断和下一步建议",
            ],
        },
        "economics_interpretation": {
            "intent_name": "经济学视角的内容效率解释",
            "keywords": ["经济学", "信息不对称", "搜索成本", "边际收益", "劣币驱逐良币", "曝光资源错配", "每千曝光"],
            "reason": "问题包含经济学解释关键词，适合用高曝光低决策进入分析来解释曝光资源是否错配。",
            "analysis_steps": [
                "识别经济学问题：社区内容是否降低搜索成本和信息不对称",
                "用高曝光低商品决策进入内容观察曝光资源错配",
                "比较套话比例、决策价值分、商品卡点击和详情页访问",
                "输出内容效率诊断和运营建议",
            ],
            "target_intent": "high_exposure_low_decision",
        },
    }

    scored: list[tuple[str, dict, list[str]]] = []
    for intent, config in rules.items():
        matched = _score_keywords(normalized, config["keywords"])
        scored.append((intent, config, matched))

    scored.sort(key=lambda item: (len(item[2]), sum(len(k) for k in item[2])), reverse=True)
    best_intent, best_config, best_matches = scored[0]

    if not best_matches:
        route = {
            "intent": "unknown",
            "intent_name": "未识别问题",
            "confidence": 0.18,
            "reason": "未命中当前规则库中的核心关键词。可以换一种问法，或从示例问题中选择。",
            "analysis_steps": [
                "尝试补充业务对象，例如潮鞋社区内容、真实体验内容或 GMV 下滑。",
                "尝试补充指标，例如曝光、商品卡点击、详情页访问、价格查看、下单转化、取消/退货。",
            ],
        }
    else:
        confidence = min(0.94, 0.48 + 0.08 * len(best_matches) + 0.01 * min(10, sum(len(k) for k in best_matches) / 4))
        route = {
            "intent": best_config.get("target_intent", best_intent),
            "raw_intent": best_intent,
            "intent_name": best_config["intent_name"],
            "confidence": round(confidence, 2),
            "reason": f"{best_config['reason']} 命中关键词：{', '.join(best_matches)}。",
            "analysis_steps": best_config["analysis_steps"],
        }

    if df is not None:
        route["filters"] = extract_filters(question, df)
    return route


def _with_unified_result(
    *,
    analysis_name: str,
    tables: dict[str, pd.DataFrame],
    conclusion: str,
    suggestions: list[str],
    filters_used: dict,
    original_row_count: int,
    filtered_row_count: int,
    filter_warning: str | None,
    extra: dict | None = None,
) -> dict:
    result = {
        "analysis_name": analysis_name,
        "tables": tables,
        "conclusion": conclusion,
        "suggestions": suggestions,
        "filters_used": filters_used,
        "original_row_count": original_row_count,
        "filtered_row_count": filtered_row_count,
        "filter_warning": filter_warning,
    }
    if extra:
        result.update(extra)
    return result


def analyze_high_exposure_low_decision(df: pd.DataFrame, filters: dict | None = None) -> dict:
    """Find high-exposure content that failed to drive product decision entry."""

    original_row_count = len(df)
    df_filtered, filters_used, filter_warning = _filter_context(df, filters)
    exposure_threshold = df_filtered["exposure"].quantile(0.75)
    click_median = df_filtered["product_card_click_rate"].median()
    detail_median = df_filtered["product_detail_uv_rate"].median()

    mask = (
        (df_filtered["exposure"] >= exposure_threshold)
        & (df_filtered["product_card_click_rate"] < click_median)
        & (df_filtered["product_detail_uv_rate"] < detail_median)
    )
    subset = df_filtered.loc[mask].copy()

    type_distribution = (
        subset["content_type"]
        .value_counts(normalize=True)
        .rename("占比")
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"content_type": "内容类型"})
    )

    sample_table = subset.sort_values(["exposure", "positive_cliche_ratio"], ascending=[False, False])[
        [
            "content_id", "content_type", "brand", "exposure", "product_card_click_rate",
            "product_detail_uv_rate", "decision_value_score", "positive_cliche_ratio",
        ]
    ].head(20)

    summary = pd.DataFrame(
        [{
            "高曝光阈值": round(exposure_threshold, 0),
            "问题内容数": len(subset),
            "问题内容占比": format_percent(len(subset) / len(df_filtered)) if len(df_filtered) else "0.00%",
            "平均套话比例": round(subset["positive_cliche_ratio"].mean(), 3) if len(subset) else 0,
            "平均决策价值分": round(subset["decision_value_score"].mean(), 2) if len(subset) else 0,
        }]
    )

    top_type = type_distribution.iloc[0]["内容类型"] if not type_distribution.empty else "无"
    conclusion = (
        f"在当前筛选样本 {len(df_filtered)} 条内容中，高曝光但低商品决策进入的内容共有 {len(subset)} 条，"
        f"占筛选样本 {summary.iloc[0]['问题内容占比']}。主要集中在「{top_type}」，"
        f"平均套话比例为 {summary.iloc[0]['平均套话比例']}。"
    )
    suggestions = [
        "优先复盘高曝光内容是否缺少尺码、缺点、价格判断和同类对比。",
        "降低只含“百搭、好看、闭眼入”等套话内容的推荐权重。",
        "把商品卡点击率和详情页访问率纳入社区内容质量评估，而不是只看曝光和点赞。",
    ]

    return _with_unified_result(
        analysis_name="高曝光低商品决策进入分析",
        tables={"分析摘要": summary, "主要内容类型分布": type_distribution, "样例内容": sample_table},
        conclusion=conclusion,
        suggestions=suggestions,
        filters_used=filters_used,
        original_row_count=original_row_count,
        filtered_row_count=len(df_filtered),
        filter_warning=filter_warning,
        extra={"summary": summary, "type_distribution": type_distribution, "sample_table": sample_table},
    )


def compare_real_experience_vs_cliche(df: pd.DataFrame, filters: dict | None = None) -> dict:
    """Compare real-experience content against generic positive content."""

    original_row_count = len(df)
    df_filtered, filters_used, filter_warning = _filter_context(df, filters, ignore_content_type=True)
    compare_df = df_filtered[df_filtered["content_type"].isin(["真实体验", "同质化好评"])].copy()
    metrics = [
        "decision_value_score", "product_card_click_rate", "review_view_rate",
        "price_view_rate", "purchase_conversion_rate", "cancel_return_rate",
    ]

    comparison = (
        compare_df.groupby("content_type")[metrics]
        .mean()
        .reindex(["真实体验", "同质化好评"])
        .round(4)
        .reset_index()
    )

    if comparison[metrics].isna().any().any():
        conclusion = "当前筛选条件下真实体验或同质化好评样本不足，均值对比可能不稳定。"
    else:
        real = comparison[comparison["content_type"] == "真实体验"].iloc[0]
        cliche = comparison[comparison["content_type"] == "同质化好评"].iloc[0]
        conclusion = (
            "真实体验型内容相比同质化好评，"
            f"商品卡点击率差值为 {format_percent(real['product_card_click_rate'] - cliche['product_card_click_rate'])}，"
            f"价格查看率差值为 {format_percent(real['price_view_rate'] - cliche['price_view_rate'])}，"
            f"下单转化率差值为 {format_percent(real['purchase_conversion_rate'] - cliche['purchase_conversion_rate'])}。"
        )
        if real["cancel_return_rate"] < cliche["cancel_return_rate"]:
            conclusion += " 同时取消/退货率更低，说明尺码、脚感、缺点等信息能降低购买后预期偏差。"

    suggestions = [
        "把包含尺码、脚感、缺点、价格判断、同款对比的内容作为决策型内容池。",
        "对同质化好评增加信息补全要求，例如必须补充适用脚型、尺码偏差或价格区间判断。",
        "用评论查看率、价格查看率、商品卡点击率验证内容是否真正进入购买决策。",
    ]

    return _with_unified_result(
        analysis_name="真实体验 vs 同质化好评对比",
        tables={"均值对比": comparison},
        conclusion=conclusion,
        suggestions=suggestions,
        filters_used=filters_used,
        original_row_count=original_row_count,
        filtered_row_count=len(df_filtered),
        filter_warning=filter_warning,
        extra={"comparison": comparison},
    )


def diagnose_conversion_drop(df: pd.DataFrame, filters: dict | None = None) -> dict:
    """Diagnose where the content-to-transaction funnel is weak."""

    original_row_count = len(df)
    df_filtered, filters_used, filter_warning = _filter_context(df, filters)
    avg = df_filtered[
        [
            "exposure", "product_card_click_rate", "product_detail_uv_rate", "price_view_rate",
            "review_view_rate", "purchase_conversion_rate", "cancel_return_rate",
        ]
    ].mean()

    click_median = df_filtered["product_card_click_rate"].median()
    conversion_median = df_filtered["purchase_conversion_rate"].median()
    return_median = df_filtered["cancel_return_rate"].median()

    click_low = avg["product_card_click_rate"] < click_median
    click_high = avg["product_card_click_rate"] >= click_median
    conversion_low = avg["purchase_conversion_rate"] < conversion_median
    conversion_ok = avg["purchase_conversion_rate"] >= conversion_median
    return_high = avg["cancel_return_rate"] > return_median

    if click_low:
        diagnosis = "主要问题在内容层：平均商品卡点击率低于筛选样本中位数，说明曝光没有充分转化为商品决策入口。"
        suggestions = [
            "检查高曝光内容是否缺少尺码、缺点、价格判断和同类对比。",
            "调低泛好评和疑似软广的流量权重，提升真实体验和对比测评占比。",
            "把商品卡点击率和详情页访问率作为社区分发的核心反馈指标。",
        ]
    elif click_high and conversion_low:
        diagnosis = "主要问题在商品页承接：用户已经进入商品链路，但平均下单转化率低于筛选样本中位数。"
        suggestions = [
            "检查商品详情页价格是否高于用户心理价位或近期波动过大。",
            "检查热门尺码供给、发货时效和商品评论区负反馈。",
            "在内容侧补充到手价、尺码建议和购买时机判断，降低跳失。",
        ]
    elif conversion_ok and return_high:
        diagnosis = "主要问题在交易后体验：平均下单转化率不低，但取消/退货率高于筛选样本中位数。"
        suggestions = [
            "检查履约时效、售后反馈和退货原因。",
            "强化尺码偏差、脚型适配、材质缺点等购买前提示。",
            "把取消/退货率反哺内容质量分，避免只奖励短期转化。",
        ]
    else:
        diagnosis = "链路没有单一断点，更可能是内容决策价值、商品页承接和交易体验共同偏弱。"
        suggestions = [
            "按品牌和内容类型拆分漏斗，定位具体弱项。",
            "优先治理高曝光低决策价值内容，再检查价格和尺码供给。",
            "建立内容曝光、商品点击、价格查看、下单、取消/退货的周度监控。",
        ]

    funnel_table = pd.DataFrame(
        [{
            "平均曝光": round(avg["exposure"], 0),
            "商品卡点击率": round(avg["product_card_click_rate"], 4),
            "商品卡点击率中位数": round(click_median, 4),
            "详情页访问率": round(avg["product_detail_uv_rate"], 4),
            "价格查看率": round(avg["price_view_rate"], 4),
            "评论查看率": round(avg["review_view_rate"], 4),
            "下单转化率": round(avg["purchase_conversion_rate"], 4),
            "下单转化率中位数": round(conversion_median, 4),
            "取消/退货率": round(avg["cancel_return_rate"], 4),
            "取消/退货率中位数": round(return_median, 4),
        }]
    )

    return _with_unified_result(
        analysis_name="GMV 或转化下降链路诊断",
        tables={"链路阶段平均指标": funnel_table},
        conclusion=diagnosis,
        suggestions=suggestions,
        filters_used=filters_used,
        original_row_count=original_row_count,
        filtered_row_count=len(df_filtered),
        filter_warning=filter_warning,
        extra={"funnel_table": funnel_table, "diagnosis": diagnosis},
    )
