from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st

from dewu_agent import (
    analyze_high_exposure_low_decision,
    build_decision_value_matrix,
    compare_real_experience_vs_cliche,
    diagnose_conversion_drop,
    extract_filters,
    route_question,
)
from gpt_diagnosis import generate_gpt_diagnosis, has_openai_api_key


DATA_PATH = Path("data/dewu_decision_content_mock.csv")

EXAMPLE_QUESTIONS = [
    "上周潮鞋类目里，哪些高曝光社区内容没有带来商品点击和详情页访问？",
    "带有尺码、脚感、缺点、价格判断、同款对比的真实体验型内容，是否比普通好评内容更能推动商品决策？",
    "最近潮鞋 GMV 下滑，问题主要出在社区内容没有把用户带进商品决策，还是商品页之后被价格、尺码、履约因素劝退？",
    "从经济学的信息不对称和搜索成本角度看，高曝光低商品点击内容是不是一种曝光资源错配？",
]


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到模拟数据文件，请先运行：python generate_mock_data.py")
    return pd.read_csv(DATA_PATH)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def show_metric_row(df: pd.DataFrame) -> None:
    cols = st.columns(6)
    cols[0].metric("总内容数", f"{len(df):,}")
    cols[1].metric("平均曝光", f"{df['exposure'].mean():,.0f}")
    cols[2].metric("平均决策价值分", f"{df['decision_value_score'].mean():.1f}")
    cols[3].metric("平均商品卡点击率", pct(df["product_card_click_rate"].mean()))
    cols[4].metric("平均下单转化率", pct(df["purchase_conversion_rate"].mean()))
    cols[5].metric("平均取消/退货率", pct(df["cancel_return_rate"].mean()))


def build_type_performance(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("content_type")[
            [
                "exposure",
                "decision_value_score",
                "product_card_click_rate",
                "purchase_conversion_rate",
                "cancel_return_rate",
            ]
        ]
        .mean()
        .sort_values("decision_value_score", ascending=False)
        .round(
            {
                "exposure": 0,
                "decision_value_score": 2,
                "product_card_click_rate": 4,
                "purchase_conversion_rate": 4,
                "cancel_return_rate": 4,
            }
        )
    )


def render_horizontal_bar_chart(data: pd.DataFrame, metric: str, title: str) -> None:
    chart_data = data.reset_index().rename(columns={"content_type": "内容类型"})
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            y=alt.Y("内容类型:N", sort="-x", title=None),
            x=alt.X(f"{metric}:Q", title=title),
            tooltip=["内容类型", alt.Tooltip(f"{metric}:Q", title=title)],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")


def render_analysis_path(steps: list[str]) -> None:
    with st.expander("DataAgent 分析路径", expanded=True):
        for idx, step in enumerate(steps, start=1):
            st.markdown(f"{idx}. {step}")


def run_python_analysis(intent: str, df: pd.DataFrame, filters: dict | None = None) -> dict:
    if intent == "high_exposure_low_decision":
        return analyze_high_exposure_low_decision(df, filters=filters)
    if intent == "real_experience_vs_cliche":
        return compare_real_experience_vs_cliche(df, filters=filters)
    if intent == "conversion_drop_diagnosis":
        return diagnose_conversion_drop(df, filters=filters)
    raise ValueError(f"Unsupported intent: {intent}")


def render_filter_summary(filters: dict, analysis_result: dict | None = None) -> None:
    st.subheader("解析出的筛选条件")
    filter_table = pd.DataFrame(
        [
            {
                "品牌": filters.get("brand") or "未指定",
                "品类": filters.get("category") or "未指定",
                "内容类型": filters.get("content_type") or "未指定",
                "时间范围": filters.get("time_range") or "未指定",
                "指标词": "、".join(filters.get("metric_terms") or []) or "未指定",
                "原始样本数": analysis_result.get("original_row_count") if analysis_result else "-",
                "筛选后样本数": analysis_result.get("filtered_row_count") if analysis_result else "-",
            }
        ]
    )
    st.dataframe(filter_table, width="stretch", hide_index=True)
    if analysis_result and analysis_result.get("filter_warning"):
        st.info(analysis_result["filter_warning"])


def render_result_tables(tables: dict[str, pd.DataFrame]) -> None:
    st.subheader("Python 确定性计算结果")
    st.caption("以下表格由本地 Python 基于模拟数据计算得到；GPT 不参与指标计算。")
    for table_name, table in tables.items():
        st.markdown(f"**{table_name}**")
        st.dataframe(table, width="stretch")


def render_decision_value_matrix(df: pd.DataFrame, filters: dict) -> None:
    st.header("内容决策价值矩阵：找出虚热内容和被埋没的决策内容")
    st.write(
        "得物社区的问题不只是内容质量高低，而是曝光资源是否分配给了真正帮助用户决策的内容。"
        "这个矩阵把内容放到“曝光占用程度 × 购买决策价值”两个维度中，帮助运营识别两类关键对象："
        "一类是获得很多曝光但没有推动购买决策的虚热内容，另一类是决策价值高但没有被充分分发的被埋没内容。"
    )
    st.caption(
        "横轴表示曝光占用程度，纵轴表示购买决策价值。矩阵不是判断内容真假，而是判断曝光资源是否分配给了真正帮助用户购买决策的内容。"
    )

    matrix_result = build_decision_value_matrix(df, filters=filters)
    matrix_df = matrix_result["matrix_df"]
    exposure_threshold = matrix_result["exposure_threshold"]
    decision_threshold = matrix_result["decision_threshold"]
    counts = matrix_result["summary"]["quadrant_counts"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("被埋没的决策内容", f"{counts.get('被埋没的决策内容', 0)} 条")
    metric_cols[1].metric("虚热内容 / 曝光资源错配", f"{counts.get('虚热内容 / 曝光资源错配', 0)} 条")
    metric_cols[2].metric("已验证样板内容", f"{counts.get('已验证样板内容', 0)} 条")
    metric_cols[3].metric("低优先级观察内容", f"{counts.get('低优先级观察内容', 0)} 条")

    fig = px.scatter(
        matrix_df,
        x="exposure",
        y="decision_value_score",
        color="quadrant",
        hover_data=[
            "content_id",
            "brand",
            "content_type",
            "product_card_click_rate",
            "product_detail_uv_rate",
            "purchase_conversion_rate",
            "cancel_return_rate",
        ],
        title="Decision Value × Exposure Allocation Matrix",
    )
    fig.add_vline(x=exposure_threshold, line_dash="dash")
    fig.add_hline(y=decision_threshold, line_dash="dash")
    fig.update_layout(
        xaxis_title="曝光量 / 曝光占用程度",
        yaxis_title="购买决策价值分",
        legend_title_text="象限",
        height=560,
    )
    st.plotly_chart(fig, width="stretch")

    tab_names = [
        "被埋没的决策内容",
        "虚热内容 / 曝光资源错配",
        "已验证样板内容",
        "低优先级观察内容",
    ]
    list_keys = [
        "buried_decision_content",
        "overheated_mismatch_content",
        "verified_sample_content",
        "low_priority_content",
    ]
    tabs = st.tabs(tab_names)
    for tab, key in zip(tabs, list_keys):
        with tab:
            table = matrix_result["lists"][key]
            if table.empty:
                st.info("当前筛选条件下没有符合该象限特征的内容。")
            else:
                st.dataframe(table, width="stretch", hide_index=True)


def render_diagnosis_block(diagnosis: dict, rule_based: bool = False) -> None:
    title = "规则版诊断" if rule_based else "GPT 动态诊断"
    st.subheader(title)

    if diagnosis.get("fallback_reason"):
        st.info(diagnosis["fallback_reason"])

    if diagnosis.get("model"):
        st.caption(f"诊断模型：{diagnosis['model']}")

    st.write(diagnosis.get("diagnosis", "暂无诊断内容。"))

    key_findings = diagnosis.get("key_findings") or []
    if key_findings:
        st.markdown("**关键发现**")
        for item in key_findings:
            st.markdown(f"- {item}")

    next_steps = diagnosis.get("next_steps") or []
    st.subheader("下一步建议")
    for item in next_steps:
        st.markdown(f"- {item}")

    risk_notes = diagnosis.get("risk_notes") or []
    st.subheader("数据边界提醒")
    for item in risk_notes:
        st.markdown(f"- {item}")


def build_rule_diagnosis(result: dict) -> dict:
    return {
        "enabled": False,
        "model": "rule-based",
        "diagnosis": result["conclusion"],
        "key_findings": ["当前使用规则版诊断；所有指标均由 Python 确定性计算得到。"],
        "next_steps": result["suggestions"],
        "risk_notes": [
            "本项目使用模拟数据，不能代表得物真实业务表现。",
            "相关性不能直接解释为因果关系，正式决策需要真实数据和实验验证。",
        ],
        "fallback_reason": None,
    }


def render_routed_result(question: str, route: dict, df: pd.DataFrame, enable_gpt: bool) -> None:
    st.subheader("用户问题")
    st.write(question)

    st.subheader("意图识别结果")
    intent_cols = st.columns([2, 1, 4])
    intent_cols[0].metric("识别到的分析类型", route["intent_name"])
    intent_cols[1].metric("置信度", f"{route['confidence']:.2f}")
    intent_cols[2].write(f"识别理由：{route['reason']}")

    filters = route.get("filters") or extract_filters(question, df)

    if route["intent"] == "unknown":
        render_filter_summary(filters)
        st.warning("暂时无法识别这个问题。请换一种问法，或点击示例问题后再开始分析。")
        st.markdown("**可尝试的问题：**")
        for example in EXAMPLE_QUESTIONS[:3]:
            st.markdown(f"- {example}")
        return

    analysis_result = run_python_analysis(route["intent"], df, filters=filters)
    render_filter_summary(filters, analysis_result)
    render_analysis_path(route["analysis_steps"])
    render_result_tables(analysis_result["tables"])

    if enable_gpt:
        diagnosis = generate_gpt_diagnosis(
            user_question=question,
            route_result=route,
            analysis_name=analysis_result["analysis_name"],
            analysis_steps=route["analysis_steps"],
            result_tables=analysis_result["tables"],
            rule_based_conclusion=analysis_result["conclusion"],
            rule_based_suggestions=analysis_result["suggestions"],
        )
        render_diagnosis_block(diagnosis, rule_based=not diagnosis.get("enabled"))
    else:
        render_diagnosis_block(build_rule_diagnosis(analysis_result), rule_based=True)

    render_decision_value_matrix(df, analysis_result.get("filters_used", filters))


def set_example_question(example: str) -> None:
    st.session_state["business_question"] = example


def render_example_buttons() -> None:
    st.caption("示例问题")
    cols = st.columns(2)
    for idx, example in enumerate(EXAMPLE_QUESTIONS):
        cols[idx % 2].button(
            example,
            key=f"example_{idx}",
            width="stretch",
            on_click=set_example_question,
            args=(example,),
        )


def main() -> None:
    st.set_page_config(
        page_title="Dewu Decision Content Agent MVP",
        page_icon="👟",
        layout="wide",
    )

    st.title("Dewu Decision Content Agent MVP")
    st.write(
        "这是一个模拟得物潮鞋社区内容决策价值分析的 AI 数据产品 MVP。"
        "它不判断内容真伪，而是识别内容是否具有购买决策参考价值，"
        "并用内容表现、商品点击、交易反馈进行交叉验证。"
    )

    df = load_data()

    st.header("核心洞察")
    insight_cols = st.columns(3)
    insight_cols[0].info("得物社区应成为商品页之外的决策信息层")
    insight_cols[1].info("高曝光不等于高决策价值")
    insight_cols[2].info("真正有价值的内容应帮助用户判断适不适合、值不值得、会不会后悔")

    st.header("数据概览")
    show_metric_row(df)

    st.header("内容类型表现对比")
    type_performance = build_type_performance(df)
    st.dataframe(type_performance, width="stretch")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("不同内容类型的决策价值分")
        render_horizontal_bar_chart(type_performance, "decision_value_score", "决策价值分")
    with chart_cols[1]:
        st.subheader("不同内容类型的下单转化率")
        render_horizontal_bar_chart(type_performance, "purchase_conversion_rate", "下单转化率")

    st.header("DataAgent 自然语言问数")
    st.write(
        "本 Demo 采用混合架构：Python 负责确定性查数和指标计算，GPT 负责基于计算结果生成动态诊断。"
        "这样既保证数据口径可信，又让分析结论不再是固定模板。GPT 不参与指标计算，只解释 Python 的计算结果。"
    )
    st.caption(
        "当前版本支持规则版实体抽取，可识别品牌、内容类型和部分时间表达；"
        "真实生产环境可替换为 LLM + 指标字典 + 业务实体库。"
    )

    api_key_available = has_openai_api_key()
    enable_gpt = st.checkbox("启用 GPT 动态诊断", value=api_key_available)
    if not api_key_available:
        st.info("未检测到 OPENAI_API_KEY，将使用规则版诊断。设置 API Key 后可启用 GPT 动态诊断。")

    if "business_question" not in st.session_state:
        st.session_state["business_question"] = EXAMPLE_QUESTIONS[0]

    st.text_area("请输入你的业务问题", key="business_question", height=110)
    render_example_buttons()

    if st.button("开始分析", type="primary", width="stretch"):
        question = st.session_state["business_question"].strip()
        route = route_question(question, df=df)
        render_routed_result(question, route, df, enable_gpt)

    with st.expander("原始数据浏览：前 50 行"):
        st.dataframe(df.head(50), width="stretch")


if __name__ == "__main__":
    main()
