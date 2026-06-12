# Dewu Decision Content Agent MVP

## 项目背景

这是一个面向得物潮鞋品类运营的 AI 数据产品 Demo。项目不使用真实得物数据，而是通过模拟数据、规则逻辑和可选 GPT 动态诊断，展示“得物社区内容是否真正帮助用户完成购买决策”的分析链路。

## 产品洞察

得物社区不应只是泛种草内容社区，而应成为“商品页之外的决策信息层”。真正有价值的社区内容，不是同质化的“百搭、好看、闭眼入”，而是包含缺点、尺码反馈、价格判断、同类对比、适用场景等信息，帮助用户判断：

- 这双鞋适不适合我
- 现在值不值得买
- 买了会不会后悔

## 五类决策价值信号

- `decision_info_score`：决策信息密度，衡量内容是否包含尺码、缺点、价格、对比、场景等有效信息。
- `effective_interaction_score`：有效互动质量，衡量评论等互动是否更接近购买决策讨论。
- `decision_path_entry_score`：商品决策链路进入能力，衡量内容是否能推动用户点击商品卡、进入详情页。
- `transaction_feedback_score`：交易反馈质量，衡量内容是否有助于降低购买后取消或退货。
- `author_credibility_score`：作者长期可信度，模拟作者是否长期输出可信、有参考价值的内容。

综合决策价值分：

```text
decision_value_score =
0.25 * decision_info_score
+ 0.20 * effective_interaction_score
+ 0.25 * decision_path_entry_score
+ 0.20 * transaction_feedback_score
+ 0.10 * author_credibility_score
```

## 三个 DataAgent 问题

1. 高曝光低决策价值内容识别  
   找出潮鞋类目里高曝光但没有带来商品点击和详情页访问的社区内容，并分析是否集中在同质化好评或疑似软广。

2. 真实体验 vs 同质化好评  
   比较包含尺码、脚感、缺点、价格判断、同款对比的真实体验型内容，是否比普通好评带来更高商品卡点击率、评论查看率、价格查看率和下单转化率。

3. 转化下降链路诊断  
   用曝光、商品卡点击、详情页访问、价格查看、评论查看、下单转化、取消/退货等指标，判断问题主要在内容层、商品页承接层，还是交易后体验层。

## GPT 动态诊断模式

本项目默认可以在无 API Key 的情况下以规则版运行。如果设置了 `OPENAI_API_KEY`，页面可以启用 GPT 动态诊断。

混合架构原则：

- Python 负责确定性查数和指标计算。
- GPT 不负责计算指标，不生成 SQL，不编造数字。
- GPT 只基于 Python 已经计算出的结构化结果，生成自然语言诊断、关键发现、下一步建议和数据边界提醒。
- 本项目使用模拟数据，不能代表得物真实业务数据。

默认模型为 `gpt-5.4-mini`。如果账号不可用，可用环境变量覆盖：

```bash
OPENAI_MODEL="your_model_name"
```

## 如何运行项目

```bash
pip install -r requirements.txt
python generate_mock_data.py
streamlit run app.py
```

如果本机 `streamlit` 不在 PATH，可使用：

```bash
python -m streamlit run app.py
```

## 本地设置 API Key

Windows PowerShell:

```powershell
setx OPENAI_API_KEY "your_api_key"
```

设置后需要重新打开一个 PowerShell 窗口。

Mac/Linux:

```bash
export OPENAI_API_KEY="your_api_key"
```

Streamlit Cloud:

在 `App settings -> Secrets` 中添加：

```toml
OPENAI_API_KEY = "your_api_key"
```

## 面试演示方式

1. 启动页面后，先展示顶部的数据概览和内容类型横向柱状图，说明“高曝光不等于高决策价值”。
2. 在 DataAgent 输入框点击示例问题，展示规则版意图识别和分析路径。
3. 说明 Python 先完成确定性计算，页面展示结果表格。
4. 如果没有 API Key，展示规则版诊断和 fallback 提示。
5. 如果有 API Key，勾选“启用 GPT 动态诊断”，展示 GPT 基于结果表格生成的动态诊断、关键发现、建议和风险提醒。

## 文件结构

```text
.
├── app.py
├── dewu_agent.py
├── generate_mock_data.py
├── gpt_diagnosis.py
├── requirements.txt
├── README.md
└── data/
    └── dewu_decision_content_mock.csv
```
