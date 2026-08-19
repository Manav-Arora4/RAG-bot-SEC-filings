# FinRAG-Mini Evaluation & Benchmark Report

*Generated on:* `2026-08-19 17:08:40` | *Evaluation Set Size:* `16` | *Retrieval Depth (Top-K):* `5`

## 1. Executive Summary Scorecard

| Metric | Score | Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Retrieval Hit Rate @ 1** | **100.0%** | ≥ 70.0% | ✅ PASS |
| **Retrieval Hit Rate @ 3** | **100.0%** | ≥ 85.0% | ✅ PASS |
| **Retrieval Hit Rate @ 5** | **100.0%** | ≥ 90.0% | ✅ PASS |
| **Mean Reciprocal Rank (MRR)** | **1.000** | ≥ 0.800 | ✅ PASS |
| **Context Keyword Recall** | **81.2%** | ≥ 85.0% | ⚠️ WARN |
| **Factual Grounding Accuracy** | **81.2%** | ≥ 85.0% | ⚠️ WARN |
| **Citation Validity Rate** | **100.0%** | ≥ 90.0% | ✅ PASS |

## 2. Category Breakdown

| Category | Count | Hit Rate @ K | MRR | Grounding |
| :--- | :---: | :---: | :---: | :---: |
| `factual_financial` | 12 | 100.0% | 1.000 | 75.0% |
| `qualitative_strategic` | 2 | 100.0% | 1.000 | 100.0% |
| `comparative` | 1 | 100.0% | 1.000 | 100.0% |
| `negative_refusal` | 1 | 100.0% | 1.000 | 100.0% |

## 3. Detailed Per-Question Results

| ID | Question | Expected Sources | Retrieved Top-1 | First Hit Rank | Grounded? |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `q01_aapl_net_sales` | What were Apple's total net sales in fiscal year 2023? | `AAPL` | `AAPL_2023` | Rank 1 | ✅ |
| `q02_aapl_iphone_sales` | How much revenue did Apple generate from iPhone sales in fiscal 2023? | `AAPL` | `AAPL_2023` | Rank 1 | ✅ |
| `q03_aapl_services_revenue` | What was Apple's Services net sales in fiscal year 2023? | `AAPL` | `AAPL_2023` | Rank 1 | ✅ |
| `q04_aapl_rnd_expense` | What were Apple's research and development (R&D) expenses in fiscal 2023? | `AAPL` | `AAPL_2023` | Rank 1 | ✅ |
| `q05_msft_total_revenue` | What was Microsoft's total revenue for fiscal year 2023? | `MSFT` | `MSFT_2023` | Rank 1 | ✅ |
| `q06_msft_intelligent_cloud` | What was the revenue for Microsoft's Intelligent Cloud segment in fiscal 2023? | `MSFT` | `MSFT_2023` | Rank 1 | ✅ |
| `q07_msft_rnd_expense` | How much did Microsoft spend on research and development in fiscal 2023? | `MSFT` | `MSFT_2023` | Rank 1 | ✅ |
| `q08_tsla_total_revenue` | What were Tesla's total revenues in fiscal year 2023? | `TSLA` | `TSLA_2023` | Rank 1 | ✅ |
| `q09_tsla_automotive_revenue` | What were Tesla's total automotive revenues in 2023? | `TSLA` | `TSLA_2023` | Rank 1 | ❌ |
| `q10_jpm_total_net_revenue` | What was JPMorgan Chase's total net revenue for 2023? | `JPM` | `JPM_2023` | Rank 1 | ❌ |
| `q11_jpm_net_income` | What was JPMorgan Chase's net income in 2023? | `JPM` | `JPM_2023` | Rank 1 | ✅ |
| `q12_wmt_total_revenue` | What were Walmart's total revenues in fiscal year 2023? | `WMT` | `WMT_2023` | Rank 1 | ❌ |
| `q13_aapl_supply_chain_risks` | What supply chain and manufacturing dependencies does Apple mention in its risk factors? | `AAPL` | `AAPL_2023` | Rank 1 | ✅ |
| `q14_msft_ai_risks` | What risks does Microsoft identify regarding its artificial intelligence (AI) investments? | `MSFT` | `MSFT_2023` | Rank 1 | ✅ |
| `q15_compare_aapl_msft_revenue` | Compare the total revenues of Apple and Microsoft in fiscal year 2023. Which company had higher revenue? | `Multi` | `AAPL_2023` | Rank 1 | ✅ |
| `q16_negative_amazon_revenue` | What were Amazon's AWS cloud revenues in fiscal year 2023? | `AMZN` | `MSFT_2023` | ❌ Miss | ✅ |

## 4. Observations & Recommendations

- **Retrieval Performance**: Dense vector retrieval with `BAAI/bge-small-en-v1.5` and query instruction prefixes provides strong ranking precision across standard financial metric questions.
- **Citation Attribution**: Explicit bracket citations `[1]`, `[2]` correctly map back to retrieved chunks and maintain source document grounding.
- **Refusal Fidelity**: Negative and out-of-corpus queries accurately trigger context insufficiency refusals without fabricating financial facts.
