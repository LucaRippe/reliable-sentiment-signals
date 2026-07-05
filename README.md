# Sentiment-Certainty Signals

**Does earnings-call transcript sentiment carry a tradable signal beyond conventional factors — and can LLM uncertainty quantification make that signal better?**

This project predicts post-earnings stock drift from earnings-call transcripts using a local large language model (LLM). Beyond a raw sentiment score, it measures the model's *uncertainty* (via Monte-Carlo sampling: mean entropy, confidence, agreement), *calibrates* that uncertainty (zero-shot reasoning + active learning, evaluated with AURC), builds a cross-sectional trading signal, tests it against conventional factors (momentum, size, value, earnings surprise), and finally couples a trading strategy to the uncertainty metrics. Results are explored in an interactive Streamlit dashboard.

> Research disclaimer: This is a research project, not investment advice. Backtested performance does not guarantee future results.

---



## 1. Research questions

1. **Signal existence:** Does transcript sentiment predict the post-earnings drift (return over t+1 to t+5 trading days)?
2. **Incremental value:** Is that signal robust *after controlling* for momentum, size, value, and the standardized earnings surprise (SUE) plus the announcement-day return? Is there a genuine alpha beyond well-known effects?
3. **Uncertainty value:** Do LLM uncertainty metrics (entropy, confidence, agreement) identify *when* the signal is trustworthy — i.e. does uncertainty-aware trading add return / Sharpe / lower drawdown?
4. **Calibration:** Can we improve uncertainty calibration (lower AURC) with zero-shot reasoning and active learning?

---



## 2. Methodology



### 2.1 Data (free stack)

The default pipeline uses **only free data sources and tools** — no paid API subscriptions required for the core research workflow. Provider choices and paths are defined in [`config/config.yaml`](config/config.yaml).

| Layer | Provider | Notes |
| --- | --- | --- |
| **Universe** | [index-constitution](https://github.com/unliftedq/index-constitution) | Point-in-time S&P 500 membership (`constituents_at("sp500", date)`). Free Wikipedia-based alternatives: [fja05680/sp500](https://github.com/fja05680/sp500), [bkestelman/sp500_historical_components](https://github.com/bkestelman/sp500_historical_components). |
| **Transcripts** | [Hugging Face `kurry/sp500_earnings_transcripts`](https://huggingface.co/datasets/kurry/sp500_earnings_transcripts) | ~33k S&P 500 earnings calls (2005–2025), full text + speaker-segmented dialogues. Research/educational license. |
| **Prices** | [`yfinance`](https://github.com/ranaroussi/yfinance) | Adjusted OHLCV, earnings dates. Delisted tickers may be missing — survivorship bias must be documented. |
| **Fundamentals** | [SimFin](https://simfin.com/) (primary) + `yfinance` (fallback) | Market cap, book equity → book-to-market. SimFin requires free registration (`SIMFIN_API_KEY` in `.env`). |
| **SUE** | Computed in-pipeline | Lagged-EPS method (actual vs. same-quarter prior year); `yfinance` earnings surprise as fallback when available. Analyst-consensus SUE would require paid data and is **not** used. |
| **Factor controls & alpha** | [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | FF5 + momentum (daily), free download. In-pipeline: 12-1 month momentum, size, value, SUE, announcement-day return. |

**Look-ahead safety:** every transcript is timestamped to when it became public; signals only use information available at (or after) that time. Trades are entered on t+1.

**Known limitations of the free stack:** (1) yfinance lacks history for many delisted index members; (2) Wikipedia-derived index data may have gaps before ~2000; (3) SimFin fundamentals are quarterly — align carefully to announcement dates. See `limitations` in `config/config.yaml`.

### 2.2 Sentiment + uncertainty (core)

A **local** instruction-tuned LLM (default: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) via Hugging Face Transformers; alternatives: Llama 3.1 8B, Mistral 7B, or [Ollama](https://ollama.com/) for serving) classifies the forward-looking sentiment of each transcript (and optionally its sections: prepared remarks vs. Q&A) zero-shot into an ordinal outlook (bearish / neutral / bullish) plus a self-reported confidence. No paid inference APIs — all Monte-Carlo sampling runs on local hardware (~8–16 GB RAM/VRAM for 7B models).

Uncertainty is estimated by **Monte-Carlo sampling** (N stochastic generations, temperature > 0):

- **Confidence** — mean probability mass on the predicted (modal) class.
- **Mean entropy** — average predictive entropy of the answer distribution across samples; high entropy = uncertain.
- **Agreement** — fraction of samples that agree with the modal label (variation ratio); low agreement = uncertain.



### 2.3 Uncertainty calibration

- **Zero-shot reasoning:** generate chain-of-thought before the verdict and test whether reasoning-augmented confidence is better calibrated.
- **Selective prediction / AURC:** the trading task provides ground truth (did the sentiment-implied direction match the realized drift?). We build the **Risk–Coverage curve** (selective error vs. fraction of predictions retained, ordered by confidence) and report **AURC** (Area Under the Risk–Coverage curve; **lower is better**). Reliability diagrams and ECE complement this.
- **Active learning:** iteratively select the most uncertain / most informative transcripts for stronger-teacher (or human) labeling, refine few-shot exemplars / a lightweight classifier head, and re-measure AURC to demonstrate calibration improvement.



### 2.4 Signal construction & factor controls

- Raw signal = ordinal sentiment (optionally confidence-weighted), standardized cross-sectionally per earnings window.
- **Incremental test:** panel and Fama-MacBeth regressions of forward returns on the signal **with controls** (momentum, size, value, SUE, announcement return). The key output is the sentiment coefficient's sign, magnitude, and significance *after* controls.
- **Orthogonalized signal:** residual of the signal regressed on the controls, used to isolate the marginal effect.
- **Portfolio sorts:** quantile long/short portfolios; alpha vs. FF5 + momentum.



### 2.5 Trading strategy coupled to uncertainty

- **Base strategy:** long high-sentiment, short low-sentiment names, hold t+1..t+5.
- **Uncertainty coupling (compared head-to-head):**
  - *Filter:* trade only when uncertainty is below a threshold (selective prediction).
  - *Sizing:* position size scaled by confidence / inverse uncertainty.
- **Evaluation:** cumulative return, Sharpe, max drawdown, hit rate, turnover, transaction-cost sensitivity, and alpha vs. factors — base vs. uncertainty-aware.



### 2.6 Dashboard (Streamlit)

Cumulative virtual return vs. benchmark, trades table with per-trade sentiment & uncertainty, signal distribution, calibration plots (reliability diagram, risk–coverage curve with AURC), factor-regression results, and an interactive uncertainty-threshold slider. Runs locally for free; optional deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) (free tier).

---



## 3. Free tooling stack

All analysis and visualization dependencies are open source (see `pyproject.toml`):

| Component | Tool |
| --- | --- |
| Data processing | numpy, pandas, pyarrow |
| Econometrics | statsmodels (panel, Fama-MacBeth) |
| ML / calibration | scikit-learn (active learning, ECE) |
| Plots | matplotlib, plotly |
| Config | pyyaml + `config/config.yaml` |
| Data fetch (optional extra) | yfinance, datasets (`pip install -e ".[data]"`) |
| LLM (optional extra) | torch, transformers, accelerate (`pip install -e ".[llm]"`) |
| Dashboard | streamlit |

**Optional paid sources** (not used by default): Financial Modeling Prep transcripts, Refinitiv/Zacks analyst consensus, CRSP/Norgate for delisted-ticker price history. The free stack is sufficient for a research MVP; upgrade paths are documented under `limitations` in `config/config.yaml`.

---



## 4. Repository structure

```text
sentiment-certainty-signals/
├── README.md
├── pyproject.toml            # dependencies & tooling
├── config/
│   └── config.yaml          
├── data/                     
├── src/
│   ├── data/                 
│   ├── nlp/                  # LLM inference, MC sampling, entropy/confidence/agreement, reasoning
│   ├── calibration/          # risk-coverage, AURC, reliability/ECE, active-learning loop
│   ├── signals/              # signal build, standardization, orthogonalization, regressions
│   ├── backtest/             # portfolio construction, strategy, uncertainty coupling, metrics
│   └── dashboard/            # streamlit app
├── scripts/                  # stage entrypoints
├── notebooks/                # exploratory analysis
└── tests/
```

