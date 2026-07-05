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

### 2.1 Data
- **Universe:** S&P 500, roughly the last 5 years. Point-in-time constituents to reduce survivorship bias.
- **Transcripts:** Financial Modeling Prep (FMP) earnings-call transcript endpoints, per ticker / quarter / year.
- **Prices & fundamentals:** Adjusted OHLCV (FMP or `yfinance`), market cap, book-to-market.
- **Factors / controls:** 12-1 month momentum, size (market cap), value (B/M), standardized unexpected earnings (SUE), announcement-day return, and Fama-French 5 + momentum factors (Ken French data library) for alpha attribution.

**Look-ahead safety:** every transcript is timestamped to when it became public; signals only use information available at (or after) that time. Trades are entered on t+1.

### 2.2 Sentiment + uncertainty (core)
A local instruction-tuned LLM (e.g. Qwen / Llama via Hugging Face) classifies the forward-looking sentiment of each transcript (and optionally its sections: prepared remarks vs. Q&A) zero-shot into an ordinal outlook (bearish / neutral / bullish) plus a self-reported confidence.

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
Cumulative virtual return vs. benchmark, trades table with per-trade sentiment & uncertainty, signal distribution, calibration plots (reliability diagram, risk–coverage curve with AURC), factor-regression results, and an interactive uncertainty-threshold slider.

---

## 3. Pipeline

```mermaid
flowchart LR
  ingest[Ingest: transcripts, prices, factors] --> nlp[LLM sentiment + MC uncertainty]
  nlp --> calib[Calibration: reasoning, AURC, active learning]
  calib --> signal[Signal + factor controls]
  signal --> backtest[Backtest + uncertainty coupling]
  backtest --> dash[Streamlit dashboard]
```

---

## 4. Repository structure

```text
sentiment-certainty-signals/
├── README.md
├── pyproject.toml            # dependencies & tooling
├── config/
│   └── config.yaml           # universe, dates, model, sampling N, thresholds, costs
├── data/                     # raw / interim / processed (gitignored)
├── src/
│   ├── data/                 # FMP transcripts, prices, constituents, factors, SUE
│   ├── nlp/                  # LLM inference, MC sampling, entropy/confidence/agreement, reasoning
│   ├── calibration/          # risk-coverage, AURC, reliability/ECE, active-learning loop
│   ├── signals/              # signal build, standardization, orthogonalization, regressions
│   ├── backtest/             # portfolio construction, strategy, uncertainty coupling, metrics
│   └── dashboard/            # streamlit app
├── scripts/                  # stage entrypoints (run_ingest.py, run_nlp.py, ...)
├── notebooks/                # exploratory analysis
└── tests/
```

---

## 5. Installation

```bash
git clone <repo-url>
cd sentiment-certainty-signals
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

Set credentials (e.g. in a `.env`, not committed):

```bash
FMP_API_KEY=your_key_here
```

> Compute note: local LLM inference with MC sampling over thousands of transcripts realistically needs a GPU. Without one, use `config.yaml` to select a smaller model, a prototype ticker sample, or an API backend.

---

## 6. Usage

```bash
python scripts/run_ingest.py       # transcripts, prices, factors, SUE
python scripts/run_nlp.py          # sentiment + uncertainty (entropy/confidence/agreement)
python scripts/run_calibration.py  # reasoning, AURC, active-learning loop
python scripts/run_signals.py      # signal + factor-control regressions
python scripts/run_backtest.py     # base vs. uncertainty-aware strategy
streamlit run src/dashboard/app.py # explore results
```

Key configuration lives in `config/config.yaml` (universe, date range, LLM name, sampling `N`, temperature, holding horizon, uncertainty thresholds, transaction costs).

---

## 7. Roadmap

- [ ] Project scaffolding, config, dependencies
- [ ] Data ingestion (constituents, transcripts, prices, factors, SUE)
- [ ] LLM sentiment + MC uncertainty (entropy / confidence / agreement)
- [ ] Calibration: zero-shot reasoning, risk-coverage / AURC, active learning
- [ ] Signal construction + factor-control regressions (incremental-alpha test)
- [ ] Backtest: base vs. uncertainty-coupled strategy
- [ ] Streamlit dashboard
- [ ] Reproducibility & documentation

---

## 8. References
- Fama & French factor models; Fama-MacBeth regressions.
- Post-earnings-announcement drift (PEAD) literature.
- Selective prediction & Risk-Coverage / AURC (e.g. Geifman & El-Yaniv).
- LLM uncertainty via sampling / self-consistency and calibration.

## 9. License
TBD.
