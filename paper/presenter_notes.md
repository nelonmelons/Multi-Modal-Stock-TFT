# Presenter Notes: Multi-Horizon Stock Prediction

**Conference Presentation (~20 minutes)**

---

## Pre-Presentation Checklist

- [ ] Test laptop connection to projector
- [ ] Check font sizes are readable from back of room
- [ ] Have backup USB with PDF
- [ ] Water/coffee ready
- [ ] Timer set for 18 minutes (leave 2 min buffer)

---

## SLIDE 1: Title Slide (30 sec)

**What to say:**
"Good morning/afternoon everyone. My name is Nelson Siu from the University of Toronto, and I'm here with Dr. Jonathan Chan from KMUTT in Thailand. Today we're presenting our work on benchmarking transformers and baseline models for multi-horizon stock return prediction."

**Key point:** Establish credibility, set the stage

---

## SLIDE 2: Outline (15 sec)

**What to say:**
"We'll cover our motivation, hypotheses, methodology, results, and what this means for both practitioners and researchers in the field."

**Action:** Quickly scan through - don't linger

---

## SLIDE 3: The Challenge (1 min)

**What to say:**
"Stock prediction is notoriously difficult. We have extremely noisy data with low signal-to-noise ratios. Patterns change across different market regimes - what works in a bull market may fail in a bear market. And markets are efficient, limiting predictability.

But this matters enormously for trading strategies, risk management, and understanding market dynamics.

Our key question is: Do complex transformer models actually outperform simpler baselines on daily stock data? To answer this, we benchmark across multiple time horizons, model classes, and market conditions."

**Emphasis:** Pause after "key question" - this is the hook

---

## SLIDE 4: Research Gap (1 min)

**What to say:**
"Looking at the literature, we found four major gaps.

First, most studies don't do fair comparisons - they test transformers, RNNs, and tabular models on different features or datasets.

Second, event features like earnings announcements are rarely integrated into deep learning models, despite known predictive power.

Third, many protocols risk look-ahead bias - accidentally using future information.

And fourth, there's limited evaluation across different market conditions.

Our contribution is an apples-to-apples comparison with strict temporal validation, a fixed universe to avoid survivorship bias, and comprehensive regime and event analysis."

**Emphasis:** "apples-to-apples" and "strict temporal validation"

---

## SLIDE 5: Our Contributions (1.5 min)

**What to say:**
"Let me highlight our four main contributions.

First, robust experimental design. We use a strict temporal split - training from 2016 to 2019, holding out 2020 as validation, and testing on 2021 to 2024. This covers multiple market regimes. We fixed the Dow 30 constituents to avoid survivorship bias, and used forward-only validation with 5-day embargos to prevent leakage.

Second, fair model comparison. Every model sees identical features. We test RNNs, transformers, and tabular models, running each three times with different random seeds for reliability.

Third, comprehensive evaluation using multiple metrics - RMSE, R-squared, and directional accuracy - across all horizons, with specific analysis of bear versus bull markets and earnings windows.

And fourth, we found some surprising insights that challenge common assumptions about complex models."

**Action:** Gesture to each point as you mention it

---

## SLIDE 6: Three Core Hypotheses (1.5 min)

**What to say:**
"We tested three specific hypotheses.

H1: Model class advantage. We hypothesized that the Temporal Fusion Transformer would outperform baselines at the 21-day horizon, based on the assumption that attention mechanisms handle long-term dependencies better. We set clear criteria: at least 2 percentage points improvement in directional accuracy or 5% reduction in RMSE.

H2: Event features add value. We predicted that adding earnings features would improve directional accuracy without worsening error, based on well-documented post-earnings-announcement drift.

H3: Regime dependence. Following the Adaptive Markets Hypothesis, we expected model rankings to differ between bear and bull markets - what works in calm conditions may fail in volatility.

These hypotheses give us concrete, testable predictions rather than just reporting results."

**Pause:** After each hypothesis, give audience time to absorb

---

## SLIDE 7: The Model Landscape (1 min)

**What to say:**
"Before diving into our work, let's quickly orient ourselves in the model landscape.

Traditional approaches include ARIMA and Ridge regression - simple, fast, interpretable, but they miss temporal patterns and assume linearity.

Deep learning approaches like LSTM, GRU, and transformers can capture complex temporal dependencies, but they need more data and are prone to overfitting with weak signals.

The key insight from the literature is that there's no universal winner - performance is highly task and data dependent. That's exactly what we wanted to test rigorously."

**Action:** Gesture to left side, then right side of slide

---

## SLIDE 8: Recent Trends: Why Transformers? (1 min)

**What to say:**
"Transformers have revolutionized NLP and time-series forecasting. PatchTST improves long-horizon forecasting, TimesFM enables zero-shot prediction, and financial LLMs like BloombergGPT integrate text and prices.

The TFT architecture has several appealing features: static encoders for metadata, variable selection networks, multi-head attention for learning dependencies, and interpretable attention weights.

Our question is: Do these advantages actually hold on small, daily financial panels? That's what we're here to find out."

**Emphasis:** "small, daily financial panels" - this is the key constraint

---

## SLIDE 9: Experimental Design Overview (1.5 min)

**What to say:**
"Here's our experimental setup. We focus on the Dow 30 - 30 large US stocks, fixed as of 2018.

The timeline is crucial. Blue shows training from 2016 to 2019. Orange is 2020, held out for validation - this COVID year helps select hyperparameters without leaking into test. Red is our test period, 2021 to 2024, spanning multiple market regimes.

We predict at three horizons: h equals 1, 5, and 21 trading days - next day, one week, and one month. Our target is log returns, which are more statistically well-behaved than raw returns."

**Action:** Point to each colored region on the timeline

---

## SLIDE 10: Data & Features (1 min)

**What to say:**
"We use two feature groups. Technical indicators include momentum, moving averages, volatility measures like rolling standard deviation, RSI, MACD, and volume features.

Earnings features include surprise percentage, days to or from earnings announcements, event flags, and indicators for ±3 day windows around announcements.

Data comes from yfinance for prices, API Ninjas for earnings, and FRED for macro variables.

Critically, we have strong leakage controls: after-market data is shifted to t+1, we use forward-only blocks with 5-day embargos, scaling is fit on training data only. Models train globally on pooled data from all 30 stocks."

**Emphasis:** "leakage controls" - credibility point

---

## SLIDE 11: Models Tested (1.5 min)

**What to say:**
"On the left, our tabular models. Ridge regression is our linear baseline with L2 regularization. Random Forest with 100 trees and depth 5 is a strong tabular learner. XGBoost uses gradient boosting with shallow trees. These are trained separately for each horizon.

On the right, sequence models. LSTM and GRU both have 64 hidden units, one layer, and look back 60 days. The Temporal Fusion Transformer uses 4 attention heads, 1 layer, hidden dimension 16, with about 50,000 parameters. Sequence models predict all three horizons simultaneously.

This gives us a nice range from simple to complex, tabular to sequential."

**Action:** Gesture left, then right

---

## SLIDE 12: TFT Architecture (30 sec)

**What to say:**
"Here's the TFT architecture in more detail. Static encoders handle stock metadata, historical encoders process past prices and technical indicators, future encoders handle known covariates. Multi-head attention learns temporal dependencies, and separate heads predict each horizon simultaneously.

We kept the configuration compact to prevent overfitting on this daily panel."

**Action:** Trace the flow with laser pointer if available

---

## SLIDE 13: Evaluation Metrics (1 min)

**What to say:**
"We use three complementary metrics.

RMSE measures prediction precision in log-return units - interpretable and common.

R-squared is computed against a zero-return baseline, representing the naive strategy of always predicting zero. Negative values are common in finance due to low signal-to-noise and actually reflect market efficiency rather than model failure.

Directional accuracy is the fraction of correct sign predictions. This is most important for trading - getting the direction right matters more than the exact magnitude. 50% is random; anything above 50% shows predictive power."

**Emphasis:** "Directional accuracy is most important for trading"

---

## SLIDE 14: Main Results (2 min)

**What to say:**
"Here are our main results across all three horizons.

At h=1, next day, LSTM achieves the lowest RMSE at 0.01622, just barely ahead of GRU. Random Forest has the highest directional accuracy at 51.9%. TFT is actually worst here at 48.9% DA.

At h=5, one week, Ridge regression dominates on RMSE at 0.03651 - about 37% lower than the deep models. This suggests weekly signals are largely linear. But LSTM has the best directional accuracy at 55.4%.

At h=21, one month - our primary horizon - LSTM wins both metrics: lowest RMSE at 0.05084 and highest DA at 55.0%. It's 31 to 38% better than tabular baselines on RMSE. TFT is competitive at 0.05096 RMSE and 53.9% DA, but it doesn't lead.

The key takeaway: LSTM wins at short and long horizons, Ridge dominates at medium, and TFT is competitive but not superior."

**Action:** Point to bold numbers as you mention them

---

## SLIDE 15: Directional Accuracy Across Horizons (45 sec)

**What to say:**
"This plot shows a clear pattern. Sequence models - LSTM and GRU in dark blue and orange - improve from about 51-52% at h=1 to about 55% at longer horizons.

Tabular baselines stay relatively flat or even degrade slightly.

This suggests temporal sequence models benefit more from longer context when predicting multi-week moves, whereas tabular models capture less horizon-dependent signal."

**Emphasis:** "improve" and "stay flat" - the divergence is key

---

## SLIDE 16: Why Are All R² Values Negative? (1 min)

**What to say:**
"You might be wondering why all our R-squared values are negative or near zero. This is actually normal in finance!

R-squared measures performance versus a zero-return baseline - the naive strategy of always predicting zero return. Negative R-squared means the model performs worse than this naive strategy.

This reflects market efficiency and extremely low signal-to-noise ratios, not model failure.

Look at the differences: LSTM's near-zero values around -0.002 suggest better calibration and that it's finding real signal. Random Forest's highly negative values at -0.248 indicate severe overfitting.

Despite negative R-squared, RMSE and directional accuracy reveal meaningful, consistent differences. DA above 50% can be economically significant even when R-squared is negative."

**Emphasis:** "This is normal" and "economically significant"

---

## SLIDE 17: Hypothesis 1 - TFT Superiority? (1 min)

**What to say:**
"Let's evaluate hypothesis 1. We predicted TFT would outperform baselines at the 21-day horizon.

Results: LSTM achieves RMSE 0.05084 and DA 55.0%. TFT gets 0.05096 and 53.9%.

The verdict is clear: H1 is NOT supported. LSTM beats TFT by 0.24% on RMSE and 1.1 percentage points on directional accuracy. These differences don't meet our thresholds of 2 percentage points or 5% improvement.

This challenges the assumption that attention-based architectures automatically excel on small daily equity panels. The additional complexity of attention doesn't help here."

**Emphasis:** "NOT supported" and "challenges the assumption"

---

## SLIDE 18: Hypothesis 2 - Earnings Features Matter? (1.5 min)

**What to say:**
"Hypothesis 2 is more interesting. We predicted earnings features would improve directional accuracy without worsening error.

The ablation study on the left shows that for TFT, adding earnings features improves DA from 52.2% to 53.3% - a 1.1 percentage point gain - while slightly reducing RMSE. XGBoost shows similar patterns.

But look at the earnings window analysis on the right. This is striking: during the ±3 day window around earnings announcements, directional accuracy jumps to 61.4% compared to 53.9% outside these windows. That's a 7.5 percentage point improvement!

Yes, RMSE increases slightly due to higher volatility around earnings, but the directional signal is much stronger.

The verdict: H2 is SUPPORTED. Earnings features provide modest overall gains but substantial improvements during announcement windows, confirming post-earnings-announcement drift."

**Emphasis:** "61.4%" and "7.5 percentage point improvement"

---

## SLIDE 19: Hypothesis 3 - Regime Dependence (1.5 min)

**What to say:**
"Hypothesis 3 predicted model rankings would differ across market regimes.

This heatmap shows RMSE by model in two distinct periods. Darker means better - lower error.

In the 2022 bear market on the left, GRU achieves the lowest RMSE at 0.0636. But look at directional accuracy - all models fall below 50%. They're essentially coin flips in the bear market.

In the 2023-24 rally on the right, LSTM dominates with RMSE 0.0475. Both LSTM and GRU achieve 56.2% directional accuracy - significantly better than in the bear market.

The verdict: H3 is SUPPORTED. Model rankings change dramatically across regimes. Sequence models thrive in bull markets with clear trends but fail to sustain accuracy in high-volatility downturns. This aligns perfectly with the Adaptive Markets Hypothesis."

**Action:** Point to darkest cells in each heatmap

---

## SLIDE 20: Why Did Simple Models Win? (1 min)

**What to say:**
"So why did simpler models match or beat the complex transformer?

Four reasons. First, data constraints: 30 stocks at daily frequency is relatively small for deep learning. TFT's 50,000 parameters may be too many.

Second, signal characteristics: extremely low signal-to-noise ratio. Short-term patterns appear to be quasi-linear, which explains why Ridge dominates at h=5.

Third, model capacity trade-offs: more parameters means harder to train with weak signals. LSTM and GRU have about 25,000 parameters versus TFT's 50,000. Simpler models achieve better bias-variance trade-off.

Fourth, feature engineering matters: our strong technical indicators capture most of the signal, and tabular models exploit these effectively. Deep models don't add much on top."

**Key phrase:** "better bias-variance trade-off"

---

## SLIDE 21: When Do Transformers Shine? (1 min)

**What to say:**
"Our negative result suggests transformers need more to succeed.

They need larger datasets - TimesFM was trained on 100 billion time points; we have about 30 stocks times 8 years.

They need richer modalities: text from news and earnings calls, social media, alternative data like satellite imagery. BloombergGPT and FinGPT show the power of multimodal integration.

They may benefit from longer horizons or higher frequency data - intraday gives more samples, longer-term forecasting may have clearer patterns.

And they need pretraining - transfer learning from related tasks or foundation models for finance.

The key insight: transformers aren't universally better. They shine when you have scale and richness."

**Emphasis:** "scale and richness"

---

## SLIDE 22: Practical Recommendations (1 min)

**What to say:**
"What should practitioners do?

For model selection by horizon: use LSTM or simple tabular methods for daily predictions, Ridge regression for weekly - linear is sufficient - and LSTM or GRU for monthly predictions to capture temporal patterns.

Only use transformers if you have large diverse datasets, multimodal inputs, computational resources for training, and the ability to leverage pretraining.

And don't neglect feature engineering! Strong technical indicators and earnings features capture most of the signal. Traditional feature engineering still matters enormously."

**Action:** This is actionable advice - slow down slightly

---

## SLIDE 23: Economic Significance vs Statistical Accuracy (45 sec)

**What to say:**
"Important caveat: we focus on statistical accuracy - RMSE, R-squared, directional accuracy - not economic profitability.

We don't test trading strategies, transaction costs, slippage, position sizing, or risk management.

Reality check: 55% directional accuracy might be profitable, but it depends heavily on trading frequency, spreads, commissions, and market impact. Daily trading incurs high costs.

Full backtesting is needed to assess economic value. Statistical improvements may not survive real trading costs."

**Tone:** Honest and cautious

---

## SLIDE 24: Limitations (45 sec)

**What to say:**
"Let me acknowledge our limitations.

Limited universe: Dow 30 only - large-cap US stocks may not generalize to small-cap, international, or other assets.

Daily frequency only: we don't explore intraday or test weekly/monthly frequencies.

Hyperparameter tuning was lightweight on the validation set and may favor simpler models. More extensive tuning could change results.

Single split: main results use one temporal split. Rolling or expanding window validation is recommended for robustness.

And extreme events: we include COVID in validation, but black swan events remain challenging for all models."

**Tone:** This builds credibility - acknowledging limitations shows rigor

---

## SLIDE 25: Key Findings Summary (1 min)

**What to say:**
"Let me summarize our four key findings.

One: Simple models match or beat transformers on this dataset. LSTM wins at h=1 and h=21, Ridge at h=5, TFT is competitive but not superior.

Two: Earnings features add value. 1.1 percentage points overall improvement, but 7.5 percentage points during earnings windows - from 53.9% to 61.4%. This confirms post-earnings-announcement drift.

Three: Performance is regime-dependent. GRU best in bear markets, LSTM best in bull markets. All models struggle in high-volatility downturns.

Four: Complexity is not always better. More parameters don't equal better performance. Data constraints favor simpler models. Feature engineering remains crucial."

**Pace:** Slower - these are take-homes

---

## SLIDE 26: Contributions to the Field (45 sec)

**What to say:**
"What makes this work valuable?

Rigorous benchmarking: apples-to-apples comparison across model classes with strict temporal validation and multiple random seeds.

Honest negative results: transformers don't always win. This is important for setting realistic expectations and guiding future research.

Comprehensive evaluation: multi-metric, regime analysis, event windows, ablations.

And we provide a baseline for future work: researchers can compare against our results using similar methodology."

**Emphasis:** "honest negative results" - this is scientifically important

---

## SLIDE 27: Future Directions (1 min)

**What to say:**
"The primary extension is multi-modal integration: news sentiment, earnings call transcripts, social media, macroeconomic text. Our hypothesis is that transformers will excel when processing this heterogeneous data.

Alternative data sources: satellite imagery, credit card data, web scraping, order flow.

Methodological extensions: larger stock universes like the S&P 500, intraday or high-frequency data, longer horizons, cross-asset learning."

**Action:** Keep moving - don't linger on future work

---

## SLIDE 28: Future Work (Continued) (45 sec)

**What to say:**
"Advanced architectures: patch-wise transformers like PatchTST, foundation models with pretraining, hybrid architectures, graph neural networks for sector relationships.

Interpretability: attention weight analysis, feature importance via SHAP and LIME, regime detection, understanding when and why models fail.

And economic evaluation: full backtesting with transaction costs, portfolio optimization, risk-adjusted returns, real-world deployment."

---

## SLIDE 29: Take-Home Messages (1 min)

**What to say:**
"Let me close with take-home messages.

For practitioners: Start simple - LSTM and Ridge can match transformers on daily data. Don't neglect feature engineering. Earnings events provide predictable opportunities. Monitor regime changes carefully. Always validate with strict temporal splits.

For researchers: Transformers need scale - more data and richer modalities. Negative results are valuable - publish them! Multi-modal integration is a promising direction. Benchmark against strong, well-tuned baselines. Economic evaluation is the crucial next step.

Thank you for your attention. I'm happy to take questions."

**Pace:** Slow down for final slide - let each point land

---

## SLIDE 30: Thank You / Q&A

**What to say:**
"Thank you very much. I'll be happy to answer any questions."

**Then:** Wait for questions. Don't rush to fill silence.

---

## Anticipated Questions & Answers

### Q: Why didn't you test on more stocks?

**A:** "Great question. We used the Dow 30 to have a fixed, well-defined universe without survivorship bias. Testing on larger universes like the S&P 500 is definitely valuable future work, but we wanted to ensure rigorous controls first. The Dow 30 gives us 30 stocks times roughly 2000 trading days, which is about 60,000 stock-day observations."

### Q: Have you done any backtesting with real trading costs?

**A:** "Not yet - this study focused on statistical accuracy as a necessary first step. We explicitly acknowledge in the paper that 55% directional accuracy might not survive transaction costs, especially for daily trading. Full economic evaluation with realistic costs, slippage, and position sizing is crucial future work."

### Q: Why use log returns instead of raw returns?

**A:** "Log returns have several advantages: they're more statistically well-behaved, approximately normally distributed for short horizons, time-additive, and symmetric for gains and losses. They're also standard in academic finance research."

### Q: Could the TFT do better with more hyperparameter tuning?

**A:** "Possibly, though we did tune on the validation set. The deeper issue is that TFT has about 50,000 parameters on a dataset where the signal-to-noise ratio is extremely low. More tuning might help marginally, but it's unlikely to overcome the fundamental bias-variance trade-off. The simpler LSTM with 25,000 parameters seems better suited to this constraint."

### Q: What about using technical indicators with transformers differently?

**A:** "That's an interesting idea. We fed technical indicators the same way to all models for fairness. Transformers might benefit from treating raw prices and indicators as separate modalities with different encoders. This goes back to our point about richer modalities - transformers may need more architectural customization to excel."

### Q: How do you explain the ridge regression success at h=5?

**A:** "Ridge regression is essentially finding linear combinations of our engineered technical features. The fact that it dominates at h=5 suggests that weekly signals are largely captured by these linear relationships. The features themselves - momentum, moving averages, volatility - already encode nonlinear transformations of prices. Ridge exploits these effectively without overfitting."

### Q: Why did all models fail in the 2022 bear market?

**A:** "The 2022 bear market had extremely high volatility and regime shifts - the Fed raised rates aggressively, inflation spiked, and correlations broke down. Pattern-based supervised learning struggles when historical patterns become unreliable. This highlights a fundamental limitation of pure ML approaches without macroeconomic understanding."

### Q: Could you use sentiment from news to improve predictions?

**A:** "Absolutely - that's our primary future direction. We strongly believe that transformers will show their advantage when processing heterogeneous data: news sentiment, earnings call transcripts, social media, combined with prices. The attention mechanism should excel at fusing these different modalities. That's where we expect to see TFT outperform simpler models."

---

## Timing Guide (Total: ~20 minutes)

- **Introduction (Slides 1-5):** 5 minutes
- **Hypotheses & Background (Slides 6-8):** 3.5 minutes
- **Methodology (Slides 9-13):** 6 minutes
- **Results (Slides 14-19):** 8.5 minutes
- **Discussion (Slides 20-24):** 4 minutes
- **Conclusions (Slides 25-29):** 4 minutes
- **Q&A (Slide 30):** Remaining time

**Total presentation: 18-19 minutes, leaving 1-2 minutes buffer**

---

## Final Tips

1. **Pace yourself:** If running behind, you can skip:

   - Slide 8 (Recent Trends) - less critical
   - Slide 12 (TFT Architecture) - can describe verbally
   - Slide 23 (Economic Significance) - already mentioned in limitations
   - Slide 28 (Future Work Continued) - combine with Slide 27

2. **If running ahead:** Elaborate more on:

   - Hypothesis evaluation (Slides 17-19) - core contribution
   - Why simple models won (Slide 20) - key insight
   - Practical recommendations (Slide 22) - audience value

3. **Engagement:**

   - Make eye contact with different parts of the room
   - Use hand gestures for emphasis
   - Pause after key findings
   - Smile - enthusiasm is contagious!

4. **Technical issues:**

   - If figures don't display: "As you can see in the paper..."
   - If laser pointer fails: "In the top left corner..."
   - If time display fails: Ask audience member to signal at 15 min

5. **Nerves:**
   - Take deep breaths before starting
   - Focus on sharing exciting findings, not performing
   - Remember: you know this material better than anyone in the room
   - It's okay to say "That's a great question - let me think for a moment"

---

**Good luck! You've got this! 🎤📊**
