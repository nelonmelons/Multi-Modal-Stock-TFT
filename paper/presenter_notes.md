# Presenter Notes: Multi-Horizon Stock Prediction

**IEEE ICITEE 2025 Conference Presentation (~20 minutes)**

---

## Pre-Talk Setup

- Test projector connection and backup slides on USB
- Have water ready
- Timer set to 18 minutes (2 min buffer for Q&A)
- Take a breath—you've got this!

---

## SLIDE 1: Title (30 seconds)

Good morning everyone. I'm Nelson Siu from the University of Toronto, and I'm here today with my collaborator Dr. Jonathan Chan from King Mongkut's University of Technology Thonburi in Thailand.

We're going to talk about something that's been a bit of a hot topic in quantitative finance lately—whether these fancy transformer models that have been dominating natural language processing actually help us predict stock returns better than simpler approaches.

---

## SLIDE 2: The Challenge & Our Approach (2 minutes)

So the question we wanted to answer is whether transformers actually beat simpler models on daily stock data.

This is hard because stock returns are brutally noisy, and what works in a bull market can completely fail in a bear market. Plus you've got market efficiency working against you the whole time.

But this matters a lot for trading, risk management, portfolio optimization—we really need to know what actually works in practice.

So our approach was to do a fair comparison where every model gets the exact same features. We use strict temporal validation, test at three horizons—next day, one week, and one month—and we break down the results by market regime and look specifically at earnings events.

We're using the Dow 30 from 2016 to 2024, which gives us a bunch of different market conditions to test against.

---

## SLIDE 3: Three Core Hypotheses (2 minutes)

Instead of just throwing models at data, we set up three specific hypotheses to test.

First hypothesis is that TFT beats the baselines at 21 days because attention mechanisms should handle long-term dependencies better. We set a clear bar—we need at least 2 percentage points better directional accuracy or 5% lower error to call it a win.

Second hypothesis is that adding earnings features improves directional accuracy without making the errors worse. This is based on the fact that markets don't immediately price in earnings—there's this documented drift that happens after announcements.

Third hypothesis is that model rankings shift across different market conditions. So what works in calm markets might completely fail when things get volatile, and we'd expect to see different winners in the 2022 bear market versus the 2023-24 rally.

The nice thing is we've got concrete criteria we can actually test against.

---

## SLIDE 4: Literature & Research Gaps (90 seconds)

Let me give you quick background on what's been tried in this space.

Classical methods like ARIMA and VAR assume linearity, and they work okay for short horizons but really struggle when you hit regime shifts. Tree ensembles like XGBoost and Random Forest are strong on cross-sectional tasks, and RNNs like LSTM and GRU are good at capturing temporal patterns and often beat the classical approaches.

More recently we've seen transformers come in. TFT uses attention for multi-horizon forecasting, PatchTST improves long-horizon prediction, and there's foundation models like TimesFM that do zero-shot forecasting. BloombergGPT even integrates text and prices together. These all work well when you have scale and multimodal data.

But there are some real gaps in the literature. Very few studies actually compare all these models with the same features, earnings features are pretty underused in deep learning models, and evaluation protocols sometimes have look-ahead bias or they're missing regime analysis entirely.

So that's what we're addressing here—doing a fair comparison where identical features go to all models, combining earnings with technical inputs, using strict temporal validation, and breaking things down by regime.

---

## SLIDE 5: Experimental Design (90 seconds)

So we're using the Dow 30 stocks, and we fixed the composition as of 2018 to avoid survivorship bias.

The timeline works like this: we train on 2016 to 2019, then we hold out 2020 for validation—that COVID year actually helps us tune hyperparameters without leaking into the test set. Then we test on 2021 to 2024, which gives us four years of test data spanning really different market regimes.

We're predicting at three horizons: 1, 5, and 21 days, so that's next day, one week, and one month out. We use log returns because they behave better statistically.

The key thing is we make predictions at every single time point, so this multi-year test period lets us see how the models actually perform across different conditions and it prevents them from just overfitting to one particular year.

---

## SLIDE 6: Data & Features (90 seconds)

We've got two main feature groups here. Technical indicators include your standard momentum, moving averages, volatility measures, RSI, and MACD. Then earnings features include surprise percentage, days until or since the announcement, and event flags for that plus or minus 3 day window around earnings.

Data comes from yfinance for prices, API Ninjas for earnings data, and FRED for the macro variables.

Now the really critical part is leakage control, and this is where a lot of studies actually get it wrong. We shift any after-market data to the next day, we use forward-only validation with 5-day embargos between splits, and the scaling parameters get fit only on training data—absolutely no peeking at the future.

The models train on pooled data from all 30 stocks, so we're learning shared parameters across both time and stocks.

---

## SLIDE 7: Models Tested (2 minutes)

On the tabular side we've got Ridge regression as our linear baseline, Random Forest with 100 trees and depth capped at 5, and XGBoost with shallow trees. These models train separately for each horizon.

For sequence models we're using LSTM and GRU, both with 64 hidden units and a 60-day lookback window. Then there's TFT—the Temporal Fusion Transformer. We kept it pretty compact with 4 attention heads, 1 layer, and about 50,000 parameters total. We didn't go crazy with the capacity because we knew we were working with limited data. The sequence models predict all three horizons at once, which is actually nice because they can share representations across the different time scales.

We're evaluating with three metrics. RMSE measures the prediction error, R-squared is computed against a zero-return baseline—I want to flag this because all our R-squared values are going to be negative or near zero, and that's completely normal in finance. It reflects market efficiency, not model failure. Then directional accuracy is just the fraction of times we get the sign right, and this actually matters most for trading. Fifty percent would be random coin-flipping, anything above 50% means you've got real signal.

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

## SLIDE 8: Main Results (2.5 minutes)

Alright, so the results are honestly pretty surprising.

At the one-day horizon, LSTM gets the lowest RMSE at 0.01622, just barely ahead of GRU. Random Forest actually has the highest directional accuracy at 51.9%. And TFT is the worst performer here with only 48.9% directional accuracy.

At five days things get really interesting. Ridge completely dominates on RMSE at 0.03651, which is about 37% lower than the deep models. What this tells us is that the weekly signal is actually being captured by linear combinations of our engineered features—you don't need deep learning for this. But LSTM still has the best directional accuracy at 55.4%.

Now at twenty-one days, which is our primary horizon, LSTM wins on both metrics. It gets 0.05084 RMSE and 55% directional accuracy, beating Ridge by 31 to 38 percent on the error. TFT is competitive here—0.05096 RMSE and 53.9% accuracy—but it doesn't actually lead.

So the big takeaway is LSTM wins at both short and long horizons, Ridge dominates at the medium horizon, and TFT is competitive throughout but never actually superior.

---

## SLIDE 9: Key Observations (90 seconds)

So this plot really tells the story here. You can see the sequence models—LSTM and GRU—improving from around 51-52% at day 1 all the way up to 55% at the longer horizons. There's a clear upward trend there.

The tabular models, on the other hand, they stay pretty flat or even degrade a bit. Ridge, Random Forest, XGBoost—they're not getting any better as the horizon extends.

TFT stays competitive throughout but it never actually takes the lead at any horizon.

What this suggests is that temporal models are benefiting from that longer context when you're trying to predict multi-week moves. They're actually learning something about the dynamics. Tabular models are mostly capturing instantaneous relationships in the feature space, not these horizon-dependent patterns.

And yes, all the R-squared values are negative or near zero, but that's completely normal in finance—returns are just incredibly noisy. LSTM being near zero actually means better calibration. Random Forest at -0.248 is clearly overfitting.

---

## SLIDE 10: Hypothesis Testing (2 minutes)

Let's talk about the hypotheses.

Hypothesis 1 was about TFT superiority, and it's not supported. At 21 days, LSTM gets 55% directional accuracy while TFT only gets 53.9%, and LSTM also has slightly lower RMSE. This doesn't meet our threshold for saying TFT is better. What we're seeing is that attention mechanisms aren't providing any advantage on this daily stock panel, and LSTM's simpler structure is actually better suited to these noisy returns.

Hypothesis 3 about regime dependence is supported. In the 2022 bear market, GRU performs best on RMSE, but here's the kicker—all the models fall below 50% directional accuracy during that period. They're basically coin flips. But in the 2023-24 rally, LSTM completely dominates with 56.2% directional accuracy. So the rankings completely change across regimes, and what we're seeing is that sequence models thrive when there are clear trends but they fail in high volatility environments.

---

## SLIDE 11: Earnings Features (2 minutes)

Hypothesis 2 is where things get really interesting.

We ran an ablation study comparing technical features only versus technical plus earnings. At the 21-day horizon, adding earnings features gives us about a 1.1 percentage point improvement, which is modest but consistent across the board.

But look at what happens during those earnings windows—the plus or minus 3 days around announcements. Directional accuracy jumps all the way up to 61.4% compared to 53.9% outside those windows. That's 7.5 percentage points, which is huge.

Now RMSE does increase a bit during earnings windows because volatility is naturally higher around those events. But the directional signal is so much stronger that it more than makes up for it. This really confirms the post-earnings drift phenomenon—markets genuinely don't immediately price in earnings information.

So Hypothesis 2 is supported. Earnings features give you modest overall gains, but you get substantial improvements during the announcement windows themselves. If you're trading around earnings, these features really matter.

---

## SLIDE 12: Why Simple Models Win (2 minutes)

So why did simpler models match or even beat TFT? There are four main reasons.

First is data constraints. We're working with 30 stocks at daily frequency, which is pretty small for deep learning. TFT has 50,000 parameters, and that might just be too many for this amount of data. You really need a lot of data to properly train those attention mechanisms.

Second is signal characteristics. There's extremely low signal-to-noise ratio in stock returns, and it looks like the short-term patterns are actually quasi-linear, which explains why Ridge completely dominates at the 5-day horizon. You just don't need complex nonlinear models for that.

Third is model capacity trade-offs. More parameters means harder training when your signals are this weak. LSTM and GRU have about 25,000 parameters compared to TFT's 50,000, and they're achieving a better bias-variance trade-off on this particular dataset.

Fourth is feature engineering. Our technical indicators are already capturing most of the signal, and tabular models can exploit these really effectively. The deep models aren't adding much on top of that good feature engineering.

So when do transformers actually help? They need scale—we're talking 100 billion time points like TimesFM has. They need multimodal inputs like text combined with prices. And they benefit a lot from pretraining and transfer learning. We just don't have any of that here.

---

## SLIDE 13: Key Takeaways & Future Work (2 minutes)

Let me wrap up with the main findings.

First, simple models are genuinely competitive here. LSTM and Ridge can match transformers on this daily stock data, so you don't need to overcomplicate things.

Second, earnings features are really valuable, especially during those announcement windows where we saw that 7.5 percentage point boost in directional accuracy.

Third, performance is highly regime-dependent. What works great in a bull market can completely fail in a bear market, so you need to actively monitor this.

Fourth, transformers aren't magic—they need scale to work well. They shine when you have massive amounts of data and rich, heterogeneous inputs.

For future work, the obvious next step is multimodal integration where you're combining text from news and earnings calls with the price data. That's really where transformers might pull ahead. We could also scale this up to the S&P 500, test on intraday data, explore newer architectures like PatchTST. And critically, we need economic evaluation—actual trading simulations with transaction costs, not just statistical metrics.

Bottom line is start simple, engineer your features carefully, and validate rigorously. Transformers excel when you have scale and multimodal data, but not on small daily panels like this.

---

## SLIDE 14: Q&A

Thank you very much. I'm happy to answer any questions.

[Wait for questions. Don't rush to fill silence. Make eye contact. Smile.]

---

---

# BACKUP: Anticipated Questions & Natural Responses

## Q: Why didn't you test on more stocks?

Good question. We used Dow 30 for a fixed universe—no survivorship bias. S&P 500 is valuable future work, but we wanted rigorous controls first. Dow 30 gives us 30 stocks times 2000 days, about 60,000 observations. Substantial for initial study.

---

## Q: Have you done any backtesting with real trading costs?

Not yet. This focused on statistical accuracy first. We acknowledge in the paper that 55% accuracy might not survive transaction costs, especially daily trading with spreads and commissions. Economic evaluation with costs, slippage, position sizing is crucial future work. Can't claim profitability without that.

---

## Q: Why use log returns instead of raw returns?

Log returns are better behaved statistically. Approximately normal for short horizons. Time-additive—you can sum them across days. Symmetric for gains and losses. Also standard in academic finance, makes our results comparable.

---

## Q: Could the TFT do better with more hyperparameter tuning?

Possibly, though we did tune on validation. Deeper issue: TFT has 50,000 parameters where signal-to-noise is extremely low—R-squared around 1%. More tuning might help marginally, but won't overcome the bias-variance trade-off. LSTM with 25,000 parameters better suited. Data problem, not tuning problem.

---

## Q: What about using technical indicators with transformers differently?

Interesting idea. We fed indicators the same way to all models for fairness. Transformers might benefit from treating prices and indicators as separate modalities with different encoders, maybe cross-attention. Goes back to richer modalities—transformers may need more customization. Worth exploring.

---

## Q: How do you explain the Ridge regression success at h=5?

Ridge finds linear combinations of our engineered features. That it dominates at 5 days suggests weekly signals are captured by linear relationships. The features themselves—momentum, moving averages, volatility—already encode nonlinear transformations. Ridge exploits these without overfitting.

---

## Q: Why did all models fail in the 2022 bear market?

2022 had extreme volatility and regime shifts—Fed raised rates aggressively, inflation spiked, correlations broke down. Pattern-based learning struggles when historical patterns become unreliable. Fundamental limitation of ML without macro understanding. Can't predict regimes you've never seen.

---

## Q: Could you use sentiment from news to improve predictions?

Absolutely—primary future direction. Transformers will show advantage with heterogeneous data: news sentiment, call transcripts, social media, combined with prices. Attention should excel at fusing modalities. That's where TFT might outperform. Text and prices together—real test.

---

## Q: Why is your R-squared so low—only 1-2%?

Welcome to stock prediction. This is typical for daily returns. Stock returns are extremely noisy—dominated by unpredictable shocks. Academic literature shows R-squared in this range. Even 1-2% can be economically significant. Key is directional accuracy—getting the sign right—and that's 53-55%. Low R-squared reflects fundamental unpredictability of daily moves.

---

## Q: What about using longer sequences—maybe attention with 500 days instead of 60?

We chose 60 days—3 months—based on typical lookback in technical analysis. Longer sequences could help, transformers designed for long-range dependencies. But more parameters to estimate, more computational cost. With our data constraints, worried about overfitting. Worth exploring with sufficient data—where attention might shine.

---

## Q: Did you try any ensembles?

Didn't formally test ensembles, but natural next step. Stacking could combine strengths—LSTM for short horizons, Ridge for medium, TFT for specific scenarios. Given different models excel in different regimes, regime-adaptive ensemble could be powerful.

---

---

## Timing Guide (Total: ~20 minutes)

- **Introduction (Slides 1-5):** 5 minutes
- **Hypotheses & Background (Slides 6-8):** 3.5 minutes
- **Methodology (Slides 9-13):** 6 minutes
- **Results (Slides 14-19):** 8.5 minutes
- **Discussion (Slides 20-24):** 4 minutes
- **Conclusions (Slides 25-29):** 4 minutes
- **Q&A (Slide 30):** Remaining time

---

# PRESENTATION TIMING GUIDE

**Target: 18-20 minutes total**

- Slide 1 (Title): 30 seconds
- Slide 2 (Introduction & Approach): 2 minutes
- Slide 3 (Hypotheses): 1.5 minutes
- Slide 4 (Literature & Research Gaps): 1.5 minutes
- Slide 5 (Experimental Design): 1.5 minutes
- Slide 6 (Data & Features): 1.5 minutes
- Slide 7 (Models Tested): 1.5 minutes
- Slide 8 (Main Results - RMSE): 2 minutes
- Slide 9 (Key Observations): 1.5 minutes
- Slide 10 (Hypothesis Testing - H1 & H3): 2 minutes
- Slide 11 (H2 - Earnings Features): 2 minutes
- Slide 12 (Why Simple Models Win): 2 minutes
- Slide 13 (Key Takeaways & Future): 2 minutes
- Slide 14 (Q&A): Transition

**Total: ~21 minutes**

---

# DELIVERY TIPS

## Pacing

- If running behind: Speed up on Slide 4 (Literature) or combine with intro—it's context, not core results. Also speed up on Slides 6 (Data) and 7 (Models)—they're descriptive
- If running ahead: Elaborate on Slides 10-11 (hypothesis testing) and Slide 12 (why simple models win)—those are your main contributions

## Engagement

- Make eye contact with different parts of the room
- Use hand gestures for emphasis on key findings
- Pause after surprising results (Ridge winning at h=5, TFT being worst at h=1)
- Smile—show enthusiasm for the findings!

## Handling Questions

- Listen fully before responding
- Repeat the question if the room is large
- It's okay to say "That's a great question—let me think for a moment"
- If you don't know: "That's an interesting point I hadn't considered—worth exploring"
- Bridge back to your core findings when possible

## Technical Issues

- If figures don't display clearly: "As you can see in the paper, Figure X shows..."
- If laser pointer fails: "In the top left corner..." or "The blue line..."
- If running low on time: Skip slide details, hit the key number

## Managing Nerves

- Take a deep breath before starting
- Focus on sharing exciting findings, not performing perfectly
- Remember: you know this material better than anyone in the room
- Speak to friendly faces first to build confidence
- It's a conversation about your work, not a performance

---

**You've got this! Good luck! 🎤📊**

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
