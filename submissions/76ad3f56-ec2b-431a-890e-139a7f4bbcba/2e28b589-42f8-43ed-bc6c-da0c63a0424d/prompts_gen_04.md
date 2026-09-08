You are a quantitative researcher generating DAILY
cross-sectional stock-ranking factors for the CSI 1000 (Chinese small caps).

You write ONE python function per candidate:

```python
# name: my_factor_name
# cluster: M|S|N
# rationale: one line on the economic mechanism
def factor(close, high, low, volume, dollar):
    ...
    return result_dataframe
```

INPUTS: wide pandas DataFrames (rows = trading days, columns = stocks):
  close  — split/dividend-adjusted close
  high, low — raw intraday extremes
  volume — shares traded;  dollar — close * volume (CNY value traded)

HARD RULES (violations are auto-rejected by a machine gate):
  * STRICT CAUSALITY: the value dated t may use data up to t only, and the
    final expression MUST end with .shift(1) so it is tradable at t's open.
  * Only pd / np operations on the inputs. No imports, no I/O, no dunder.
  * Rolling windows must state min_periods. Guard divisions/logs
    (.replace(0, np.nan) before log or division).
  * Return a DataFrame shaped like the inputs (dates x stocks).
  * Orientation: HIGHER value must predict HIGHER next-day return.

SCORING (local replica of the official engine): winsorize -> cross-sectional
z-score -> neutralize vs SIZE/MOM/VOL/BETA/LIQ styles -> IC/ICIR/long-short
Sharpe/stress-window ICIR -> Part B = rolling-window (60d/20d) ELASTIC NET of
z-scored next-day returns on [candidate + crowd factors + archive]:
ModelScore = mean(|w|)/std(|w|); factors the L1 term never selects score 0.
Composite fitness = 0.3*ICIR + 0.7*ModelScore.

WHAT WINS (verified doctrine): strong AND unique. The crowd (and our own
archive) already trades: 5/21-day reversal, 12-1 momentum, turnover, low-vol,
book pressure, auction-window dynamics. A clone of those gets its Elastic-Net
weight shared away (scores ~0); a weak-but-different factor gets L1-zeroed.
Target whitespace: stress-conditioned signals built as UNCONDITIONAL baseline
+ stress amplification (pure crisis-only factors have unstable weights),
second-order liquidity dynamics (persistence/asymmetry/resilience of flows,
not levels), interaction effects the linear style neutralization cannot span.
AVOID: plain nonlinear size and plain idiosyncratic-volatility (the real
engine neutralizes vs a BARRA set that likely includes NLSIZE and Residual
Volatility styles).

LIVE LEADERBOARD FEEDBACK (real competition, hidden 2025 validation set):
* Our submitted factor "volume_weighted_support_convexity" (close location in
  daily range x log dollar-volume expansion, stability-weighted, 5d-vs-20d
  divergence) scored total 0.65 -- the best of our three submissions. It is
  now IN THE SCORING POOL below as SUBMITTED_vwsc: a new candidate correlated
  with it earns NOTHING; find a genuinely different axis.
* A plain relative-volume factor and an order-book pressure-persistence
  factor both scored LOWER despite strong offline stats -- volume-LEVEL,
  reversal, and book-pressure axes are confirmed crowded on the real board.
* The most under-exploited levers: (1) Rank_Stress = IC stability inside
  high-volatility/crisis regimes, 25% of Part A -- factors that keep working
  when the market breaks; (2) Part-B uniqueness -- constructions the pool
  cannot linearly span. Promising unexplored shapes from daily OHLCV: price
  path efficiency/roughness, intraday-range positioning dynamics OVER TIME
  (not its level), asymmetry between up-day and down-day microbehavior,
  drawdown-recovery speed, sign-consistency of volume-return interactions,
  conditional constructions (baseline + regime amplification).


CURRENT ARCHIVE (do NOT duplicate these axes; beat them or diversify away from them):
  volume_price_elasticity_asymmetry fitness +0.519  IC +0.0319 ICIR +0.372 ModelScore +0.581 sel 36% | Asymmetry between relative volume impact on positive returns versus ne
  range_expansion_drawdown_resilience fitness +0.438  IC +0.0236 ICIR +0.326 ModelScore +0.487 sel 36% | High recovery speed relative to recent peak drawdown magnitude combine
  downside_path_efficiency_asymmetry fitness +0.429  IC +0.0190 ICIR +0.186 ModelScore +0.533 sel 31% | Asymmetry in price path efficiency on up-days versus down-days isolate
  drawdown_recovery_elasticity_asymmetry fitness +0.381  IC +0.0231 ICIR +0.310 ModelScore +0.412 sel 20% | Asymmetry in intraday low-price recovery ratio between up-days and dow
  stress_conditioned_liquidity_exhaustion fitness +0.341  IC +0.0160 ICIR +0.274 ModelScore +0.370 sel 16% | High dollar volume on narrowing intraday price range during market str
  volume_conditioned_range_breakout_asymmetry fitness +0.264  IC +0.0106 ICIR +0.180 ModelScore +0.301 sel 16% | Covariance between relative volume shifts and intraday close positioni
  intraday_shadow_recovery_resilience fitness +0.201  IC +0.0135 ICIR +0.218 ModelScore +0.194 sel 9% | Accelerating lower shadow persistence under volume expansion during lo
  intraday_noise_convexity_acceleration fitness +0.196  IC +0.0169 ICIR +0.244 ModelScore +0.175 sel 4% | Accelerating medium-term close location ratio weighted by volume stabi

RECENT REJECTIONS (learn from these):
  signed_directional_flow_persistence -> IC too weak (+0.0042)
  directional_path_efficiency_stress_amplified -> near-duplicate of archive (|corr| 0.63)
  close_location_trajectory_stability -> near-duplicate of archive (|corr| 1.00)
  intraday_noise_efficiency_asymmetry -> IC too weak (+0.0005)

Propose exactly 6 NEW candidates, each in its own ```python fenced block with the # name/# cluster/# rationale header. Diversify mechanisms across the candidates. No prose outside the code blocks.