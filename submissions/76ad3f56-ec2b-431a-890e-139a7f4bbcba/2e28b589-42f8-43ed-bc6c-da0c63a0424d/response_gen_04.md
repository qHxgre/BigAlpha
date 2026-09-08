```python
# name: fractal_roughness_asymmetry
# cluster: N
# rationale: High efficiency on up-days relative to down-days signals institutional accumulation with minimal down-day panic.
def factor(close, high, low, volume, dollar):
    ret = close.pct_change()
    price_range = (high - low).replace(0, np.nan)
    net_move = (close - close.shift(1)).abs()
    efficiency = net_move / price_range
    up_eff = efficiency.where(ret > 0, np.nan)
    dn_eff = efficiency.where(ret < 0, np.nan)
    up_eff_mean = up_eff.rolling(20, min_periods=5).mean()
    dn_eff_mean = dn_eff.rolling(20, min_periods=5).mean()
    asym = (up_eff_mean - dn_eff_mean) / (up_eff_mean + dn_eff_mean).replace(0, np.nan)
    vol_scale = volume / volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
    res = asym * vol_scale
    return res.shift(1)
```

```python
# name: volume_weighted_shadow_skewness
# cluster: S
# rationale: Dominant lower shadow volume during market volatility stress identifies institutional price support.
def factor(close, high, low, volume, dollar):
    prev_close = close.shift(1)
    upper_wick = high - np.maximum(close, prev_close)
    lower_wick = np.minimum(close, prev_close) - low
    total_range = (high - low).replace(0, np.nan)
    wick_skew = (lower_wick - upper_wick) / total_range
    vol_rel = dollar / dollar.rolling(20, min_periods=10).mean().replace(0, np.nan)
    weighted_skew = wick_skew * vol_rel
    skew_mean = weighted_skew.rolling(15, min_periods=5).mean()
    mkt_vol = (high - low).div(close.replace(0, np.nan)).median(axis=1)
    mkt_stress = mkt_vol / mkt_vol.rolling(60, min_periods=20).mean().replace(0, np.nan)
    res = skew_mean.mul(mkt_stress, axis=0)
    return res.shift(1)
```

```python
# name: volatility_of_volatility_convexity
# cluster: N
# rationale: Orderly volume accumulation characterized by compressed range volatility-of-volatility relative to return volatility.
def factor(close, high, low, volume, dollar):
    range_ratio = (high - low) / close.replace(0, np.nan)
    range_vol = range_ratio.rolling(5, min_periods=3).std()
    range_vol_vol = range_vol.rolling(20, min_periods=10).std()
    ret_vol = close.pct_change().rolling(20, min_periods=10).std()
    vol_momo = volume / volume.rolling(20, min_periods=10).mean().replace(0, np.nan)
    vol_momo_smooth = vol_momo.rolling(5, min_periods=3).mean()
    vol_of_vol_ratio = range_vol_vol / ret_vol.replace(0, np.nan)
    res = -1.0 * vol_of_vol_ratio * vol_momo_smooth
    return res.shift(1)
```

```python
# name: cross_lag_volume_price_lead
# cluster: M
# rationale: Positive cross-lag asymmetry where volume changes lead price changes indicates informed directional positioning.
def factor(close, high, low, volume, dollar):
    d_vol = dollar.pct_change()
    ret = close.pct_change()
    cov_vol_lead = d_vol.shift(1).rolling(20, min_periods=10).cov(ret)
    cov_price_lead = ret.shift(1).rolling(20, min_periods=10).cov(d_vol)
    vol_std = d_vol.rolling(20, min_periods=10).std()
    ret_std = ret.rolling(20, min_periods=10).std()
    denom = (vol_std * ret_std).replace(0, np.nan)
    lead_asym = (cov_vol_lead - cov_price_lead) / denom
    return lead_asym.shift(1)
```

```python
# name: downside_volume_decay_acceleration
# cluster: S
# rationale: Rapid volume drying up on down-days during market stress signals selling exhaustion and impending reversal.
def factor(close, high, low, volume, dollar):
    ret = close.pct_change()
    dollar_ma = dollar.rolling(5, min_periods=2).mean().replace(0, np.nan)
    dn_vol_ratio = (dollar / dollar_ma).where(ret < 0, np.nan)
    decay_mean = dn_vol_ratio.rolling(15, min_periods=5).mean()
    decay_accel = decay_mean - decay_mean.shift(5)
    mkt_ret = ret.median(axis=1)
    mkt_down_stress = (-1.0 * mkt_ret).clip(lower=0)
    res = -1.0 * decay_accel.mul(mkt_down_stress, axis=0)
    return res.shift(1)
```

```python
# name: intraday_range_intraday_jump_ratio
# cluster: N
# rationale: Low ratio of intraday range volatility to close-to-close variance highlights continuous momentum with minimal intraday friction.
def factor(close, high, low, volume, dollar):
    log_hl = np.log((high / low.replace(0, np.nan)).replace(0, np.nan))
    parkinson_proxy = (log_hl ** 2) / (4.0 * np.log(2.0))
    ret_log = np.log((close / close.shift(1).replace(0, np.nan)).replace(0, np.nan))
    close_vol_proxy = ret_log ** 2
    p_sum = parkinson_proxy.rolling(20, min_periods=10).sum()
    c_sum = close_vol_proxy.rolling(20, min_periods=10).sum()
    jump_ratio = p_sum / c_sum.replace(0, np.nan)
    ret_5d = close / close.shift(5).replace(0, np.nan) - 1.0
    res = ret_5d / jump_ratio.replace(0, np.nan)
    return res.shift(1)
```