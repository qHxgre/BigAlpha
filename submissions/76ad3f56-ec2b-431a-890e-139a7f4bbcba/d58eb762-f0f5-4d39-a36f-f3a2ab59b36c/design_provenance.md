# Design provenance — close_execution_chunkiness_shift

This factor was NOT produced by the team's Gemini evolutionary miner. Its
logic was composed by an **Anthropic Claude agent** (the "factor-designer"
role) inside the team's multi-agent research workflow on 2026-07-27, then
adversarially reviewed by an independent Claude "verifier" agent, then
evaluated UNCHANGED on the platform. This file is the verbatim audit trail
(the equivalent of the `prompts_gen_*/response_gen_*` pair that accompanies
our Gemini-mined factors). Workflow run id: `wf_2be85604-83a` (local session
archive; full JSON preserved in the team repo).

## 1. Designer agent output (verbatim, structured JSON fields)

**name:** `close_execution_chunkiness_shift`

**hypothesis:** "Institutional execution in CSI1000 concentrates into the
final continuous-trading window (VWAP/close-benchmark algos finishing parent
orders), while retail flow dominates earlier in the session. Average deal
size (volume/deal_number) is the retail-vs-institutional fragmentation proxy
no live or field factor uses. A stock whose late-session flow (14:27-14:56,
deliberately EXCLUDING the 14:57-15:00 closing auction to dodge the crowded
auction-window axis) becomes CHUNKIER relative to its own day baseline is
being accumulated by institutions into the close -> positive next-day
cross-sectional return; a late fragmentation spike (retail chase/panic) ->
negative. The within-stock late-vs-base ratio cancels price level, share
count, and total-volume scale, so the factor is a pure timing-of-granularity
signal. 07's num_orders success (ic +0.0098) proves the order-granularity
information family carries alpha on this platform; this is its
executed-trade mirror on disjoint columns."

**sign_hypothesis:** +1 (OVERRIDDEN to −1 by the mandatory on-platform
first-run check — see §3; the empirical orientation matches the A-share
smart-money/reversal literature the verifier cited, i.e. the designer's
accumulation narrative was backwards, exactly as the verifier warned).

**factor_sql:** identical to the `FACTOR_SQL` in `submission.ipynb`
(byte-preserved; no human edit).

**biggest_risk (designer's own):** "Sign ambiguity: the late-chunkiness
premium could invert (chunky late prints = distribution rather than
accumulation in small caps). Mitigated by the mandatory first-run sign
check. Secondary risk: late-window ADS is noisy for illiquid names."

## 2. Adversarial verifier verdict (verbatim excerpts)

VERDICT: PARTIALLY_SOUND. "The design survives the fatal-flaw hunt — no
arithmetic error, no look-ahead, no banned construct, no INT32 trap."
Conditions demanded before submission: (a) measured H1-2024 Spearman corr vs
05 AND 06; (b) measured late-window NULL rate; (c) sign-story caution: "the
best-known per-trade-size factor family in this exact market finds high
chunkiness predicts REVERSAL, i.e., the opposite" (this is precisely what
the platform sign check then confirmed).

## 3. Measurements that closed the verifier's conditions (2026-07-27)

- Platform eval (2024, `endgame/chunkiness.log`): raw +1 → ic_mean
  −0.009233, ic_ir −0.270935, LS sharpe −1.542595, stress_ic_ir −0.433334.
  SIGN flipped once to **−1**; submit orientation: ic +0.0092 / icir +0.271
  / stress +0.433 / sharpe +1.543.
- Correlations (H1-2024 mean daily Spearman, submitted signs,
  `endgame/corr_chunk.log`): corr to 05 = +0.0119, to 06 = −0.0124,
  to 07 = −0.0040.
- Coverage: min 0.900 / mean 0.979 (late-window sparsity measured).

## 4. Human-written parts

Data plumbing only, reused verbatim from the verified 07 template:
month-chunked `main()`, constituents merge, gates, eval cell. No human edit
to the factor logic; the single change to the designer's artifact is the
platform-verified SIGN constant, per the team's standing first-run protocol.
