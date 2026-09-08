# AI Application Statement — BigAlpha 2026 Track 1 (AI Intelligent Track)

## AI technology
- **Type:** LLM Agent (hypothesis generation + critic) + HMM regime detection
- **Role:** AI dominates factor *generation and iterative revision*; humans set operator grammar, risk constraints, and submission policy.

## AI-CORE locations
| Step | File | Role |
|---|---|---|
| Prompt construction | `src/agent/prompts.py` | Feed metrics + regime into LLM |
| Formula validation | `src/factors/dsl.py` | Restrict AST to legal operators |
| Expression → panel | `src/factors/executor.py` | Execute DSL on minute bars + PIT financials |
| Critic loop + gates | `src/agent/loop.py` | Build panel → metrics → lineage gates → revise |
| Regime labeling | `src/regime/hmm_regime.py` | Stress-day definition |

## Offline vs online
LLM calls run **offline** (local). BigQuant notebooks are network-isolated; submitted code only **executes frozen formulas** via DAI/SQL/UDF.

## Lineage methodology transfer (`global_factor_lineage_audit.xlsx`)
That workbook audits **global peer returns → 工业富联/立讯** (research-only, NO-GO for trading). It is **not** competition data and **must not** be queried inside the notebook (no Yahoo / US peers / external network).

What we reuse as **submission discipline** (see `src/evaluation/lineage_gates.py`):
| Audit rule | Track-1 mapping |
|---|---|
| One economic family (F1–F9) | Families M1–M5; ≤1 live factor per family |
| Tier A: weight P10>0 | Rolling ModelScore path P10>0 |
| Coverage / structure gate | Daily non-null ≥60% (miss ≤40%) |
| Conditional drop-column / joint model | Multi-factor Elastic Net vs `factorlib`+own pool (B proxy) |
| Withdrawn non-identifiable atom nets | Do not submit collinear siblings of the same family |
| PASS_RESEARCH_ONLY / no lockbox | Treat public-board IC as exploratory; freeze ≤2 for private board |

## Reproducibility
- Prompt templates versioned in `configs/agent.yaml`
- Seeds, thresholds, and operator whitelist logged per candidate in `experiments/`
- Final submitted formulas listed in `submission/selected_factors.json`

## AI participation self-score
- Generation: high
- Evaluation objective design: human+AI
- Final economic interpretation: human review required for finals report
