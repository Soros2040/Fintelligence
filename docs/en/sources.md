# Sources and research lineage

[Home](../../README.md) · [中文](../zh/sources.md)

## Software foundations

| Source | Role in this project | Local evidence |
|---|---|---|
| [DeerFlow](https://github.com/bytedance/deer-flow) | Frontend, agent framework, gateway and development tooling | [Root MIT license](../../LICENSE), retained source namespaces |
| [Qlib](https://github.com/microsoft/qlib) | Data handling, model adapter and portfolio evaluation | [Qlib MIT license](../../third_party/qlib/LICENSE), pinned runtime dependency |
| [LightGBM](https://github.com/microsoft/LightGBM) | Current gradient-boosted tree training backend | Qlib `LGBModel` configuration and dependency |
| [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful graph execution and streaming | Backend dependency declarations and graph registration |

The source inventory and recovery backup retain the original working snapshot, including research edits. This package provides selected implementation files with sanitized configuration; source revision and local modifications must be recorded again for any future experiment.

## Manuscript relation

An earlier project manuscript uses the name **Alex-Fin** and describes a factor-DAG, retrieval, formula/code generation, and Bandit research design. **BenjaminAgent** is the presentation name used here; `Fintelligence` remains the directory/repository lineage name.

Read the [complete manuscript](https://github.com/Soros2040/julius-future/blob/main/works/benjamin-agent/manuscript.md) or [PDF](https://github.com/Soros2040/julius-future/blob/main/works/benjamin-agent/manuscript.pdf) for the research narrative, equations, result tables, and appendices. The [work entry](https://github.com/Soros2040/julius-future/tree/main/works/benjamin-agent/) keeps the manuscript alongside its publication context. The [original figure guide](../assets/manuscript/README.md) and [hash manifest](../assets/manuscript/provenance.json) identify the three method images used in this repository.

Use three evidence categories while reading. A manuscript passage establishes what the author proposed or reported. A source-code inspection establishes what a particular implementation expresses. A saved execution bundle establishes what ran under a particular environment and configuration. The tutorials connect the first two categories; historical performance values retain their manuscript attribution until data, logs, costs, and configurations can support a current result claim.

| Question to reconcile | Manuscript evidence | Current source reading |
|---|---|---|
| Scheduling state | Eight dimensions through Sharpe | Nine fields, adding Calmar and changing weights |
| Label and portfolio protocol | Main §4.1 and appendix describe different horizons and selection rules | Inspect configured Qlib handler, label, top-k/drop, split, and costs |
| Annualization and drawdown | Appendix uses compound ARR; MDD equation is a positive loss magnitude | Pipeline uses `252 × mean(return)` and a negative drawdown |
| Reported result attribution | Main Table 3 and §4.2 prose assign the strongest row differently | Preserve the table's attribution and record the unresolved prose mismatch |
| Knowledge persistence | Fat node and centralized research memory | Serialized IC-family node fields plus a separate model-experiment log |

[Case 1](case-01-task-to-backtest.md) works through evaluation definitions and the result-table example. [Case 2](case-02-factor-lifecycle.md) covers retrieval, admission, state construction, and posterior-update arithmetic. The [status ledger](status.md) records implementation gaps. A source review should retain exact section/table references and identify which configuration a number belongs to.

## Learning organization

[diy-llm](https://github.com/datawhalechina/diy-llm) and [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) inform the progression from objectives and prerequisites to implementation, examples, exercises, and evidence-based contribution. The explanations and examples in this project are original. These references establish a teaching style, not an organizational affiliation.

## How to add a reference

Link the primary source, identify the claim it supports, and record the relevant function, section, or equation. For an implementation claim, point to current code. For an empirical claim, include a reproducible result bundle. Keep third-party full text under its own distribution terms; a bibliographic link does not transfer those rights.
