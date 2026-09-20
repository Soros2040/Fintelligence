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

The manuscript is design context. Its reported performance requires reconciliation with current code, model choice, label horizon, cost definitions, logs, and dataset versions before it can become a current result claim. This documentation maps design ideas to concrete source modules and marks implementation gaps in the [status ledger](status.md). Private manuscript metadata and full manuscript material remain outside this package.

## Learning organization

[diy-llm](https://github.com/datawhalechina/diy-llm) and [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) inform the progression from objectives and prerequisites to implementation, examples, exercises, and evidence-based contribution. The explanations and examples in this project are original. These references establish a teaching style, not an organizational affiliation.

## How to add a reference

Link the primary source, identify the claim it supports, and record the relevant function, section, or equation. For an implementation claim, point to current code. For an empirical claim, include a reproducible result bundle. Keep third-party full text under its own distribution terms; a bibliographic link does not transfer those rights.
