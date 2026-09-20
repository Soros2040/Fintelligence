# Architecture and data contracts

[Home](../../README.md) · [中文](../zh/architecture.md)

BenjaminAgent has an interaction layer, an orchestration layer, and a quantitative execution layer. The framework retains upstream module names so imports and build tooling remain traceable.

![Original manuscript Figure 2: Alex-Fin architecture](../assets/manuscript/figure-02-system-architecture.jpeg)

*Original manuscript Figure 2, “Alex-Fin架构图.” The original method diagram presents the historical eight-stage research loop and eight-dimensional scheduling design. Current source uses a nine-field state and implements the stages across separate handlers. [Image provenance](../assets/manuscript/README.md) records the unchanged original asset.*

Read the design from the research question outward. A task needs configuration and data; measurements inform whether a factor is useful; a graph preserves what was tried; retrieval supplies context for a new proposal; admission manages the active pool; scheduling allocates the next research effort. This explains the method's intended feedback loop. The [complete manuscript](https://github.com/Soros2040/julius-future/blob/main/works/benjamin-agent/manuscript.md) develops the motivation and equations. The table below maps those ideas to current implementation boundaries, and [status](status.md) records the integration work still needed.

| Boundary | Concrete source | Data crossing it |
|---|---|---|
| Chat → stream | [hooks.ts](../../frontend/src/core/threads/hooks.ts) | Human message, thread ID, runtime context |
| Stream → graph | [api-client.ts](../../frontend/src/core/api/api-client.ts), [langgraph.json](../../backend/langgraph.json) | `lead_agent` run and incremental events |
| Agent → tools | [tools.py](../../backend/packages/harness/deerflow/tools/tools.py), [config example](../../config.example.yaml) | Resolved tool object and validated arguments |
| Tool → session | [thread_state.py](../../backend/packages/harness/deerflow/agents/thread_state.py), [quant_analyze_tool.py](../../backend/packages/harness/deerflow/tools/builtins/quant/quant_analyze_tool.py) | Configuration, history, metrics, DAG state |
| Session → experiment | [qlib_pipeline.py](../../backend/packages/harness/deerflow/tools/builtins/quant/qlib_pipeline.py) | Dataset, train/validation/test segments, model and strategy parameters |
| Factor → knowledge | [knowledge_graph.py](../../backend/packages/harness/deerflow/tools/builtins/quant/dag/knowledge_graph.py) | Formula, code, ancestry, measurements, feedback |

## Contracts to inspect

A factor output is a single-column pandas DataFrame with a two-level `(datetime, instrument)` index. A prediction aligns with the test label index before daily cross-sectional correlation. A session holds configuration separately from human-readable messages; the message alone is insufficient to reproduce a run. A DAG node contains both formula and code, plus provenance and measured quality.

The frontend consumes tool events and state updates; it does not implement Qlib training in the browser. The FastAPI gateway provides related APIs while the default streaming run is served by LangGraph. The local reverse proxy routes both under one browser origin.

## Read the original roadmap alongside the implementation

![Original manuscript Figure 1: technical roadmap](../assets/manuscript/figure-01-technical-roadmap.jpeg)

*Original manuscript Figure 1, “技术路线图,” preserved with its historical labels, thresholds, and eight-dimensional scheduling context. Current metric definitions and thresholds are explained in [Case 1](case-01-task-to-backtest.md) and [Case 2](case-02-factor-lifecycle.md). The diagram is a research-design source; verification of an executed lifecycle requires an evidence bundle.*

The manuscript treats the DAG as the central research-memory object. Current code also contains a separate model-experiment log, and the serialized factor node retains IC-family metrics but lacks declared portfolio-metric fields. Follow both storage paths when asking whether a later decision can recover all earlier evidence. Likewise, `run_mining_loop` currently schedules and retrieves; the other stage handlers must be orchestrated explicitly before a complete loop can be claimed.

## Review question

Where should you look if the chat says an experiment completed but no portfolio metrics are present?

**Answer.** Follow the tool update into `run_full_pipeline` and `_calc_backtest_metrics`. Missing benchmark data can leave an IC-only result. Check the structured metrics and log before changing the UI or interpreting absent metrics as zero.
