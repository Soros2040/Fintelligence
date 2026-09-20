# Learning path

[Home](../../README.md) · [中文](../zh/learning-path.md)

The goal is to explain one experiment well enough that another person can inspect its data, reproduce its transformations, and challenge its conclusion.

## Prerequisites

| Concept | You should be able to do | Preparation task |
|---|---|---|
| Python and pandas | Read a function and align a MultiIndex | Build a two-day, three-stock table |
| Statistics | Compute mean, standard deviation, correlation | Finish the IC example in Case 1 |
| Time-series evaluation | Keep fitting and evaluation periods separate | Draw train, validation, and test intervals |
| Web application flow | Distinguish a UI event from a server tool call | Trace `sendMessage` to `thread.submit` |
| Research engineering | Record versions and failed runs | Complete one evidence record |

Prior knowledge of LangGraph, Qlib, or contextual Bandits is useful but not assumed. Each case introduces its data structures before using them.

## Suggested sequence

1. **Orientation, 30–45 minutes.** Read the [architecture](architecture.md). Find the frontend stream client, backend graph registration, and tool loader. Deliver a five-step request trace.
2. **Baseline reasoning, 60–90 minutes.** Complete [Case 1](case-01-task-to-backtest.md), including both exercises. Deliver a metric-definition table and a fixed chronological split.
3. **Factor reasoning, 60–90 minutes.** Complete [Case 2](case-02-factor-lifecycle.md). Deliver a factor node, its parent relation, a retrieval calculation, and an admission decision.
4. **Environment and execution.** Follow the [run guide](run-guide.md) in a dedicated environment. Begin with import and configuration checks. Save exact commands and versions before running a model-backed task.
5. **Contribution.** Select a [roadmap](roadmap.md) item, open an Issue, and submit one reviewable change with evidence.

## A research note that earns its conclusion

State the question and comparison first. Then record dataset coverage, source revision, actual features, label horizon, split, model parameters, costs, and run identifiers. Explain uncertainty and a failed attempt if either affects the conclusion. A plot without the underlying measurement definition is incomplete.

## Exercise

You have a valid training log and a screenshot of a high Sharpe ratio, but the feature calculation used the full date range to standardize values. Can the result be registered as a held-out baseline?

**Answer.** The evaluation has a leakage concern: preprocessing may have learned from future observations. Refit preprocessing using the training segment, reproduce the run, and record both configurations. A successful application run establishes execution; the split and transformation audit establishes the evaluation claim.
