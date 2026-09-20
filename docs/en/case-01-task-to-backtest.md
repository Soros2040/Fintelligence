# Case 1 — From a research task to a Qlib baseline

[Home](../../README.md) · [中文](../zh/case-01-task-to-backtest.md) · [Next case](case-02-factor-lifecycle.md)

## Learning objective and prerequisites

After this case, you should be able to identify where a request becomes a tool call, specify a chronological experiment, and explain what the reported metrics actually measure. You need basic Python, pandas indexing, and correlation. A full execution additionally requires the environment and data in the [run guide](run-guide.md).

The worked example is a **source walkthrough with illustrative arithmetic**. It does not represent an executed investment experiment. The current model is Qlib's `LGBModel`, backed by LightGBM; a mixture-of-experts model is a research direction requiring its own implementation and evidence.

## 1. Frame the task before opening the interface

A useful request names an experiment rather than only an outcome:

> Initialize a CSI 300 research session. Show the data location, feature handler, label horizon, train/validation/test intervals, strategy, and cost assumptions. After those are checked, run the LightGBM baseline and return the structured metrics and experiment record.

The initial inspection is possible without training. `quant_analyze(action="init_session")` constructs a session and does not call `qlib.init`. Its defaults come from [QuantConfig](../../backend/packages/harness/deerflow/config/quant_config.py):

| Setting | Current default | What the experiment record must clarify |
|---|---|---|
| Universe / benchmark | `csi300` / `SH000300` | Coverage and historical membership |
| Training | 2010-01-01 to 2021-12-31 | Data availability and fitted preprocessing |
| Validation | 2022-01-01 to 2023-06-30 | Hyperparameter-selection rule |
| Test | 2023-07-01 to 2026-04-30 | Actual available last date and untouched holdout |
| Model | `LGBModel` | Full resolved parameters and seed policy |
| Portfolio | `topk=50`, `n_drop=5` | Ranking, execution and rebalance assumptions |
| Costs | Open 0.0005, close 0.0015, minimum 5 | How the displayed return incorporates costs |

These are configuration defaults, not proof that the dataset covers all dates. Override the window before a run when coverage differs. Reserve the final test segment for a final evaluation; repeatedly selecting factors using its score turns it into a selection set.

## 2. Follow the request through the application

1. [chat-box.tsx](../../frontend/src/components/workspace/chats/chat-box.tsx) and [use-thread-chat.ts](../../frontend/src/components/workspace/chats/use-thread-chat.ts) connect the chat UI and thread identity.
2. `sendMessage` in [hooks.ts](../../frontend/src/core/threads/hooks.ts) builds a human message, handles optional uploads, and calls `thread.submit` with context. `useStream` names the assistant `lead_agent` and listens for tool-end and state-update events.
3. [api-client.ts](../../frontend/src/core/api/api-client.ts) creates the LangGraph SDK client and normalizes stream options. [langgraph.json](../../backend/langgraph.json) maps `lead_agent` to `deerflow.agents:make_lead_agent`.
4. [tools.py](../../backend/packages/harness/deerflow/tools/tools.py) resolves enabled tool objects from [config.example.yaml](../../config.example.yaml). The four quant entries are included in this package's example configuration.
5. [quant_analyze_tool.py](../../backend/packages/harness/deerflow/tools/builtins/quant/quant_analyze_tool.py) dispatches by `action` and returns a `Command` containing a `ToolMessage` and structured state updates. The session survives in `quant_session`.

This chain gives a debugging method. If the tool never appears, inspect configuration and model tool selection. If the tool appears but fails on data, inspect the provider and Qlib initialization. If a result appears but does not update the display, compare the structured event with the UI handler.

## 3. Separate the displayed factor library from the training handler

`init_session` can select the `ALPHA20` or `ALPHA158` formula library. `calculate_factors` calls Qlib `D.features` for expressions in that library. However, the default `_build_task_config` in [qlib_pipeline.py](../../backend/packages/harness/deerflow/tools/builtins/quant/qlib_pipeline.py) constructs an **Alpha158 handler** for model training.

Consequently, a session whose displayed library says `ALPHA20` does not establish that the model trained on exactly those 20 factors. A reproduction should record the resolved handler and actual feature columns. The handler's fit interval is set to the training dates; DatasetH supplies separate train, valid, and test segments.

The model path is `init_instance_by_config` → `model.fit(dataset)` → `model.predict(dataset)`. The prediction is then aligned with the test label. Consult the selected Qlib handler's label expression to establish the return horizon; a term such as “future return” alone is underspecified.

## 4. Understand IC before interpreting a portfolio

For each date, IC is the cross-sectional Pearson correlation between prediction scores and future labels:

$$IC_t=\frac{\sum_i (p_{i,t}-\bar p_t)(y_{i,t}-\bar y_t)}{\sqrt{\sum_i(p_{i,t}-\bar p_t)^2\sum_i(y_{i,t}-\bar y_t)^2}}.$$

Rank IC applies the same correlation to ranks. Mean IC summarizes the daily values. The pipeline computes ICIR as the mean daily IC divided by its standard deviation; check the code and sample length before comparing it with a separately annualized statistic.

**Hand example.** For one date, predictions for three stocks are `[1, 2, 3]`, and labels are `[0.02, 0.01, 0.03]`. Centered predictions are `[-1, 0, 1]`; centered labels are `[0, -0.01, 0.01]`. The numerator is `0.01`, and the denominator is `sqrt(2 × 0.0002)=0.02`, so IC is `0.5`. The rank vectors have the same ordering pattern, giving Rank IC `0.5` here.

This measures ordering association on one date. It does not include trading costs, position constraints, or evidence of stability over many dates.

## 5. Read the portfolio report literally

`_calc_backtest_metrics` constructs a Qlib `TopkDropoutStrategy`. It returns no portfolio report when the benchmark series is unavailable; the full pipeline can therefore return IC-only metrics.

For daily report returns `r_t`, the current summary computes:

$$ARR=252\,\bar r,\qquad IR=Sharpe=\sqrt{252}\,\bar r/s_r,$$
$$V_t=\prod_{u\leq t}(1+r_u),\quad MDD=\min_t(V_t/\max_{u\leq t}V_u-1),\quad Calmar=ARR/|MDD|.$$

The current fields `ir` and `sharpe` use the same return series and formula; `ir` is not an independently calculated benchmark-excess information ratio. The annualized return is arithmetic, not compound annual growth. Cost parameters are passed to Qlib's exchange, while these summaries use `report["return"]`; a net-of-cost performance claim requires inspection of the report's return and cost fields.

**Hand example.** With mean daily return `0.001` and daily standard deviation `0.02`, arithmetic ARR is `0.252` and the ratio is approximately `0.794`. If maximum drawdown is `-0.20`, Calmar is `1.26`. These numbers illustrate formulas; they are not project performance.

## 6. Save an evidence bundle

For a meaningful baseline, record the code revision, dependency versions, dataset source and calendar, universe membership, label expression, actual feature columns, all split dates, seed policy, model settings, strategy, cost interpretation, prediction output, structured metrics, logs, duration, and peak memory. Save failed runs with their reason if they affect model or factor selection.

The session's `experiment_history` and best-record fields are useful bookkeeping. They do not replace a dataset identifier or reproducibility record. The session selects a best record using Sharpe when available; the DAG's “SOTA” accessor has different semantics, covered in Case 2.

## Exercises and answers

**Exercise A.** A session displays ALPHA20 and returns a LightGBM score. What evidence establishes the actual inputs?

**Answer.** The resolved Qlib task, handler class and feature columns. In the current default model path the handler is Alpha158; the displayed formula library alone is insufficient.

**Exercise B.** IC is present, ARR is missing, and the tool says the full pipeline completed. What are the next two checks?

**Answer.** Inspect benchmark availability and `_calc_backtest_metrics` logs, then inspect the structured metrics for absent values. Report “IC evaluation completed; portfolio evaluation unavailable” when that is what the evidence supports.

**Exercise C.** A researcher chooses a new factor after every test-period Sharpe calculation. Is the period still an untouched test?

**Answer.** It has become part of the selection process. Use validation for iteration, freeze the factor and model, then evaluate once on a separate final holdout.

## Contribution deliverable

Create a small fixture that verifies one metric definition or one data-contract boundary. Include a hand calculation, exact function reference, expected result, observed result, and a note on which broader claims the fixture does not test. Follow [CONTRIBUTING](../../CONTRIBUTING.md).
