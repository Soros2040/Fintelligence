# Case 1 — From a research task to a Qlib baseline

[Home](../../README.md) · [中文](../zh/case-01-task-to-backtest.md) · [Next case](case-02-factor-lifecycle.md)

## Learning objective and prerequisites

After this case, you should be able to identify where a request becomes a tool call, specify a chronological experiment, and explain what the reported metrics actually measure. You need basic Python, pandas indexing, and correlation. A full execution additionally requires the environment and data in the [run guide](run-guide.md).

The worked example is a **source walkthrough with illustrative arithmetic**. The current model is Qlib's `LGBModel`, backed by LightGBM. The [complete manuscript](https://github.com/Soros2040/julius-future/blob/main/works/benjamin-agent/manuscript.md) and [PDF](https://github.com/Soros2040/julius-future/blob/main/works/benjamin-agent/manuscript.pdf) supply the research motivation, methods, historical results, and appendices. The manuscript uses the historical name Alex-Fin. Its results keep their original attribution; the arithmetic in this tutorial is a manual teaching example.

The central question is how to connect a research intention to evidence. A chat answer, a configured experiment, a fitted model, and a measured portfolio are different artifacts. By following the values between them, a reader can locate the exact point at which a claim gains support or becomes uncertain.

## 1. Frame the task before opening the interface

A useful request names an experiment rather than only an outcome:

> For a CSI 300 baseline, identify the data provider, actual training features, future-return label, chronological split, LightGBM configuration, portfolio rule, and metric definitions. Trace where each choice is stored and where it is consumed.

The manuscript motivates joint factor/model optimization because a better isolated factor need not improve the model or portfolio that uses it. A baseline gives “better” a reference point. This reading task specifies that reference before assessing any performance claim. Even a precise request is still an intention; the resolved configuration and data path establish what the program would evaluate.

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

The thread ID connects a visible conversation to backend state. The graph registration identifies the program serving that thread, and tool configuration identifies the operations available to it. Finally, the action chooses a specific branch of the quantitative tool. These boundaries explain why a working chat interface alone cannot establish that an experiment reached Qlib.

The `Command` update also separates explanation from data: a `ToolMessage` can describe completion while structured metrics contain absent portfolio fields. Read both. If a tool never appears, inspect configuration and tool selection. If it fails on data, inspect the provider and Qlib initialization. If the display disagrees with a returned result, compare structured events with the UI handler.

## 3. Separate the displayed factor library from the training handler

`init_session` can select the `ALPHA20` or `ALPHA158` formula library. `calculate_factors` calls Qlib `D.features` for expressions in that library. However, the default `_build_task_config` in [qlib_pipeline.py](../../backend/packages/harness/deerflow/tools/builtins/quant/qlib_pipeline.py) constructs an **Alpha158 handler** for model training.

Consequently, a session whose displayed library says `ALPHA20` does not establish that the model trained on exactly those 20 factors. A reproduction should record the resolved handler and actual feature columns. The handler's fit interval is set to the training dates; DatasetH supplies separate train, valid, and test segments.

The model path is `init_instance_by_config` → `model.fit(dataset)` → `model.predict(dataset)`. The prediction is then aligned with the test label. Consult the selected Qlib handler's label expression to establish the return horizon; a term such as “future return” alone is underspecified.

The manuscript makes this distinction especially useful. Its main evaluation section describes a **20-day future-return label**, a top-20% portfolio, and a 20-day holding period. The appendix's trading settings describe a **one-day label**, top-50 selection, five dropouts, and buy/sell costs of 0.05%/0.15%. The code matches parts of the appendix, but its default test end is April 30, 2026, while the manuscript states a May endpoint. These descriptions must remain associated with their own source locations until a versioned experiment record resolves them.

A static contribution can already identify the exact handler construction and compare those statements. Establishing the observed feature matrix or executed label requires a later run record; reading the source should not be reported as observing an experiment.

## 4. Understand IC before interpreting a portfolio

For each date, IC is the cross-sectional Pearson correlation between prediction scores and future labels:

$$IC_t=\frac{\sum_i (p_{i,t}-\bar p_t)(y_{i,t}-\bar y_t)}{\sqrt{\sum_i(p_{i,t}-\bar p_t)^2\sum_i(y_{i,t}-\bar y_t)^2}}.$$

Rank IC applies the same correlation to ranks. Mean IC summarizes the daily values. The pipeline computes ICIR as the mean daily IC divided by its standard deviation; check the code and sample length before comparing it with a separately annualized statistic.

**Hand example.** For one date, predictions for three stocks are `[1, 2, 3]`, and labels are `[0.02, 0.01, 0.03]`. Centered predictions are `[-1, 0, 1]`; centered labels are `[0, -0.01, 0.01]`. The numerator is `0.01`, and the denominator is `sqrt(2 × 0.0002)=0.02`, so IC is `0.5`. The rank vectors have the same ordering pattern, giving Rank IC `0.5` here.

The centering operation removes the mean level; dividing by both scales makes Pearson correlation insensitive to positive rescaling. Rank IC additionally removes information about distances between values. Their equality in this three-stock example is a coincidence of the ordering, not a general identity.

This measures association on one date. Portfolio construction introduces turnover, position constraints, and implementation costs. A stable positive IC can motivate that next question, but cannot answer it by itself.

## 5. Read the portfolio report literally

`_calc_backtest_metrics` constructs a Qlib `TopkDropoutStrategy`. It returns no portfolio report when the benchmark series is unavailable; the full pipeline can therefore return IC-only metrics.

For daily report returns `r_t`, the current summary computes:

$$ARR=252\,\bar r,\qquad IR=Sharpe=\sqrt{252}\,\bar r/s_r,$$
$$V_t=\prod_{u\leq t}(1+r_u),\quad MDD=\min_t(V_t/\max_{u\leq t}V_u-1),\quad Calmar=ARR/|MDD|.$$

The current fields `ir` and `sharpe` use the same return series and formula; `ir` is not an independently calculated benchmark-excess information ratio. The annualized return is arithmetic, not compound annual growth. Cost parameters are passed to Qlib's exchange, while these summaries use `report["return"]`; a net-of-cost performance claim requires inspection of the report's return and cost fields.

The manuscript appendix defines ARR using compounded growth and describes MDD as a loss magnitude. The code above uses arithmetic annualization and a negative drawdown. Therefore identical daily observations can produce differently named or signed summaries across the two sources. Record the formula alongside the metric instead of comparing the field names alone. The pipeline can also shorten its requested end to the available trading calendar; configured dates and effective dates belong in separate fields of an evidence record.

**Hand example.** With mean daily return `0.001` and daily standard deviation `0.02`, arithmetic ARR is `0.252` and the ratio is approximately `0.794`. If maximum drawdown is `-0.20`, Calmar is `1.26`. These numbers illustrate formulas; they are not project performance.

## 6. Read the manuscript's table as evidence

Table 3 reports these CSI 300 rows. They are existing manuscript observations reproduced here for source interpretation; this tutorial has not repeated the experiments.

| Table 3 row | IC | ARR | IR | MDD |
|---|---:|---:|---:|---:|
| LightGBM | 0.0277 | 0.0397 | 0.5664 | -0.0855 |
| Alex-Fin (DeepSeekV4) | 0.0497 | 0.1144 | 1.3167 | -0.0811 |
| Alex-Fin (Qwen3.6Plus) | 0.0532 | 0.1421 | 1.7382 | -0.0742 |

The §4.2 narrative attributes IC `0.0532`, ARR `14.21%`, and IR around `1.74` to DeepSeek, while Table 3 assigns that combination to Qwen. A review should preserve the table's row label and record the narrative mismatch. It should also keep Table 4 and the appendix tables associated with their own units and protocols rather than merging their values into one benchmark.

This gives two useful conclusions with different scopes. The manuscript reports comparative results and leaves attribution/protocol questions to reconcile. The current source exposes specific metric calculations. Connecting its checkout to the reported table still requires dataset, configuration, revision, and run evidence.

## 7. Save an evidence bundle

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

Choose one metric definition, data boundary, or manuscript statement. Submit a static reading note with the original section/table, exact source symbol, input/output contract, manual derivation where useful, and the wording supported by that evidence. Keep source inspection and observed execution in separate fields. Update both language versions and use the [contribution record](../../CONTRIBUTING.md#contribution-record) to link the Issue, review, and accepted change.
