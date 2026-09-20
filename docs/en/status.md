# Implementation and evidence status

[Home](../../README.md) · [中文](../zh/status.md)

This ledger describes the selected working-tree snapshot, including its research edits. Source presence, component checks, integrated execution, and measured research results are separate evidence levels.

| Component | Source evidence | Acceptance still required |
|---|---|---|
| Frontend and gateway | Chat streams, task state, files and API code retained | Fresh dependency install, typecheck/build, browser-to-tool run |
| Qlib / LightGBM | Dataset, training, prediction, IC and portfolio functions | Fixed-data baseline with resolved handler, labels, costs and logs |
| Factor generation | Prompt preparation, code execution and format checks | Formula/code equivalence, temporal leakage tests, controlled execution |
| Factor DAG | Node records, ancestry and JSON persistence | Complete metric round-trip and semantic tests |
| Retrieval | ICIR quality, pool scoring, optional embeddings, quotas | Actual factor-vector correlation and calibrated interpretation |
| Admission | Quality/diversity routes and pool capacity | Measured-correlation fixtures and boundary tests |
| Bandit | Two arms, nine state fields, sampled linear scores | Posterior update, reward direction/scaling and repeatable comparison |
| Mining orchestration | Stage handlers; convenience loop schedules/retrieves | Explicit end-to-end lifecycle trace |
| Generated-factor training | Multiple integration functions exist | Prove extra columns reach the fitted model and change a controlled ablation |
| MoE research | Design direction | Implemented training path and independently reproducible evidence |

## Open implementation findings

1. Session formula-library selection and the default Alpha158 training handler are separate; record actual features.
2. `ir` and `sharpe` currently share a formula, and summary cost handling needs explicit interpretation.
3. `FactorDAG.get_sota` selects the latest decision-marked node; dynamic portfolio metrics are omitted from node serialization, and code-experience lookup is a stub.
4. Retrieval uses a scalar-IC correlation heuristic; admission can also fall back to an IC-sign heuristic.
5. Bandit uses nine raw metric dimensions. Drawdown direction, scale, and the precision update need review.
6. `run_mining_loop` does not execute every lifecycle stage. `pipeline_with_new_factors` needs proof that generated features enter training.
7. Quantitative dependencies need integration into the inherited backend lock.

See the two [cases](case-01-task-to-backtest.md) for source links and hand examples. These findings are review tasks; the snapshot's research algorithms are preserved for analysis.

## Evidence required for a result

Include source revision, local modifications, environment versions, data provenance and access conditions, universe and calendar, preprocessing fit dates, label horizon, feature columns, split policy, model parameters, seeds, evaluation formulas, cost treatment, run commands, structured outputs, duration, and resource usage. Mark a skipped stage explicitly. A historical manuscript table or successful interface response is not sufficient to establish this record.

## Verification record

The local handoff report records performed static and component checks. No research training, model API experiment, or full frontend/backend deployment was run as part of the documentation packaging work. Integrated execution and research performance remain pending until their evidence bundles are reviewed.
