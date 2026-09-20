# Roadmap with acceptance gates

[Home](../../README.md) · [中文](../zh/roadmap.md)

Progress is recorded by evidence. A task is complete when its deliverable can be independently checked.

| Order | Work package | Deliverable and acceptance gate | Status |
|---|---|---|---|
| 1 | Dependency integration | Quant dependencies in a reproducible lock; fresh import check | Open |
| 2 | Baseline definition | Resolved handler, labels, splits, costs, and metric unit tests | Open |
| 3 | Integrated baseline | One UI-to-LightGBM run with commands, data ID, logs and structured output | Open |
| 4 | Factor contracts | Formula/code and leakage tests; generated feature verified in training matrix | Open |
| 5 | Knowledge correctness | DAG round-trip, real vector correlation, admission boundary fixtures | Open |
| 6 | Scheduler semantics | Posterior update tests, reward normalization/direction, fixed-seed ablation | Open |
| 7 | Complete research cycle | Explicit stage trace with failure handling and reproducible result bundle | Open |
| 8 | Model expansion | Separate proposal and controlled baseline comparison for each new model | Planned |

## First reviewable tasks

- **Documentation:** reproduce the three-stock IC arithmetic and link the exact calculation function in both languages.
- **Data contracts:** verify that serialization preserves every metric needed by the scheduling state; state what is currently lost.
- **Research methods:** propose a validation/holdout policy and a cost-aware metric table before another performance experiment.

Claim a task in an Issue with scope, expected evidence, and proposed reviewer. Work on a branch, link the Issue in a PR, and register the merged evidence through the [contribution guide](../../CONTRIBUTING.md). Update both language versions when changing a conceptual explanation.
