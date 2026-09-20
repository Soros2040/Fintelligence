# BenjaminAgent

**A research workspace for turning financial questions into traceable factor experiments.**

[中文](README_zh.md) · [Start here](docs/en/learning-path.md) · [Full manuscript](https://github.com/Soros2040/julius-future/blob/main/works/benjamin-agent/manuscript.md) · [Contribute](CONTRIBUTING.md)

BenjaminAgent connects a streaming research interface, a LangGraph agent runtime, Qlib evaluation, and experimental factor knowledge management. It builds on [DeerFlow](https://github.com/bytedance/deer-flow) and [Qlib](https://github.com/microsoft/qlib). The current training path uses **LightGBM**. The project is an implementation under validation, with source-based tutorials and a concrete verification backlog.

## Architecture at a glance

![Original manuscript Figure 2: Alex-Fin architecture](docs/assets/manuscript/figure-02-system-architecture.jpeg)

*Original manuscript Figure 2, “Alex-Fin架构图,” extracted unchanged. Alex-Fin is the historical research name. This method diagram uses an eight-dimensional scheduling state; the current BenjaminAgent implementation uses nine fields. Read the [architecture guide](docs/en/architecture.md) for the source mapping and [image provenance](docs/assets/manuscript/README.md) for the original-file record.*

The [complete manuscript and PDF](https://github.com/Soros2040/julius-future/tree/main/works/benjamin-agent/) provide the research narrative, equations, tables, and appendices. The tutorials connect that narrative to current source code. The [implementation status](docs/en/status.md) identifies lifecycle stages that still need end-to-end orchestration.

## What you can learn and do

- Trace a request from the chat interface to a configured tool and a Qlib experiment.
- Separate factor quality, predictive quality, and portfolio outcomes using explicit definitions.
- Inspect how factor ancestry, retrieval, admission, and scheduling interact.
- Contribute a source review, a hand derivation, a figure-caption check, or a bilingual documentation improvement.

The tutorials use small hand-worked examples. Any illustrative numbers are teaching examples; measured research results require the evidence listed in the [status ledger](docs/en/status.md).

## Explore the project

| Track | Existing material | Next deliverable | Current status |
|---|---|---|---|
| Learn the system | [Architecture](docs/en/architecture.md), [learning path](docs/en/learning-path.md) | A source map from one request to its configured tool | Source walkthrough available |
| Understand evaluation | [Case 1: task to backtest](docs/en/case-01-task-to-backtest.md) and Qlib pipeline | Review label horizon, costs, and metric definitions against the manuscript | Code present; integrated run pending |
| Study factor evolution | [Case 2: factor lifecycle](docs/en/case-02-factor-lifecycle.md) | Check a retrieval or scheduling calculation by hand | Experimental components present |
| Preserve research evidence | [Sources](docs/en/sources.md), [roadmap](docs/en/roadmap.md), [maintenance](docs/maintenance.md) | Review a manuscript table, original figure, or matched translation | Documentation review open |

## Two practical cases

**[1. A user task becomes a baseline experiment](docs/en/case-01-task-to-backtest.md).** Follow the UI, stream, tool configuration, session, chronological split, LightGBM fit, IC calculation, and portfolio report. Work through a three-stock correlation example and an annualization example.

**[2. A factor becomes reusable research knowledge](docs/en/case-02-factor-lifecycle.md).** Follow a momentum hypothesis through formula and code, validation, DAG ancestry, retrieval scores, admission gates, and a nine-dimensional Bandit state. Work through selection and admission by hand, then inspect the precise implementation boundaries.

## First contribution

1. Read either case and choose one claim with a source link.
2. Open the linked source and the relevant manuscript passage; record the symbol or section, input, output, and assumptions.
3. Work through a small example by hand or compare a figure/table with its caption. State whether each claim is a design statement, a source-code finding, or a reported manuscript result.
4. Open an Issue, claim the bounded task, and submit a focused PR with a [contribution record](contributions/README.md). Update the English and Chinese pages together when their content overlaps.

| Starter task | Suggested location | Acceptance evidence | Claim status |
|---|---|---|---|
| Check the IC and return definitions | Case 1 and its Chinese counterpart | A hand calculation, exact function references, and the manuscript formula | Open |
| Review an original figure or result table | [Sources](docs/en/sources.md) and [figure guide](docs/assets/manuscript/README.md) | Figure/table number, source location, and a caption consistent with its historical context | Open |
| Explain a retrieval or scheduling calculation | Case 2 and its Chinese counterpart | Inputs, every arithmetic step, source symbol, and matched bilingual explanation | Open |

These first tasks require reading and reasoning. The [contribution guide](CONTRIBUTING.md) explains review and attribution; the [run guide](docs/en/run-guide.md) remains a separate reference for a future, explicitly scoped execution task.

## Repository map

```text
frontend/                       Streaming research workspace
backend/app/                    Gateway APIs
backend/packages/harness/       Agent runtime and quantitative tools
third_party/                    Dependency provenance and license notices
scripts/                        Local setup and service launch helpers
docker/                         Service and proxy configurations
skills/                         Upstream runtime skills
docs/en/ and docs/zh/            Matched tutorials and project documentation
```

The folder and historical repository name remain `Fintelligence`; the project is presented as **BenjaminAgent**. Historical manuscript naming and source provenance are explained in [Sources and research lineage](docs/en/sources.md).

## Current research boundary

The baseline pipeline has model training, prediction, IC evaluation, and portfolio evaluation code. Factor generation, DAG storage, retrieval, admission, and Bandit modules are present. Their integration and metric semantics have open verification tasks. In particular, `run_mining_loop` currently schedules and retrieves; the detailed lifecycle handlers require orchestration and validation. Consult the [status ledger](docs/en/status.md) before interpreting an output as a completed experiment.

## People, maintenance, and licenses

Julius maintains the research direction, reviews tasks, and coordinates releases. See [roles](docs/en/roles.md), [maintenance and handoff](docs/maintenance.md), [roadmap](docs/en/roadmap.md), and [third-party notices](THIRD_PARTY_NOTICES.md). Upstream and original code retain the applicable [MIT license](LICENSE). Original project documentation is licensed under [CC BY-NC-SA 4.0](LICENSE-DOCS); third-party documentation retains its own terms. Datawhale-style learning organization informs the tutorials; project participation or endorsement is recorded only when confirmed.
