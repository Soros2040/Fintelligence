# Case 2 — From a factor hypothesis to research memory

[Home](../../README.md) · [中文](../zh/case-02-factor-lifecycle.md) · [Previous case](case-01-task-to-backtest.md)

## Objective and prerequisites

Follow a factor through generation, executable representation, validation, ancestry, retrieval, admission, and research scheduling. You need the data-contract and metric concepts in Case 1, plus the idea that a graph edge records a parent-child relation. This is a source-based walkthrough; the hand calculations use invented observations.

The main sources are [factor_generate_tool.py](../../backend/packages/harness/deerflow/tools/builtins/quant/factor_generate_tool.py), [quant_mining_tool.py](../../backend/packages/harness/deerflow/tools/builtins/quant/quant_mining_tool.py), and the `dag`, `retrieval`, `admission`, and `bandit` subdirectories beside them. The [complete manuscript](https://github.com/Soros2040/julius-future/blob/main/works/benjamin-agent/manuscript.md) gives the historical Alex-Fin design context and reports an **eight-dimensional** scheduling vector. The current scheduler has **nine fields**. This case preserves that distinction between design and implementation.

The lifecycle is easier to understand as one evidence path. A hypothesis becomes a formula and executable code; validation produces measurements; a DAG preserves ancestry and rejected ideas; retrieval chooses context; admission decides whether a node enters the active pool; scheduling decides whether the next effort targets a factor or a model. Each boundary has a data contract, and a successful earlier boundary does not imply that later ones ran.

![Original manuscript Figure 3: DAG-aware factor generation](../assets/manuscript/figure-03-dag-aware-generation.png)

*Original manuscript Figure 3, “全局感知因子挖掘架构图,” extracted unchanged. It depicts the historical Alex-Fin method: retrieve research context before proposing a factor, then preserve feedback for the next proposal. The historical scheduler used eight dimensions; the current scheduler has nine. See [image provenance](../assets/manuscript/README.md) and the implementation boundaries below.*

## 1. Express one falsifiable hypothesis

Suppose the hypothesis is: “Recent price momentum contains information about a specified future return.” A simple candidate is

```math
f_{i,t}=P_{i,t}/P_{i,t-5}-1.
```

If today's close is 105 and the close five observations ago is 100, the feature is `0.05`. A Qlib-style representation is `$close / Ref($close, 5) - 1`. Decide when the close is observable and when a trade can occur before using it. The label belongs to a later interval and must never enter feature construction.

The implementation keeps both a formula and Python code. The execution contract expects `calculate_factor(df)` to produce a Series or a single-column DataFrame with the original stock/date index. Group by instrument when applying a lag; shifting a flattened table can accidentally cross from one stock to another.

`generate_factors` prepares generation prompts and saves pending context. It does not, by itself, establish that a new formula has been executed or evaluated. A later `validate_and_add_factor` action accepts the proposed code and measurements follow that path. Treat the pending prompt, generated text, executed output, and admitted node as four different evidence records.

The manuscript motivates formula-first generation because a financial hypothesis should remain inspectable before its executable representation is accepted. It also motivates global DAG context: a proposal should account for earlier branches, failure feedback, and pool composition. The implementation exposes prompt preparation, validation, and graph insertion as separate actions, so a reviewer can trace those transitions individually. The original diagram explains why the stages exist; the source establishes which transitions currently occur.

## 2. Validate the artifact at the correct level

`_execute_factor_code` writes a temporary Python program, runs it as a subprocess with a 300-second timeout, reads a Parquet result, and removes temporary files. This path inherits the process environment and runs on the host. A timeout is an execution limit; it is not an isolation boundary. Review generated code and use an isolated environment before executing untrusted proposals.

[evaluators.py](../../backend/packages/harness/deerflow/tools/builtins/quant/evaluators.py) supplies six structural checks:

| Check | Current behavior | What it establishes |
|---|---|---|
| Infinite values | Counts `inf` | No infinite numeric entries |
| Single column | Requires one column | One factor output |
| Basic format | Checks two named index levels and one column | Expected table shape |
| Daily frequency | Median unique-date gap at most two days | A frequency heuristic |
| Row count | At least 99% of reference when supplied | Approximate coverage |
| Index consistency | At least 99% row ratio and overlap when supplied | Alignment with a reference |

When a reference is absent, corresponding checks can report a pass with a skipped-comparison explanation. Passing these checks does not prove absence of leakage, economic significance, or formula/code equivalence. Those require distinct tests. A factor can therefore be structurally well-formed and still be unusable at decision time.

## 3. Record ancestry and evidence in a DAG

[DualRepFactorNode](../../backend/packages/harness/deerflow/tools/builtins/quant/dag/knowledge_graph.py) stores formula, code, parent and child IDs, depth, description, IC-family metrics, selection count, status, hypothesis, feedback, and a decision flag. `FactorDAG.insert` rejects duplicate IDs and requires the parent to exist before linking a child.

For example, node `momentum_5d` can have a child `momentum_5d_scaled`, whose transformation changes scale or conditioning. A path to the root explains the sequence of ideas, while measurements explain whether each idea helped. Similar names do not establish this relation; the parent ID does. Rejected and evicted nodes remain useful history, because a later hypothesis may need to understand why a branch stopped.

The manuscript describes a fat node containing formula, code, eight metrics, topology, trace, and knowledge. The current `DualRepFactorNode` retains formula/code, ancestry, depth, IC-family metrics, feedback, status, and decision fields. It does not declare ARR, IR, drawdown, Sharpe, or Calmar; dynamically attached values are not retained by its serializer. A reader should therefore distinguish the manuscript's intended knowledge object from the fields currently round-tripped by code.

Two details matter for interpretation. `get_sota` returns the latest node marked `decision=True`, not a general metric argmax. `query_code_experience` currently returns an empty list. Start a static serialization review by listing each field written by mining code and comparing it with `to_dict` and `from_dict`; this makes a future round-trip test precise.

## 4. Retrieve useful starting points

[BayesianFactorRetriever](../../backend/packages/harness/deerflow/tools/builtins/quant/retrieval/bayesian_retriever.py) considers active candidates and combines an ICIR-derived quality term with a pool-quality term. A simplified view of the implemented quality factor is

```math
q_i=\sigma(z(|ICIR_i|))\,0.95^{depth_i}\,0.9^{\max(selected_i-2,0)},
```
```math
score_i=clip(q_i\,pool_i).
```

Here `z` standardizes values across candidates and `σ` is the logistic function. The implementation normalizes pool scores, optionally reserves separate quotas for leaves and non-leaves, and increments the selection counts of retrieved nodes. With too few candidates it returns the available nodes directly.

**Hand example.** Suppose the standardized ICIR values are `[-1, 0, 1]`, depths are `[0, 1, 2]`, and selection counts are `[0, 3, 2]`. Then the three quality terms are approximately `0.269`, `0.5×0.95×0.9=0.428`, and `0.731×0.95²=0.660`. If pool terms are `[0.9, 0.6, 0.3]`, the combined scores are about `[0.242, 0.257, 0.198]`. The middle candidate wins in a single global ranking; quota splitting can change the final selection.

The class name is not a proof of a calibrated posterior probability. Its current response-correlation helper uses scalar IC values, with `abs(IC_i IC_j)/(abs(IC_i) abs(IC_j)+epsilon)`, which is near one for any two nonzero ICs. It does not measure correlation of factor-output vectors. Optional semantic similarity uses embeddings and can make an external API call. Edit-distance scoring is disabled by default. These distinctions define concrete validation tasks. Retrieval also mutates selection counts, so a read of the retriever can change later scores when executed in a live session.

## 5. Apply the admission rule

[DualAdmissionChecker](../../backend/packages/harness/deerflow/tools/builtins/quant/admission/dual_checker.py) has two routes with default thresholds:

- **Quality route A:** absolute IC at least `0.006`, and, when a parent exists, absolute ICIR greater than the parent's.
- **Diversity route B:** a parent exists, IC passes, absolute ICIR exceeds `0.70 × parent ICIR`, and maximum mutual correlation is below both `0.45` and the parent's value.
- Both routes additionally require maximum mutual correlation at most `0.9`; a full pool evicts the lowest absolute-IC member.

**Hand example.** Parent ICIR is `0.50`; candidate absolute IC is `0.008` and ICIR is `0.40`. Route A fails because `0.40 < 0.50`. Route B's ICIR gate passes because `0.40 > 0.35`. If measured candidate maximum correlation is `0.30` and the parent's is `0.60`, both diversity comparisons pass, so B can admit the candidate.

When two factor Parquet series are available, the checker aligns their indices and estimates daily Pearson correlation using the first column, subject to a minimum overlap. Its fallback uses only the signs of scalar ICs: same sign gives one, opposite signs give zero. Label the fallback explicitly in an experiment record; it cannot substantiate measured diversity.

The manuscript's thresholds describe a different research configuration: its quality floor is `IC > 0.10`, the exploration comparison uses `0.90 × parent ICIR`, and its mutual-correlation threshold is `0.70`. Its retrieval diversity discussion also includes response, semantic, and edit-distance thresholds. Record the source and configuration for each threshold; copying the historical numbers into a current-code explanation would change the rule being taught.

## 6. Schedule factor work or model work

[BanditScheduler](../../backend/packages/harness/deerflow/tools/builtins/quant/bandit/scheduler.py) selects between `factor` and `model` using a sampled linear score. The implemented state has **nine** entries:

```math
x=[IC,ICIR,RankIC,RankICIR,ARR,IR,-MDD,Sharpe,Calmar].
```

The reward helper is `r = wᵀx`, with weights `[.10,.10,.05,.05,.25,.15,.10,.15,.05]`. For each arm, the scheduler samples coefficients from its stored mean and inverse precision, then picks the arm with the larger dot product. At an all-zero state both scores are zero, so insertion order resolves the tie to `factor`.

Raw metrics have different scales. Also, current MDD is negative, so `-MDD` is positive: a deeper drawdown can increase the raw reward term. These semantics need a deliberate reward-design review before interpreting scheduling as research improvement.

The historical manuscript writes an eight-entry vector `[IC, ICIR, RankIC, RankICIR, ARR, IR, -MDD, Sharpe]` with weights `[.1,.1,.05,.05,.25,.15,.1,.2]`. The current implementation adds `Calmar`, uses weights `[.10,.10,.05,.05,.25,.15,.10,.15,.05]`, and builds `-mdd` from its negative drawdown field. Captions for historical figures should retain that eight-dimensional context; current code and current prose should use the nine-entry contract.

For a standard Bayesian linear update, one expects

```math
P'=P+xx^T/\sigma^2,\qquad \mu'=(P')^{-1}(P\mu+xr/\sigma^2).
```

The current `record` function uses the updated precision in its right-hand side. Compare a second update with a nonzero prior mean to expose the difference; a first update from a zero mean can hide it. This tutorial records the discrepancy without changing the research algorithm.

**Two updates by hand.** Reduce to one dimension and use the implementation's prior variance `10` and noise variance `0.5`, so `P₀ = 0.1` and `μ₀ = 0`. Observe `x = 1, r = 1` twice. The first update gives `P₁ = 2.1` and `μ₁ = 2/2.1 = 20/21` under either calculation. On the second observation, `P₂ = 4.1`. The standard update gives `μ₂ = (2.1 × 20/21 + 2)/4.1 = 40/41 ≈ 0.97561`. Using the new precision on the right gives `μ₂ = (4.1 × 20/21 + 2)/4.1 = 20/21 + 20/41 ≈ 1.44019`. This is a derivation from the source expressions, not an observed run.

Also trace the state builder used by the caller. The scheduler helper constructs `-mdd`, while the separate metrics helper's `to_state_vector` includes raw `mdd`. A nine-field list with the same labels is insufficient if two construction paths disagree on sign.

## 7. Understand orchestration and completion

`quant_mining_tool` exposes staged handlers for assembly, scheduling, retrieval, generation, verification, backtesting, admission, and update. Its `run_mining_loop` convenience action currently performs scheduling and retrieval only. A complete research cycle must explicitly orchestrate the remaining stages and save their evidence. The manuscript's eight-stage loop is thus a design map; the current convenience action is a partial implementation of that map.

Likewise, `factor_generate`'s `pipeline_with_new_factors` counts active nodes but currently calls `run_full_pipeline(config)` without passing factor values. The message alone does not prove that generated factors entered model training. The extra-factor integration path requires a feature-column test and an ablation before a performance claim.

## Exercises and answers

**A.** A candidate passes all six structural checks. Can it use tomorrow's close?

**Answer.** The structure checks do not inspect temporal information use. A dedicated leakage test must verify feature availability at decision time.

**B.** Two factors both have IC `0.03`. Does the retriever's current near-one “correlation” prove their output series are redundant?

**Answer.** It follows from the scalar heuristic, not from aligned series. Compute correlation on actual factor outputs before claiming redundancy.

**C.** Why is a two-step Bandit update test more informative than a one-step test starting at zero?

**Answer.** After the first update, the mean can be nonzero, so using `P'μ` instead of `Pμ` affects the result. The initial zero mean removes that distinction.

## Contribution deliverable

Choose one lifecycle boundary and submit a static source review before attempting a run. Include a manual expected result, the exact source symbol, the evidence type (design description, code path, or observed output), and the broader claim it does not establish. Good first tasks include node serialization, vector-based diversity, reward scaling, and tracing whether a generated factor reaches the training matrix. Register the work through [CONTRIBUTING](../../CONTRIBUTING.md).
