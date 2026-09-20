# Case 2 — From a factor hypothesis to research memory

[Home](../../README.md) · [中文](../zh/case-02-factor-lifecycle.md) · [Previous case](case-01-task-to-backtest.md)

## Objective and prerequisites

Follow a factor through generation, executable representation, validation, ancestry, retrieval, admission, and research scheduling. You need the data-contract and metric concepts in Case 1, plus the idea that a graph edge records a parent-child relation. This is a source-based walkthrough; the hand calculations use invented observations.

The main sources are [factor_generate_tool.py](../../backend/packages/harness/deerflow/tools/builtins/quant/factor_generate_tool.py), [quant_mining_tool.py](../../backend/packages/harness/deerflow/tools/builtins/quant/quant_mining_tool.py), and the `dag`, `retrieval`, `admission`, and `bandit` subdirectories beside them.

## 1. Express one falsifiable hypothesis

Suppose the hypothesis is: “Recent price momentum contains information about a specified future return.” A simple candidate is

$$f_{i,t}=P_{i,t}/P_{i,t-5}-1.$$

If today's close is 105 and the close five observations ago is 100, the feature is `0.05`. A Qlib-style representation is `$close / Ref($close, 5) - 1`. Decide when the close is observable and when a trade can occur before using it. The label belongs to a later interval and must never enter feature construction.

The implementation keeps both a formula and Python code. The execution contract expects `calculate_factor(df)` to produce a Series or a single-column DataFrame with the original stock/date index. Group by instrument when applying a lag; shifting a flattened table can accidentally cross from one stock to another.

`generate_factors` prepares generation prompts and saves pending context. It does not, by itself, establish that a new formula has been executed or evaluated. A later `validate_and_add_factor` action accepts the proposed code and measurements follow that path.

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

When a reference is absent, corresponding checks can report a pass with a skipped-comparison explanation. Passing these checks does not prove absence of leakage, economic significance, or formula/code equivalence. Those require distinct tests.

## 3. Record ancestry and evidence in a DAG

[DualRepFactorNode](../../backend/packages/harness/deerflow/tools/builtins/quant/dag/knowledge_graph.py) stores formula, code, parent and child IDs, depth, description, IC-family metrics, selection count, status, hypothesis, feedback, and a decision flag. `FactorDAG.insert` rejects duplicate IDs and requires the parent to exist before linking a child.

For example, node `momentum_5d` can have a child `momentum_5d_scaled`, whose transformation changes scale or conditioning. A path to the root explains the sequence of ideas, while measurements explain whether each idea helped. Similar names do not establish this relation; the parent ID does.

Two details matter for interpretation. `get_sota` returns the latest node marked `decision=True`, not a general metric argmax. `query_code_experience` currently returns an empty list. In addition, ARR, IR, drawdown, Sharpe, and Calmar are not declared node fields, although mining code may attach them dynamically; the node serializer does not retain those extra attributes. A round-trip test is therefore a meaningful first contribution.

## 4. Retrieve useful starting points

[BayesianFactorRetriever](../../backend/packages/harness/deerflow/tools/builtins/quant/retrieval/bayesian_retriever.py) considers active candidates and combines an ICIR-derived quality term with a pool-quality term. A simplified view of the implemented quality factor is

$$q_i=\sigma(z(|ICIR_i|))\,0.95^{depth_i}\,0.9^{\max(selected_i-2,0)},$$
$$score_i=clip(q_i\,pool_i).$$

Here `z` standardizes values across candidates and `σ` is the logistic function. The implementation normalizes pool scores, optionally reserves separate quotas for leaves and non-leaves, and increments the selection counts of retrieved nodes. With too few candidates it returns the available nodes directly.

**Hand example.** Suppose the standardized ICIR values are `[-1, 0, 1]`, depths are `[0, 1, 2]`, and selection counts are `[0, 3, 2]`. Then the three quality terms are approximately `0.269`, `0.5×0.95×0.9=0.428`, and `0.731×0.95²=0.660`. If pool terms are `[0.9, 0.6, 0.3]`, the combined scores are about `[0.242, 0.257, 0.198]`. The middle candidate wins in a single global ranking; quota splitting can change the final selection.

The class name is not a proof of a calibrated posterior probability. Its current response-correlation helper uses scalar IC values, with `abs(IC_i IC_j)/(abs(IC_i) abs(IC_j)+epsilon)`, which is near one for any two nonzero ICs. It does not measure correlation of factor-output vectors. Optional semantic similarity uses embeddings and can make an external API call. Edit-distance scoring is disabled by default. These distinctions define concrete validation tasks.

## 5. Apply the admission rule

[DualAdmissionChecker](../../backend/packages/harness/deerflow/tools/builtins/quant/admission/dual_checker.py) has two routes with default thresholds:

- **Quality route A:** absolute IC at least `0.006`, and, when a parent exists, absolute ICIR greater than the parent's.
- **Diversity route B:** a parent exists, IC passes, absolute ICIR exceeds `0.70 × parent ICIR`, and maximum mutual correlation is below both `0.45` and the parent's value.
- Both routes additionally require maximum mutual correlation at most `0.9`; a full pool evicts the lowest absolute-IC member.

**Hand example.** Parent ICIR is `0.50`; candidate absolute IC is `0.008` and ICIR is `0.40`. Route A fails because `0.40 < 0.50`. Route B's ICIR gate passes because `0.40 > 0.35`. If measured candidate maximum correlation is `0.30` and the parent's is `0.60`, both diversity comparisons pass, so B can admit the candidate.

When two factor Parquet series are available, the checker aligns their indices and estimates daily Pearson correlation using the first column, subject to a minimum overlap. Its fallback uses only the signs of scalar ICs: same sign gives one, opposite signs give zero. Label the fallback explicitly in an experiment record; it cannot substantiate measured diversity.

## 6. Schedule factor work or model work

[BanditScheduler](../../backend/packages/harness/deerflow/tools/builtins/quant/bandit/scheduler.py) selects between `factor` and `model` using a sampled linear score. The implemented state has **nine** entries:

$$x=[IC,ICIR,RankIC,RankICIR,ARR,IR,-MDD,Sharpe,Calmar].$$

The reward helper is `r = wᵀx`, with weights `[.10,.10,.05,.05,.25,.15,.10,.15,.05]`. For each arm, the scheduler samples coefficients from its stored mean and inverse precision, then picks the arm with the larger dot product. At an all-zero state both scores are zero, so insertion order resolves the tie to `factor`.

Raw metrics have different scales. Also, current MDD is negative, so `-MDD` is positive: a deeper drawdown can increase the raw reward term. These semantics need a deliberate reward-design review before interpreting scheduling as research improvement.

For a standard Bayesian linear update, one expects

$$P'=P+xx^T/\sigma^2,\qquad \mu'=(P')^{-1}(P\mu+xr/\sigma^2).$$

The current `record` function uses the updated precision in its right-hand side. Compare a second update with a nonzero prior mean to expose the difference; a first update from a zero mean can hide it. This tutorial records the discrepancy without changing the research algorithm.

## 7. Understand orchestration and completion

`quant_mining_tool` exposes staged handlers for assembly, scheduling, retrieval, generation, verification, backtesting, admission, and update. Its `run_mining_loop` convenience action currently performs scheduling and retrieval only. A complete research cycle must explicitly orchestrate the remaining stages and save their evidence.

Likewise, `factor_generate`'s `pipeline_with_new_factors` counts active nodes but currently calls `run_full_pipeline(config)` without passing factor values. The message alone does not prove that generated factors entered model training. The extra-factor integration path requires a feature-column test and an ablation before a performance claim.

## Exercises and answers

**A.** A candidate passes all six structural checks. Can it use tomorrow's close?

**Answer.** The structure checks do not inspect temporal information use. A dedicated leakage test must verify feature availability at decision time.

**B.** Two factors both have IC `0.03`. Does the retriever's current near-one “correlation” prove their output series are redundant?

**Answer.** It follows from the scalar heuristic, not from aligned series. Compute correlation on actual factor outputs before claiming redundancy.

**C.** Why is a two-step Bandit update test more informative than a one-step test starting at zero?

**Answer.** After the first update, the mean can be nonzero, so using `P'μ` instead of `Pμ` affects the result. The initial zero mean removes that distinction.

## Contribution deliverable

Choose one lifecycle boundary. Produce a small deterministic fixture, a manual expected result, a saved structured output, and the corresponding source reference. Good first tasks include node serialization, vector-based diversity, reward scaling, and verifying that a generated factor reaches the actual training matrix. Register the work through [CONTRIBUTING](../../CONTRIBUTING.md).
