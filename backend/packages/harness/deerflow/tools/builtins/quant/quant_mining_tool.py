from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState
from deerflow.config.quant_config import QuantConfig
from deerflow.tools.builtins.quant.admission import DualAdmissionChecker
from deerflow.tools.builtins.quant.bandit import BanditScheduler, build_state_vector, compute_reward
from deerflow.tools.builtins.quant.dag import DualRepFactorNode, FactorDAG
from deerflow.tools.builtins.quant.dag.model_log import ModelExperiment, ModelExperimentLog
from deerflow.tools.builtins.quant.metrics import QuantMetrics
from deerflow.tools.builtins.quant.prompts import (
    FACTOR_CODE_FIX_SYSTEM,
    FACTOR_CODE_FIX_USER,
    FACTOR_EVAL_FEEDBACK_TEMPLATE,
    FACTOR_GENERATION_SYSTEM,
    FACTOR_GENERATION_USER,
)
from deerflow.tools.builtins.quant.retrieval import BayesianFactorRetriever
from deerflow.tools.builtins.quant.scenario import QuantScenario

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROUNDS = 20
IC_DEDUP_THRESHOLD = 0.99
CONSECUTIVE_FAIL_LIMIT = 3
CONVERGENCE_PATIENCE = 5

STEP_TRANSITIONS = {
    "step0_assemble": {0},
    "step1_bandit": {0, 7},
    "step2_retrieve": {1},
    "step3_generate": {2},
    "step4_verify": {3},
    "step5_backtest": {4},
    "step6_admit": {5},
    "step7_update": {6},
}

STEP_NEXT = {
    0: "step1_bandit",
    1: "step2_retrieve",
    2: "step3_generate",
    3: "step4_verify",
    4: "step5_backtest",
    5: "step6_admit",
    6: "step7_update",
    7: "step1_bandit",
}


def _get_session(state: ThreadState) -> dict | None:
    return state.get("quant_session")


def _ensure_session(state: ThreadState) -> dict:
    session = _get_session(state)
    if session is None:
        raise ValueError("请先调用 quant_analyze(action='init_session') 初始化研究上下文。")
    return session


def _load_dag(session: dict) -> FactorDAG:
    dag_state = session.get("dag_state")
    if dag_state:
        return FactorDAG.from_json(dag_state)
    return FactorDAG()


def _save_dag(session: dict, dag: FactorDAG) -> None:
    session["dag_state"] = dag.to_json()


def _load_bandit(session: dict) -> BanditScheduler:
    bandit_state = session.get("bandit_state")
    if bandit_state:
        return BanditScheduler.from_json(bandit_state)
    return BanditScheduler()


def _save_bandit(session: dict, bandit: BanditScheduler) -> None:
    session["bandit_state"] = bandit.to_json()


def _load_model_log(session: dict) -> ModelExperimentLog:
    log_state = session.get("model_experiment_log")
    if log_state:
        return ModelExperimentLog.from_json(log_state)
    return ModelExperimentLog()


def _save_model_log(session: dict, log: ModelExperimentLog) -> None:
    session["model_experiment_log"] = log.to_json()


def _load_retriever(session: dict) -> BayesianFactorRetriever:
    return BayesianFactorRetriever()


def _load_admission_checker(session: dict) -> DualAdmissionChecker:
    return DualAdmissionChecker()


def _check_step_guard(action: str, mining_state: dict) -> str | None:
    if action in ("get_mining_status", "run_mining_loop", "step0_assemble"):
        return None
    current_step = mining_state.get("step", -1)
    allowed_steps = STEP_TRANSITIONS.get(action, set())
    if current_step not in allowed_steps:
        next_action = STEP_NEXT.get(current_step, "step0_assemble")
        return (
            f"步骤守卫: 当前在Step {current_step}，不允许执行{action}。"
            f"请按顺序执行，下一步应为: {next_action}。"
            f"或调用 get_mining_status 查看当前状态。"
        )
    return None


def _check_ic_dedup(node: DualRepFactorNode, dag: FactorDAG) -> bool:
    if not node.formula or node.ic == 0.0:
        return False
    for existing in dag.active_nodes():
        if existing.node_id == node.node_id:
            continue
        if existing.ic == 0.0:
            continue
        if existing.formula and node.formula and existing.formula.strip() == node.formula.strip():
            return True
        ni_ic, ej_ic = abs(node.ic), abs(existing.ic)
        if ni_ic > 0 and ej_ic > 0:
            ic_ratio = min(ni_ic, ej_ic) / max(ni_ic, ej_ic)
            if ic_ratio >= IC_DEDUP_THRESHOLD and (node.ic > 0) == (existing.ic > 0):
                return True
    return False


def _build_traces_text(dag: FactorDAG, node_id: str | None) -> str:
    if node_id is None:
        return "（无已有路径，这是第一个因子。请从基础算子开始，确保维度为0。）"
    path = dag.path_to_root(node_id)
    if not path:
        return "（无已有路径）"
    lines = []
    for i, node in enumerate(path):
        ic_str = f"IC={node.ic:.4f}, ICIR={node.icir:.4f}" if node.ic != 0.0 else "未评估"
        fb_str = ""
        if node.feedback:
            fb_str = f"\n      反馈: {node.feedback[:120]}"
        desc_str = f"\n      描述: {node.description[:100]}" if node.description else ""
        formula_str = node.formula or node.description or node.node_id
        lines.append(f"  [{i}] {formula_str}\n      指标: ({ic_str}){desc_str}{fb_str}")
    lines.append("\n学习轨迹中的优化模式，基于原始表达式和生成轨迹生成新表达式。"
                 "如果轨迹显示IC逐步提升，继续沿此方向优化；"
                 "如果轨迹显示IC下降或停滞，尝试完全不同的方向。")
    return "\n".join(lines)


def _build_retrieval_context(retrieved_nodes: list[DualRepFactorNode], dag: FactorDAG) -> str:
    if not retrieved_nodes:
        return "（无检索结果）"
    lines = []
    for i, node in enumerate(retrieved_nodes):
        path = dag.path_to_root(node.node_id)
        path_str = " → ".join(
            n.formula or n.description or n.node_id[:8] for n in path
        )
        ic_str = f"IC={node.ic:.4f}, ICIR={node.icir:.4f}" if node.ic != 0.0 else "未评估"
        ctx_parts = [f"  [{i}] {node.formula or node.description or node.node_id} ({ic_str})"]
        ctx_parts.append(f"      演化轨迹: {path_str}")
        if node.feedback:
            ctx_parts.append(f"      反馈: {node.feedback[:120]}")
        if node.code_experience:
            ctx_parts.append(f"      代码经验: {json.dumps(node.code_experience, ensure_ascii=False)[:120]}")
        if node.model_context:
            ctx_parts.append(f"      模型引用: {node.model_context}")
        lines.append("\n".join(ctx_parts))
    return "\n\n".join(lines)


def _build_model_retrieval_context(model_log: ModelExperimentLog, dag: FactorDAG) -> str:
    sota_model = model_log.get_sota_model()
    sota_factor = dag.get_sota()

    lines = []
    if sota_model:
        lines.append(f"  SOTA模型: {sota_model.architecture}")
        lines.append(f"  假设: {sota_model.hypothesis[:200]}")
        if sota_model.feedback:
            lines.append(f"  反馈: {sota_model.feedback[:200]}")
        if sota_model.metrics:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in sota_model.metrics.items() if isinstance(v, (int, float)))
            lines.append(f"  指标: {metrics_str}")
    else:
        lines.append("  （无SOTA模型，将使用LightGBM基线）")

    if sota_factor:
        lines.append(f"\n  SOTA因子: {sota_factor.formula or sota_factor.description}")
        lines.append(f"  IC={sota_factor.ic:.4f}, ICIR={sota_factor.icir:.4f}")

    return "\n".join(lines)


def _build_specific_trace(dag: FactorDAG, model_log: ModelExperimentLog, action: str) -> str:
    hist = dag.get_hist()
    if not hist:
        return "（无历史实验）"

    lines = []
    if action == "factor":
        factor_nodes = [n for n in hist if n.action == "factor"]
        for node in factor_nodes[-15:]:
            status_mark = "✅" if node.decision else ("❌" if node.status == "rejected" else "⏳")
            lines.append(f"  {status_mark} [{node.action}] {node.formula or node.description or node.node_id[:8]} "
                         f"IC={node.ic:.4f} ICIR={node.icir:.4f} decision={node.decision}")
            if node.feedback:
                lines.append(f"      反馈: {node.feedback[:100]}")
        sota_model = model_log.get_sota_model()
        if sota_model and sota_model.decision:
            lines.append(f"  [model-SOTA] {sota_model.architecture} "
                         f"hypothesis={sota_model.hypothesis[:80]}")
    else:
        for exp in model_log.experiments:
            lines.append(f"  [model] {exp.architecture} decision={exp.decision}")
        sota_factor = dag.get_sota()
        if sota_factor and sota_factor.decision:
            lines.append(f"  [factor-SOTA] {sota_factor.formula or sota_factor.description} "
                         f"IC={sota_factor.ic:.4f} ICIR={sota_factor.icir:.4f}")

    return "\n".join(lines)


def _build_rag_strategy(action: str, hist_len: int) -> str:
    if action == "factor":
        if hist_len < 3:
            return "先尝试最简单最快的因子，使用基础算子（Rank/Zscore/TsMean/TsStd）和短窗口（5/10/20日）。每次生成2-3个因子，确保维度为0。"
        elif hist_len < 6:
            return "尝试中等复杂度因子，结合截面操作（Rank/Zscore）和时间序列操作（TsCorr/TsDelta/TsRank），窗口可扩展到60/120日。探索动量反转、波动率、流动性等不同方向。"
        elif hist_len < 12:
            return "需要尝试高IC因子，不要包含与SOTA因子库相似的因子。尝试：(1)多算子组合如Zscore(Mul(Rank(x), Rank(y)))；(2)不同窗口参数如%%d占位符（5/10/20/30自动搜索最优）；(3)跨维度交互如Div(TsMean(close,%%d), TsStd(volume,%%d))。"
        else:
            return "深度探索阶段：尝试与已有因子完全不同的方向。建议：(1)基本面/价值方向因子；(2)波动率形态因子；(3)量价背离因子；(4)使用%%d占位符让系统自动搜索最优窗口。避免重复已有因子的变体。"
    return "GRU/LSTM适合时序数据，不生成GNN模型。"


def _build_baseline_factors_text(session: dict) -> str:
    base_features = session.get("base_features", {})
    if not base_features:
        return "（无基线因子）"
    items = list(base_features.items())[:10]
    lines = [f"  {name}: {expr}" for name, expr in items]
    if len(base_features) > 10:
        lines.append(f"  ... 共{len(base_features)}个因子")
    return "\n".join(lines)


def _build_failure_history(dag: FactorDAG, limit: int = 5) -> str:
    rejected = [n for n in dag.get_hist() if n.status == "rejected" and n.feedback]
    if not rejected:
        return "（无失败记录）"
    recent = rejected[-limit:]
    lines = []
    for node in recent:
        formula = node.formula or node.description or node.node_id[:8]
        lines.append(f"  ❌ {formula} IC={node.ic:.4f} → {node.feedback[:100]}")
    return "\n".join(lines)


def _action_step0_assemble(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    config = QuantConfig(**session["config"])
    dag = _load_dag(session)
    bandit = _load_bandit(session)
    model_log = _load_model_log(session)

    if dag.size() == 0:
        from deerflow.tools.builtins.quant.factor_baseline_library_tool import ALPHA20
        base_features = session.get("base_features", ALPHA20)
        for name, expr in base_features.items():
            node = DualRepFactorNode(
                node_id=FactorDAG.new_node_id(),
                formula=expr,
                description=name,
                topic="baseline",
                code_verified=True,
                decision=True,
                action="factor",
                timestamp=time.time(),
            )
            dag.insert(node)

        _save_dag(session, dag)

        try:
            from deerflow.tools.builtins.quant.qlib_pipeline import run_full_pipeline
            baseline_metrics = run_full_pipeline(config)
            baseline_dict = baseline_metrics.model_dump()
            session["sota_metrics"] = baseline_dict
            for node in dag.active_nodes():
                node.ic = baseline_metrics.ic or 0.0
                node.icir = baseline_metrics.icir or 0.0
                node.rank_ic = baseline_metrics.rank_ic or 0.0
                node.rank_icir = baseline_metrics.rank_icir or 0.0
                node.arr = baseline_metrics.arr or 0.0
                node.ir = baseline_metrics.ir or 0.0
                node.mdd = baseline_metrics.mdd or 0.0
                node.sharpe = baseline_metrics.sharpe or 0.0
                node.calmar = baseline_metrics.calmar or 0.0
            _save_dag(session, dag)
        except Exception as e:
            logger.warning(f"ALPHA20 cold start backtest failed: {e}")

    scenario = QuantScenario(config)
    system_prompt = FACTOR_GENERATION_SYSTEM
    scenario_text = scenario.get_scenario_all_desc()

    mining_state = {
        "step": 0,
        "max_rounds": kwargs.get("max_rounds", DEFAULT_MAX_ROUNDS),
        "current_round": 0,
        "action": None,
        "retrieved_nodes": [],
        "generated_factors": [],
        "current_metrics": None,
        "admission_result": None,
        "started_at": datetime.now().isoformat(),
    }
    session["mining_state"] = mining_state
    _save_bandit(session, bandit)
    _save_model_log(session, model_log)

    msg = (
        f"Step 0 完成：上下文组装完毕。\n"
        f"DAG节点数: {dag.size()}\n"
        f"活跃节点: {len(dag.active_nodes())}\n"
        f"模型实验数: {len(model_log.experiments)}\n"
        f"最大迭代轮数: {mining_state['max_rounds']}\n\n"
        f"下一步: 调用 step1_bandit 开始Bandit调度。\n\n"
        f"=== System Prompt ===\n{system_prompt}\n\n"
        f"=== 场景背景 ===\n{scenario_text}"
    )
    return msg, {"quant_session": session}


def _action_step1_bandit(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    dag = _load_dag(session)
    bandit = _load_bandit(session)
    model_log = _load_model_log(session)
    mining_state = session.get("mining_state", {})

    prev_metrics = mining_state.get("current_metrics")
    if prev_metrics:
        state_vector = build_state_vector(prev_metrics)
    else:
        sota = dag.get_sota()
        if sota:
            metrics_dict = {
                "ic": sota.ic or 0.0, "icir": sota.icir or 0.0,
                "rank_ic": sota.rank_ic or 0.0, "rank_icir": sota.rank_icir or 0.0,
                "arr": sota.arr or 0.0, "ir": sota.ir or 0.0,
                "mdd": sota.mdd or 0.0, "sharpe": sota.sharpe or 0.0,
                "calmar": sota.calmar or 0.0,
            }
            sota_metrics = session.get("sota_metrics")
            if sota_metrics:
                metrics_dict.update(sota_metrics)
            state_vector = build_state_vector(metrics_dict)
        else:
            state_vector = build_state_vector(None)

    action = bandit.decide(state_vector)

    mining_state["step"] = 1
    mining_state["action"] = action
    mining_state["current_round"] = mining_state.get("current_round", 0) + 1
    mining_state["state_vector"] = state_vector
    session["mining_state"] = mining_state
    _save_bandit(session, bandit)

    current_round = mining_state["current_round"]
    max_rounds = mining_state.get("max_rounds", DEFAULT_MAX_ROUNDS)

    msg = (
        f"Step 1 Bandit调度完成（第{current_round}/{max_rounds}轮）。\n"
        f"决策: {action}\n"
        f"状态向量: [{', '.join(f'{v:.4f}' for v in state_vector)}]\n"
        f"Bandit历史: {len(bandit.history)}次决策\n\n"
        f"下一步: 调用 step2_retrieve 执行{'因子' if action == 'factor' else '模型'}检索。"
    )
    return msg, {"quant_session": session}


def _action_step2_retrieve(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    dag = _load_dag(session)
    model_log = _load_model_log(session)
    retriever = _load_retriever(session)
    mining_state = session.get("mining_state", {})
    action = mining_state.get("action", "factor")

    if action == "factor":
        top_k = kwargs.get("top_k") or 4
        retrieved_nodes = retriever.retrieve(dag, top_k=top_k)

        _save_dag(session, dag)

        retrieval_context = _build_retrieval_context(retrieved_nodes, dag)
        specific_trace = _build_specific_trace(dag, model_log, action)
        rag_strategy = _build_rag_strategy(action, len(dag.get_hist()))

        mining_state["step"] = 2
        mining_state["retrieved_nodes"] = [n.node_id for n in retrieved_nodes]
        session["mining_state"] = mining_state

        msg = (
            f"Step 2 因子检索完成。\n"
            f"检索到 {len(retrieved_nodes)} 个种子节点。\n\n"
            f"=== 检索上下文 ===\n{retrieval_context}\n\n"
            f"=== 特定轨迹过滤 ===\n{specific_trace}\n\n"
            f"=== RAG策略 ===\n{rag_strategy}\n\n"
            f"下一步: 调用 step3_generate 生成新因子。"
        )
        return msg, {"quant_session": session}

    model_context = _build_model_retrieval_context(model_log, dag)
    specific_trace = _build_specific_trace(dag, model_log, action)
    rag_strategy = _build_rag_strategy(action, len(model_log.experiments))

    mining_state["step"] = 2
    session["mining_state"] = mining_state

    msg = (
        f"Step 2 模型检索完成。\n\n"
        f"=== 模型上下文 ===\n{model_context}\n\n"
        f"=== 特定轨迹过滤 ===\n{specific_trace}\n\n"
        f"=== RAG策略 ===\n{rag_strategy}\n\n"
        f"下一步: 调用 step3_generate 生成新模型假设和代码。"
    )
    return msg, {"quant_session": session}


def _action_step3_generate(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    dag = _load_dag(session)
    mining_state = session.get("mining_state", {})
    action = mining_state.get("action", "factor")

    topic = kwargs.get("topic", "动量与反转")
    num_factors = kwargs.get("num_factors", 3)

    if action == "factor":
        retrieved_ids = mining_state.get("retrieved_nodes", [])
        parent_node = None
        if retrieved_ids:
            parent_node = dag.get_node(retrieved_ids[0])

        parent_id = parent_node.node_id if parent_node else None
        traces = _build_traces_text(dag, parent_id)
        baseline_text = _build_baseline_factors_text(session)
        failure_history = _build_failure_history(dag)

        generation_prompt = FACTOR_GENERATION_USER.format(
            topic=topic,
            traces=traces,
            baseline_factors=baseline_text,
            failure_history=failure_history,
            num_factors=num_factors,
        )

        mining_state["step"] = 3
        mining_state["parent_id"] = parent_id
        session["mining_state"] = mining_state

        msg = (
            f"Step 3 因子生成请求已准备。\n"
            f"研究主题: {topic}\n"
            f"生成数量: {num_factors}\n"
            f"父节点: {parent_id or '根节点'}\n\n"
            f"请使用LLM以下列Prompt生成因子：\n\n"
            f"=== System Prompt ===\n{FACTOR_GENERATION_SYSTEM}\n\n"
            f"=== User Prompt ===\n{generation_prompt}\n\n"
            f"生成后，请通过 step4_verify action 提交每个因子进行验证。"
        )
        return msg, {"quant_session": session}

    model_hypothesis_prompt = _build_model_hypothesis_prompt(session, dag)
    mining_state["step"] = 3
    session["mining_state"] = mining_state

    msg = (
        f"Step 3 模型生成请求已准备。\n\n"
        f"请使用LLM以下列Prompt生成模型假设和代码：\n\n"
        f"=== Model Hypothesis Prompt ===\n{model_hypothesis_prompt}\n\n"
        f"生成后，请通过 step4_verify action 提交模型代码进行验证。"
    )
    return msg, {"quant_session": session}


def _build_model_hypothesis_prompt(session: dict, dag: FactorDAG) -> str:
    model_log = _load_model_log(session)
    sota_model = model_log.get_sota_model()
    sota_factor = dag.get_sota()

    parts = [
        "你是一位量化模型研究专家。请基于以下信息生成新的模型假设。\n",
        "## 当前SOTA模型",
    ]
    if sota_model:
        parts.append(f"架构: {sota_model.architecture}")
        parts.append(f"假设: {sota_model.hypothesis}")
        if sota_model.feedback:
            parts.append(f"反馈: {sota_model.feedback[:200]}")
    else:
        parts.append("LightGBM基线模型（Qlib默认）")

    if sota_factor:
        parts.append(f"\n## 当前SOTA因子")
        parts.append(f"公式: {sota_factor.formula or sota_factor.description}")
        parts.append(f"IC={sota_factor.ic:.4f}, ICIR={sota_factor.icir:.4f}")

    parts.append("\n## 输出格式")
    parts.append('严格输出JSON: {"hypothesis": "假设文本", "architecture": "模型架构", "code": "Python代码"}')
    parts.append("\n注意: GRU/LSTM适合时序数据，不生成GNN模型。")

    return "\n".join(parts)


def _action_step4_verify(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    config = QuantConfig(**session["config"])
    dag = _load_dag(session)
    mining_state = session.get("mining_state", {})
    action = mining_state.get("action", "factor")

    if action == "factor":
        return _verify_factor(session, config, dag, mining_state, **kwargs)
    return _verify_model(session, config, dag, mining_state, **kwargs)


def _verify_factor(session: dict, config: QuantConfig, dag: FactorDAG, mining_state: dict, **kwargs) -> Command:
    from deerflow.tools.builtins.quant.factor_generate_tool import _execute_factor_code, _evaluate_single_factor

    factor_name = kwargs.get("name", "unnamed_factor")
    formula = kwargs.get("formula")
    description = kwargs.get("description", "")
    code = kwargs.get("code", "")
    topic = kwargs.get("topic", "")
    parent_id = kwargs.get("parent_id") or mining_state.get("parent_id")

    if not code:
        return "错误：必须提供因子的Python实现代码（code参数）。", {}

    factor_df, exec_error = _execute_factor_code(code, config)

    if exec_error:
        node = DualRepFactorNode(
            node_id=FactorDAG.new_node_id(),
            formula=formula,
            code=code,
            topic=topic,
            description=f"执行失败: {exec_error[:200]}",
            parent_id=parent_id,
            status="rejected",
            action="factor",
            timestamp=time.time(),
        )
        parent_node = dag.get_node(parent_id) if parent_id else None
        dag.insert(node, parent_node)
        _save_dag(session, dag)

        mining_state["step"] = 4
        mining_state["verify_result"] = "fail"
        mining_state["verify_node_id"] = node.node_id
        mining_state["verify_error"] = exec_error
        session["mining_state"] = mining_state

        fix_prompt = FACTOR_CODE_FIX_USER.format(
            factor_name=factor_name,
            formula=formula or "N/A",
            description=description,
            original_code=code,
            error_message=exec_error,
            eval_feedback="代码执行失败，需要修复。",
        )

        msg = (
            f"Step 4 因子验证失败。\n"
            f"错误: {exec_error[:300]}\n\n"
            f"修复Prompt:\n=== System ===\n{FACTOR_CODE_FIX_SYSTEM}\n\n=== User ===\n{fix_prompt}\n\n"
            f"修复后请通过 step4_verify 重新提交（提供 fixed_code 参数）。"
        )
        return msg, {"quant_session": session}

    metrics, eval_results, eval_error = _evaluate_single_factor(factor_df, config)

    if metrics is not None:
        node = DualRepFactorNode(
            node_id=FactorDAG.new_node_id(),
            formula=formula,
            code=code,
            topic=topic,
            description=description,
            ic=metrics.ic,
            icir=metrics.icir,
            rank_ic=metrics.rank_ic,
            rank_icir=metrics.rank_icir,
            parent_id=parent_id,
            status="active",
            code_verified=True,
            action="factor",
            timestamp=time.time(),
        )

        is_duplicate = _check_ic_dedup(node, dag)
        if is_duplicate:
            node.status = "rejected"
            node.decision = False
            node.description = f"IC去重拒绝: 与池中因子IC相关性≥{IC_DEDUP_THRESHOLD}"

        parent_node = dag.get_node(parent_id) if parent_id else None
        dag.insert(node, parent_node)
        _save_dag(session, dag)

        mining_state["step"] = 4
        mining_state["verify_result"] = "pass" if not is_duplicate else "dedup_reject"
        mining_state["verify_node_id"] = node.node_id
        mining_state["current_metrics"] = metrics.model_dump()
        session["mining_state"] = mining_state

        dedup_note = f"\n⚠ IC去重: 因子与池中已有因子IC相关性≥{IC_DEDUP_THRESHOLD}，已标记为rejected。" if is_duplicate else ""
        msg = (
            f"Step 4 因子验证通过！\n"
            f"公式: {formula or 'N/A'}\n"
            f"IC: {metrics.ic:.4f}, ICIR: {metrics.icir:.4f}\n"
            f"Rank IC: {metrics.rank_ic:.4f}, Rank ICIR: {metrics.rank_icir:.4f}"
            f"{dedup_note}\n\n"
            f"下一步: 调用 step5_backtest 执行回测评估。"
        )
        return msg, {"quant_session": session, "quant_metrics": metrics.model_dump()}

    node = DualRepFactorNode(
        node_id=FactorDAG.new_node_id(),
        formula=formula,
        code=code,
        topic=topic,
        description=f"评估失败: {eval_error[:200]}",
        parent_id=parent_id,
        status="rejected",
        action="factor",
        timestamp=time.time(),
    )
    parent_node = dag.get_node(parent_id) if parent_id else None
    dag.insert(node, parent_node)
    _save_dag(session, dag)

    mining_state["step"] = 4
    mining_state["verify_result"] = "fail"
    mining_state["verify_node_id"] = node.node_id
    mining_state["verify_error"] = eval_error
    session["mining_state"] = mining_state

    msg = (
        f"Step 4 因子评估未通过。\n"
        f"错误: {eval_error[:300]}\n\n"
        f"该因子已标记为rejected，继续下一步。"
    )
    return msg, {"quant_session": session}


def _verify_model(session: dict, config: QuantConfig, dag: FactorDAG, mining_state: dict, **kwargs) -> Command:
    model_log = _load_model_log(session)

    hypothesis = kwargs.get("hypothesis", "")
    architecture = kwargs.get("architecture", "LSTM")
    code = kwargs.get("code", "")

    if not code:
        return "错误：必须提供模型的Python实现代码（code参数）。", {}

    experiment = ModelExperiment(
        experiment_id=ModelExperimentLog.new_experiment_id(),
        hypothesis=hypothesis,
        architecture=architecture,
        code=code,
        timestamp=time.time(),
    )
    model_log.add_model(experiment)
    _save_model_log(session, model_log)

    mining_state["step"] = 4
    mining_state["verify_result"] = "pass"
    mining_state["current_model_id"] = experiment.experiment_id
    session["mining_state"] = mining_state

    msg = (
        f"Step 4 模型验证通过（代码已记录）。\n"
        f"架构: {architecture}\n"
        f"假设: {hypothesis[:100]}\n\n"
        f"下一步: 调用 step5_backtest 执行回测评估。"
    )
    return msg, {"quant_session": session}


def _action_step5_backtest(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    config = QuantConfig(**session["config"])
    dag = _load_dag(session)
    mining_state = session.get("mining_state", {})
    action = mining_state.get("action", "factor")

    from deerflow.tools.builtins.quant.qlib_pipeline import run_full_pipeline, build_pipeline_with_new_factors

    try:
        active_custom = [n for n in dag.active_nodes() if n.code_verified and n.code and n.parent_id is not None]
        if active_custom:
            metrics = build_pipeline_with_new_factors(config, active_custom)
        else:
            metrics = run_full_pipeline(config)
        metrics_dict = metrics.model_dump()

        if action == "factor":
            node_id = mining_state.get("verify_node_id")
            if node_id:
                node = dag.get_node(node_id)
                if node:
                    node.arr = metrics.arr or 0.0
                    node.ir = metrics.ir or 0.0
                    node.mdd = metrics.mdd or 0.0
                    node.sharpe = metrics.sharpe or 0.0
                    node.calmar = metrics.calmar or 0.0
                    _save_dag(session, dag)

        mining_state["step"] = 5
        mining_state["current_metrics"] = metrics_dict
        session["mining_state"] = mining_state

        session["experiment_history"].append({
            "round": len(session.get("experiment_history", [])) + 1,
            "action": f"mining_{action}",
            "metrics": metrics_dict,
            "timestamp": datetime.now().isoformat(),
        })

        if session.get("sota_metrics") is None:
            session["sota_metrics"] = metrics_dict
        else:
            sota_sharpe = session["sota_metrics"].get("sharpe", 0)
            if metrics_dict.get("sharpe", 0) > sota_sharpe:
                session["sota_metrics"] = metrics_dict

        msg = (
            f"Step 5 回测完成。\n\n{metrics.to_display()}\n\n"
            f"下一步: 调用 step6_admit 执行双条件准入判断。"
        )
        return msg, {"quant_session": session, "quant_metrics": metrics_dict}

    except Exception as e:
        logger.exception("step5_backtest failed")
        mining_state["step"] = 5
        mining_state["current_metrics"] = None
        mining_state["backtest_failed"] = True
        session["mining_state"] = mining_state
        return f"Step 5 回测失败: {e}\n\n跳过准入，直接进入 step7_update。", {"quant_session": session}


def _action_step6_admit(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    dag = _load_dag(session)
    model_log = _load_model_log(session)
    checker = _load_admission_checker(session)
    mining_state = session.get("mining_state", {})
    action = mining_state.get("action", "factor")

    if action == "factor":
        node_id = mining_state.get("verify_node_id")
        node = dag.get_node(node_id) if node_id else None

        if node is None or node.status == "rejected":
            mining_state["step"] = 6
            mining_state["admission_result"] = {"decision": False, "type": "reject", "reason": "节点不存在或已被拒绝"}
            session["mining_state"] = mining_state
            msg = (
                f"Step 6 准入判断: 拒绝（节点不存在或已被拒绝）\n\n"
                f"下一步: 调用 step7_update 更新知识。"
            )
            return msg, {"quant_session": session}

        parent = dag.get_node(node.parent_id) if node.parent_id else None
        pool = dag.active_nodes()
        result = checker.check(node, parent, pool)

        if result["decision"]:
            can_enter = checker.can_enter_pool(node, pool)
            if can_enter or len(pool) < checker.capacity:
                node.decision = True
                node.status = "active"
                if len(pool) >= checker.capacity:
                    worst = min(pool, key=lambda f: abs(f.ic))
                    worst.status = "evicted"
            else:
                result = {"decision": False, "type": "reject", "reason": "池已满且MutIC不满足入池条件"}
                node.decision = False
                node.status = "rejected"
        else:
            node.decision = False
            node.status = "rejected"

        _save_dag(session, dag)

        mining_state["step"] = 6
        mining_state["admission_result"] = result
        session["mining_state"] = mining_state

        decision_str = "✅ 准入" if result["decision"] else "❌ 拒绝"
        msg = (
            f"Step 6 双条件准入判断: {decision_str}\n"
            f"类型: {result['type']}\n"
            f"原因: {result['reason']}\n\n"
            f"下一步: 调用 step7_update 更新DAG知识。"
        )
        return msg, {"quant_session": session}

    model_id = mining_state.get("current_model_id")
    experiment = None
    for exp in model_log.experiments:
        if exp.experiment_id == model_id:
            experiment = exp
            break

    if experiment is None:
        mining_state["step"] = 6
        mining_state["admission_result"] = {"decision": False, "type": "reject", "reason": "模型实验不存在"}
        session["mining_state"] = mining_state
        return "Step 6 模型准入: 拒绝（模型实验不存在）\n\n下一步: 调用 step7_update。", {"quant_session": session}

    current_metrics = mining_state.get("current_metrics", {})
    sota_model = model_log.get_sota_model()
    if sota_model and sota_model.metrics:
        sota_sharpe = sota_model.metrics.get("sharpe", 0)
        current_sharpe = current_metrics.get("sharpe", 0)
        decision = current_sharpe > sota_sharpe
    else:
        decision = True

    experiment.decision = decision
    if decision:
        experiment.metrics = current_metrics
    _save_model_log(session, model_log)

    result = {"decision": decision, "type": "exploit" if decision else "reject",
              "reason": "模型Sharpe优于SOTA" if decision else "模型Sharpe未超过SOTA"}
    mining_state["step"] = 6
    mining_state["admission_result"] = result
    session["mining_state"] = mining_state

    decision_str = "✅ 准入" if decision else "❌ 拒绝"
    msg = (
        f"Step 6 模型准入判断: {decision_str}\n"
        f"原因: {result['reason']}\n\n"
        f"下一步: 调用 step7_update 更新知识。"
    )
    return msg, {"quant_session": session}


def _action_step7_update(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    dag = _load_dag(session)
    bandit = _load_bandit(session)
    model_log = _load_model_log(session)
    mining_state = session.get("mining_state", {})
    action = mining_state.get("action", "factor")
    admission_result = mining_state.get("admission_result", {})

    current_metrics = mining_state.get("current_metrics")
    backtest_failed = mining_state.get("backtest_failed", False)
    if current_metrics is None or backtest_failed:
        if backtest_failed:
            reward = -1.0
            state_vector = build_state_vector(None)
        else:
            sota = dag.get_sota()
            if sota:
                current_metrics = {
                    "ic": sota.ic or 0.0, "icir": sota.icir or 0.0,
                    "rank_ic": sota.rank_ic or 0.0, "rank_icir": sota.rank_icir or 0.0,
                    "arr": sota.arr or 0.0, "ir": sota.ir or 0.0,
                    "mdd": sota.mdd or 0.0, "sharpe": sota.sharpe or 0.0,
                    "calmar": sota.calmar or 0.0,
                }
                sota_metrics = session.get("sota_metrics")
                if sota_metrics:
                    current_metrics.update(sota_metrics)
            state_vector = build_state_vector(current_metrics)
            reward = compute_reward(state_vector)
    else:
        state_vector = build_state_vector(current_metrics)
        reward = compute_reward(state_vector)
    bandit.record(action, state_vector, reward)
    _save_bandit(session, bandit)

    if action == "factor":
        node_id = mining_state.get("verify_node_id")
        node = dag.get_node(node_id) if node_id else None
        if node:
            node.timestamp = time.time()
            node.action = action
            node.feedback = kwargs.get("feedback") or admission_result.get("reason", "")
            node.hypothesis = kwargs.get("hypothesis", "")
            sota_model = model_log.get_sota_model()
            if sota_model:
                node.model_context = sota_model.experiment_id
            _save_dag(session, dag)
    else:
        model_id = mining_state.get("current_model_id")
        for exp in model_log.experiments:
            if exp.experiment_id == model_id:
                exp.feedback = kwargs.get("feedback") or admission_result.get("reason", "")
                break
        _save_model_log(session, model_log)

    current_round = mining_state.get("current_round", 1)
    max_rounds = mining_state.get("max_rounds", DEFAULT_MAX_ROUNDS)
    remaining = max_rounds - current_round

    mining_state["step"] = 7
    mining_state.pop("backtest_failed", None)
    session["mining_state"] = mining_state

    if remaining > 0:
        msg = (
            f"Step 7 知识更新完成。\n"
            f"Bandit后验已更新，reward={reward:.4f}\n"
            f"DAG节点数: {dag.size()}, 活跃: {len(dag.active_nodes())}\n"
            f"模型实验数: {len(model_log.experiments)}\n"
            f"剩余轮次: {remaining}\n\n"
            f"下一步: 调用 step1_bandit 开始第{current_round + 1}轮调度。"
        )
    else:
        sota = dag.get_sota()
        sota_model = model_log.get_sota_model()
        summary_lines = [
            f"=== MVP3 挖掘完成（{max_rounds}轮） ===",
            f"DAG总节点: {dag.size()}, 活跃: {len(dag.active_nodes())}",
            f"模型实验: {len(model_log.experiments)}",
        ]
        if sota:
            summary_lines.append(f"SOTA因子: {sota.formula or sota.description}")
            summary_lines.append(f"  IC={sota.ic:.4f}, ICIR={sota.icir:.4f}, Sharpe={sota.sharpe:.4f}, Calmar={sota.calmar:.4f}")
        if sota_model:
            summary_lines.append(f"SOTA模型: {sota_model.architecture}")
        summary_lines.append(f"Bandit历史: {len(bandit.history)}次决策")

        msg = "\n".join(summary_lines)

    return msg, {"quant_session": session}


def _action_get_mining_status(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    mining_state = session.get("mining_state", {})

    if not mining_state:
        return "尚未启动挖掘流程。请调用 step0_assemble 初始化。", {}

    dag = _load_dag(session)
    bandit = _load_bandit(session)
    model_log = _load_model_log(session)

    step = mining_state.get("step", 0)
    current_round = mining_state.get("current_round", 0)
    max_rounds = mining_state.get("max_rounds", DEFAULT_MAX_ROUNDS)
    action = mining_state.get("action", "N/A")

    lines = [
        f"=== 挖掘状态 ===",
        f"当前步骤: Step {step}",
        f"当前轮次: {current_round}/{max_rounds}",
        f"当前方向: {action}",
        f"DAG节点: {dag.size()} (活跃: {len(dag.active_nodes())})",
        f"模型实验: {len(model_log.experiments)}",
        f"Bandit决策历史: {len(bandit.history)}",
    ]

    admission = mining_state.get("admission_result")
    if admission:
        lines.append(f"最近准入结果: {'✅' if admission.get('decision') else '❌'} {admission.get('reason', '')}")

    return "\n".join(lines), {}


def _action_run_mining_loop(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    config = QuantConfig(**session["config"])
    dag = _load_dag(session)
    bandit = _load_bandit(session)
    model_log = _load_model_log(session)
    retriever = _load_retriever(session)
    checker = _load_admission_checker(session)

    max_rounds = kwargs.get("max_rounds", DEFAULT_MAX_ROUNDS)
    topic = kwargs.get("topic", "动量与反转")

    if dag.size() == 0:
        from deerflow.tools.builtins.quant.factor_baseline_library_tool import ALPHA20
        base_features = session.get("base_features", ALPHA20)
        for name, expr in base_features.items():
            node = DualRepFactorNode(
                node_id=FactorDAG.new_node_id(),
                formula=expr,
                description=name,
                topic="baseline",
                code_verified=True,
                decision=True,
                action="factor",
                timestamp=time.time(),
            )
            dag.insert(node)
        _save_dag(session, dag)

        try:
            from deerflow.tools.builtins.quant.qlib_pipeline import run_full_pipeline
            baseline_metrics = run_full_pipeline(config)
            baseline_dict = baseline_metrics.model_dump()
            session["sota_metrics"] = baseline_dict
            for node in dag.active_nodes():
                node.ic = baseline_metrics.ic or 0.0
                node.icir = baseline_metrics.icir or 0.0
                node.rank_ic = baseline_metrics.rank_ic or 0.0
                node.rank_icir = baseline_metrics.rank_icir or 0.0
                node.arr = baseline_metrics.arr or 0.0
                node.ir = baseline_metrics.ir or 0.0
                node.mdd = baseline_metrics.mdd or 0.0
                node.sharpe = baseline_metrics.sharpe or 0.0
                node.calmar = baseline_metrics.calmar or 0.0
            _save_dag(session, dag)
        except Exception as e:
            logger.warning(f"ALPHA20 cold start backtest failed: {e}")

    round_results = []
    consecutive_no_improve = 0
    best_sharpe = 0.0
    consecutive_errors = 0
    prev_metrics = None

    for round_idx in range(1, max_rounds + 1):
        try:
            if prev_metrics:
                state_vector = build_state_vector(prev_metrics)
            else:
                sota = dag.get_sota()
                if sota:
                    metrics_dict = {
                        "ic": sota.ic or 0.0, "icir": sota.icir or 0.0,
                        "rank_ic": sota.rank_ic or 0.0, "rank_icir": sota.rank_icir or 0.0,
                        "arr": sota.arr or 0.0, "ir": sota.ir or 0.0,
                        "mdd": sota.mdd or 0.0, "sharpe": sota.sharpe or 0.0,
                        "calmar": sota.calmar or 0.0,
                    }
                    sota_metrics = session.get("sota_metrics")
                    if sota_metrics:
                        metrics_dict.update(sota_metrics)
                    state_vector = build_state_vector(metrics_dict)

                    current_sharpe = metrics_dict.get("sharpe", 0.0) or 0.0
                    if current_sharpe > best_sharpe:
                        best_sharpe = current_sharpe
                        consecutive_no_improve = 0
                    else:
                        consecutive_no_improve += 1
                else:
                    state_vector = build_state_vector(None)
                    consecutive_no_improve += 1

            if consecutive_no_improve >= CONVERGENCE_PATIENCE:
                round_results.append({
                    "round": round_idx,
                    "info": f"收敛检测: 连续{CONVERGENCE_PATIENCE}轮无SOTA提升，提前终止",
                })
                break

            consecutive_errors = 0

            action = bandit.decide(state_vector)

            if action == "factor":
                retrieved_nodes = retriever.retrieve(dag, top_k=4)
                _save_dag(session, dag)

                parent_node = retrieved_nodes[0] if retrieved_nodes else None
                parent_id = parent_node.node_id if parent_node else None

                round_results.append({
                    "round": round_idx,
                    "action": "factor",
                    "bandit_decision": "factor",
                    "parent_id": parent_id,
                    "retrieved_count": len(retrieved_nodes),
                })
            else:
                round_results.append({
                    "round": round_idx,
                    "action": "model",
                    "bandit_decision": "model",
                })

        except Exception as e:
            logger.exception(f"Mining loop round {round_idx} failed: {e}")
            consecutive_errors += 1
            round_results.append({
                "round": round_idx,
                "error": str(e),
            })
            if consecutive_errors >= CONSECUTIVE_FAIL_LIMIT:
                round_results.append({
                    "round": round_idx,
                    "info": f"连续{CONSECUTIVE_FAIL_LIMIT}次错误，提前终止",
                })
                break

    _save_dag(session, dag)
    _save_bandit(session, bandit)
    _save_model_log(session, model_log)

    mining_state = {
        "step": 7,
        "max_rounds": max_rounds,
        "current_round": max_rounds,
        "action": None,
        "started_at": datetime.now().isoformat(),
    }
    session["mining_state"] = mining_state

    factor_rounds = sum(1 for r in round_results if r.get("action") == "factor")
    model_rounds = sum(1 for r in round_results if r.get("action") == "model")
    error_rounds = sum(1 for r in round_results if "error" in r)

    sota = dag.get_sota()
    sota_model = model_log.get_sota_model()

    summary_lines = [
        f"=== MVP3 自动挖掘完成（{max_rounds}轮） ===",
        f"因子方向: {factor_rounds}轮, 模型方向: {model_rounds}轮, 错误: {error_rounds}轮",
        f"DAG总节点: {dag.size()}, 活跃: {len(dag.active_nodes())}",
        f"模型实验: {len(model_log.experiments)}",
        f"Bandit历史: {len(bandit.history)}次决策",
    ]
    if sota:
        summary_lines.append(f"SOTA因子: {sota.formula or sota.description}")
        summary_lines.append(f"  IC={sota.ic:.4f}, ICIR={sota.icir:.4f}, Sharpe={sota.sharpe:.4f}, Calmar={sota.calmar:.4f}")
    if sota_model:
        summary_lines.append(f"SOTA模型: {sota_model.architecture}")

    msg = "\n".join(summary_lines)
    return msg, {"quant_session": session}


ACTION_HANDLERS = {
    "step0_assemble": _action_step0_assemble,
    "step1_bandit": _action_step1_bandit,
    "step2_retrieve": _action_step2_retrieve,
    "step3_generate": _action_step3_generate,
    "step4_verify": _action_step4_verify,
    "step5_backtest": _action_step5_backtest,
    "step6_admit": _action_step6_admit,
    "step7_update": _action_step7_update,
    "get_mining_status": _action_get_mining_status,
    "run_mining_loop": _action_run_mining_loop,
}


@tool("quant_mining", parse_docstring=True)
def quant_mining_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    action: str,
    max_rounds: int | None = None,
    topic: str | None = None,
    num_factors: int | None = None,
    top_k: int | None = None,
    name: str | None = None,
    formula: str | None = None,
    description: str | None = None,
    code: str | None = None,
    parent_id: str | None = None,
    hypothesis: str | None = None,
    architecture: str | None = None,
    feedback: str | None = None,
) -> Command:
    """MVP3量化因子-模型联合挖掘工具。通过action参数选择8步工作流的具体步骤。

    8步工作流：
    - step0_assemble: 组装上下文，冷启动Alpha20因子到DAG
    - step1_bandit: LinearThompson Bandit调度（factor/model方向选择）
    - step2_retrieve: BayesianFactorRetriever检索种子节点（因子方向）或模型上下文（模型方向）
    - step3_generate: LLM生成因子公式+代码（因子方向）或模型假设+代码（模型方向）
    - step4_verify: 代码验证（执行+格式校验+IC评估）
    - step5_backtest: Qlib回测→9维Metrics（IC/ICIR/RankIC/RankICIR/ARR/IR/MDD/Sharpe/Calmar）
    - step6_admit: DualAdmissionChecker双条件准入判断
    - step7_update: DAG知识更新+Bandit后验更新

    辅助action：
    - get_mining_status: 查看当前挖掘状态
    - run_mining_loop: 自动运行完整挖掘循环（Bandit调度+检索，不含LLM生成）

    典型使用流程：
    1. quant_mining(action="step0_assemble", max_rounds=20) — 初始化
    2. quant_mining(action="step1_bandit") — Bandit调度
    3. quant_mining(action="step2_retrieve") — 检索
    4. quant_mining(action="step3_generate", topic="动量与反转") — 生成
    5. quant_mining(action="step4_verify", code="...", formula="...") — 验证
    6. quant_mining(action="step5_backtest") — 回测
    7. quant_mining(action="step6_admit") — 准入
    8. quant_mining(action="step7_update") — 更新
    9. 回到step1继续下一轮

    Args:
        action: 操作名称
        max_rounds: 最大迭代轮数（step0_assemble/run_mining_loop使用，默认20）
        topic: 研究主题（step3_generate使用）
        num_factors: 生成因子数量（step3_generate使用，默认3）
        top_k: 检索返回数量（step2_retrieve使用，默认4）
        name: 因子名称（step4_verify使用）
        formula: Qlib表达式公式（step4_verify使用）
        description: 因子/模型描述（step4_verify使用）
        code: Python实现代码（step4_verify使用）
        parent_id: 父节点ID（step4_verify使用）
        hypothesis: 模型假设（step4_verify模型方向使用）
        architecture: 模型架构（step4_verify模型方向使用）
        feedback: 反馈文本（step7_update使用）
    """
    state = runtime.state or {}
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(ACTION_HANDLERS.keys())
        return Command(
            update={"messages": [ToolMessage(f"未知action '{action}'。可用action: {available}", tool_call_id=tool_call_id)]},
        )

    kwargs = {k: v for k, v in {
        "max_rounds": max_rounds,
        "topic": topic,
        "num_factors": num_factors,
        "top_k": top_k,
        "name": name,
        "formula": formula,
        "description": description,
        "code": code,
        "parent_id": parent_id,
        "hypothesis": hypothesis,
        "architecture": architecture,
        "feedback": feedback,
    }.items() if v is not None}

    session = _get_session(state)
    if session is not None:
        mining_state = session.get("mining_state", {})
        guard_msg = _check_step_guard(action, mining_state)
        if guard_msg:
            return Command(
                update={"messages": [ToolMessage(guard_msg, tool_call_id=tool_call_id)]},
            )

    try:
        msg, state_updates = handler(state, **kwargs)
        updates = {"messages": [ToolMessage(str(msg), tool_call_id=tool_call_id)]}
        updates.update(state_updates)
        return Command(update=updates)
    except Exception as e:
        logger.exception(f"quant_mining action '{action}' failed")
        import traceback
        tb = traceback.format_exc()
        recovery = _suggest_recovery(action, e)
        return Command(
            update={"messages": [ToolMessage(
                f"执行action '{action}'时出错: {e}\n\n"
                f"错误详情:\n```\n{tb[-500:]}\n```\n\n"
                f"恢复建议: {recovery}\n\n"
                f"⚠ 请勿重复调用同一action，按恢复建议操作。",
                tool_call_id=tool_call_id,
            )]},
        )


def _suggest_recovery(action: str, error: Exception) -> str:
    err_msg = str(error).lower()
    if "nonetype" in err_msg and ("//" in err_msg or "int" in err_msg):
        return "参数为空导致计算错误。请直接重试，或调用 get_mining_status 检查当前状态后再继续。"
    if action == "step2_retrieve":
        return "检索失败。请调用 get_mining_status 检查DAG状态，然后尝试 step3_generate 跳过检索直接生成。"
    if action == "step4_verify":
        return "验证失败。请检查代码是否有语法错误，修复后重新提交。"
    if action == "step5_backtest":
        return "回测失败。可能是Qlib数据问题。请调用 step7_update 跳过本轮，进入下一轮迭代。"
    if action == "step6_admit":
        return "准入判断失败。请调用 step7_update 跳过准入，直接更新知识。"
    return f"请调用 get_mining_status 检查状态，或跳到下一步继续。"
