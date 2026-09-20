from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState
from deerflow.config.quant_config import QuantConfig
from deerflow.tools.builtins.quant.dag import FactorDAG, DualRepFactorNode
from deerflow.tools.builtins.quant.evaluators import run_all_evaluators
from deerflow.tools.builtins.quant.metrics import QuantMetrics
from deerflow.tools.builtins.quant.prompts import (
    FACTOR_CODE_FIX_SYSTEM,
    FACTOR_CODE_FIX_USER,
    FACTOR_EVAL_FEEDBACK_TEMPLATE,
    FACTOR_GENERATION_SYSTEM,
    FACTOR_GENERATION_USER,
)
from deerflow.tools.builtins.quant.qlib_pipeline import (
    ensure_qlib_init,
    run_full_pipeline,
)
from deerflow.tools.builtins.quant.scenario import QuantScenario

logger = logging.getLogger(__name__)

MAX_CODE_FIX_ROUNDS = 3
MAX_GENERATION_ROUNDS = 5


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


def _build_traces(dag: FactorDAG, node_id: str | None) -> str:
    if node_id is None:
        return "（无已有路径，这是第一个因子）"
    path = dag.path_to_root(node_id)
    if not path:
        return "（无已有路径）"
    lines = []
    for i, node in enumerate(path):
        ic_str = f"IC={node.ic:.4f}, ICIR={node.icir:.4f}" if node.ic != 0.0 else "未评估"
        lines.append(f"  [{i}] {node.formula or node.description or node.node_id} ({ic_str})")
    return "\n".join(lines)


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
    failed = [n for n in dag.all_nodes() if n.status == "failed"]
    if not failed:
        return "（无历史失败记录）"
    recent = sorted(failed, key=lambda n: n.timestamp, reverse=True)[:limit]
    lines = []
    for n in recent:
        lines.append(f"  - {n.formula or n.node_id}: {n.description or '未知错误'}")
    return "\n".join(lines)


def _search_node_to_optimize(dag: FactorDAG) -> DualRepFactorNode | None:
    active = dag.active_nodes()
    if not active:
        return None
    scored = []
    for node in active:
        if node.icir == 0.0:
            score = 0.0
        else:
            depth_penalty = 0.1 * node.depth
            score = abs(node.icir) - depth_penalty
        scored.append((score, node))
    scored.sort(key=lambda x: x[0])
    return scored[0][1] if scored else None


def _get_raw_ohlcv(config: QuantConfig) -> pd.DataFrame:
    from qlib.data import D
    ensure_qlib_init(config)
    instruments = D.instruments(config.market)
    fields = ["$open", "$close", "$high", "$low", "$volume", "$vwap"]
    df = D.features(instruments, fields, start_time=config.train_start, end_time=config.test_end or None)
    df.columns = [c.lstrip("$") for c in df.columns]
    df = df.sort_index()
    return df


def _execute_factor_code(code: str, config: QuantConfig) -> tuple[pd.DataFrame | None, str]:
    ensure_qlib_init(config)
    try:
        raw_df = _get_raw_ohlcv(config)
    except Exception as e:
        return None, f"获取原始OHLCV数据失败: {e}"

    stock_data_path = tempfile.mktemp(suffix=".parquet")
    code_path = tempfile.mktemp(suffix=".py")
    result_path = tempfile.mktemp(suffix=".parquet")

    try:
        raw_df.to_parquet(stock_data_path)

        wrapped_code = f"""
import pandas as pd
import numpy as np
import sys
import json
import traceback

_stock_data_path = {repr(stock_data_path)}
_result_path = {repr(result_path)}

{code}

if __name__ == "__main__":
    try:
        df = pd.read_parquet(_stock_data_path)
        result = calculate_factor(df)
        if isinstance(result, pd.Series):
            result = result.to_frame("factor_value")
        result.to_parquet(_result_path)
    except Exception as e:
        print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()}}), file=sys.stderr)
        sys.exit(1)
"""
        Path(code_path).write_text(wrapped_code, encoding="utf-8")

        env = {
            **dict(__import__("os").environ),
            "OPENBLAS_NUM_THREADS": "2",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
        proc = subprocess.run(
            [sys.executable, code_path],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        if proc.returncode != 0:
            error_msg = proc.stderr.strip() or proc.stdout.strip()
            return None, f"代码执行失败 (exit={proc.returncode}): {error_msg[:500]}"

        if not Path(result_path).exists():
            return None, "代码执行完成但未生成结果文件"

        result_df = pd.read_parquet(result_path)
        return result_df, ""

    except subprocess.TimeoutExpired:
        return None, "代码执行超时（300秒）"
    except Exception as e:
        return None, f"执行异常: {e}"
    finally:
        for p in [stock_data_path, code_path, result_path]:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


def _evaluate_single_factor(
    factor_df: pd.DataFrame,
    config: QuantConfig,
) -> tuple[QuantMetrics | None, list, str]:
    try:
        raw_df = _get_raw_ohlcv(config)
    except Exception as e:
        return None, [], f"获取原始OHLCV数据失败: {e}"

    expected_rows = len(raw_df)
    reference_index = raw_df.index

    eval_results = run_all_evaluators(
        factor_df,
        expected_rows=expected_rows,
        reference_index=reference_index,
    )

    format_ok = all(r.passed or r.level == "warning" for r in eval_results)
    if not format_ok:
        feedback_lines = []
        for r in eval_results:
            status = "✅" if r.passed else ("⚠️" if r.level == "warning" else "❌")
            feedback_lines.append(f"  {status} {r.name}: {r.message}")
        return None, eval_results, "格式校验未通过:\n" + "\n".join(feedback_lines)

    try:
        next_ret = raw_df["close"].groupby(level="instrument").transform(
            lambda x: x.shift(-1) / x - 1
        )
        factor_values = factor_df.iloc[:, 0]

        common_idx = factor_values.dropna().index.intersection(next_ret.dropna().index)
        if len(common_idx) < 100:
            return None, eval_results, f"有效数据点不足: {len(common_idx)}"

        fv = factor_values.loc[common_idx]
        nr = next_ret.loc[common_idx]

        daily_ic = fv.groupby(level="datetime").apply(
            lambda x: x.corr(nr.loc[x.index], method="pearson")
        ).dropna()
        daily_rank_ic = fv.groupby(level="datetime").apply(
            lambda x: x.corr(nr.loc[x.index], method="spearman")
        ).dropna()

        ic = float(daily_ic.mean()) if len(daily_ic) > 0 else 0.0
        icir = float(daily_ic.mean() / daily_ic.std()) if len(daily_ic) > 1 and daily_ic.std() > 0 else 0.0
        rank_ic = float(daily_rank_ic.mean()) if len(daily_rank_ic) > 0 else 0.0
        rank_icir = float(daily_rank_ic.mean() / daily_rank_ic.std()) if len(daily_rank_ic) > 1 and daily_rank_ic.std() > 0 else 0.0

        metrics = QuantMetrics(
            ic=ic, icir=icir, rank_ic=rank_ic, rank_icir=rank_icir,
            arr=0.0, ir=0.0, mdd=0.0, sharpe=0.0,
        )
        return metrics, eval_results, ""
    except Exception as e:
        logger.exception("IC evaluation failed")
        return None, eval_results, f"IC评估失败: {e}"


def _build_eval_feedback(metrics: QuantMetrics | None, eval_results: list, error: str) -> str:
    if error and metrics is None:
        return error

    eval_map = {}
    for r in eval_results:
        eval_map[f"check_{r.name}"] = f"{'✅' if r.passed else '❌'} {r.message}"

    return FACTOR_EVAL_FEEDBACK_TEMPLATE.format(
        check_inf=eval_map.get("check_inf检测", "N/A"),
        check_single_column=eval_map.get("check单列输出", "N/A"),
        check_format_basic=eval_map.get("check格式检查", "N/A"),
        check_daily_frequency=eval_map.get("check日频校验", "N/A"),
        check_row_count=eval_map.get("check行数匹配", "N/A"),
        check_index_consistency=eval_map.get("check索引一致性", "N/A"),
        ic=metrics.ic if metrics else 0.0,
        icir=metrics.icir if metrics else 0.0,
        rank_ic=metrics.rank_ic if metrics else 0.0,
        rank_icir=metrics.rank_icir if metrics else 0.0,
        correlation_info="（暂无相关性分析）",
        decision="因子通过格式校验和IC评估" if metrics else "因子未通过评估",
    )


def _parse_llm_json_response(text: str) -> dict | list | None:
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace_start = text.find("{")
        bracket_start = text.find("[")
        if brace_start == -1 and bracket_start == -1:
            return None
        start = min(i for i in [brace_start, bracket_start] if i != -1)
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            return None


def _action_generate_factors(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    config = QuantConfig(**session["config"])
    dag = _load_dag(session)

    topic = kwargs.get("topic", "动量与反转")
    num_factors = kwargs.get("num_factors", 3)

    parent_node = _search_node_to_optimize(dag)
    parent_id = parent_node.node_id if parent_node else None

    traces = _build_traces(dag, parent_id)
    baseline_text = _build_baseline_factors_text(session)
    failure_text = _build_failure_history(dag)

    generation_prompt = FACTOR_GENERATION_USER.format(
        topic=topic,
        traces=traces,
        baseline_factors=baseline_text,
        failure_history=failure_text,
        num_factors=num_factors,
    )

    msg = (
        f"因子生成请求已准备。\n"
        f"研究主题: {topic}\n"
        f"生成数量: {num_factors}\n"
        f"优化节点: {parent_id or '根节点（首次生成）'}\n"
        f"DAG节点数: {dag.size()}\n\n"
        f"请使用LLM以下列Prompt生成因子：\n\n"
        f"=== System Prompt ===\n{FACTOR_GENERATION_SYSTEM}\n\n"
        f"=== User Prompt ===\n{generation_prompt}\n\n"
        f"生成后，请将每个因子的JSON结果通过 validate_and_add_factor action 逐个验证并添加到DAG中。"
    )

    session["pending_generation"] = {
        "topic": topic,
        "parent_id": parent_id,
        "system_prompt": FACTOR_GENERATION_SYSTEM,
        "user_prompt": generation_prompt,
    }

    return msg, {"quant_session": session}


def _action_validate_and_add_factor(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    config = QuantConfig(**session["config"])
    dag = _load_dag(session)

    factor_name = kwargs.get("name", "unnamed_factor")
    formula = kwargs.get("formula")
    description = kwargs.get("description", "")
    code = kwargs.get("code", "")
    topic = kwargs.get("topic", "")
    parent_id = kwargs.get("parent_id") or session.get("pending_generation", {}).get("parent_id")

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
            status="failed",
        )
        parent_node = dag.get_node(parent_id) if parent_id else None
        dag.insert(node, parent_node)
        _save_dag(session, dag)

        fix_prompt = FACTOR_CODE_FIX_USER.format(
            factor_name=factor_name,
            formula=formula or "N/A",
            description=description,
            original_code=code,
            error_message=exec_error,
            eval_feedback="代码执行失败，需要修复。",
        )

        session["pending_fix"] = {
            "node_id": node.node_id,
            "fix_round": 1,
            "system_prompt": FACTOR_CODE_FIX_SYSTEM,
            "user_prompt": fix_prompt,
        }

        msg = (
            f"因子 '{factor_name}' 执行失败。\n\n"
            f"错误信息: {exec_error[:300]}\n\n"
            f"已将失败记录添加到DAG。请使用LLM以下列Prompt修复代码：\n\n"
            f"=== System Prompt ===\n{FACTOR_CODE_FIX_SYSTEM}\n\n"
            f"=== User Prompt ===\n{fix_prompt}\n\n"
            f"修复后请通过 fix_factor_code action 提交修复后的代码。"
        )
        return msg, {"quant_session": session}

    metrics, eval_results, eval_error = _evaluate_single_factor(factor_df, config)
    feedback = _build_eval_feedback(metrics, eval_results, eval_error)

    if metrics is not None:
        node = DualRepFactorNode(
            node_id=FactorDAG.new_node_id(),
            formula=formula,
            code=code,
            topic=topic,
            description=description,
            ic=metrics.ic,
            icir=metrics.icir,
            parent_id=parent_id,
            status="active",
        )
        parent_node = dag.get_node(parent_id) if parent_id else None
        dag.insert(node, parent_node)
        _save_dag(session, dag)

        session["experiment_history"].append({
            "round": len(session["experiment_history"]) + 1,
            "action": "generate_factor",
            "factor_name": factor_name,
            "formula": formula,
            "metrics": metrics.model_dump(),
            "timestamp": datetime.now().isoformat(),
        })

        msg = (
            f"因子 '{factor_name}' 验证通过并已添加到DAG！\n\n"
            f"公式: {formula or 'N/A'}\n"
            f"IC: {metrics.ic:.4f}, ICIR: {metrics.icir:.4f}\n"
            f"Rank IC: {metrics.rank_ic:.4f}, Rank ICIR: {metrics.rank_icir:.4f}\n\n"
            f"DAG节点数: {dag.size()}\n\n"
            f"详细评估反馈:\n{feedback}"
        )
        return msg, {"quant_session": session, "quant_metrics": metrics.model_dump()}

    node = DualRepFactorNode(
        node_id=FactorDAG.new_node_id(),
        formula=formula,
        code=code,
        topic=topic,
        description=f"评估失败: {eval_error[:200]}",
        ic=0.0,
        icir=0.0,
        parent_id=parent_id,
        status="failed",
    )
    parent_node = dag.get_node(parent_id) if parent_id else None
    dag.insert(node, parent_node)
    _save_dag(session, dag)

    fix_prompt = FACTOR_CODE_FIX_USER.format(
        factor_name=factor_name,
        formula=formula or "N/A",
        description=description,
        original_code=code,
        error_message=eval_error,
        eval_feedback=feedback,
    )

    session["pending_fix"] = {
        "node_id": node.node_id,
        "fix_round": 1,
        "system_prompt": FACTOR_CODE_FIX_SYSTEM,
        "user_prompt": fix_prompt,
    }

    msg = (
        f"因子 '{factor_name}' 评估未通过。\n\n"
        f"评估反馈:\n{feedback}\n\n"
        f"已将失败记录添加到DAG。请使用LLM以下列Prompt修复代码：\n\n"
        f"=== System Prompt ===\n{FACTOR_CODE_FIX_SYSTEM}\n\n"
        f"=== User Prompt ===\n{fix_prompt}\n\n"
        f"修复后请通过 fix_factor_code action 提交修复后的代码。"
    )
    return msg, {"quant_session": session}


def _action_fix_factor_code(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    config = QuantConfig(**session["config"])
    dag = _load_dag(session)

    pending_fix = session.get("pending_fix")
    if not pending_fix:
        return "错误：没有待修复的因子。请先通过 validate_and_add_factor 提交因子。", {}

    fixed_code = kwargs.get("fixed_code", "")
    fix_explanation = kwargs.get("fix_explanation", "")
    node_id = pending_fix["node_id"]
    fix_round = pending_fix.get("fix_round", 1)

    if not fixed_code:
        return "错误：必须提供修复后的代码（fixed_code参数）。", {}

    node = dag.get_node(node_id)
    if node is None:
        return f"错误：DAG中找不到节点 {node_id}。", {}

    factor_df, exec_error = _execute_factor_code(fixed_code, config)

    if exec_error:
        if fix_round >= MAX_CODE_FIX_ROUNDS:
            node.status = "failed"
            node.description = f"修复{fix_round}轮后仍失败: {exec_error[:200]}"
            _save_dag(session, dag)
            session.pop("pending_fix", None)
            return f"因子修复已达到最大轮次({MAX_CODE_FIX_ROUNDS})，放弃修复。错误: {exec_error[:300]}", {"quant_session": session}

        new_fix_prompt = FACTOR_CODE_FIX_USER.format(
            factor_name=node.formula or node.node_id,
            formula=node.formula or "N/A",
            description=node.description or "",
            original_code=fixed_code,
            error_message=exec_error,
            eval_feedback=f"第{fix_round}轮修复仍失败。",
        )

        session["pending_fix"] = {
            "node_id": node_id,
            "fix_round": fix_round + 1,
            "system_prompt": FACTOR_CODE_FIX_SYSTEM,
            "user_prompt": new_fix_prompt,
        }

        msg = (
            f"第{fix_round}轮修复失败。\n\n"
            f"错误: {exec_error[:300]}\n\n"
            f"请继续修复，或放弃此因子。剩余修复轮次: {MAX_CODE_FIX_ROUNDS - fix_round - 1}"
        )
        return msg, {"quant_session": session}

    metrics, eval_results, eval_error = _evaluate_single_factor(factor_df, config)
    feedback = _build_eval_feedback(metrics, eval_results, eval_error)

    if metrics is not None:
        node.code = fixed_code
        node.ic = metrics.ic
        node.icir = metrics.icir
        node.status = "active"
        node.description = fix_explanation or node.description
        _save_dag(session, dag)
        session.pop("pending_fix", None)

        session["experiment_history"].append({
            "round": len(session["experiment_history"]) + 1,
            "action": "fix_factor",
            "factor_name": node.formula or node.node_id,
            "fix_round": fix_round,
            "metrics": metrics.model_dump(),
            "timestamp": datetime.now().isoformat(),
        })

        msg = (
            f"因子修复成功！（第{fix_round}轮）\n\n"
            f"公式: {node.formula or 'N/A'}\n"
            f"IC: {metrics.ic:.4f}, ICIR: {metrics.icir:.4f}\n\n"
            f"详细评估反馈:\n{feedback}"
        )
        return msg, {"quant_session": session, "quant_metrics": metrics.model_dump()}

    if fix_round >= MAX_CODE_FIX_ROUNDS:
        node.status = "failed"
        node.description = f"修复{fix_round}轮后评估仍失败"
        _save_dag(session, dag)
        session.pop("pending_fix", None)
        return f"因子修复已达到最大轮次({MAX_CODE_FIX_ROUNDS})，评估仍不通过。", {"quant_session": session}

    new_fix_prompt = FACTOR_CODE_FIX_USER.format(
        factor_name=node.formula or node.node_id,
        formula=node.formula or "N/A",
        description=node.description or "",
        original_code=fixed_code,
        error_message=eval_error,
        eval_feedback=feedback,
    )

    session["pending_fix"] = {
        "node_id": node_id,
        "fix_round": fix_round + 1,
        "system_prompt": FACTOR_CODE_FIX_SYSTEM,
        "user_prompt": new_fix_prompt,
    }

    msg = (
        f"第{fix_round}轮修复后评估未通过。\n\n"
        f"评估反馈:\n{feedback}\n\n"
        f"剩余修复轮次: {MAX_CODE_FIX_ROUNDS - fix_round - 1}"
    )
    return msg, {"quant_session": session}


def _action_get_dag(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    dag = _load_dag(session)

    if dag.size() == 0:
        return "DAG为空，尚未生成任何因子。", {}

    lines = [f"因子知识图谱（共{dag.size()}个节点）：\n"]
    for node in dag.all_nodes():
        status_icon = "✅" if node.status == "active" else "❌"
        ic_str = f"IC={node.ic:.4f}, ICIR={node.icir:.4f}" if node.ic != 0.0 else "未评估"
        parent_str = f"← {node.parent_id[:8]}" if node.parent_id else "(根)"
        lines.append(f"  {status_icon} {node.node_id[:8]} | {node.formula or node.description or 'N/A'} | {ic_str} | {parent_str}")

    active = len(dag.active_nodes())
    failed = dag.size() - active
    lines.append(f"\n活跃: {active}, 失败: {failed}")

    return "\n".join(lines), {}


def _action_evolve_factors(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    dag = _load_dag(session)

    topic = kwargs.get("topic", "动量与反转")
    num_rounds = kwargs.get("num_rounds", 3)

    active = dag.active_nodes()
    if not active:
        return (
            "DAG中没有活跃因子节点。请先通过 generate_factors 生成初始因子，"
            "或通过 validate_and_add_factor 添加因子到DAG。",
            {},
        )

    parent_node = _search_node_to_optimize(dag)
    if parent_node is None:
        return "DAG中没有可优化的节点。", {}

    parent_id = parent_node.node_id
    traces = _build_traces(dag, parent_id)
    baseline_text = _build_baseline_factors_text(session)
    failure_text = _build_failure_history(dag)

    generation_prompt = FACTOR_GENERATION_USER.format(
        topic=topic,
        traces=traces,
        baseline_factors=baseline_text,
        failure_history=failure_text,
        num_factors=1,
    )

    msg = (
        f"因子进化请求已准备。\n"
        f"优化目标节点: {parent_id} (IC={parent_node.ic:.4f}, ICIR={parent_node.icir:.4f})\n"
        f"进化轮次: {num_rounds}\n"
        f"DAG活跃节点数: {len(active)}\n\n"
        f"请使用LLM以下列Prompt生成进化因子：\n\n"
        f"=== System Prompt ===\n{FACTOR_GENERATION_SYSTEM}\n\n"
        f"=== User Prompt ===\n{generation_prompt}\n\n"
        f"生成后，请将因子的JSON结果通过 validate_and_add_factor action 验证并添加到DAG中，"
        f"parent_id 参数设为 '{parent_id}'。"
    )

    session["pending_generation"] = {
        "topic": topic,
        "parent_id": parent_id,
        "system_prompt": FACTOR_GENERATION_SYSTEM,
        "user_prompt": generation_prompt,
    }

    return msg, {"quant_session": session}


def _action_pipeline_with_new_factors(state: ThreadState, **kwargs) -> Command:
    session = _ensure_session(state)
    config = QuantConfig(**session["config"])
    dag = _load_dag(session)

    active_nodes = dag.active_nodes()
    if not active_nodes:
        return "DAG中没有活跃因子。请先生成并验证因子后再运行管道。", {}

    try:
        metrics = run_full_pipeline(config)
        metrics_dict = metrics.model_dump()

        session["experiment_history"].append({
            "round": len(session["experiment_history"]) + 1,
            "action": "pipeline_with_new_factors",
            "factor_count": len(active_nodes),
            "metrics": metrics_dict,
            "timestamp": datetime.now().isoformat(),
        })

        if session.get("sota_metrics") is None:
            session["sota_metrics"] = metrics_dict
        else:
            sota_sharpe = session["sota_metrics"].get("sharpe", 0)
            if metrics_dict.get("sharpe", 0) > sota_sharpe:
                session["sota_metrics"] = metrics_dict

        msg = f"包含{len(active_nodes)}个新因子的完整管道执行完成。\n\n{metrics.to_display()}"
        return msg, {"quant_session": session, "quant_metrics": metrics_dict}
    except Exception as e:
        logger.exception("pipeline_with_new_factors failed")
        return f"管道执行失败: {e}", {}


ACTION_HANDLERS = {
    "generate_factors": _action_generate_factors,
    "validate_and_add_factor": _action_validate_and_add_factor,
    "fix_factor_code": _action_fix_factor_code,
    "get_dag": _action_get_dag,
    "evolve_factors": _action_evolve_factors,
    "pipeline_with_new_factors": _action_pipeline_with_new_factors,
}


@tool("factor_generate", parse_docstring=True)
def factor_generate_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    action: str,
    topic: str | None = None,
    num_factors: int | None = None,
    name: str | None = None,
    formula: str | None = None,
    description: str | None = None,
    code: str | None = None,
    parent_id: str | None = None,
    fixed_code: str | None = None,
    fix_explanation: str | None = None,
    num_rounds: int | None = None,
) -> Command:
    """因子生成与进化工具。通过action参数选择具体操作。

    可用的action：
    - generate_factors: 准备因子生成Prompt，基于主题和DAG路径生成新因子
    - validate_and_add_factor: 验证因子代码并添加到DAG（执行代码→格式校验→IC评估）
    - fix_factor_code: 提交修复后的因子代码（最多3轮修复）
    - get_dag: 查看当前因子知识图谱状态
    - evolve_factors: 基于DAG中最弱节点进化新因子
    - pipeline_with_new_factors: 用DAG中所有活跃因子运行完整管道

    典型使用流程：
    1. factor_generate(action="generate_factors", topic="动量与反转") — 生成因子
    2. factor_generate(action="validate_and_add_factor", name="...", code="...") — 验证并添加
    3. factor_generate(action="get_dag") — 查看DAG状态
    4. factor_generate(action="evolve_factors") — 进化优化
    5. factor_generate(action="pipeline_with_new_factors") — 运行完整管道

    Args:
        action: 操作名称
        topic: 研究主题（generate_factors/evolve_factors使用）
        num_factors: 生成因子数量（默认3）
        name: 因子名称（validate_and_add_factor使用）
        formula: Qlib表达式公式（validate_and_add_factor使用）
        description: 因子描述（validate_and_add_factor使用）
        code: Python实现代码（validate_and_add_factor使用）
        parent_id: 父节点ID（validate_and_add_factor使用）
        fixed_code: 修复后的代码（fix_factor_code使用）
        fix_explanation: 修复说明（fix_factor_code使用）
        num_rounds: 进化轮次（evolve_factors使用，默认3）
    """
    state = runtime.state or {}
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(ACTION_HANDLERS.keys())
        return Command(
            update={"messages": [ToolMessage(f"未知action '{action}'。可用action: {available}", tool_call_id=tool_call_id)]},
        )

    kwargs = {
        "topic": topic,
        "num_factors": num_factors,
        "name": name,
        "formula": formula,
        "description": description,
        "code": code,
        "parent_id": parent_id,
        "fixed_code": fixed_code,
        "fix_explanation": fix_explanation,
        "num_rounds": num_rounds,
    }

    try:
        msg, state_updates = handler(state, **kwargs)
        updates = {"messages": [ToolMessage(str(msg), tool_call_id=tool_call_id)]}
        updates.update(state_updates)
        return Command(update=updates)
    except Exception as e:
        logger.exception(f"factor_generate action '{action}' failed")
        return Command(
            update={"messages": [ToolMessage(f"执行action '{action}'时出错: {e}", tool_call_id=tool_call_id)]},
        )
