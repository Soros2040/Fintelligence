import json
import logging
from datetime import datetime
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState
from deerflow.config.quant_config import QuantConfig
from deerflow.tools.builtins.quant.evaluators import run_all_evaluators
from deerflow.tools.builtins.quant.metrics import QuantMetrics
from deerflow.tools.builtins.quant.qlib_pipeline import (
    calculate_baseline_factors,
    run_backtest,
    run_full_pipeline,
    run_ic_evaluation,
)
from deerflow.tools.builtins.quant.scenario import QuantScenario

logger = logging.getLogger(__name__)

QUANT_WORKSPACE = "/mnt/user-data/workspace/quant"


def _get_quant_session(state: ThreadState) -> dict | None:
    return state.get("quant_session")


def _build_config_from_overrides(overrides: dict) -> QuantConfig:
    base = QuantConfig()
    valid_fields = set(base.model_fields.keys())
    filtered = {k: v for k, v in overrides.items() if k in valid_fields and v is not None}
    return base.model_copy(update=filtered)


def _action_init_session(state: ThreadState, **kwargs) -> Command:
    overrides = {k: v for k, v in kwargs.items() if v is not None}
    config = _build_config_from_overrides(overrides)
    scenario = QuantScenario(config)

    from deerflow.tools.builtins.quant.factor_baseline_library_tool import ALPHA20, ALPHA158
    if config.factor_set == "ALPHA20":
        base_features = ALPHA20
    elif config.factor_set == "ALPHA158":
        base_features = {k: v for k, v in ALPHA158.items() if k != "VWAP0"}
    else:
        base_features = ALPHA20

    quant_session = {
        "scenario": scenario.get_scenario_all_desc(),
        "config": config.model_dump(),
        "base_features": base_features,
        "base_feature_codes": {},
        "experiment_history": [],
        "sota_metrics": None,
        "sota_features": None,
        "dag_state": None,
        "factor_pool": None,
    }

    msg = (
        f"量化研究上下文已初始化。\n"
        f"股票池: {config.market}\n"
        f"因子集: {config.factor_set} ({len(base_features)}个因子)\n"
        f"训练期: {config.train_start} ~ {config.train_end}\n"
        f"验证期: {config.valid_start} ~ {config.valid_end}\n"
        f"测试期: {config.test_start} ~ {config.test_end or '最新'}\n"
        f"模型: {config.model_class}\n"
        f"下一步: 调用 full_pipeline 执行完整管道，或调用 calculate_factors 计算基准因子。"
    )
    return msg, {"quant_session": quant_session}


def _action_calculate_factors(state: ThreadState, **kwargs) -> Command:
    session = _get_quant_session(state)
    if session is None:
        return "错误：请先调用 init_session 初始化研究上下文。", {}

    config = QuantConfig(**session["config"])
    try:
        df = calculate_baseline_factors(config)
        rows, cols = df.shape
        msg = (
            f"基准因子矩阵计算完成。\n"
            f"因子数量: {cols}\n"
            f"数据行数: {rows}\n"
            f"时间范围: {df.index.get_level_values('datetime').min().strftime('%Y-%m-%d')} ~ "
            f"{df.index.get_level_values('datetime').max().strftime('%Y-%m-%d')}\n"
            f"股票数量: {len(df.index.get_level_values('instrument').unique())}\n"
            f"缺失值比例: {df.isnull().mean().mean():.4f}"
        )
        return msg, {}
    except Exception as e:
        logger.exception("calculate_factors failed")
        return f"计算基准因子失败: {e}", {}


def _action_validate_factor(state: ThreadState, **kwargs) -> Command:
    session = _get_quant_session(state)
    if session is None:
        return "错误：请先调用 init_session 初始化研究上下文。", {}

    config = QuantConfig(**session["config"])
    try:
        df = calculate_baseline_factors(config)
        results = run_all_evaluators(df)
        lines = ["因子验证结果："]
        for r in results:
            status = "✅" if r.passed else ("⚠️" if r.level == "warning" else "❌")
            lines.append(f"  {status} {r.name}: {r.message}")
        passed_count = sum(1 for r in results if r.passed)
        lines.append(f"\n通过: {passed_count}/{len(results)}")
        msg = "\n".join(lines)
        return msg, {}
    except Exception as e:
        logger.exception("validate_factor failed")
        return f"因子验证失败: {e}", {}


def _action_evaluate_ic(state: ThreadState, **kwargs) -> Command:
    session = _get_quant_session(state)
    if session is None:
        return "错误：请先调用 init_session 初始化研究上下文。", {}

    config = QuantConfig(**session["config"])
    overrides = {k: v for k, v in kwargs.items() if v is not None}
    if overrides:
        config = _build_config_from_overrides({**session["config"], **overrides})

    try:
        metrics = run_ic_evaluation(config)
        msg = f"IC评估完成。\n\n{metrics.to_display()}"
        metrics_dict = metrics.model_dump()
        return msg, {"quant_metrics": metrics_dict}
    except Exception as e:
        logger.exception("evaluate_ic failed")
        return f"IC评估失败: {e}", {}


def _action_backtest(state: ThreadState, **kwargs) -> Command:
    session = _get_quant_session(state)
    if session is None:
        return "错误：请先调用 init_session 初始化研究上下文。", {}

    config = QuantConfig(**session["config"])
    overrides = {k: v for k, v in kwargs.items() if v is not None}
    if overrides:
        config = _build_config_from_overrides({**session["config"], **overrides})

    try:
        metrics = run_backtest(config)
        msg = f"回测完成。\n\n{metrics.to_display()}"
        metrics_dict = metrics.model_dump()

        session["experiment_history"].append({
            "round": len(session["experiment_history"]) + 1,
            "action": "backtest",
            "metrics": metrics_dict,
            "timestamp": datetime.now().isoformat(),
        })

        if session["sota_metrics"] is None:
            session["sota_metrics"] = metrics_dict
            session["sota_features"] = session["base_features"]
        else:
            sota_sharpe = session["sota_metrics"].get("sharpe")
            new_sharpe = metrics_dict.get("sharpe")
            if new_sharpe is not None and (sota_sharpe is None or new_sharpe > sota_sharpe):
                session["sota_metrics"] = metrics_dict
                session["sota_features"] = session["base_features"]

        return msg, {"quant_session": session, "quant_metrics": metrics_dict}
    except Exception as e:
        logger.exception("backtest failed")
        return f"回测失败: {e}", {}


def _action_full_pipeline(state: ThreadState, **kwargs) -> Command:
    session = _get_quant_session(state)
    if session is None:
        return "错误：请先调用 init_session 初始化研究上下文。", {}

    config = QuantConfig(**session["config"])
    overrides = {k: v for k, v in kwargs.items() if v is not None}
    if overrides:
        config = _build_config_from_overrides({**session["config"], **overrides})

    try:
        metrics = run_full_pipeline(config)
        msg = f"完整管道执行完成。\n\n{metrics.to_display()}"
        metrics_dict = metrics.model_dump()

        session["experiment_history"].append({
            "round": len(session["experiment_history"]) + 1,
            "action": "full_pipeline",
            "metrics": metrics_dict,
            "timestamp": datetime.now().isoformat(),
        })

        if session["sota_metrics"] is None:
            session["sota_metrics"] = metrics_dict
            session["sota_features"] = session["base_features"]
        else:
            sota_sharpe = session["sota_metrics"].get("sharpe")
            new_sharpe = metrics_dict.get("sharpe")
            if new_sharpe is not None and (sota_sharpe is None or new_sharpe > sota_sharpe):
                session["sota_metrics"] = metrics_dict
                session["sota_features"] = session["base_features"]

        return msg, {"quant_session": session, "quant_metrics": metrics_dict}
    except Exception as e:
        logger.exception("full_pipeline failed")
        return f"完整管道执行失败: {e}", {}


def _action_get_sota(state: ThreadState, **kwargs) -> Command:
    session = _get_quant_session(state)
    if session is None:
        return "错误：请先调用 init_session 初始化研究上下文。", {}

    sota_metrics = session.get("sota_metrics")
    sota_features = session.get("sota_features")

    if sota_metrics is None:
        return "尚无SOTA记录。请先执行 full_pipeline 或 backtest。", {}

    metrics = QuantMetrics(**sota_metrics)
    feature_names = list(sota_features.keys()) if sota_features else []
    msg = (
        f"当前SOTA指标：\n\n{metrics.to_display()}\n\n"
        f"SOTA因子集（{len(feature_names)}个）：{', '.join(feature_names[:10])}"
        f"{'...' if len(feature_names) > 10 else ''}"
    )
    return msg, {}


def _action_generate_feedback(state: ThreadState, **kwargs) -> Command:
    return "generate_feedback action 预留给后续MVP实现。当前版本请根据Metrics结果自行分析。", {}


ACTION_HANDLERS = {
    "init_session": _action_init_session,
    "calculate_factors": _action_calculate_factors,
    "validate_factor": _action_validate_factor,
    "evaluate_ic": _action_evaluate_ic,
    "backtest": _action_backtest,
    "full_pipeline": _action_full_pipeline,
    "get_sota": _action_get_sota,
    "generate_feedback": _action_generate_feedback,
}


@tool("quant_analyze", parse_docstring=True)
def quant_analyze_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    action: str,
    provider_uri: str | None = None,
    market: str | None = None,
    benchmark: str | None = None,
    train_start: str | None = None,
    train_end: str | None = None,
    valid_start: str | None = None,
    valid_end: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
    model_class: str | None = None,
    model_module: str | None = None,
    topk: int | None = None,
    n_drop: int | None = None,
    account: float | None = None,
    limit_threshold: float | None = None,
    open_cost: float | None = None,
    close_cost: float | None = None,
    min_cost: float | None = None,
    deal_price: str | None = None,
    factor_set: str | None = None,
) -> Command:
    """量化因子分析工具。通过action参数选择具体操作。

    可用的action：
    - init_session: 初始化研究上下文，创建QuantScenario和quant_session（不调用qlib.init）
    - calculate_factors: 计算基准因子矩阵（首次调用时延迟执行qlib.init）
    - validate_factor: 调用6个评估器验证因子质量
    - evaluate_ic: 独立IC/ICIR评估
    - backtest: 独立回测执行
    - full_pipeline: 一键执行全管道（含模型训练+IC评估+回测），返回8维Metrics（IC/ICIR/Rank_IC/Rank_ICIR/ARR/IR/MDD/Sharpe）
    - get_sota: 查询SOTA因子和最佳指标
    - generate_feedback: 生成结构化反馈（预留给后续MVP）

    典型使用流程：
    1. quant_analyze(action="init_session") — 初始化
    2. quant_analyze(action="full_pipeline") — 执行完整管道

    Args:
        action: 要执行的操作名称（init_session/calculate_factors/validate_factor/evaluate_ic/backtest/full_pipeline/get_sota/generate_feedback）
        provider_uri: Qlib数据路径（覆盖默认值）
        market: 股票池（如csi300）
        benchmark: 基准指数代码
        train_start: 训练集开始日期
        train_end: 训练集结束日期
        valid_start: 验证集开始日期
        valid_end: 验证集结束日期
        test_start: 测试集开始日期
        test_end: 测试集结束日期
        model_class: 模型类名
        model_module: 模型模块路径
        topk: TopK策略的K值
        n_drop: 每次调仓丢弃的股票数
        account: 初始资金
        limit_threshold: 涨跌停阈值
        open_cost: 买入交易成本
        close_cost: 卖出交易成本
        min_cost: 最小交易成本
        deal_price: 成交价格字段
        factor_set: 因子集名称（ALPHA20/ALPHA158）
    """
    state = runtime.state or {}
    handler = ACTION_HANDLERS.get(action)
    if handler is None:
        available = ", ".join(ACTION_HANDLERS.keys())
        return Command(
            update={"messages": [ToolMessage(f"未知action '{action}'。可用action: {available}", tool_call_id=tool_call_id)]},
        )

    kwargs = {
        "provider_uri": provider_uri,
        "market": market,
        "benchmark": benchmark,
        "train_start": train_start,
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": test_end,
        "model_class": model_class,
        "model_module": model_module,
        "topk": topk,
        "n_drop": n_drop,
        "account": account,
        "limit_threshold": limit_threshold,
        "open_cost": open_cost,
        "close_cost": close_cost,
        "min_cost": min_cost,
        "deal_price": deal_price,
        "factor_set": factor_set,
    }

    try:
        msg, state_updates = handler(state, **kwargs)
        updates = {"messages": [ToolMessage(str(msg), tool_call_id=tool_call_id)]}
        updates.update(state_updates)
        return Command(update=updates)
    except Exception as e:
        logger.exception(f"quant_analyze action '{action}' failed")
        return Command(
            update={"messages": [ToolMessage(f"执行action '{action}'时出错: {e}", tool_call_id=tool_call_id)]},
        )
