import logging
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd

from deerflow.config.quant_config import QuantConfig
from deerflow.tools.builtins.quant.metrics import QuantMetrics

logger = logging.getLogger(__name__)

_qlib_initialized = False


def ensure_qlib_init(config: QuantConfig) -> None:
    global _qlib_initialized
    if _qlib_initialized:
        return
    import qlib
    qlib.init(provider_uri=config.provider_uri, region=config.region)
    _qlib_initialized = True
    logger.info(f"qlib initialized: provider_uri={config.provider_uri}, region={config.region}")


def calculate_baseline_factors(config: QuantConfig) -> pd.DataFrame:
    from qlib.data import D
    from deerflow.tools.builtins.quant.factor_baseline_library_tool import ALPHA20, ALPHA158

    ensure_qlib_init(config)

    if config.factor_set == "ALPHA20":
        feature_dict = ALPHA20
    elif config.factor_set == "ALPHA158":
        feature_dict = {k: v for k, v in ALPHA158.items() if k != "VWAP0"}
    else:
        feature_dict = ALPHA20

    expressions = list(feature_dict.values())
    instruments = D.instruments(config.market)
    df = D.features(instruments, expressions, start_time=config.train_start, end_time=config.test_end or None)
    df.columns = list(feature_dict.keys())
    return df


def _train_and_predict(config: QuantConfig, extra_factor_values: dict[str, pd.Series] | None = None) -> tuple:
    from qlib.utils import init_instance_by_config

    ensure_qlib_init(config)

    os.environ["QLIB_SKIP_SITECUSTOMIZE"] = "1"

    task_config = _build_task_config(config, extra_factor_values=extra_factor_values)
    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])

    logger.info("Training model...")
    model.fit(dataset)

    pred = model.predict(dataset)
    logger.info(f"Prediction shape: {pred.shape}")
    return model, dataset, pred


def _calc_ic_metrics(pred: pd.Series, dataset) -> QuantMetrics:
    from qlib.contrib.eva.alpha import calc_ic

    label = dataset.prepare("test", col_set="label")
    label = label.iloc[:, 0]

    pred_aligned = pred.reindex(label.index)
    pred_aligned = pred_aligned.dropna()
    label_aligned = label.reindex(pred_aligned.index)

    if len(pred_aligned) == 0 or len(label_aligned) == 0:
        logger.warning("No valid prediction/label data for IC calculation")
        return QuantMetrics()

    ic_series, ric_series = calc_ic(pred_aligned, label_aligned)

    ic = float(ic_series.mean()) if len(ic_series) > 0 else None
    icir = float(ic_series.mean() / ic_series.std()) if len(ic_series) > 1 and ic_series.std() > 0 else None
    rank_ic = float(ric_series.mean()) if len(ric_series) > 0 else None
    rank_icir = float(ric_series.mean() / ric_series.std()) if len(ric_series) > 1 and ric_series.std() > 0 else None

    return QuantMetrics(ic=ic, icir=icir, rank_ic=rank_ic, rank_icir=rank_icir)


def _calc_backtest_metrics(pred: pd.Series, config: QuantConfig) -> QuantMetrics | None:
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    from qlib.data import D

    benchmark_ok = False
    try:
        bench_df = D.features([config.benchmark], ["$close"], start_time=config.test_start, end_time=config.test_end or None)
        if bench_df is not None and len(bench_df) > 0:
            benchmark_ok = True
    except Exception:
        pass

    if not benchmark_ok:
        logger.warning(f"Benchmark {config.benchmark} not available, skipping backtest")
        return None

    try:
        cal = D.calendar()
        bt_end = config.test_end
        if bt_end is None or bt_end >= str(cal[-3]):
            bt_end = str(cal[-3].date()) if hasattr(cal[-3], 'date') else str(cal[-3])[:10]
            logger.info(f"Adjusted backtest end_time to {bt_end} (3 trading days before calendar end) to avoid future calendar issue")

        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": pred,
                "topk": config.topk,
                "n_drop": config.n_drop,
            },
        }

        exchange_config = {
            "freq": "day",
            "limit_threshold": config.limit_threshold,
            "deal_price": config.deal_price,
            "open_cost": config.open_cost,
            "close_cost": config.close_cost,
            "min_cost": config.min_cost,
        }

        from qlib.utils import init_instance_by_config
        strategy = init_instance_by_config(strategy_config)

        report_normal, positions_normal = backtest_daily(
            start_time=config.test_start,
            end_time=bt_end,
            account=config.account,
            benchmark=config.benchmark,
            exchange_kwargs=exchange_config,
            strategy=strategy,
        )

        if report_normal is None or len(report_normal) == 0:
            logger.warning("Backtest returned empty report")
            return None

        arr = None
        ir = None
        mdd = None
        sharpe = None
        calmar = None

        try:
            if "return" in report_normal.columns:
                daily_ret = report_normal["return"].dropna()
                if len(daily_ret) > 0:
                    arr = float(daily_ret.mean() * 252)
                    ir_val = daily_ret.mean() / daily_ret.std() * (252 ** 0.5) if daily_ret.std() > 0 else None
                    ir = float(ir_val) if ir_val is not None else None
                    cum_ret = (1 + daily_ret).cumprod()
                    running_max = cum_ret.cummax()
                    drawdown = (cum_ret - running_max) / running_max
                    mdd = float(drawdown.min())
                    sharpe_val = daily_ret.mean() / daily_ret.std() * (252 ** 0.5) if daily_ret.std() > 0 else None
                    sharpe = float(sharpe_val) if sharpe_val is not None else None
                    if arr is not None and mdd is not None and mdd != 0:
                        calmar = float(arr / abs(mdd))
        except Exception as e:
            logger.warning(f"Manual risk calculation failed: {e}")

        return QuantMetrics(
            arr=arr,
            ir=ir,
            mdd=mdd,
            sharpe=sharpe,
            calmar=calmar,
        )
    except Exception as e:
        logger.warning(f"Backtest failed: {e}", exc_info=True)
        return None


def run_full_pipeline(config: QuantConfig, extra_factor_values: dict[str, pd.Series] | None = None) -> QuantMetrics:
    model, dataset, pred = _train_and_predict(config, extra_factor_values=extra_factor_values)

    ic_metrics = _calc_ic_metrics(pred, dataset)
    logger.info(f"IC metrics: {ic_metrics}")

    bt_metrics = _calc_backtest_metrics(pred, config)

    if bt_metrics is not None:
        metrics = QuantMetrics(
            ic=ic_metrics.ic,
            icir=ic_metrics.icir,
            rank_ic=ic_metrics.rank_ic,
            rank_icir=ic_metrics.rank_icir,
            arr=bt_metrics.arr,
            ir=bt_metrics.ir,
            mdd=bt_metrics.mdd,
            sharpe=bt_metrics.sharpe,
            calmar=bt_metrics.calmar,
        )
    else:
        metrics = ic_metrics

    logger.info(f"Full pipeline completed. Metrics: {metrics}")
    return metrics


def run_ic_evaluation(config: QuantConfig) -> QuantMetrics:
    model, dataset, pred = _train_and_predict(config)
    metrics = _calc_ic_metrics(pred, dataset)
    logger.info(f"IC evaluation completed. Metrics: {metrics}")
    return metrics


def run_backtest(config: QuantConfig) -> QuantMetrics:
    model, dataset, pred = _train_and_predict(config)

    ic_metrics = _calc_ic_metrics(pred, dataset)
    bt_metrics = _calc_backtest_metrics(pred, config)

    if bt_metrics is not None:
        metrics = QuantMetrics(
            ic=ic_metrics.ic,
            icir=ic_metrics.icir,
            rank_ic=ic_metrics.rank_ic,
            rank_icir=ic_metrics.rank_icir,
            arr=bt_metrics.arr,
            ir=bt_metrics.ir,
            mdd=bt_metrics.mdd,
            sharpe=bt_metrics.sharpe,
            calmar=bt_metrics.calmar,
        )
    else:
        logger.warning("Backtest unavailable, returning IC metrics only")
        metrics = ic_metrics

    logger.info(f"Backtest completed. Metrics: {metrics}")
    return metrics


def _build_task_config(config: QuantConfig, extra_factor_values: dict[str, pd.Series] | None = None) -> dict:
    if extra_factor_values:
        handler_config = _build_handler_with_extra_factors(config, extra_factor_values)
    else:
        handler_config = {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": config.train_start,
                "end_time": config.test_end or None,
                "fit_start_time": config.train_start,
                "fit_end_time": config.train_end,
                "instruments": config.market,
            },
        }
    return {
        "model": {
            "class": config.model_class,
            "module_path": config.model_module,
            "kwargs": config.model_kwargs or {},
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler_config,
                "segments": {
                    "train": (config.train_start, config.train_end),
                    "valid": (config.valid_start, config.valid_end),
                    "test": (config.test_start, config.test_end or None),
                },
            },
        },
    }


def _build_handler_with_extra_factors(config: QuantConfig, extra_factor_values: dict[str, pd.Series]) -> dict:
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.processor import Processor

    class ExtraFactorProcessor(Processor):
        def __init__(self, extra_values: dict[str, pd.Series]):
            self._extra_values = extra_values

        def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
            for name, series in self._extra_values.items():
                if name not in df.columns:
                    aligned = series.reindex(df.index).fillna(0)
                    df[name] = aligned
            return df

    extra_processor = ExtraFactorProcessor(extra_factor_values)

    class Alpha158WithExtras(Alpha158):
        def _init_processor(self):
            super()._init_processor()
            if hasattr(self, 'infer_processors') and self.infer_processors is not None:
                self.infer_processors.append(extra_processor)
            else:
                self.infer_processors = [extra_processor]

    module_name = Alpha158WithExtras.__module__
    class_name = Alpha158WithExtras.__qualname__

    _EXTRA_PROCESSOR_REGISTRY[config] = extra_processor

    return {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
        "kwargs": {
            "start_time": config.train_start,
            "end_time": config.test_end or None,
            "fit_start_time": config.train_start,
            "fit_end_time": config.train_end,
            "instruments": config.market,
        },
    }


_EXTRA_PROCESSOR_REGISTRY: dict = {}


def build_pipeline_with_new_factors(config: QuantConfig, new_factor_nodes: list) -> QuantMetrics:
    ensure_qlib_init(config)
    from qlib.data import D
    from qlib.contrib.data.handler import Alpha158

    instruments = D.instruments(config.market)
    handler = Alpha158(
        start_time=config.train_start,
        end_time=config.test_end or None,
        fit_start_time=config.train_start,
        fit_end_time=config.train_end,
        instruments=instruments,
    )
    handler.setup_data()

    feature_df = handler.fetch(selector=None, level=None)
    if feature_df is None:
        logger.warning("Failed to fetch Alpha158 features, falling back to baseline")
        return run_full_pipeline(config)

    extra_cols = {}
    for node in new_factor_nodes:
        if not node.code or node.status != "active":
            continue
        try:
            from deerflow.tools.builtins.quant.factor_generate_tool import _execute_factor_code
            factor_df, exec_error = _execute_factor_code(node.code, config)
            if exec_error or factor_df is None or factor_df.empty:
                logger.warning(f"Factor {node.node_id} execution failed for pipeline injection: {exec_error}")
                continue
            factor_series = factor_df.iloc[:, 0]
            col_name = f"custom_{node.node_id}"
            extra_cols[col_name] = factor_series
        except Exception as e:
            logger.warning(f"Factor {node.node_id} injection failed: {e}")
            continue

    if not extra_cols:
        logger.info("No extra factors to inject, running baseline pipeline")
        return run_full_pipeline(config)

    for col_name, series in extra_cols.items():
        aligned = series.reindex(feature_df.index).fillna(0)
        feature_df[col_name] = aligned

    logger.info(f"Pipeline with {len(extra_cols)} extra factors, total features: {len(feature_df.columns)}")

    from qlib.data.dataset import DatasetH
    from qlib.utils import init_instance_by_config

    model_config = {
        "class": config.model_class,
        "module_path": config.model_module,
        "kwargs": config.model_kwargs or {},
    }
    model = init_instance_by_config(model_config)

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (config.train_start, config.train_end),
            "valid": (config.valid_start, config.valid_end),
            "test": (config.test_start, config.test_end or None),
        },
    )

    model.fit(dataset)
    pred = model.predict(dataset)

    ic_metrics = _calc_ic_metrics(pred, dataset)
    bt_metrics = _calc_backtest_metrics(pred, config)

    if bt_metrics is not None:
        metrics = QuantMetrics(
            ic=ic_metrics.ic,
            icir=ic_metrics.icir,
            rank_ic=ic_metrics.rank_ic,
            rank_icir=ic_metrics.rank_icir,
            arr=bt_metrics.arr,
            ir=bt_metrics.ir,
            mdd=bt_metrics.mdd,
            sharpe=bt_metrics.sharpe,
            calmar=bt_metrics.calmar,
        )
    else:
        metrics = ic_metrics

    logger.info(f"Pipeline with extra factors completed. Metrics: {metrics}")
    return metrics
