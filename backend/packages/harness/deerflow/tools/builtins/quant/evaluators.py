import numpy as np
import pandas as pd
from pydantic import BaseModel


class EvaluatorResult(BaseModel):
    name: str
    passed: bool
    level: str
    message: str


def check_inf(df: pd.DataFrame) -> EvaluatorResult:
    inf_count = int(np.isinf(df.values).sum())
    if inf_count == 0:
        return EvaluatorResult(name="inf检测", passed=True, level="ok", message=f"未发现inf值")
    return EvaluatorResult(name="inf检测", passed=False, level="error", message=f"发现{inf_count}个inf值")


def check_single_column(df: pd.DataFrame) -> EvaluatorResult:
    if len(df.columns) == 1:
        return EvaluatorResult(name="单列输出", passed=True, level="ok", message=f"输出为单列: {df.columns[0]}")
    return EvaluatorResult(name="单列输出", passed=False, level="error", message=f"输出有{len(df.columns)}列，期望1列")


def check_format_basic(df: pd.DataFrame) -> EvaluatorResult:
    issues = []
    if not isinstance(df.index, pd.MultiIndex):
        issues.append("索引不是MultiIndex")
    else:
        if len(df.index.names) != 2:
            issues.append(f"MultiIndex应有2层，实际有{len(df.index.names)}层")
        else:
            names_lower = [n.lower() if n else "" for n in df.index.names]
            if "instrument" not in names_lower or "datetime" not in names_lower:
                issues.append(f"索引名称应包含instrument和datetime，实际为{df.index.names}")
    if df.shape[1] != 1:
        issues.append(f"应有1列，实际有{df.shape[1]}列")
    if not issues:
        return EvaluatorResult(name="格式检查", passed=True, level="ok", message="DataFrame格式正确")
    return EvaluatorResult(name="格式检查", passed=False, level="error", message="; ".join(issues))


def check_daily_frequency(df: pd.DataFrame) -> EvaluatorResult:
    if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names:
        return EvaluatorResult(name="日频校验", passed=False, level="warning", message="无法检查：索引不含datetime层")
    dates = df.index.get_level_values("datetime")
    if len(dates) < 2:
        return EvaluatorResult(name="日频校验", passed=True, level="ok", message="数据不足2天，跳过频率检查")
    diff = pd.Series(dates.unique()).sort_values().diff().dropna()
    median_diff = diff.median()
    if median_diff <= pd.Timedelta(days=2):
        return EvaluatorResult(name="日频校验", passed=True, level="ok", message="数据频率为日频")
    return EvaluatorResult(name="日频校验", passed=False, level="warning", message=f"数据频率可能非日频，中位数间隔={median_diff}")


def check_row_count(df: pd.DataFrame, expected_rows: int | None = None) -> EvaluatorResult:
    if expected_rows is None or expected_rows == 0:
        return EvaluatorResult(name="行数匹配", passed=True, level="ok", message=f"行数={len(df)}，无基准行数可比较")
    ratio = len(df) / expected_rows
    if ratio >= 0.99:
        return EvaluatorResult(name="行数匹配", passed=True, level="ok", message=f"行数={len(df)}，基准={expected_rows}，比率={ratio:.4f}")
    return EvaluatorResult(name="行数匹配", passed=False, level="warning", message=f"行数={len(df)}，基准={expected_rows}，比率={ratio:.4f}（<0.99）")


def check_index_consistency(df: pd.DataFrame, reference_index: pd.MultiIndex | None = None) -> EvaluatorResult:
    if reference_index is None:
        return EvaluatorResult(name="索引一致性", passed=True, level="ok", message="无参考索引，跳过检查")
    if not isinstance(df.index, pd.MultiIndex):
        return EvaluatorResult(name="索引一致性", passed=False, level="error", message="因子索引不是MultiIndex")
    ratio = len(df.index) / len(reference_index) if len(reference_index) > 0 else 0
    if ratio >= 0.99:
        overlap = df.index.intersection(reference_index)
        overlap_ratio = len(overlap) / len(df.index) if len(df.index) > 0 else 0
        if overlap_ratio >= 0.99:
            return EvaluatorResult(name="索引一致性", passed=True, level="ok", message=f"索引高度一致，重叠率={overlap_ratio:.4f}")
        return EvaluatorResult(name="索引一致性", passed=False, level="warning", message=f"索引重叠率={overlap_ratio:.4f}（<0.99）")
    return EvaluatorResult(name="索引一致性", passed=False, level="warning", message=f"行数比率={ratio:.4f}（<0.99）")


def run_all_evaluators(
    df: pd.DataFrame,
    expected_rows: int | None = None,
    reference_index: pd.MultiIndex | None = None,
) -> list[EvaluatorResult]:
    results = [
        check_inf(df),
        check_single_column(df),
        check_format_basic(df),
        check_daily_frequency(df),
        check_row_count(df, expected_rows),
        check_index_consistency(df, reference_index),
    ]
    return results
