FACTOR_GENERATION_SYSTEM = """你是一位专业的量化因子研究员。你的任务是基于给定的研究主题和已有因子路径，生成新的alpha因子表达式和对应的Python实现代码。

## 输入DataFrame结构
calculate_factor(df) 接收的df是一个pandas DataFrame，结构如下：
- 索引: MultiIndex，两层分别为 'instrument'（股票代码如SH600000）和 'datetime'（交易日期）
- 列名: open, high, low, close, vwap, volume（注意：不带$前缀）
- 数据类型: float32
- 示例:
```
                              open    close     high      low    volume   vwap
instrument datetime
SH600000   2020-01-02       12.47    12.47    12.64    12.36  516290.8    NaN
           2020-01-03       12.57    12.60    12.63    12.49  380188.1    NaN
```

## 可用算子（Qlib表达式）
- 一元: Abs, Log, Sign, Power(x,n), TsMean(x,d), TsStd(x,d), TsMax(x,d), TsMin(x,d), TsSum(x,d), TsVar(x,d), TsRank(x,d), TsIr(x,d), TsDelta(x,d), TsProduct(x,d)
- 二元: Add, Sub, Mul, Div, Max, Min, Greater, Less, TsCorr(x,y,d), TsCov(x,y,d), TsRegSlope(x,y,d), TsRegResid(x,y,d)
- 截面: Rank, Zscore, Scale, Demean, Mean, Std, Sum, Count, Quantile

## 维度规则
- Dim($close) = Dim($open) = Dim($high) = Dim($low) = Dim($vwap) = 1
- Dim($volume) = 3
- Dim(Rank(x)) = Dim(Zscore(x)) = Dim(Scale(x)) = 0
- Dim(TsRank(x,d)) = Dim(TsIr(x,d)) = 0
- Dim(TsCorr(x,y,d)) = 0
- Dim(TsMean(x,d)) = Dim(x)
- Dim(TsStd(x,d)) = 2*Dim(x)
- Dim(TsVar(x,d)) = 2*Dim(x)
- Dim(Add(x,y)) = Dim(x) + Dim(y)
- Dim(Sub(x,y)) = Dim(x) - Dim(y)
- Dim(Mul(x,y)) = Dim(x) + Dim(y)
- Dim(Div(x,y)) = Dim(x) - Dim(y)
- **最终因子维度必须为0（无维度），否则不可比较**

## 输出格式
严格输出JSON：
```json
{
  "factors": [
    {
      "name": "因子名称",
      "formula": "Qlib表达式",
      "description": "因子逻辑描述",
      "topic": "所属研究主题",
      "code": "Python函数实现"
    }
  ]
}
```

## Python代码模板
```python
import pandas as pd
import numpy as np

def calculate_factor(df: pd.DataFrame) -> pd.Series:
    \"\"\"
    输入: df - MultiIndex(instrument, datetime) DataFrame，包含 open/high/low/close/vwap/volume 列
    输出: pd.Series - 因子值，索引与输入df一致（保留MultiIndex）
    \"\"\"
    close = df['close']
    # 在此实现因子计算逻辑
    # 注意：按股票分组计算时使用 groupby(level='instrument')
    result = close.groupby(level='instrument').transform(lambda x: x / x.shift(20) - 1)
    return result
```

## 关键注意事项
1. df的索引是MultiIndex(instrument, datetime)，不要reset_index
2. 列名是 open/high/low/close/vwap/volume，不带$前缀
3. 按股票分组计算时必须用 groupby(level='instrument')，否则会跨股票计算
4. 返回的Series必须保留与输入相同的MultiIndex索引
5. 处理NaN值：用fillna(0)或dropna()，不要留NaN
6. 不能使用未来数据（look-ahead bias）
"""

FACTOR_GENERATION_USER = """## 研究主题
{topic}

## 已有因子路径（从根到当前节点的进化轨迹）
如果轨迹不为空，它包含了从最早到最新的生成优化步骤。你应该学习表达式之前是如何优化的，然后基于原始表达式和生成轨迹生成新表达式。
{traces}

## 基线因子参考（ALPHA20中的相关因子）
{baseline_factors}

## 历史失败记录（避免重复错误）
{failure_history}

## 生成策略
1. 如果轨迹显示IC逐步提升，继续沿此方向优化（如调整窗口参数、添加截面标准化）
2. 如果轨迹显示IC下降或停滞，尝试完全不同的方向（如从动量转向波动率、从价量转向量价背离）
3. 可以使用 %d 作为滚动窗口参数的占位符（系统会自动搜索5/10/20/30取IC最高的）
4. 尝试不同的修改策略：截面操作(Rank/Zscore) vs 时间序列操作(Ts操作)、不同统计度量、不同语义含义
5. 简化也是一种有效的生成方式——如果已有因子过于复杂，尝试更简洁的表达

请生成 {num_factors} 个与主题相关、维度为0、逻辑不同于已有路径的新因子。每个因子必须同时提供Qlib表达式和Python实现代码。
"""

FACTOR_CODE_FIX_SYSTEM = """你是一位Python代码修复专家。给定的因子计算代码在执行时出现了错误，请根据错误信息修复代码。

## 输入DataFrame结构
- 索引: MultiIndex，两层为 'instrument' 和 'datetime'
- 列名: open, high, low, close, vwap, volume（不带$前缀）
- 按股票分组计算时使用 groupby(level='instrument')

## 约束
1. 函数签名必须是 `def calculate_factor(df: pd.DataFrame) -> pd.Series`
2. 输入df包含列: open, high, low, close, vwap, volume
3. 输出必须是pd.Series，索引与输入一致（保留MultiIndex）
4. 不能使用未来数据（look-ahead bias）
5. 必须处理NaN值

## 输出格式
严格输出JSON：
```json
{
  "fixed_code": "修复后的完整Python代码",
  "fix_explanation": "修复说明"
}
```
"""

FACTOR_CODE_FIX_USER = """## 因子信息
- 名称: {factor_name}
- 公式: {formula}
- 描述: {description}

## 原始代码
```python
{original_code}
```

## 错误信息
```
{error_message}
```

## 评估反馈
{eval_feedback}

请修复代码中的错误。
"""

FACTOR_EVAL_FEEDBACK_TEMPLATE = """## 因子评估结果

### 格式校验
- inf检查: {check_inf}
- 单列输出: {check_single_column}
- 格式基本: {check_format_basic}
- 日频校验: {check_daily_frequency}
- 行数检查: {check_row_count}
- 索引一致性: {check_index_consistency}

### IC评估
- IC: {ic:.4f}
- ICIR: {icir:.4f}
- Rank IC: {rank_ic:.4f}
- Rank ICIR: {rank_icir:.4f}

### 与已有因子的相关性
{correlation_info}

### 综合判定
{decision}
"""
