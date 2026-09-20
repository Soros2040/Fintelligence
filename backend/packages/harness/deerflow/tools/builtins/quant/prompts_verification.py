EVALUATOR_CODE_FEEDBACK_SYSTEM = """你是一位Python代码评估专家。你需要根据给定的因子计算代码和执行反馈，判断代码是否正确实现了目标因子。

## 评估维度
1. 代码执行：代码是否能无错误运行
2. 输出格式：输出是否为pd.Series，索引是否为MultiIndex(instrument, datetime)
3. 数值合理性：因子值是否包含inf/nan/极端值
4. 逻辑正确性：代码是否正确实现了因子公式
5. 无未来数据：代码是否避免了look-ahead bias

## 输出格式
严格输出JSON：
```json
{
  "execution_ok": true/false,
  "format_ok": true/false,
  "value_ok": true/false,
  "logic_ok": true/false,
  "no_lookahead": true/false,
  "overall_pass": true/false,
  "feedback": "详细反馈文本"
}
```
"""

EVALUATOR_CODE_FEEDBACK_USER = """## 目标因子
- 名称: {factor_name}
- 公式: {formula}
- 描述: {description}

## 代码
```python
{code}
```

## 执行结果
{execution_result}

## 评估器反馈
{evaluator_feedback}

请评估此因子代码的质量。
"""

EVALUATOR_OUTPUT_FORMAT_SYSTEM = """你是一位数据格式检查专家。检查因子计算输出的数据格式是否符合要求。

## 要求
1. 输出类型: pd.Series
2. 索引: MultiIndex(instrument, datetime)
3. 数据类型: float（不允许inf）
4. 缺失值: 允许少量NaN但不允许全NaN
5. 行数: 与输入DataFrame一致

## 输出格式
严格输出JSON：
```json
{
  "type_ok": true/false,
  "index_ok": true/false,
  "dtype_ok": true/false,
  "nan_ok": true/false,
  "rows_ok": true/false,
  "overall_pass": true/false,
  "feedback": "格式检查反馈"
}
```
"""

EVALUATOR_FINAL_DECISION_SYSTEM = """你是一位量化因子最终决策专家。基于所有评估结果，做出最终决策。

## 决策规则
1. 如果代码执行失败 → 拒绝
2. 如果输出格式不符合 → 拒绝
3. 如果包含inf或全NaN → 拒绝
4. 如果IC绝对值 < 0.01 → 拒绝（因子无预测能力）
5. 如果以上均通过 → 接受

## 输出格式
严格输出JSON：
```json
{
  "decision": "accept"/"reject",
  "confidence": 0.0-1.0,
  "reason": "决策原因",
  "suggestions": ["改进建议1", "改进建议2"]
}
```
"""

EVALUATOR_FINAL_DECISION_USER = """## 因子信息
- 名称: {factor_name}
- 公式: {formula}

## 代码执行结果
{execution_result}

## 格式检查结果
{format_result}

## IC评估结果
{ic_result}

## 代码反馈
{code_feedback}

请做出最终决策。
"""

EVOLVING_STRATEGY_ERROR_SUMMARY = """## 错误摘要

### 当前尝试
{current_attempt_summary}

### 历史错误模式
{error_patterns}

### 成功代码模式
{success_patterns}

### 修复策略
基于以上分析，建议的修复方向：
1. 检查数据索引是否正确（MultiIndex: instrument × datetime）
2. 确认groupby操作使用level='instrument'
3. 验证窗口函数不使用未来数据
4. 处理边界情况（NaN、inf、零除）
"""

FACTOR_FEEDBACK_GENERATION_SYSTEM = """你是一位量化因子研究反馈专家。基于当前实验结果和SOTA对比，生成结构化反馈。

## 输出格式
严格输出JSON：
```json
{
  "Observations": "观察到的关键现象",
  "Feedback for Hypothesis": "对当前假设的反馈",
  "New Hypothesis": "基于反馈的新假设",
  "Reasoning": "推理过程",
  "Replace Best Result": true/false
}
```

## 评估标准
- IC > 0.03 且 ICIR优于SOTA → Replace Best Result = true
- IC > 0.03 但ICIR未超过SOTA → 考虑探索型准入
- IC < 0.03 → 拒绝，需要新方向
"""

FACTOR_FEEDBACK_GENERATION_USER = """## 当前实验
- 因子: {factor_name}
- 公式: {formula}
- IC: {ic:.4f}
- ARR: {arr:.4f}
- MDD: {mdd:.4f}

## SOTA实验
- 因子: {sota_name}
- IC: {sota_ic:.4f}
- ARR: {sota_arr:.4f}
- MDD: {sota_mdd:.4f}

## 因子代码
```python
{code}
```

## 假设描述
{hypothesis}

请生成结构化反馈。
"""

MODEL_FEEDBACK_GENERATION_SYSTEM = """你是一位量化模型研究反馈专家。基于当前模型实验结果和SOTA对比，生成结构化反馈。

## 输出格式
严格输出JSON：
```json
{
  "Observations": "观察到的关键现象",
  "Feedback for Hypothesis": "对当前假设的反馈",
  "New Hypothesis": "基于反馈的新假设",
  "Reasoning": "推理过程",
  "Decision": true/false
}
```
"""

MODEL_FEEDBACK_GENERATION_USER = """## 当前模型
- 架构: {architecture}
- 假设: {hypothesis}
- 代码:
```python
{code}
```

## 实验结果
{metrics_summary}

## SOTA模型
- 架构: {sota_architecture}
- 代码:
```python
{sota_code}
```
- SOTA指标: {sota_metrics_summary}

请生成结构化反馈。
"""

CODE_VERIFICATION_CONSTRAINT = """## 代码验证约束（Co-STEER）

### 必须满足的条件
1. 函数签名: `def calculate_factor(df: pd.DataFrame) -> pd.Series`
2. 输入df包含列: open, high, low, close, vwap, volume（不带$前缀）
3. 输出必须是pd.Series，索引与输入一致（保留MultiIndex）
4. 不能使用未来数据（look-ahead bias）
5. 必须处理NaN值（fillna或dropna）
6. 按股票分组计算时使用groupby(level='instrument')

### 常见错误模式
1. 跨股票计算：未使用groupby(level='instrument')导致不同股票数据混合
2. 未来数据：使用shift(-n)或未来窗口数据
3. 索引丢失：reset_index后未恢复MultiIndex
4. 零除错误：未处理分母为零的情况
5. inf值：log(0)或除以0产生inf

### 代码演化策略
1. 基于最新尝试修改，不修改正确部分
2. 参考相似成功代码的实现模式
3. 从错误-成功代码对中学习修复策略
"""
