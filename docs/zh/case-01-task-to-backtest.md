# 案例一：从研究任务到 Qlib 基线

[首页](../../README_zh.md) · [English](../en/case-01-task-to-backtest.md) · [下一个案例](case-02-factor-lifecycle.md)

## 学习目标与前置知识

完成后，你应能找到请求转成工具调用的位置，说明时间顺序实验的配置，并准确解释报告指标。需要 Python、pandas 索引与相关系数基础；完整执行还需要[运行指南](run-guide.md)中的环境与数据。

本案例是**源码导读与说明性手算**，没有把示例数字当作实际投资实验。当前模型是由 LightGBM 实现的 Qlib `LGBModel`；混合专家模型属于需要独立实现和证据的研究方向。

## 1. 打开界面前先定义任务

有效请求应说明实验内容：

> 初始化沪深 300 研究会话。先展示数据位置、特征处理器、标签周期、训练/验证/测试区间、策略和成本假设。核对后运行 LightGBM 基线，并返回结构化指标与实验记录。

初始化本身无需训练。`quant_analyze(action="init_session")` 创建会话，但不调用 `qlib.init`。默认值来自 [QuantConfig](../../backend/packages/harness/deerflow/config/quant_config.py)：

| 设置 | 当前默认值 | 实验记录必须说明 |
|---|---|---|
| 股票池/基准 | `csi300` / `SH000300` | 覆盖情况及历史成分股 |
| 训练段 | 2010-01-01 至 2021-12-31 | 数据可用性及预处理拟合范围 |
| 验证段 | 2022-01-01 至 2023-06-30 | 超参数选择规则 |
| 测试段 | 2023-07-01 至 2026-04-30 | 实际最后日期与未参与选择的留出集 |
| 模型 | `LGBModel` | 完整解析参数及随机种子规则 |
| 组合 | `topk=50`、`n_drop=5` | 排序、成交与调仓假设 |
| 成本 | 买入 0.0005、卖出 0.0015、最低 5 | 展示收益如何体现成本 |

配置默认值不证明数据覆盖全部日期。覆盖不同时，应在运行前改好区间。最终测试段应留作最后评估；反复用其得分选择因子后，它就参与了选择过程。

## 2. 沿应用追踪请求

1. [chat-box.tsx](../../frontend/src/components/workspace/chats/chat-box.tsx) 与 [use-thread-chat.ts](../../frontend/src/components/workspace/chats/use-thread-chat.ts) 连接聊天界面和任务身份。
2. [hooks.ts](../../frontend/src/core/threads/hooks.ts) 中的 `sendMessage` 构造用户消息、处理可选附件，再携带上下文调用 `thread.submit`。`useStream` 指定 `lead_agent`，监听工具完成事件和状态更新。
3. [api-client.ts](../../frontend/src/core/api/api-client.ts) 创建 LangGraph SDK 客户端并规范流参数。[langgraph.json](../../backend/langgraph.json) 将 `lead_agent` 映射到 `deerflow.agents:make_lead_agent`。
4. [tools.py](../../backend/packages/harness/deerflow/tools/tools.py) 从[配置样例](../../config.example.yaml)解析启用的工具；本地包的示例配置已包含四个量化入口。
5. [quant_analyze_tool.py](../../backend/packages/harness/deerflow/tools/builtins/quant/quant_analyze_tool.py) 按 `action` 分派，返回包含 `ToolMessage` 和结构化状态更新的 `Command`。研究会话保存在 `quant_session`。

这条链也是排错方法：工具没有出现，先查配置和模型选择；出现工具但读取数据失败，检查数据位置与 Qlib 初始化；返回了结果但界面不更新，对照结构化事件和前端处理器。

## 3. 区分展示的因子库与实际训练处理器

`init_session` 可选择 `ALPHA20` 或 `ALPHA158` 公式库。`calculate_factors` 使用 Qlib `D.features` 计算库中的表达式。但是，[qlib_pipeline.py](../../backend/packages/harness/deerflow/tools/builtins/quant/qlib_pipeline.py) 的默认 `_build_task_config` 为模型训练构造 **Alpha158 处理器**。

因此，会话显示 `ALPHA20` 不能证明模型只训练了这 20 个因子。复现时要保存解析后的处理器和实际特征列。处理器拟合区间设为训练日期，DatasetH 则分别提供训练、验证和测试段。

模型路径是 `init_instance_by_config` → `model.fit(dataset)` → `model.predict(dataset)`，随后预测与测试标签对齐。标签周期应从所选 Qlib 处理器的实际表达式确认；只写“未来收益”不够具体。

## 4. 先理解 IC，再解释组合收益

每天的 IC 是预测分数与未来标签之间的截面 Pearson 相关系数：

$$IC_t=\frac{\sum_i (p_{i,t}-\bar p_t)(y_{i,t}-\bar y_t)}{\sqrt{\sum_i(p_{i,t}-\bar p_t)^2\sum_i(y_{i,t}-\bar y_t)^2}}.$$

Rank IC 对排名计算相关系数。平均 IC 汇总每日结果，当前 ICIR 是每日 IC 均值除以标准差。与另外年化的统计量比较前，应核对实现和样本长度。

**手算例子。** 某天三只股票的预测为 `[1, 2, 3]`，标签为 `[0.02, 0.01, 0.03]`。中心化后分别为 `[-1, 0, 1]` 和 `[0, -0.01, 0.01]`。分子为 `0.01`，分母为 `sqrt(2 × 0.0002)=0.02`，所以 IC 为 `0.5`。两组排名具有相同的排列关系，因此本例 Rank IC 也是 `0.5`。

这个值反映一天的排序关联，尚未涉及交易成本、仓位约束或多日稳定性。

## 5. 按实际定义阅读组合报告

`_calc_backtest_metrics` 构造 Qlib `TopkDropoutStrategy`。当基准序列不可用时，它不生成组合报告，因此完整管道可能只返回 IC 指标。

对报告中的每日收益 `r_t`，当前汇总计算：

$$ARR=252\,\bar r,\qquad IR=Sharpe=\sqrt{252}\,\bar r/s_r,$$
$$V_t=\prod_{u\leq t}(1+r_u),\quad MDD=\min_t(V_t/\max_{u\leq t}V_u-1),\quad Calmar=ARR/|MDD|.$$

当前 `ir` 与 `sharpe` 使用相同收益序列和公式；`ir` 尚未独立按基准超额收益计算。年化收益使用算术年化。成本参数传入了 Qlib 交易环境，但汇总公式读取 `report["return"]`，若要声称结果已经扣费，需要核查报告中的收益和成本字段。

**手算例子。** 日均收益为 `0.001`、日标准差为 `0.02` 时，算术 ARR 为 `0.252`，对应比率约为 `0.794`。如果最大回撤为 `-0.20`，Calmar 为 `1.26`。这些数值解释公式，不代表项目业绩。

## 6. 保存实验凭据

可评估的基线应记录：源码版本、依赖版本、数据来源与日历、股票池成分、标签表达式、实际特征列、全部划分日期、随机种子规则、模型设置、策略、成本解释、预测输出、结构化指标、日志、耗时与峰值内存。若失败运行影响了模型或因子选择，也应保存失败原因。

会话中的 `experiment_history` 和最佳记录字段有助于管理实验，但不能替代数据集标识或复现记录。会话在夏普可用时据此选择最佳结果；DAG 中名为 “SOTA” 的访问器有不同含义，案例二会解释。

## 练习与答案

**练习 A。** 会话显示 ALPHA20，并返回 LightGBM 得分。什么证据能证明实际输入？

**答案。** 解析后的 Qlib 任务、处理器类与特征列。当前默认模型路径采用 Alpha158，仅凭展示的公式库不足以确认训练输入。

**练习 B。** 有 IC，没有 ARR，工具文字显示完整管道完成。接下来检查哪两项？

**答案。** 检查基准可用性和 `_calc_backtest_metrics` 日志，再检查结构化指标缺失值。若证据如此，就应登记“IC 评估完成，组合评估不可用”。

**练习 C。** 研究者每次计算测试期夏普后都选择新因子，这段数据仍是未使用的测试集吗？

**答案。** 它已参与选择。应在验证段迭代，冻结因子和模型后，再到单独的最终留出集评估。

## 贡献交付

为一个指标定义或数据契约编写小样本验证，附上手算、准确函数引用、预期与实际结果，并说明该验证没有覆盖哪些更大的结论。按[贡献指南](../../CONTRIBUTING.zh-CN.md)提交。
