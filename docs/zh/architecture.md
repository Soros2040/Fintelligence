# 架构与数据契约

[首页](../../README_zh.md) · [English](../en/architecture.md)

BenjaminAgent 包含交互层、编排层和量化执行层。框架沿用上游模块名，使导入关系与构建工具保持可追溯。

```mermaid
flowchart TD
    U[研究问题] --> UI[Next.js 聊天工作台]
    UI --> S[LangGraph 流客户端]
    S --> A[lead_agent 与配置工具]
    UI --> G[FastAPI 网关：文件与元数据]
    A --> Q[quant_session 与量化工具]
    Q --> P[Qlib 数据集与 LightGBM]
    P --> M[预测、IC、组合报告]
    Q --> D[因子 DAG 与模型实验日志]
    D --> R[检索与准入]
    R --> B[Bandit 调度]
    M --> Q
    Q --> UI
```

图中展示已有组件及预期数据关系；挖掘便捷操作是否一次执行全部环节，应以[状态清单](status.md)记录的集成边界为准。

| 边界 | 对应源码 | 经过的数据 |
|---|---|---|
| 聊天 → 数据流 | [hooks.ts](../../frontend/src/core/threads/hooks.ts) | 用户消息、任务 ID、运行上下文 |
| 数据流 → 图 | [api-client.ts](../../frontend/src/core/api/api-client.ts)、[langgraph.json](../../backend/langgraph.json) | `lead_agent` 运行及增量事件 |
| 智能体 → 工具 | [tools.py](../../backend/packages/harness/deerflow/tools/tools.py)、[配置样例](../../config.example.yaml) | 解析后的工具对象与参数 |
| 工具 → 会话 | [thread_state.py](../../backend/packages/harness/deerflow/agents/thread_state.py)、[quant_analyze_tool.py](../../backend/packages/harness/deerflow/tools/builtins/quant/quant_analyze_tool.py) | 配置、历史、指标及 DAG 状态 |
| 会话 → 实验 | [qlib_pipeline.py](../../backend/packages/harness/deerflow/tools/builtins/quant/qlib_pipeline.py) | 数据集、训练/验证/测试段、模型和策略参数 |
| 因子 → 知识 | [knowledge_graph.py](../../backend/packages/harness/deerflow/tools/builtins/quant/dag/knowledge_graph.py) | 公式、代码、血缘、测量值及反馈 |

## 需要检查的契约

因子输出为单列 pandas DataFrame，索引为 `(datetime, instrument)` 两层。预测值在计算每日截面相关系数前，与测试标签索引对齐。会话将配置和可读消息分别保存，单凭回复文字不足以复现实验。DAG 节点同时记录公式与代码，并保存来源和质量测量。

前端消费工具事件和状态更新，Qlib 训练在服务端执行。FastAPI 网关提供相关 API，默认流式运行由 LangGraph 服务承担；本地反向代理将两者映射到同一浏览器源。

## 检查题

聊天显示实验完成，但没有组合指标，应从哪里排查？

**答案。** 从工具状态更新追踪到 `run_full_pipeline` 和 `_calc_backtest_metrics`。基准数据缺失时可能只返回 IC 结果。先核对结构化指标和日志，再决定是否修改界面；缺失指标不能按零解释。
