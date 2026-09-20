# 来源与研究脉络

[首页](../../README_zh.md) · [English](../en/sources.md)

## 软件基础

| 来源 | 在本项目中的作用 | 本地证据 |
|---|---|---|
| [DeerFlow](https://github.com/bytedance/deer-flow) | 前端、智能体框架、网关与开发工具 | [根目录 MIT 许可](../../LICENSE)、沿用的源码命名空间 |
| [Qlib](https://github.com/microsoft/qlib) | 数据处理、模型适配与组合评估 | [Qlib MIT 许可](../../third_party/qlib/LICENSE)、固定版本的运行时依赖 |
| [LightGBM](https://github.com/microsoft/LightGBM) | 当前梯度提升树训练后端 | Qlib `LGBModel` 配置与依赖 |
| [LangGraph](https://github.com/langchain-ai/langgraph) | 有状态图执行与流式返回 | 后端依赖声明及图注册 |

源码清单与恢复备份保存了原始工作快照，包括研究修改。本包提供筛选后的实现和通用配置；每次新实验仍需重新记录源码版本与本地修改。

## 与历史文稿的关系

早期项目文稿使用 **Alex-Fin** 名称，描述因子 DAG、检索、公式/代码生成及 Bandit 研究设计。本项目展示名称为 **BenjaminAgent**，目录与仓库脉络沿用 `Fintelligence`。

通过[完整文稿](https://github.com/Soros2040/julius-future/blob/main/works/benjamin-agent/manuscript.md)或 [PDF](https://github.com/Soros2040/julius-future/blob/main/works/benjamin-agent/manuscript.pdf)阅读研究叙述、公式、结果表和附录。[作品入口](https://github.com/Soros2040/julius-future/tree/main/works/benjamin-agent/)保存文稿及发布背景。[原图指南](../assets/manuscript/README.md)与[哈希清单](../assets/manuscript/provenance.json)记录本仓库使用的三张方法图。

阅读时区分三类证据：文稿段落说明作者提出或报告了什么，源码检查说明特定实现表达了什么，保存的运行证据包说明在特定环境与配置下实际执行了什么。教程联系前两类证据；历史性能数值保留文稿归属，数据、日志、成本与配置完整对应后，才能形成当前结果声明。

| 待对照问题 | 文稿证据 | 当前源码阅读 |
|---|---|---|
| 调度状态 | 截至 Sharpe 的八维状态 | 增加 Calmar 的九个字段，权重也有变化 |
| 标签与组合协议 | 正文 §4.1 与附录采用不同周期和选股规则 | 核查 Qlib 处理器、标签、top-k/drop、划分及成本 |
| 年化与回撤 | 附录 ARR 为复合年化，MDD 公式为正损失幅度 | 管道采用 `252 × mean(return)` 与负回撤 |
| 结果归属 | 主表 3 与 §4.2 正文对最高结果的归属不同 | 保留表格归属，并记录尚待解决的正文差异 |
| 知识持久化 | 胖节点与集中研究记忆 | 序列化的 IC 系列节点字段及独立模型实验日志 |

[案例一](case-01-task-to-backtest.md)推导评价口径并阅读结果表，[案例二](case-02-factor-lifecycle.md)讲解检索、准入、状态构造和后验更新。[状态清单](status.md)记录实现缺口。来源复核应保留准确章节/表格引用，并说明每个数值所属配置。

## 学习组织方式

[diy-llm](https://github.com/datawhalechina/diy-llm) 与 [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) 为“目标—前置知识—实现—例子—练习—证据贡献”的组织方式提供参考。本项目的解释与例子使用原创文字。教学风格参考不构成组织隶属关系。

## 如何增加引用

链接原始来源，说明它支持的具体表述，并记录函数、章节或公式位置。实现结论指向当前源码，实验结论附可复现凭据包。第三方全文遵循其原有再分发条件，题录链接不转移这些权利。
