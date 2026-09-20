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

历史文稿用于理解设计。其性能描述需要与当前代码、模型选择、标签周期、成本定义、日志及数据版本对应，才能成为当前结果声明。本教程将设计映射到具体源码，并在[状态清单](status.md)中记录实现缺口。文稿的私人元数据和全文材料保留在本包之外。

## 学习组织方式

[diy-llm](https://github.com/datawhalechina/diy-llm) 与 [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang) 为“目标—前置知识—实现—例子—练习—证据贡献”的组织方式提供参考。本项目的解释与例子使用原创文字。教学风格参考不构成组织隶属关系。

## 如何增加引用

链接原始来源，说明它支持的具体表述，并记录函数、章节或公式位置。实现结论指向当前源码，实验结论附可复现凭据包。第三方全文遵循其原有再分发条件，题录链接不转移这些权利。
