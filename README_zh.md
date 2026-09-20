# BenjaminAgent

**把金融研究问题变成可追溯因子实验的研究工作台。**

[English](README.md) · [从这里开始](docs/zh/learning-path.md) · [完整文稿](https://github.com/Soros2040/julius-future/blob/main/works/benjamin-agent/manuscript.md) · [参与贡献](CONTRIBUTING.zh-CN.md)

BenjaminAgent 将流式研究界面、LangGraph 智能体运行时、Qlib 评估与实验性的因子知识管理连接起来，基于 [DeerFlow](https://github.com/bytedance/deer-flow) 和 [Qlib](https://github.com/microsoft/qlib) 构建。当前训练路径使用 **LightGBM**。项目处于实现与验证阶段，提供对应真实源码的教程和明确的验证任务。

## 架构概览

![原稿图 2：Alex-Fin 架构图](docs/assets/manuscript/figure-02-system-architecture.jpeg)

*原稿图 2「Alex-Fin架构图」，按原始文件提取。Alex-Fin 是历史研究名称；图中方法采用八维调度状态，当前 BenjaminAgent 实现采用九个字段。源码映射见[架构指南](docs/zh/architecture.md)，原始文件记录见[图片来源](docs/assets/manuscript/README.md)。*

[完整文稿与 PDF](https://github.com/Soros2040/julius-future/tree/main/works/benjamin-agent/)提供研究叙述、公式、表格和附录。教程将这些内容与当前源码联系起来；[实现状态](docs/zh/status.md)列出仍需贯通编排的生命周期阶段。

## 你可以学到什么、完成什么

- 从聊天界面追踪一次请求，直到工具调用和 Qlib 实验。
- 用清晰定义区分因子质量、预测质量与组合表现。
- 理解因子血缘、检索、准入与调度之间的关系。
- 提交源码审阅、手算推导、原图说明复核或双语文档改进。

教程使用可以手算的小例子。示例数字用于讲解；研究结果需要附上[状态清单](docs/zh/status.md)中规定的证据。

## 项目导航

| 主线 | 现有材料 | 下一项交付 | 实际状态 |
|---|---|---|---|
| 理解系统 | [架构](docs/zh/architecture.md)、[学习路径](docs/zh/learning-path.md) | 从一次请求到配置工具的源码映射 | 已有源码导读 |
| 理解评估 | [案例一：任务到回测](docs/zh/case-01-task-to-backtest.md)及 Qlib 管道 | 对照文稿复核标签周期、成本与指标定义 | 已有代码，待集成运行验证 |
| 研究因子演化 | [案例二：因子生命周期](docs/zh/case-02-factor-lifecycle.md) | 手算复核一次检索或调度 | 已有实验组件 |
| 保存研究证据 | [来源](docs/zh/sources.md)、[路线图](docs/zh/roadmap.md)、[维护说明](docs/maintenance.zh-CN.md) | 复核一张文稿表格、原图或对应翻译 | 文档审阅开放认领 |

## 两个贯通案例

**[一：用户任务如何成为基线实验](docs/zh/case-01-task-to-backtest.md)。** 沿着界面、数据流、工具配置、会话、时间划分、LightGBM 训练、IC 计算和组合报告逐步阅读，用三只股票的相关系数和年化计算理解指标。

**[二：因子如何成为可复用的研究知识](docs/zh/case-02-factor-lifecycle.md)。** 从动量假设出发，经过公式与代码、格式检查、DAG 血缘、检索打分、准入规则和九维 Bandit 状态。先手算选择与准入，再核查实现边界。

## 第一次贡献

1. 阅读一个案例，选择一条附有源码链接的表述。
2. 打开链接源码及对应文稿段落，记录符号或章节、输入、输出与前提。
3. 手算一个小例子，或对照原图/表格与说明。区分设计表述、源码发现和文稿报告的结果。
4. 创建 Issue 并认领明确范围，使用[贡献记录](contributions/README.md)提交聚焦 PR；中英文内容对应的页面同步更新。

| 入门任务 | 建议位置 | 验收证据 | 认领状态 |
|---|---|---|---|
| 复核 IC 与收益定义 | 案例一及英文对应页 | 手算过程、精确函数引用与文稿公式 | 待认领 |
| 复核一张原图或结果表 | [来源页](docs/zh/sources.md)与[原图指南](docs/assets/manuscript/README.md) | 图表编号、原文位置、符合历史背景的说明 | 待认领 |
| 解释一次检索或调度计算 | 案例二及英文对应页 | 输入、逐步算式、源码符号和对应双语解释 | 待认领 |

这些入门任务通过阅读与推理即可完成。[贡献指南](CONTRIBUTING.zh-CN.md)说明审阅与署名流程；[运行指南](docs/zh/run-guide.md)单独作为后续明确范围的执行任务参考。

## 目录结构

```text
frontend/                       流式研究工作台
backend/app/                    网关 API
backend/packages/harness/       智能体运行时与量化工具
third_party/                    依赖来源与许可声明
scripts/                        本地配置与服务启动工具
docker/                         服务和代理配置
skills/                         上游运行时技能
docs/en/ 与 docs/zh/             对应的中英文教程和项目文档
```

目录和历史仓库名沿用 `Fintelligence`，对外展示名称为 **BenjaminAgent**。历史文稿命名和源码来源见[来源与研究脉络](docs/zh/sources.md)。

## 当前研究边界

基线管道已包含模型训练、预测、IC 评估和组合评估代码。因子生成、DAG、检索、准入及 Bandit 模块已经存在，模块集成与指标口径仍有待验证。尤其是 `run_mining_loop` 当前执行调度和检索，完整生命周期仍需编排与核验。解释任何实验输出前，请先核对[状态清单](docs/zh/status.md)。

## 分工、维护与许可

Julius 维护研究方向、审阅任务并协调发布。参见[角色分工](docs/zh/roles.md)、[维护与交接](docs/maintenance.zh-CN.md)、[路线图](docs/zh/roadmap.md)和[第三方声明](THIRD_PARTY_NOTICES.md)。上游代码和原创代码适用相应的 [MIT 许可](LICENSE)。原创项目文档采用 [CC BY-NC-SA 4.0](LICENSE-DOCS)，第三方文档保持其原有条款。教程借鉴 Datawhale 项目的学习组织方式；组织参与或背书以已确认信息为准。
