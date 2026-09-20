# 本地运行指南

[首页](../../README_zh.md) · [English](../en/run-guide.md)

本指南面向 Linux 或 WSL2，需 Python 3.12+、Node.js 22+、pnpm 10.26.2、uv、Make、Bash、nginx，若平台缺少某个依赖的预编译包，还可能需要 C++ 编译器。已有脚本包含 Windows 适配，完整 Windows 服务路径仍需单独验证。当前包实际完成的检查见[状态清单](status.md)。

## 1. 配置独立工作环境

在项目目录执行：

```bash
make check
make config
make install
```

`make check` 检查必需工具，不是完整测试套件。`make config` 从样例创建本地配置，并保留已有配置。启动前检查 `config.yaml` 及生成的环境文件。参考样例选择你使用的模型服务，将凭据放入环境变量。发布包中只包含占位配置。

[config.example.yaml](../../config.example.yaml) 已包含量化工具组和四项注册；解析后的 Python 对象暴露 `quant_analyze`、`factor_baseline_library`、`factor_generate`、`quant_mining` 工具名。智能体仍需要支持工具调用的模型配置。

## 2. 安装量化依赖层

继承的后端锁文件覆盖智能体平台。将量化依赖层安装到同一个后端环境：

```bash
cd backend
uv pip install --python .venv/bin/python -r requirements-quant.txt
uv run --no-sync python -c "import qlib, lightgbm, pandas, pyarrow; print('quant imports available')"
cd ..
```

[requirements-quant.txt](../../backend/requirements-quant.txt)记录源环境中实际观察到的版本：pyqlib 0.9.7、LightGBM 4.6.0、NumPy 1.26.4、pandas 2.3.3、PyArrow 23.0.1。Qlib 使用其正式发布的软件包安装。安装后应保存完整依赖解析结果，这份简短文件尚不是全部传递依赖的锁文件。

后续普通 `uv sync` 可能移除继承锁文件之外的依赖。量化依赖完整进入锁文件之前，请使用 `uv run --no-sync` 和下方启动方式。依赖集成已列入路线图。

## 3. 提供数据与可选嵌入服务

使用具有相应权限的 Qlib 格式数据，包含 `calendars`、`instruments` 和 `features` 目录。在启动后端的终端设置绝对路径：

```bash
export QLIB_PROVIDER_URI=/absolute/path/to/your/qlib_data/cn_data
```

检查市场标识、基准可用性、日期覆盖、复权处理及标签周期。[案例一](case-01-task-to-backtest.md)列出了默认日期，应根据实际数据覆盖调整。进程会缓存 Qlib 初始化状态，更换数据源时应重启服务。

语义检索可选使用 DashScope，仅在准备启用该功能时设置 `DASHSCOPE_API_KEY`。生成模型与嵌入服务是两项配置。进行无网络组件检查时，应显式关闭语义检索，不能以缺少凭据替代功能开关。

## 4. 启动并检查服务

依赖安装完成后：

```bash
make doctor
chmod +x scripts/*.sh
UV_NO_SYNC=1 bash ./scripts/serve.sh --dev --skip-install
```

默认开发路径使用前端 3000、LangGraph 2024、网关 8001、nginx 6006 端口；打开 `http://localhost:6006`。`--skip-install` 保留已安装的量化依赖，`UV_NO_SYNC=1` 阻止 `uv run` 自动重新同步。服务失败时检查生成的服务日志。

继承的本地配置会在主机接口绑定服务，应在独立开发环境使用，并将网络访问限制到预期参与者。生成因子代码在主机子进程执行，应在隔离环境中审阅和运行。样例保留默认 `allow_host_bash: false`。

先请求初始化量化会话并只展示配置，确认出现 `quant_analyze` 事件，且 `quant_session` 的日期、数据位置正确。准备运行基线时，按[案例一](case-01-task-to-backtest.md)执行；进入生成因子阶段前先保存证据。

## 5. 验证与排错

| 现象 | 首先检查 | 对应位置 |
|---|---|---|
| 看不到量化工具 | 本地 `config.yaml` 注册、模型工具调用能力 | `tools/tools.py` |
| `ModuleNotFoundError: qlib` | 是否同一 Python 环境、额外安装后是否重新同步 | 后端环境 |
| 量化依赖安装失败 | Python/平台的预编译包支持及完整解析日志 | `backend/requirements-quant.txt` |
| 数据或基准不可用 | 数据目录、标识和日期覆盖 | `qlib_pipeline.py` |
| 有 IC，无组合指标 | 基准序列与回测日志 | `_calc_backtest_metrics` |
| 新因子运行的输入未变 | 实际处理器的特征列 | `pipeline_with_new_factors` |
| 数据流或任务状态陈旧 | 任务 ID 与流事件 | 前端 `hooks.ts` |

前端检查在 `frontend` 中执行 `pnpm check`、`pnpm build`。继承的后端测试在 `backend` 中执行 `uv run --no-sync pytest`，先核对测试前提。组件检查可以不使用数据或模型凭据，完整实验则需要服务和数据。应在报告中分别登记这些验证类别。
