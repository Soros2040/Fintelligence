# Local run guide

[Home](../../README.md) · [中文](../zh/run-guide.md)

This guide targets Linux or WSL2 with Python 3.12+, Node.js 22+, pnpm 10.26.2, uv, Make, Bash, nginx; a C++ compiler may be needed if a dependency has no wheel for your platform. The included scripts also contain Windows adaptations; the complete Windows service path requires its own validation. See [status](status.md) for what has actually been checked in this package.

## 1. Configure an isolated workspace

From the project directory:

```bash
make check
make config
make install
```

`make check` checks required tools; it is not the full test suite. `make config` creates local configuration from examples and preserves an existing configuration. Inspect `config.yaml` and the generated environment files before starting services. Configure a model supported by your provider using the example entries, with its credential in an environment variable. The publication package contains placeholders only.

The quantitative group and four tool registrations are included in [config.example.yaml](../../config.example.yaml). The resolved Python objects expose the tool names `quant_analyze`, `factor_baseline_library`, `factor_generate`, and `quant_mining`. The agent still needs a configured model capable of tool use.

## 2. Install the quantitative dependency layer

The inherited backend lock covers the agent platform. Install the quantitative dependency layer into that same backend environment:

```bash
cd backend
uv pip install --python .venv/bin/python -r requirements-quant.txt
uv run --no-sync python -c "import qlib, lightgbm, pandas, pyarrow; print('quant imports available')"
cd ..
```

[requirements-quant.txt](../../backend/requirements-quant.txt) records versions observed in the source environment: pyqlib 0.9.7, LightGBM 4.6.0, NumPy 1.26.4, pandas 2.3.3, and PyArrow 23.0.1. Qlib is installed from its published package. Save the complete resolved environment after installation; this small requirements file is not a complete transitive lock.

An ordinary later `uv sync` can remove dependencies installed outside the inherited lock. Until the quantitative layer is integrated into that lock, use `uv run --no-sync` and the launch option below after installing it. Dependency integration is an explicit roadmap task.

## 3. Supply data and optional embeddings

Use a licensed Qlib-format dataset with `calendars`, `instruments`, and `features` directories. Set an absolute data path in the shell that launches the backend:

```bash
export QLIB_PROVIDER_URI=/absolute/path/to/your/qlib_data/cn_data
```

Check market identifiers, benchmark availability, date coverage, corporate-action handling, and the label horizon. The default dates are documented in [Case 1](case-01-task-to-backtest.md); override them to match the actual dataset. Qlib initialization is cached within a process, so restart services when changing providers.

Semantic retrieval optionally uses DashScope. Set `DASHSCOPE_API_KEY` only when that feature is intentionally enabled. The generation model and embedding service are separate configurations. For a no-network component check, disable semantic retrieval explicitly; missing credentials are not a controlled substitute for disabling a feature.

## 4. Start and inspect services

After dependencies are installed:

```bash
make doctor
chmod +x scripts/*.sh
UV_NO_SYNC=1 bash ./scripts/serve.sh --dev --skip-install
```

The default development route uses frontend port 3000, LangGraph port 2024, gateway port 8001, and nginx port 6006. Open `http://localhost:6006`. The script uses `--skip-install` to preserve the quantitative dependency layer and `UV_NO_SYNC=1` prevents `uv run` from resynchronizing it. Check the generated service logs if a process fails.

The inherited local configurations bind services on host interfaces. Use a dedicated development environment and restrict network exposure to intended users. Generated factor code executes in a host subprocess; use an isolated environment for reviewing and running that path. The default `allow_host_bash: false` setting remains in the example.

Begin with a request to initialize a quant session and display configuration only. Confirm that a `quant_analyze` event appears and that `quant_session` contains the expected dates and data location. Then follow [Case 1](case-01-task-to-backtest.md) when ready to run the baseline. Save evidence before moving to generated factors.

## 5. Validation and troubleshooting

| Symptom | First check | Useful source |
|---|---|---|
| Quant tools absent | Entries in local `config.yaml`, tool-use support | `tools/tools.py` |
| `ModuleNotFoundError: qlib` | Same Python environment, no resync after extra install | Backend environment |
| Quant dependency installation fails | Python/platform wheel support and full resolver output | `backend/requirements-quant.txt` |
| Dataset or benchmark unavailable | Provider directories, identifiers, date coverage | `qlib_pipeline.py` |
| IC present but portfolio metrics absent | Benchmark series and backtest log | `_calc_backtest_metrics` |
| Claimed new-factor run has unchanged inputs | Actual handler feature columns | `pipeline_with_new_factors` |
| Stream or task state is stale | Thread ID and stream events | Frontend `hooks.ts` |

For frontend checks, run `pnpm check` and `pnpm build` inside `frontend`. For inherited backend tests, run `uv run --no-sync pytest` inside `backend`; review test prerequisites first. Component checks can run without data or model credentials, while full experiments require the configured services and data. Record each category separately in the evidence report.
