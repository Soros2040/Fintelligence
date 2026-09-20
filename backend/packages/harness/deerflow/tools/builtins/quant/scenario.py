from deerflow.config.quant_config import QuantConfig

BACKGROUND_CN = """因子是量化投资中用于解释投资组合或单一资产收益和风险的特征或变量。投资者利用因子来识别和获取超额收益来源，因子是许多量化投资策略的核心。
因子中的每个数值代表某只股票在某一天的物理量值。
用户将基于前几天的因子值训练模型来预测未来几天的收益率。
因子定义包含以下部分：
1. 名称：因子的名称。
2. 描述：因子的描述。
3. 公式：因子的计算公式。
4. 变量：公式中使用的变量或函数。
因子可能不会提供以上所有部分的信息，因为某些部分可能不适用。
请明确给出因子中的所有超参数，如窗口大小、回溯期等。一个因子应静态定义一个输出和一个静态数据源。例如，过去10天动量和过去20天动量应该是两个不同的因子。"""


DATA_DESC_CN = """可用数据字段：
- $open: 开盘价
- $close: 收盘价
- $high: 最高价
- $low: 最低价
- $volume: 成交量
- $factor: 复权因子

数据格式：Qlib .bin 格式，按股票代码组织，每个交易日一条记录。
股票池：{market}（{market_desc}）
时间范围：{train_start} 至 {test_end}
数据路径：{provider_uri}"""


OUTPUT_FORMAT_CN = """你的输出应该是一个类似以下示例信息的 pandas DataFrame：
<class 'pandas.core.frame.DataFrame'>
MultiIndex: 40914 entries, (Timestamp('2020-01-02 00:00:00'), 'SH600000') to (Timestamp('2021-12-31 00:00:00'), 'SZ300059')
Data columns (total 1 columns):
 #   Column            Non-Null Count  Dtype
---  ------            --------------  -----
 0   your factor name  40914 non-null  float64
dtypes: float64(1)
注意：非空计数可以与总条目数不同，因为某些股票可能在某些日期没有因子值。"""


INTERFACE_CN = """你的 Python 代码应遵循以下接口以更好地与用户系统交互。
你的 Python 代码应包含以下部分：导入部分、函数部分和主函数部分。你应该编写一个名为 "calculate_{{function_name}}" 的主函数，并在 "if __name__ == '__main__'" 部分调用此函数。不要在 Python 代码中编写任何 try-except 块。用户会捕获异常消息并提供反馈给你。
用户会将你的 Python 代码写入一个 Python 文件并直接用 "python {{your_file_name}}.py" 执行该文件。你应该计算因子值并将结果保存到同一目录下名为 "result.h5" 的 HDF5(H5) 文件中。结果文件是一个包含 pandas DataFrame 的 HDF5(H5) 文件。DataFrame 的索引是 "datetime" 和 "instrument"，单列名称是因子名称，值是因子值。结果文件应保存在与你的 Python 文件相同的目录中。"""


SIMULATOR_CN = """因子将被送入 Qlib 平台，训练模型基于前几天的因子值预测未来几天的收益率。
Qlib 是一个面向AI的量化投资平台，旨在利用AI技术在量化投资中实现潜力、赋能研究和创造价值，从探索想法到实施生产。Qlib支持多种机器学习建模范式，包括监督学习、市场动态建模和强化学习。
用户将使用 Qlib 自动执行以下操作：
1. 基于因子值生成新的因子表。
2. 训练模型（如 LightGBM、CatBoost、LSTM 或简单的 PyTorch 模型）基于因子值预测未来几天的收益率。
3. 基于预测收益率和策略构建投资组合。
4. 评估投资组合的表现，包括收益率、夏普比率、最大回撤等。"""


EXPERIMENT_SETTING_CN = """| 数据集 | 模型 | 因子 | 数据划分 |
|--------|------|------|----------|
| {market} | {model_class} | {factor_set} | 训练: {train_start} 至 {train_end} <br> 验证: {valid_start} 至 {valid_end} <br> 测试: {test_start} 至 {test_end} |"""


MARKET_DESC = {
    "csi300": "沪深300成分股",
    "csi500": "中证500成分股",
}


class QuantScenario:
    def __init__(self, config: QuantConfig | None = None):
        self._config = config or QuantConfig()
        self._background = BACKGROUND_CN
        self._source_data = self._build_source_data()
        self._output_format = OUTPUT_FORMAT_CN
        self._interface = INTERFACE_CN
        self._simulator = SIMULATOR_CN
        self._experiment_setting = self._build_experiment_setting()

    def _build_source_data(self) -> str:
        return DATA_DESC_CN.format(
            market=self._config.market,
            market_desc=MARKET_DESC.get(self._config.market, self._config.market),
            train_start=self._config.train_start,
            test_end=self._config.test_end or "最新",
            provider_uri=self._config.provider_uri,
        )

    def _build_experiment_setting(self) -> str:
        return EXPERIMENT_SETTING_CN.format(
            market=self._config.market,
            model_class=self._config.model_class,
            factor_set=self._config.factor_set,
            train_start=self._config.train_start,
            train_end=self._config.train_end,
            valid_start=self._config.valid_start,
            valid_end=self._config.valid_end,
            test_start=self._config.test_start,
            test_end=self._config.test_end or "最新",
        )

    @property
    def background(self) -> str:
        return self._background

    @property
    def source_data(self) -> str:
        return self._source_data

    @property
    def output_format(self) -> str:
        return self._output_format

    @property
    def interface(self) -> str:
        return self._interface

    @property
    def simulator(self) -> str:
        return self._simulator

    @property
    def experiment_setting(self) -> str:
        return self._experiment_setting

    def get_scenario_all_desc(self) -> str:
        return f"""场景背景：
{self.background}

可用的源数据：
{self.source_data}

你应该遵循的代码接口：
{self.interface}

代码输出的格式：
{self.output_format}

用于测试因子的模拟器：
{self.simulator}

实验设置：
{self.experiment_setting}
"""
