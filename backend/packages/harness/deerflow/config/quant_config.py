import os
from pydantic import BaseModel


class QuantConfig(BaseModel):
    provider_uri: str = os.getenv("QLIB_PROVIDER_URI", "./data/qlib_data/cn_data")
    region: str = "cn"
    market: str = "csi300"
    benchmark: str = "SH000300"
    train_start: str = "2010-01-01"
    train_end: str = "2021-12-31"
    valid_start: str = "2022-01-01"
    valid_end: str = "2023-06-30"
    test_start: str = "2023-07-01"
    test_end: str | None = "2026-04-30"
    model_class: str = "LGBModel"
    model_module: str = "qlib.contrib.model.gbdt"
    model_kwargs: dict = {"n_jobs": 8}
    topk: int = 50
    n_drop: int = 5
    account: float = 100000000.0
    limit_threshold: float = 0.095
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.0
    deal_price: str = "close"
    factor_set: str = "ALPHA20"
