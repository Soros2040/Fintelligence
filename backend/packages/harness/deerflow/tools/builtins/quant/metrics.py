from pydantic import BaseModel


class QuantMetrics(BaseModel):
    ic: float | None = None
    icir: float | None = None
    rank_ic: float | None = None
    rank_icir: float | None = None
    arr: float | None = None
    ir: float | None = None
    mdd: float | None = None
    sharpe: float | None = None
    calmar: float | None = None

    def to_display(self) -> str:
        lines = ["=== 9维量化评估指标 ==="]
        if self.ic is not None:
            lines.append(f"  IC (信息系数):        {self.ic:.6f}")
        if self.icir is not None:
            lines.append(f"  ICIR (信息比率):      {self.icir:.6f}")
        if self.rank_ic is not None:
            lines.append(f"  Rank IC (秩信息系数): {self.rank_ic:.6f}")
        if self.rank_icir is not None:
            lines.append(f"  Rank ICIR:           {self.rank_icir:.6f}")
        if self.arr is not None:
            lines.append(f"  ARR (年化收益率):     {self.arr:.6f}")
        if self.ir is not None:
            lines.append(f"  IR (信息比率):        {self.ir:.6f}")
        if self.mdd is not None:
            lines.append(f"  MDD (最大回撤):       {self.mdd:.6f}")
        if self.sharpe is not None:
            lines.append(f"  Sharpe (夏普比率):    {self.sharpe:.6f}")
        if self.calmar is not None:
            lines.append(f"  Calmar (卡尔马比率):  {self.calmar:.6f}")
        return "\n".join(lines)

    def to_state_vector(self) -> list[float]:
        return [
            self.ic or 0.0,
            self.icir or 0.0,
            self.rank_ic or 0.0,
            self.rank_icir or 0.0,
            self.arr or 0.0,
            self.ir or 0.0,
            self.mdd or 0.0,
            self.sharpe or 0.0,
            self.calmar or 0.0,
        ]
