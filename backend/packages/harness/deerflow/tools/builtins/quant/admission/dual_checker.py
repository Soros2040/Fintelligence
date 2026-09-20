from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from deerflow.tools.builtins.quant.dag import DualRepFactorNode

logger = logging.getLogger(__name__)


class DualAdmissionChecker:
    def __init__(
        self,
        ic_threshold: float = 0.006,
        icir_decay: float = 0.70,
        ic_new_threshold: float = 0.45,
        ic_mut_threshold: float = 0.9,
        capacity: int = 50,
    ) -> None:
        self.ic_threshold = ic_threshold
        self.icir_decay = icir_decay
        self.ic_new_threshold = ic_new_threshold
        self.ic_mut_threshold = ic_mut_threshold
        self.capacity = capacity

    def check(
        self, node: DualRepFactorNode, parent: Optional[DualRepFactorNode], pool: List[DualRepFactorNode]
    ) -> dict:
        ic_abs = abs(node.ic)
        icir_abs = abs(node.icir)

        condition_a = ic_abs >= self.ic_threshold
        if parent is not None:
            parent_icir = abs(parent.icir)
            condition_a = condition_a and (icir_abs > parent_icir)

        condition_b = False
        if parent is not None:
            parent_icir = abs(parent.icir)
            parent_mut_ic = self._max_mutual_ic(parent, pool)
            node_mut_ic = self._max_mutual_ic(node, pool)
            condition_b = (
                ic_abs >= self.ic_threshold
                and icir_abs > parent_icir * self.icir_decay
                and node_mut_ic < self.ic_new_threshold
                and node_mut_ic < parent_mut_ic
            )

        mut_ic_ok = self._check_mutual_ic(node, pool)

        if condition_a and mut_ic_ok:
            return {"decision": True, "type": "exploit", "reason": f"IC={ic_abs:.4f}≥{self.ic_threshold} 且 ICIR={icir_abs:.4f}优于父节点 且 MutIC≤{self.ic_mut_threshold}"}
        elif condition_b and mut_ic_ok:
            return {"decision": True, "type": "explore", "reason": f"IC={ic_abs:.4f}≥{self.ic_threshold} 且 MutIC={self._max_mutual_ic(node, pool):.4f}低于父节点MutIC，多样性提升"}
        elif condition_a and not mut_ic_ok:
            return {"decision": False, "type": "reject", "reason": f"IC={ic_abs:.4f}达标但MutIC>{self.ic_mut_threshold}，与池中因子相关性过高"}
        elif not condition_a and mut_ic_ok:
            return {"decision": False, "type": "explore", "reason": f"IC={ic_abs:.4f}<{self.ic_threshold}，但MutIC≤{self.ic_mut_threshold}，因子多样性好但预测力不足"}
        else:
            return {"decision": False, "type": "reject", "reason": f"IC={ic_abs:.4f}<{self.ic_threshold} 且 MutIC>{self.ic_mut_threshold}，预测力不足且与池中因子相似"}

    def _compute_pearson_mutual_ic(self, node: DualRepFactorNode, other: DualRepFactorNode) -> float:
        if node.factor_values_path and other.factor_values_path:
            try:
                from pathlib import Path
                node_path = Path(node.factor_values_path)
                other_path = Path(other.factor_values_path)
                if node_path.exists() and other_path.exists():
                    node_vals = pd.read_parquet(node_path).iloc[:, 0]
                    other_vals = pd.read_parquet(other_path).iloc[:, 0]
                    common_idx = node_vals.dropna().index.intersection(other_vals.dropna().index)
                    if len(common_idx) < 50:
                        return 0.0
                    daily_corr = node_vals.loc[common_idx].groupby(level="datetime").apply(
                        lambda x: x.corr(other_vals.loc[x.index], method="pearson")
                    ).dropna()
                    return float(abs(daily_corr.mean())) if len(daily_corr) > 0 else 0.0
            except Exception as e:
                logger.debug(f"Factor value Pearson IC failed for {node.node_id} vs {other.node_id}: {e}")

        if node.ic != 0.0 and other.ic != 0.0:
            return 1.0 if (node.ic > 0) == (other.ic > 0) else 0.0
        return 0.0

    def _max_mutual_ic(self, node: DualRepFactorNode, pool: List[DualRepFactorNode]) -> float:
        if not pool:
            return 0.0
        max_mut = 0.0
        for other in pool:
            if other.node_id == node.node_id:
                continue
            mut_ic = self._compute_pearson_mutual_ic(node, other)
            max_mut = max(max_mut, mut_ic)
        return max_mut

    def _check_mutual_ic(self, node: DualRepFactorNode, pool: List[DualRepFactorNode]) -> bool:
        if not pool:
            return True
        max_mut = self._max_mutual_ic(node, pool)
        return max_mut <= self.ic_mut_threshold

    def can_enter_pool(self, node: DualRepFactorNode, pool: List[DualRepFactorNode]) -> bool:
        if len(pool) < self.capacity:
            return self._check_mutual_ic(node, pool)
        min_ic = min(abs(n.ic) for n in pool) if pool else 0.0
        node_ic = abs(node.ic)
        if node_ic <= min_ic:
            return False
        return self._check_mutual_ic(node, pool)

    def find_evict_candidate(self, pool: List[DualRepFactorNode]) -> Optional[DualRepFactorNode]:
        if not pool:
            return None
        return min(pool, key=lambda n: abs(n.ic))
