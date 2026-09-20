from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class DualRepFactorNode:
    node_id: str
    formula: Optional[str] = None
    tokens: Optional[Tuple[str, ...]] = None
    code: str = ""
    dimension: Optional[float] = None
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    depth: int = 0
    topic: Optional[str] = None
    description: Optional[str] = None
    llm_explanation: Optional[str] = None
    factor_values_path: Optional[str] = None
    ic: float = 0.0
    icir: float = 0.0
    rank_ic: float = 0.0
    rank_icir: float = 0.0
    test_ic: float = 0.0
    test_icir: float = 0.0
    times_selected: int = 0
    code_verified: bool = False
    timestamp: float = 0.0
    hypothesis: Optional[str] = None
    action: str = "factor"
    feedback: Optional[str] = None
    decision: bool = False
    status: str = "active"
    code_experience: Dict = field(default_factory=dict)
    model_context: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "formula": self.formula,
            "tokens": list(self.tokens) if self.tokens else None,
            "code": self.code,
            "dimension": self.dimension,
            "parent_id": self.parent_id,
            "children_ids": sorted(self.children_ids),
            "depth": self.depth,
            "topic": self.topic,
            "description": self.description,
            "llm_explanation": self.llm_explanation,
            "factor_values_path": self.factor_values_path,
            "ic": self.ic,
            "icir": self.icir,
            "rank_ic": self.rank_ic,
            "rank_icir": self.rank_icir,
            "test_ic": self.test_ic,
            "test_icir": self.test_icir,
            "times_selected": self.times_selected,
            "code_verified": self.code_verified,
            "timestamp": self.timestamp,
            "hypothesis": self.hypothesis,
            "action": self.action,
            "feedback": self.feedback,
            "decision": self.decision,
            "status": self.status,
            "code_experience": self.code_experience,
            "model_context": self.model_context,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DualRepFactorNode:
        return cls(
            node_id=d["node_id"],
            formula=d.get("formula"),
            tokens=tuple(d["tokens"]) if d.get("tokens") else None,
            code=d.get("code") or "",
            dimension=d.get("dimension"),
            parent_id=d.get("parent_id"),
            children_ids=set(d.get("children_ids") or []),
            depth=d.get("depth") or 0,
            topic=d.get("topic"),
            description=d.get("description"),
            llm_explanation=d.get("llm_explanation"),
            factor_values_path=d.get("factor_values_path"),
            ic=d.get("ic") or 0.0,
            icir=d.get("icir") or 0.0,
            rank_ic=d.get("rank_ic") or 0.0,
            rank_icir=d.get("rank_icir") or 0.0,
            test_ic=d.get("test_ic") or 0.0,
            test_icir=d.get("test_icir") or 0.0,
            times_selected=d.get("times_selected") or 0,
            code_verified=bool(d.get("code_verified")),
            timestamp=d.get("timestamp") or 0.0,
            hypothesis=d.get("hypothesis"),
            action=d.get("action") or "factor",
            feedback=d.get("feedback"),
            decision=bool(d.get("decision")),
            status=d.get("status") or "active",
            code_experience=d.get("code_experience") or {},
            model_context=d.get("model_context"),
        )


class FactorDAG:
    def __init__(self) -> None:
        self._nodes: Dict[str, DualRepFactorNode] = {}

    def insert(self, child: DualRepFactorNode, parent: Optional[DualRepFactorNode] = None) -> bool:
        if child.node_id in self._nodes:
            return False
        if parent is not None:
            if parent.node_id not in self._nodes:
                return False
            child.parent_id = parent.node_id
            child.depth = parent.depth + 1
            parent.children_ids.add(child.node_id)
        else:
            child.parent_id = None
            child.depth = 0
        if child.timestamp == 0.0:
            child.timestamp = time.time()
        self._nodes[child.node_id] = child
        return True

    def path_to_root(self, node_id: str) -> List[DualRepFactorNode]:
        if node_id not in self._nodes:
            return []
        path: List[DualRepFactorNode] = []
        visited: Set[str] = set()
        current_id: Optional[str] = node_id
        while current_id is not None:
            if current_id in visited:
                break
            visited.add(current_id)
            node = self._nodes.get(current_id)
            if node is None:
                break
            path.append(node)
            current_id = node.parent_id
        path.reverse()
        return path

    def search(self, formula: str) -> Optional[DualRepFactorNode]:
        for node in self._nodes.values():
            if node.formula == formula:
                return node
        return None

    def get_hist(self) -> List[DualRepFactorNode]:
        return sorted(self._nodes.values(), key=lambda n: n.timestamp)

    def get_sota(self) -> Optional[DualRepFactorNode]:
        decided = [n for n in self._nodes.values() if n.decision]
        if not decided:
            return None
        return max(decided, key=lambda n: n.timestamp)

    def query_code_experience(self, query_embedding=None, top_k: int = 5) -> List[DualRepFactorNode]:
        return []

    def get_node(self, node_id: str) -> Optional[DualRepFactorNode]:
        return self._nodes.get(node_id)

    def get_children(self, node_id: str) -> List[DualRepFactorNode]:
        node = self._nodes.get(node_id)
        if node is None:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def all_nodes(self) -> List[DualRepFactorNode]:
        return list(self._nodes.values())

    def active_nodes(self) -> List[DualRepFactorNode]:
        return [n for n in self._nodes.values() if n.status == "active"]

    def root_nodes(self) -> List[DualRepFactorNode]:
        return [n for n in self._nodes.values() if n.parent_id is None]

    def leaf_nodes(self) -> List[DualRepFactorNode]:
        return [n for n in self._nodes.values() if len(n.children_ids) == 0]

    def size(self) -> int:
        return len(self._nodes)

    def to_json(self) -> str:
        nodes_data = {nid: n.to_dict() for nid, n in self._nodes.items()}
        return json.dumps(nodes_data, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> FactorDAG:
        dag = cls()
        if not json_str:
            return dag
        nodes_data = json.loads(json_str)
        for nid, ndata in nodes_data.items():
            dag._nodes[nid] = DualRepFactorNode.from_dict(ndata)
        return dag

    @staticmethod
    def new_node_id() -> str:
        return f"f_{uuid.uuid4().hex[:12]}"
