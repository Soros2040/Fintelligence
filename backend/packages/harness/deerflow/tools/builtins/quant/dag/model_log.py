from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelExperiment:
    experiment_id: str
    timestamp: float = 0.0
    hypothesis: str = ""
    architecture: str = ""
    code: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    feedback: Optional[str] = None
    decision: bool = False
    code_experience: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "hypothesis": self.hypothesis,
            "architecture": self.architecture,
            "code": self.code,
            "metrics": self.metrics,
            "feedback": self.feedback,
            "decision": self.decision,
            "code_experience": self.code_experience,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModelExperiment:
        return cls(
            experiment_id=d["experiment_id"],
            timestamp=d.get("timestamp", 0.0),
            hypothesis=d.get("hypothesis", ""),
            architecture=d.get("architecture", ""),
            code=d.get("code", ""),
            metrics=d.get("metrics", {}),
            feedback=d.get("feedback"),
            decision=d.get("decision", False),
            code_experience=d.get("code_experience", {}),
        )


class ModelExperimentLog:
    def __init__(self) -> None:
        self.experiments: List[ModelExperiment] = []

    def add_model(self, experiment: ModelExperiment) -> None:
        if experiment.timestamp == 0.0:
            experiment.timestamp = time.time()
        self.experiments.append(experiment)

    def get_sota_model(self) -> Optional[ModelExperiment]:
        decided = [e for e in self.experiments if e.decision]
        if not decided:
            return self.experiments[-1] if self.experiments else None
        return max(decided, key=lambda e: e.timestamp)

    def update_sota(self, experiment_id: str) -> None:
        for e in self.experiments:
            if e.experiment_id == experiment_id:
                e.decision = True
                break

    def to_json(self) -> str:
        return json.dumps([e.to_dict() for e in self.experiments], ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> ModelExperimentLog:
        log = cls()
        if not json_str:
            return log
        data = json.loads(json_str)
        for d in data:
            log.experiments.append(ModelExperiment.from_dict(d))
        return log

    @staticmethod
    def new_experiment_id() -> str:
        return f"m_{uuid.uuid4().hex[:12]}"
