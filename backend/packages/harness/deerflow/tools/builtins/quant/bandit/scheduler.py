from __future__ import annotations

import json
import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

N_FEATURES = 9


class BanditScheduler:
    def __init__(
        self,
        n_features: int = N_FEATURES,
        prior_var: float = 10.0,
        noise_var: float = 0.5,
    ) -> None:
        self.n_features = n_features
        self.noise_var = noise_var
        self.mean = {
            "factor": np.zeros(n_features),
            "model": np.zeros(n_features),
        }
        self.precision = {
            "factor": np.eye(n_features) / prior_var,
            "model": np.eye(n_features) / prior_var,
        }
        self.history: list[dict] = []

    def _sample_reward(self, arm: str, x: np.ndarray) -> float:
        P = self.precision[arm]
        P = 0.5 * (P + P.T)
        eps = 1e-6
        try:
            cov = np.linalg.inv(P + eps * np.eye(self.n_features))
            L = np.linalg.cholesky(cov)
            z = np.random.randn(self.n_features)
            w_sample = self.mean[arm] + L @ z
        except np.linalg.LinAlgError:
            w_sample = self.mean[arm]
        return float(np.dot(w_sample, x))

    def decide(self, state_vector: list[float] | np.ndarray) -> str:
        state = np.asarray(state_vector, dtype=float)
        if state.shape[0] != self.n_features:
            logger.warning(f"State vector size {state.shape[0]} != {self.n_features}, padding/truncating")
            padded = np.zeros(self.n_features)
            padded[: min(len(state), self.n_features)] = state[: self.n_features]
            state = padded
        scores = {arm: self._sample_reward(arm, state) for arm in ("factor", "model")}
        return max(scores, key=scores.get)

    def record(self, action: str, state_vector: list[float] | np.ndarray, reward: float) -> None:
        state = np.asarray(state_vector, dtype=float)
        arm = action if action in ("factor", "model") else "factor"
        P = self.precision[arm]
        P += np.outer(state, state) / self.noise_var
        self.precision[arm] = P
        self.mean[arm] = np.linalg.solve(P, P @ self.mean[arm] + (reward / self.noise_var) * state)
        self.history.append({"action": action, "reward": reward})

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "noise_var": self.noise_var,
            "mean_factor": self.mean["factor"].tolist(),
            "mean_model": self.mean["model"].tolist(),
            "precision_factor": self.precision["factor"].tolist(),
            "precision_model": self.precision["model"].tolist(),
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BanditScheduler:
        prior_var = 10.0
        scheduler = cls(n_features=d.get("n_features", N_FEATURES), prior_var=prior_var, noise_var=d.get("noise_var", 0.5))
        scheduler.mean["factor"] = np.array(d.get("mean_factor", d.get("mu_factor", scheduler.mean["factor"].tolist())))
        scheduler.mean["model"] = np.array(d.get("mean_model", d.get("mu_model", scheduler.mean["model"].tolist())))
        if "precision_factor" in d:
            scheduler.precision["factor"] = np.array(d["precision_factor"])
            scheduler.precision["model"] = np.array(d["precision_model"])
        elif "Sigma_factor" in d:
            for arm, key in [("factor", "Sigma_factor"), ("model", "Sigma_model")]:
                sigma = np.array(d[key])
                try:
                    scheduler.precision[arm] = np.linalg.inv(sigma)
                except np.linalg.LinAlgError:
                    scheduler.precision[arm] = np.eye(scheduler.n_features) / prior_var
        scheduler.history = d.get("history", [])
        return scheduler

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> BanditScheduler:
        return cls.from_dict(json.loads(json_str))


def build_state_vector(metrics: dict | None) -> list[float]:
    if metrics is None:
        return [0.0] * N_FEATURES
    mdd_val = metrics.get("mdd", 0.0) or 0.0
    return [
        metrics.get("ic", 0.0) or 0.0,
        metrics.get("icir", 0.0) or 0.0,
        metrics.get("rank_ic", 0.0) or 0.0,
        metrics.get("rank_icir", 0.0) or 0.0,
        metrics.get("arr", 0.0) or 0.0,
        metrics.get("ir", 0.0) or 0.0,
        -mdd_val,
        metrics.get("sharpe", 0.0) or 0.0,
        metrics.get("calmar", 0.0) or 0.0,
    ]


REWARD_WEIGHTS = np.array([0.10, 0.10, 0.05, 0.05, 0.25, 0.15, 0.10, 0.15, 0.05])


def compute_reward(state_vector: list[float] | np.ndarray) -> float:
    return float(np.asarray(state_vector, dtype=float) @ REWARD_WEIGHTS)
