from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from deerflow.tools.builtins.quant.dag import DualRepFactorNode, FactorDAG

logger = logging.getLogger(__name__)

DEPTH_DECAY = 0.05
TIME_DECAY = 0.1
START_TIMES = 2


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    standardized = (values - values.mean()) / (values.std() + 1e-6)
    return _stable_sigmoid(standardized)


def _clip_prob(value: float) -> float:
    return float(np.clip(value, 1e-6, 1 - 1e-6))


class BayesianFactorRetriever:
    def __init__(
        self,
        depth_decay: float = DEPTH_DECAY,
        time_decay: float = TIME_DECAY,
        start_times: int = START_TIMES,
        use_res_correlation: bool = True,
        use_semantic_similarity: bool = True,
        use_edit_distance: bool = False,
        separate_leaf_non_leaf: bool = True,
    ) -> None:
        self.depth_decay = depth_decay
        self.time_decay = time_decay
        self.start_times = start_times
        self.use_res_correlation = use_res_correlation
        self.use_semantic_similarity = use_semantic_similarity
        self.use_edit_distance = use_edit_distance
        self.separate_leaf_non_leaf = separate_leaf_non_leaf

    def retrieve(self, dag: FactorDAG, top_k: int = 4) -> List[DualRepFactorNode]:
        if not isinstance(top_k, int) or top_k <= 0:
            top_k = 4
        active = dag.active_nodes()
        if not active:
            return []

        candidate_nodes = [n for n in active if (n.icir or 0.0) != 0.0 or (n.ic or 0.0) != 0.0]
        if not candidate_nodes:
            candidate_nodes = active

        num_nodes = len(candidate_nodes)
        if num_nodes <= top_k:
            for n in candidate_nodes:
                n.times_selected += 1
            return candidate_nodes

        icir_raw = np.array([abs(n.icir or 0.0) for n in candidate_nodes], dtype=float)
        finite_mask = np.isfinite(icir_raw)
        if finite_mask.any():
            mean_icir = icir_raw[finite_mask].mean()
            std_icir = icir_raw[finite_mask].std()
        else:
            mean_icir = 0.0
            std_icir = 1.0
        icir_logit = _stable_sigmoid((icir_raw - mean_icir) / (std_icir + 1e-6))

        depths = np.array([n.depth for n in candidate_nodes], dtype=float)
        times = np.array([n.times_selected for n in candidate_nodes], dtype=float)
        depth_decays = np.power(max(1 - self.depth_decay, 1e-6), depths)
        times_decays = np.power(max(1 - self.time_decay, 1e-6), np.maximum(times - self.start_times, 0))
        quality_prob = np.clip(icir_logit * depth_decays * times_decays, 1e-6, 1 - 1e-6)

        corr_matrix = self._compute_corr_matrix(candidate_nodes)

        semantic_matrix = None
        if self.use_semantic_similarity:
            semantic_matrix = self._compute_semantic_matrix(candidate_nodes)

        edit_matrix = None
        if self.use_edit_distance:
            edit_matrix = self._compute_edit_matrix(candidate_nodes)

        pool_quality_raw = np.ones(num_nodes, dtype=float)
        group_labels = np.array(["leaf"] * num_nodes, dtype=object)

        for pos, node in enumerate(candidate_nodes):
            children = dag.get_children(node.node_id)
            if children:
                pool_quality_raw[pos] = self._compute_non_leaf_quality(
                    pos, node, children, candidate_nodes, icir_raw, corr_matrix
                )
                group_labels[pos] = "non_leaf"
            else:
                pool_quality_raw[pos] = self._compute_leaf_quality(
                    pos, candidate_nodes, corr_matrix, semantic_matrix, edit_matrix
                )

        pool_quality_combined = _normalize_scores(pool_quality_raw.copy())
        pool_quality_combined = np.clip(pool_quality_combined, 1e-6, 1 - 1e-6)

        scores_combined = np.clip(quality_prob * pool_quality_combined, 1e-6, 1 - 1e-6)

        if not self.separate_leaf_non_leaf:
            ranking = np.argsort(-scores_combined)
            top_positions = ranking[:min(top_k, len(ranking))]
        else:
            half_k = max(1, top_k // 2)
            leaf_positions = [i for i, label in enumerate(group_labels) if label == "leaf"]
            non_leaf_positions = [i for i, label in enumerate(group_labels) if label == "non_leaf"]

            pool_quality_split = pool_quality_combined.copy()
            if leaf_positions:
                pool_quality_split[leaf_positions] = _normalize_scores(pool_quality_raw[leaf_positions])
            if non_leaf_positions:
                pool_quality_split[non_leaf_positions] = _normalize_scores(pool_quality_raw[non_leaf_positions])
            pool_quality_split = np.clip(pool_quality_split, 1e-6, 1 - 1e-6)

            scores_split = np.clip(quality_prob * pool_quality_split, 1e-6, 1 - 1e-6)

            leaf_sorted = sorted(leaf_positions, key=lambda p: scores_split[p], reverse=True)
            non_leaf_sorted = sorted(non_leaf_positions, key=lambda p: scores_split[p], reverse=True)

            selected_positions: list[int] = []
            selected_positions.extend(leaf_sorted[:min(half_k, len(leaf_sorted))])
            selected_positions.extend(non_leaf_sorted[:min(half_k, len(non_leaf_sorted))])
            selected_positions = list(dict.fromkeys(selected_positions))

            if len(selected_positions) < top_k:
                remaining = [
                    pos for pos in leaf_sorted[min(half_k, len(leaf_sorted)):]
                    + non_leaf_sorted[min(half_k, len(non_leaf_sorted)):]
                    if pos not in selected_positions
                ]
                remaining_sorted = sorted(remaining, key=lambda p: scores_split[p], reverse=True)
                selected_positions.extend(remaining_sorted[:top_k - len(selected_positions)])

            top_positions = selected_positions[:top_k]

        result = [candidate_nodes[pos] for pos in top_positions]
        for n in result:
            n.times_selected += 1
        return result

    def _compute_corr_matrix(self, nodes: List[DualRepFactorNode]) -> np.ndarray:
        n = len(nodes)
        corr_matrix = np.zeros((n, n), dtype=float)
        if not self.use_res_correlation:
            return corr_matrix
        for i in range(n):
            for j in range(i + 1, n):
                ni_ic, nj_ic = nodes[i].ic, nodes[j].ic
                if ni_ic != 0.0 and nj_ic != 0.0:
                    corr_val = abs(ni_ic * nj_ic) / (abs(ni_ic) * abs(nj_ic) + 1e-8)
                    corr_val = min(corr_val, 1.0)
                else:
                    corr_val = 0.0
                corr_matrix[i, j] = corr_matrix[j, i] = corr_val
        return corr_matrix

    def _compute_semantic_matrix(self, nodes: List[DualRepFactorNode]) -> Optional[np.ndarray]:
        try:
            from deerflow.tools.builtins.quant.embedding import DashScopeEmbedding
            embedder = DashScopeEmbedding()
            texts = []
            for n in nodes:
                text = n.llm_explanation or n.description or n.formula or n.node_id
                texts.append(text)
            embeddings = embedder.embed_batch(texts)
            valid_embs = [e for e in embeddings if e is not None]
            if len(valid_embs) < len(nodes):
                logger.warning(f"Only {len(valid_embs)}/{len(nodes)} embeddings succeeded")
                return None
            emb_matrix = np.stack(valid_embs)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
            emb_normed = emb_matrix / norms
            semantic_matrix = emb_normed @ emb_normed.T
            semantic_matrix = np.clip(semantic_matrix, -1.0, 1.0)
            return semantic_matrix
        except Exception as e:
            logger.warning(f"Semantic matrix computation failed: {e}")
            return None

    def _compute_edit_matrix(self, nodes: List[DualRepFactorNode]) -> Optional[np.ndarray]:
        try:
            from Levenshtein import distance as lev_distance
            n = len(nodes)
            edit_matrix = np.zeros((n, n), dtype=float)
            expr_strings = [n.formula or n.description or n.node_id for n in nodes]
            lengths = np.array([len(s) for s in expr_strings], dtype=float)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = lev_distance(expr_strings[i], expr_strings[j])
                    length_sum = lengths[i] + lengths[j]
                    ratio = dist / (length_sum + 1e-6)
                    edit_matrix[i, j] = edit_matrix[j, i] = _clip_prob(ratio)
            return edit_matrix
        except ImportError:
            return None

    def _compute_leaf_quality(
        self,
        pos: int,
        nodes: List[DualRepFactorNode],
        corr_matrix: np.ndarray,
        semantic_matrix: Optional[np.ndarray],
        edit_matrix: Optional[np.ndarray],
    ) -> float:
        others = [j for j in range(len(nodes)) if j != pos]

        corr_score = 1.0
        if self.use_res_correlation and others:
            corr_vals = corr_matrix[pos, others]
            corr_score = _clip_prob(1.0 - float(np.mean(corr_vals)))

        semantic_score = 1.0
        if semantic_matrix is not None and others:
            sims = semantic_matrix[pos, others]
            mapped = ((np.clip(sims, -1.0, 1.0) + 1.0) * 0.5).mean()
            semantic_score = _clip_prob(1.0 - float(mapped))

        edit_score = 1.0
        if edit_matrix is not None and others:
            edit_score = _clip_prob(float(edit_matrix[pos, others].mean()))

        return _clip_prob(corr_score * semantic_score * edit_score)

    def _compute_non_leaf_quality(
        self,
        pos: int,
        node: DualRepFactorNode,
        children: List[DualRepFactorNode],
        candidate_nodes: List[DualRepFactorNode],
        icir_raw: np.ndarray,
        corr_matrix: np.ndarray,
    ) -> float:
        if not children:
            return self._compute_leaf_quality(pos, candidate_nodes, corr_matrix, None, None)

        parent_icir = abs(icir_raw[pos])
        denom = max(parent_icir, 1e-6)

        child_icirs = []
        child_positions = []
        for child in children:
            for cp, cn in enumerate(candidate_nodes):
                if cn.node_id == child.node_id:
                    child_positions.append(cp)
                    child_icirs.append(abs(icir_raw[cp]))
                    break
            else:
                child_icirs.append(abs(child.icir))

        child_icirs = np.array(child_icirs, dtype=float)
        percent_gains = (child_icirs - parent_icir) / denom
        gain_scores = _normalize_scores(percent_gains)
        mean_gain = _clip_prob(float(gain_scores.mean()))

        parent_child_corrs = []
        for cp in child_positions:
            corr_val = float(np.clip(corr_matrix[pos, cp], 0.0, 1.0))
            parent_child_corrs.append(corr_val)

        parent_child_avg = float(np.mean(parent_child_corrs)) if parent_child_corrs else 0.0
        parent_child_avg = float(np.clip(parent_child_avg, 0.0, 1.0))

        sibling_corrs = []
        if len(child_positions) > 1:
            for i in range(len(child_positions)):
                for j in range(i + 1, len(child_positions)):
                    corr_val = float(np.clip(corr_matrix[child_positions[i], child_positions[j]], 0.0, 1.0))
                    sibling_corrs.append(corr_val)
        child_child_avg = float(np.mean(sibling_corrs)) if sibling_corrs else 0.0
        child_child_avg = float(np.clip(child_child_avg, 0.0, 1.0))

        sparsity_score = _clip_prob(1.0 - parent_child_avg * child_child_avg)
        return _clip_prob(mean_gain * sparsity_score)
