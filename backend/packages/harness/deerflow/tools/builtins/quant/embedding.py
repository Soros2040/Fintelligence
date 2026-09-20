from __future__ import os
import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_EMBEDDING_MODEL = "qwen3-vl-embedding"
DASHSCOPE_TEXT_EMBEDDING_MODEL = "text-embedding-v3"
EMBEDDING_DIMENSIONS = 1024
NATIVE_EMBEDDING_DIMENSIONS = 2560


class DashScopeEmbedding:
    def __init__(
        self,
        api_key: str = DASHSCOPE_API_KEY,
        model: str = DASHSCOPE_EMBEDDING_MODEL,
        dimensions: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._use_native = model in ("qwen3-vl-embedding", "qwen2.5-vl-embedding")
        if self._use_native:
            self.dimensions = None
        else:
            self.dimensions = dimensions or EMBEDDING_DIMENSIONS

    def embed(self, text: str) -> Optional[np.ndarray]:
        if self._use_native:
            return self._embed_native(text)
        return self._embed_openai_compat(text)

    def embed_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        results = []
        for text in texts:
            results.append(self.embed(text))
        return results

    def _embed_native(self, text: str) -> Optional[np.ndarray]:
        try:
            import dashscope
            dashscope.api_key = self.api_key

            from dashscope import MultiModalEmbedding
            call_kwargs = {
                "model": self.model,
                "input": [{"text": text}],
                "embedding_type": "text",
            }
            if self.dimensions is not None:
                call_kwargs["dimensions"] = self.dimensions
            rsp = MultiModalEmbedding.call(**call_kwargs)

            if rsp.status_code == 200:
                embeddings = rsp.output.get("embeddings", [])
                if embeddings and len(embeddings) > 0:
                    emb_data = embeddings[0].get("embedding", embeddings[0].get("text_embedding", {}))
                    if isinstance(emb_data, dict):
                        emb_list = emb_data.get("embedding", [])
                    elif isinstance(emb_data, list):
                        emb_list = emb_data
                    else:
                        emb_list = []
                    if emb_list:
                        return np.array(emb_list, dtype=np.float32)

                if hasattr(rsp.output, "embeddings"):
                    emb_list = rsp.output["embeddings"]
                    if emb_list and len(emb_list) > 0:
                        emb = emb_list[0]
                        if isinstance(emb, dict):
                            text_emb = emb.get("text_embedding", {})
                            if isinstance(text_emb, dict):
                                vec = text_emb.get("embedding", [])
                            else:
                                vec = text_emb
                        else:
                            vec = emb
                        if isinstance(vec, (list, np.ndarray)) and len(vec) > 0:
                            return np.array(vec, dtype=np.float32)

                logger.warning(f"DashScope native embedding returned unexpected format: {type(rsp.output)}")
                return None
            else:
                logger.warning(f"DashScope native embedding failed: {rsp.code} - {rsp.message}")
                return None
        except ImportError:
            logger.warning("dashscope package not installed, falling back to text-embedding-v3")
            self.model = DASHSCOPE_TEXT_EMBEDDING_MODEL
            self._use_native = False
            return self._embed_openai_compat(text)
        except Exception as e:
            logger.warning(f"DashScope native embedding error: {e}")
            return None

    def _embed_openai_compat(self, text: str) -> Optional[np.ndarray]:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            create_kwargs = {
                "model": self.model,
                "input": text,
            }
            if self.dimensions is not None:
                create_kwargs["dimensions"] = self.dimensions
            response = client.embeddings.create(**create_kwargs)
            if response.data and len(response.data) > 0:
                return np.array(response.data[0].embedding, dtype=np.float32)
            return None
        except ImportError:
            logger.warning("openai package not installed, DashScope embedding unavailable")
            return None
        except Exception as e:
            logger.warning(f"DashScope OpenAI-compat embedding failed: {e}")
            return None


class InMemoryEmbeddingIndex:
    def __init__(self, embedder: Optional[DashScopeEmbedding] = None) -> None:
        self._embedder = embedder or DashScopeEmbedding()
        self._keys: list[str] = []
        self._embeddings: list[np.ndarray] = []

    def add(self, key: str, text: str) -> bool:
        emb = self._embedder.embed(text)
        if emb is None:
            return False
        self._keys.append(key)
        self._embeddings.append(emb)
        return True

    def add_with_embedding(self, key: str, embedding: np.ndarray) -> None:
        self._keys.append(key)
        self._embeddings.append(embedding)

    def query(self, query_text: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_emb = self._embedder.embed(query_text)
        if query_emb is None:
            return []
        return self.query_by_embedding(query_emb, top_k)

    def query_by_embedding(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._embeddings:
            return []
        embeddings_matrix = np.stack(self._embeddings)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = embeddings_matrix / (np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-8)
        similarities = emb_norms @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self._keys[i], float(similarities[i])) for i in top_indices]

    def size(self) -> int:
        return len(self._keys)

    def clear(self) -> None:
        self._keys.clear()
        self._embeddings.clear()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rebuild_embedding_index_from_dag(dag, embedder: Optional[DashScopeEmbedding] = None) -> InMemoryEmbeddingIndex:
    index = InMemoryEmbeddingIndex(embedder=embedder)
    for node in dag.all_nodes():
        if node.llm_explanation:
            key = f"factor_explain:{node.node_id}"
            index.add(key, node.llm_explanation)
        elif node.description:
            key = f"factor_desc:{node.node_id}"
            index.add(key, node.description)
    return index
