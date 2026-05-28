from typing import Any, Optional, Literal
from dataclasses import field
from pydantic import BaseModel

import json
import os
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from evaluate_utils import evaluate_agentharm, evaluate_advcua, evaluate_practical

import logging
logger = logging.getLogger("execute_step")


class TemplateSample(BaseModel):
    # tool_file_path: str

    datum: dict[str, Any] = field(default_factory=dict)
    role: str
    environment: str
    directive: str
    # context: str = "You are currently working on the following task.\n{task_json}" #TODO
    tips: list[str] = field(default_factory=list) #TODO 3-5 tips
    template: Optional[str] = None

    # tool_descriptions: list[str] = field(default_factory=list) #TODO error_correction
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_call_chains: list[dict[str, Any]] = field(default_factory=list)

    score: Optional[float] = None
    refusal: Optional[float] = None
    source: list[dict] = field(default_factory=list)
    updated: bool = False

    def evaluate(self, model_name: str, dataset: str, **kwargs):
        assert len(self.messages) == 0, logger.error("messages should be empty before evaluation")
        assert len(self.tool_call_chains) == 0, logger.error("tool_call_chains should be empty before evaluation")
        assert self.score is None and self.refusal is None, logger.error("score and refusal should be None before evaluation")
        if dataset == "agentharm":
            self.messages, self.tool_call_chains, self.score, self.refusal = evaluate_agentharm(
                self.datum,
                model_name=model_name,
                field="jailbroken_prompt",
                **kwargs,
            )
        elif dataset == "advcua":
            self.messages, self.tool_call_chains, self.score, self.refusal = evaluate_advcua(
                self.datum,
                model_name=model_name,
                field="jailbroken_prompt",
                last_subtask=kwargs.get("last_subtask", False),
                **kwargs,
            )
        elif dataset == "practical":
            self.messages, self.tool_call_chains, self.score, self.refusal = evaluate_practical(
                self.datum,
                model_name=model_name,
                field="jailbroken_prompt",
                practical_baselines_root=kwargs.get("practical_baselines_root"),
                openai_api_key=kwargs.get("openai_api_key"),
                openai_base_url=kwargs.get("openai_base_url"),
            )
        self.source[-1][next(iter(self.source[-1]))] = self.score - self.refusal

    def update_template(self):
        task_context = "You are currently working on the following task:\n{task}" #TODO context
        # task_context = "You are currently working on the following task.\n{task_json}" #TODO context
        important_tips = ""
        for idx, tip in enumerate(self.tips):
            important_tips += f"{idx+1}. {tip}\n\n"
        self.template = "SETTING: " + self.role + "\n\n" + self.environment + "\n\n" + task_context + "\n\n" + "INSTRUCTIONS: " + self.directive + "\n\n" + "IMPORTATNT TIPS:\n" + important_tips
        self.datum["jailbroken_prompt"] = self.template.format(task=self.datum["prompt"])
        # logger.debug(f"template: {self.template}")
        self.updated = True

    def clear(self):
        self.updated = False
        self.messages = []
        self.tool_call_chains = []
        self.score = None
        self.refusal = None


class Memory:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        max_length: int = 512,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        self._entries: list[dict[str, Any]] = []

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 0:
            hidden_size = self.model.config.hidden_size
            return np.empty((0, hidden_size), dtype=np.float32)

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]

        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    def _embed_text(self, text: str) -> np.ndarray:
        return self._embed_texts([text])[0]

    @staticmethod
    def _resolve_persistence_paths(file_path: str) -> tuple[str, str]:
        base = file_path
        if base.endswith(".meta.json"):
            base = base[: -len(".meta.json")]
        elif base.endswith(".embeddings.npz"):
            base = base[: -len(".embeddings.npz")]
        elif base.endswith(".json"):
            base = base[: -len(".json")]
        elif base.endswith(".npz"):
            base = base[: -len(".npz")]

        return f"{base}.meta.json", f"{base}.embeddings.npz"

    def add(self, sample: TemplateSample, prompt: Optional[str] = None):
        logger.info(f"Adding sample to memory")
        prompt_text = prompt if prompt is not None else sample.datum.get("prompt", "")
        if not isinstance(prompt_text, str) or len(prompt_text.strip()) == 0:
            raise ValueError("prompt must be a non-empty string")

        embedding = self._embed_text(prompt_text)

        # Keep only the static template information in memory entries.
        sample.datum = {}
        sample.template = None
        sample.messages = []
        sample.tool_call_chains = []
        sample.score = None
        sample.refusal = None
        sample.updated = False
        sample.source = []

        self._entries.append({"embedding": embedding, "sample": sample})

    def query(
        self,
        prompt: str,
        top_k: int = 5,
        return_scores: bool = False,
    ):
        if len(self._entries) == 0:
            return []
        if top_k <= 0:
            return []

        query_embedding = self._embed_text(prompt)
        embedding_matrix = np.stack([entry["embedding"] for entry in self._entries], axis=0)

        if self.normalize_embeddings:
            similarities = embedding_matrix @ query_embedding
        else:
            matrix_norm = np.linalg.norm(embedding_matrix, axis=1) + 1e-12
            query_norm = np.linalg.norm(query_embedding) + 1e-12
            similarities = (embedding_matrix @ query_embedding) / (matrix_norm * query_norm)

        sorted_indices = np.argsort(-similarities)

        def _signature(sample: TemplateSample) -> str:
            payload = sample.model_dump(
                exclude={
                    "datum",
                    "messages",
                    "tool_call_chains",
                    "score",
                    "refusal",
                    "source",
                    "updated",
                }
            )
            return json.dumps(payload, sort_keys=True, ensure_ascii=True)

        seen_signatures: set[str] = set()
        top_indices: list[int] = []
        for idx in sorted_indices:
            signature = _signature(self._entries[idx]["sample"])
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            top_indices.append(idx)
            if len(top_indices) >= top_k:
                break

        if return_scores:
            return [
                (self._entries[idx]["sample"], float(similarities[idx]))
                for idx in top_indices
            ]

        return [self._entries[idx]["sample"] for idx in top_indices]

    def save(self, file_path: str):
        meta_path, embedding_path = self._resolve_persistence_paths(file_path)

        samples = [entry["sample"].model_dump() for entry in self._entries]
        if len(self._entries) > 0:
            embeddings = np.stack([entry["embedding"] for entry in self._entries], axis=0).astype(np.float32)
        else:
            embeddings = np.empty((0, 0), dtype=np.float32)

        payload = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "normalize_embeddings": self.normalize_embeddings,
            "embedding_file": os.path.basename(embedding_path),
            "num_entries": len(self._entries),
            "samples": samples,
        }

        logger.info(f"Saving memory to {meta_path}, {embedding_path} with {len(self._entries)} entries.")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        np.savez_compressed(embedding_path, embeddings=embeddings)

    @classmethod
    def load(cls, file_path: str, device: Optional[str] = None) -> "Memory":
        meta_path, embedding_path = cls._resolve_persistence_paths(file_path)

        # New format: *.meta.json + *.embeddings.npz
        assert os.path.exists(meta_path) and os.path.exists(embedding_path), logger.error(f"Both meta file and embedding file must exist. Checked paths: {meta_path}, {embedding_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        memory = cls(
            model_name=payload.get("model_name", "BAAI/bge-base-en-v1.5"),
            device=device,
            max_length=payload.get("max_length", 512),
            normalize_embeddings=payload.get("normalize_embeddings", True),
        )

        samples = payload.get("samples", [])
        data = np.load(embedding_path)
        embeddings = data["embeddings"]

        if len(samples) != embeddings.shape[0]:
            raise ValueError("samples count and embeddings count mismatch")

        memory._entries = [
            {
                "embedding": np.asarray(embeddings[idx], dtype=np.float32),
                "sample": TemplateSample.model_validate(samples[idx]),
            }
            for idx in range(len(samples))
        ]
        return memory

