"""EL-TSBPO task similarity implementation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


PREPROCESSING = "pure-task-instruction-lasttoken-l2-v1"


def resolve_model(path):
    path = Path(path).expanduser().resolve()
    if (path / "config.json").exists():
        return path
    snapshots = sorted((path / "snapshots").glob("*/config.json"))
    if len(snapshots) != 1:
        raise ValueError(f"Cannot resolve one embedding snapshot under {path}")
    return snapshots[0].parent


def model_identity(path):
    path = resolve_model(path)
    records = []
    for file in sorted(path.iterdir()):
        if file.name in {"config.json", "tokenizer.json", "tokenizer_config.json"}:
            records.append([file.name, hashlib.sha256(file.read_bytes()).hexdigest()])
        elif file.suffix == ".safetensors":
            stat = file.stat()
            records.append([file.name, stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def text_digest(text):
    return hashlib.sha256(str(text).strip().encode()).hexdigest()


def prepare_task_vectors(texts, model_path, cache_dir, *, device="cuda", batch_size=64,
                         namespace="webshop-seed0-synthetic-1k"):
    import time
    start = time.perf_counter()
    folder = Path(cache_dir)
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(model=model_identity(model_path), preprocessing=PREPROCESSING,
                    namespace=str(namespace))
    index_path, vectors_path = folder / "index.json", folder / "vectors.npy"
    rows, old_vectors = [], None
    if index_path.exists() and vectors_path.exists():
        previous = json.loads(index_path.read_text())
        if previous["identity"] == identity:
            rows = previous["rows"]
            old_vectors = np.load(vectors_path)
    by_ref = {row["task_ref"]: i for i, row in enumerate(rows)}
    missing = [str(ref) for ref, text in texts.items()
               if str(ref) not in by_ref or rows[by_ref[str(ref)]]["text_sha256"] != text_digest(text)]
    timing = dict(tasks=len(texts), hits=len(texts)-len(missing), misses=len(missing),
                  forward_calls=0, device=device, preprocessing=PREPROCESSING)
    if missing:
        import torch
        from transformers import AutoModel, AutoTokenizer
        model_dir = resolve_model(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left")
        model = AutoModel.from_pretrained(model_dir, local_files_only=True,
                                         torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                                         attn_implementation="sdpa").to(device).eval()
        model.requires_grad_(False)
        arrays = []
        encode_start = time.perf_counter()
        with torch.inference_mode():
            for lo in range(0, len(missing), batch_size):
                chunk = [str(texts[ref]).strip() for ref in missing[lo:lo+batch_size]]
                lengths = tokenizer(chunk, truncation=False)["input_ids"]
                if any(len(ids) > 1024 for ids in lengths):
                    raise ValueError("Task instruction exceeds embedding limit; refusing silent truncation")
                batch = tokenizer(chunk, padding=True, return_tensors="pt").to(device)
                output = model(**batch, use_cache=False).last_hidden_state[:, -1].float()
                arrays.append(torch.nn.functional.normalize(output, dim=1).cpu().numpy())
                timing["forward_calls"] += 1
        new_vectors = np.concatenate(arrays)
        timing["encode_seconds"] = time.perf_counter()-encode_start
        dim = new_vectors.shape[1]
        values = [row.copy() for row in rows]
        vectors = [v for v in old_vectors] if old_vectors is not None else []
        for ref, vector in zip(missing, new_vectors):
            record = dict(task_ref=ref, text_sha256=text_digest(texts[ref]))
            if ref in by_ref:
                pos = by_ref[ref]
                values[pos], vectors[pos] = record, vector
            else:
                by_ref[ref] = len(values)
                values.append(record)
                vectors.append(vector)
        matrix = np.asarray(vectors, dtype=np.float32).reshape(-1, dim)

        with (folder / "vectors.npy.tmp").open("wb") as stream:
            np.save(stream, matrix)
        os.replace(folder / "vectors.npy.tmp", vectors_path)
        data = dict(identity=identity, rows=values, embedding_dim=dim,
                    model_path=str(model_dir))
        (folder / "index.json.tmp").write_text(json.dumps(data, ensure_ascii=False))
        os.replace(folder / "index.json.tmp", index_path)
        timing["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    timing["total_seconds"] = time.perf_counter()-start
    (folder / "prepare_result.json").write_text(json.dumps(timing, indent=2)+"\n")
    return timing


def validate_task_cache(texts, model_path, cache_dir, *, namespace):
    prior = TaskSimilarityPrior(cache_dir)
    expected = dict(model=model_identity(model_path), preprocessing=PREPROCESSING,
                    namespace=str(namespace))
    if prior.index.get("identity") != expected:
        raise RuntimeError("Task embedding cache identity does not match the model, preprocessing, or dataset")
    if len(prior.rows) != len(prior.index["rows"]):
        raise RuntimeError("Task embedding cache contains duplicate task references")
    shape = (len(prior.rows), int(prior.index["embedding_dim"]))
    if prior.vectors.ndim != 2 or prior.vectors.shape != shape or shape[1] <= 0:
        raise RuntimeError("Task embedding cache dimensions do not match its index")
    if not np.isfinite(prior.vectors).all() or not np.allclose(np.linalg.norm(prior.vectors, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Task embedding cache must contain finite L2-normalized vectors")
    prior.validate_texts(texts)


class TaskSimilarityPrior:

    def __init__(self, cache_dir, *, threshold=0.95, top_k=8, max_strength=4.0):
        if not 0.0 <= float(threshold) < 1.0:
            raise ValueError("Task similarity threshold must be in [0,1)")
        if int(top_k) < 1 or not math.isfinite(float(max_strength)) or float(max_strength) < 0:
            raise ValueError("Invalid root shared-prior limits")
        folder = Path(cache_dir)
        self.index = json.loads((folder / "index.json").read_text())
        self.vectors = np.load(folder / "vectors.npy", mmap_mode="r")
        self.rows = {row["task_ref"]: i for i, row in enumerate(self.index["rows"])}
        self.threshold, self.top_k, self.max_strength = float(threshold), int(top_k), float(max_strength)
        self._step = None
        self._answers = {}

    def validate_texts(self, texts):
        for ref, text in texts.items():
            pos = self.rows.get(str(ref))
            if pos is None or self.index["rows"][pos]["text_sha256"] != text_digest(text):
                raise RuntimeError(f"Missing/stale embedding for task {ref}; prepare task cache before training")

    def _refresh(self, memory, step):
        if self._step == int(step):
            return
        refs, positions, counts = [], [], []
        for ref, key in sorted(memory.root_state_keys.items()):
            if str(ref) not in self.rows:
                continue
            success, failure = memory.evidence_counts(key, step)
            mass = success + failure
            if mass <= 1e-8:
                continue

            scale = min(1.0, 6.0/mass)
            refs.append(str(ref))
            positions.append(self.rows[str(ref)])
            counts.append([success*scale, failure*scale])
        self._refs = np.asarray(refs, dtype=object)
        self._vectors = np.asarray(self.vectors[positions])
        self._counts = np.asarray(counts, dtype=np.float64).reshape(-1, 2)
        self._answers = {}
        self._step = int(step)

    def parameters(self, memory, ref, key, step):
        self._refresh(memory, step)
        ref = str(ref)
        if ref not in self.rows:
            raise RuntimeError(f"No frozen task vector for root {ref}")
        if ref not in self._answers:
            shared_s = shared_f = 0.0
            metadata = dict(neighbors=0.0, max_similarity=0.0, shared_mass=0.0,
                            semantic_success_mass=0.0, semantic_failure_mass=0.0)
            if len(self._refs):
                scores = self._vectors @ self.vectors[self.rows[ref]]
                eligible = np.flatnonzero((self._refs != ref) & (scores >= self.threshold))
                chosen = eligible[np.argsort(scores[eligible])[-self.top_k:]]
                if len(chosen):
                    weights = np.maximum(0.0, (scores[chosen]-self.threshold)/(1-self.threshold))**2
                    values = (weights[:, None]*self._counts[chosen]).sum(0)
                    mass = float(values.sum())
                    scale = min(1.0, self.max_strength/max(mass, 1e-12))
                    shared_s, shared_f = (values*scale).tolist()
                    metadata = dict(neighbors=float(len(chosen)), max_similarity=float(scores[chosen].max()),
                                    shared_mass=float(shared_s+shared_f),
                                    semantic_success_mass=float(shared_s),
                                    semantic_failure_mass=float(shared_f))
            self._answers[ref] = shared_s, shared_f, metadata
        shared_s, shared_f, metadata = self._answers[ref]
        success, failure = memory.evidence_counts(key, step)
        return memory.alpha+success+shared_s, memory.beta+failure+shared_f, dict(metadata)


def root_frontier_probability(memory, key, step, alpha, beta, group_size):
    fail = success = 1.0
    for i in range(max(1, int(group_size))):
        fail *= (beta+i)/(alpha+beta+i)
        success *= (alpha+i)/(alpha+beta+i)
    beta_mixed = max(0.0, 1.0-fail-success)
    own_s, own_f = memory.evidence_counts(key, step)
    own_mass = own_s+own_f
    weight = own_mass/(own_mass+memory.alpha+memory.beta)
    return weight*memory.dps_belief(key, step)[1] + (1-weight)*beta_mixed
