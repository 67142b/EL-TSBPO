"""EL-TSBPO branch similarity implementation."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time

import numpy as np


def state_texts(goal, observation):
    goal, observation = str(goal or '').strip(), str(observation or '').strip()
    if not goal or not observation:
        return None
    return f'Task:\n{goal}\nObservation:\n{observation}', f'Task:\n{goal}'


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _host_memory():
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024

    result = dict(raw_free=None, reclaimable=0, available=available)
    for limit, current in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                           ('/sys/fs/cgroup/memory/memory.limit_in_bytes',
                            '/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            cap = Path(limit).read_text().strip()
            if cap != 'max':
                remaining = max(0, int(cap) - int(Path(current).read_text()))
                stat_path = Path(current).with_name('memory.stat')
                try:
                    values = dict(line.split() for line in stat_path.read_text().splitlines())

                    prefix = 'total_' if 'total_inactive_file' in values else ''
                    inactive = int(values.get(prefix + 'inactive_file', 0))
                    dirty = int(values.get(prefix + 'file_dirty', values.get(prefix + 'dirty', 0)))
                    writeback = int(values.get(prefix + 'file_writeback', values.get(prefix + 'writeback', 0)))
                    reclaimable = max(0, inactive - dirty - writeback)
                except (OSError, ValueError):
                    reclaimable = 0
                effective = remaining + reclaimable
                available = effective if available is None else min(available, effective)
                result.update(raw_free=remaining, reclaimable=reclaimable, available=available)
        except (OSError, ValueError):
            pass
    return result


def _host_available():
    return _host_memory()['available']


class BranchSimilarity:
    def __init__(self, model_path, cache_dir, **options):
        from .task_similarity import model_identity, resolve_model
        self.options = dict(device='auto', batch_size=4, max_length=1024,
                            gpu_memory_gib=2.5, gpu_reserve_gib=4., cpu_threads=2,
                            min_host_free_gib=3., timeout_seconds=600,
                            threshold=.95, goal_threshold=.9, top_k=8,
                            max_strength=2., cache_entries=20000)
        self.options.update(options)
        validate_options(self.options)
        self.model = str(resolve_model(model_path))
        identity = json.dumps(dict(model=model_identity(self.model),
                                   preprocessing='goal-observation-head-tail-lasttoken-l2-v1',
                                   max_length=int(self.options['max_length'])), sort_keys=True)
        self.folder = Path(cache_dir) / hashlib.sha256(identity.encode()).hexdigest()[:20]
        self.folder.mkdir(parents=True, exist_ok=True)
        self.database = self.folder / 'vectors.sqlite3'
        with self._connect() as db:
            db.execute('PRAGMA auto_vacuum=FULL')
            db.execute('CREATE TABLE IF NOT EXISTS vectors '
                       '(key TEXT PRIMARY KEY, dim INTEGER NOT NULL, value BLOB NOT NULL, touched INTEGER NOT NULL)')
        self.stats = {}
        self.reset_stats()

    @contextmanager
    def _connect(self):


        db = sqlite3.connect(str(self.database), timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def reset_stats(self):
        self._failed_keys = set()
        self.stats = dict(cache_hits=0, cache_misses=0, encoded=0, encode_seconds=0.,
                          cpu_calls=0, cuda_calls=0, fallback_calls=0, skipped=0,
                          retry_suppressed=0,
                          truncated=0, cuda_peak_mib=0., missing_goal=0,
                          host_raw_free_gib=0., host_reclaimable_gib=0., host_available_gib=0.,
                          host_memory_skips=0, cuda_budget_mib=0., cuda_free_before_mib=0.,
                          cuda_oom_retries=0)

    def _read(self, keys):
        result = {}
        keys = list(set(keys))
        with self._connect() as db:
            for lo in range(0, len(keys), 400):
                chunk = keys[lo:lo + 400]
                rows = db.execute('SELECT key, dim, value FROM vectors WHERE key IN (' +
                                  ','.join('?' for _ in chunk) + ')', chunk)
                for key, dim, value in rows:
                    vector = np.frombuffer(value, dtype=np.float16).astype(np.float32)
                    if len(vector) != dim or not np.isfinite(vector).all():
                        continue
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        result[key] = vector / norm
        return result

    def _run_encoder(self, texts):
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix='encode-', dir=self.folder) as temporary:
            directory = Path(temporary)
            request = directory / 'request.json'
            request.write_text(json.dumps(dict(model=self.model, texts=texts,
                                                options=self.options)), encoding='utf-8')
            env = dict(os.environ)
            for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS']:
                env[name] = str(self.options['cpu_threads'])
            env['TOKENIZERS_PARALLELISM'] = 'false'
            env['HF_HUB_OFFLINE'] = '1'


            for name in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                         'PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF']:
                env.pop(name, None)
            device = self.options['device']
            host = _host_memory()
            for field, source in [('host_raw_free_gib', 'raw_free'),
                                  ('host_reclaimable_gib', 'reclaimable'),
                                  ('host_available_gib', 'available')]:
                self.stats[field] = float(host[source] or 0) / 2**30
            available = host['available']
            if available is not None and available < float(self.options['min_host_free_gib']) * 2**30:
                self.stats['host_memory_skips'] += 1
                print('[EL-TSBPO-DPS] semantic encoder skipped: low cgroup host memory '
                      f"(raw={self.stats['host_raw_free_gib']:.2f} GiB, "
                      f"clean_inactive_file={self.stats['host_reclaimable_gib']:.2f} GiB, "
                      f"available={self.stats['host_available_gib']:.2f} GiB)", flush=True)
            else:
                with (directory / 'worker.log').open('w') as log:
                    try:
                        proc = subprocess.run(
                            [sys.executable, str(Path(__file__).resolve()), str(request), device],
                            env=env, stdout=log, stderr=subprocess.STDOUT,
                            timeout=float(self.options['timeout_seconds']), check=False)
                        ok = proc.returncode == 0
                    except subprocess.TimeoutExpired:
                        ok = False
                if ok:
                    info = json.loads((directory / 'result.json').read_text())
                    vectors = np.load(directory / 'vectors.npy', allow_pickle=False)
                    if vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
                        raise ValueError('Invalid semantic encoder result')
                    self.stats[info['device'] + '_calls'] += 1
                    self.stats['truncated'] += info['truncated']
                    self.stats['cuda_peak_mib'] = max(self.stats['cuda_peak_mib'], info['cuda_peak_mib'])
                    for field in ['cuda_budget_mib', 'cuda_free_before_mib']:
                        self.stats[field] = max(self.stats[field], info.get(field, 0.))
                    self.stats['cuda_oom_retries'] += info.get('cuda_oom_retries', 0)
                    self.stats['encoded'] += len(texts)
                    self.stats['encode_seconds'] += time.perf_counter() - started
                    print(f"[EL-TSBPO-DPS] semantic encoder device={info['device']} texts={len(texts)} "
                          f"seconds={time.perf_counter()-started:.2f} "
                          f"reason={info.get('device_reason', 'legacy')} "
                          f"cuda_budget_mib={info.get('cuda_budget_mib', 0.):.0f} "
                          f"cuda_peak_mib={info['cuda_peak_mib']:.0f}", flush=True)
                    return vectors
                self.stats['fallback_calls'] += 1
                tail = (directory / 'worker.log').read_text(errors='replace')[-1200:]
                status_path = directory / 'worker_status.json'
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                print(f'[EL-TSBPO-DPS] semantic encoder {device} failed; '
                      f"using exact evidence only (selected_gpu={status.get('cuda_visible_device_selected')}, "
                      f"reason={status.get('device_reason')}, status={status.get('status')}): {tail}",
                      flush=True)
        self.stats['encode_seconds'] += time.perf_counter() - started
        self.stats['skipped'] += len(texts)
        return None

    def register(self, memory, descriptors):
        texts, bindings = {}, {}
        live_states = {p['state'] for p in memory.pools.values()}
        for state, goal, observation in descriptors:
            if state not in live_states:
                continue
            pair = state_texts(goal, observation)
            if pair is None:
                self.stats['missing_goal'] += 1
                continue
            state_text, goal_text = pair
            keys = dict(state=_digest(state_text), goal=_digest(goal_text))
            old = memory.semantic_keys.get(state)
            if old is not None and old != keys:
                raise ValueError('Exact state identity changed its semantic task/observation')
            bindings[state] = keys
            texts[keys['state']], texts[keys['goal']] = state_text, goal_text
        cached = self._read(texts)
        self.stats['cache_hits'] += len(cached)
        missing = {key: text for key, text in texts.items() if key not in cached}
        self.stats['cache_misses'] += len(missing)
        pending = {key: text for key, text in missing.items() if key not in self._failed_keys}
        self.stats['retry_suppressed'] += len(missing) - len(pending)
        if pending:
            vectors = self._run_encoder(list(pending.values()))
            if vectors is not None:
                now = time.time_ns()
                with self._connect() as db:
                    db.executemany('INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)',
                                   [(key, len(v), np.asarray(v, dtype=np.float16).tobytes(), now)
                                    for key, v in zip(pending, vectors)])
                cached.update(dict(zip(pending, vectors)))
            else:
                self._failed_keys.update(pending)

        for state, keys in bindings.items():
            if all(key in cached for key in keys.values()):
                memory.semantic_keys[state] = keys
        with self._connect() as db:
            now = time.time_ns()
            db.executemany('UPDATE vectors SET touched=? WHERE key=?', [(now, key) for key in cached])
            db.execute('DELETE FROM vectors WHERE key IN (SELECT key FROM vectors '
                       'ORDER BY touched DESC, key LIMIT -1 OFFSET ?)', (int(self.options['cache_entries']),))

    def prepare(self, memory, step, decay):
        return SemanticDonors(memory, self._read(
            key for keys in memory.semantic_keys.values() for key in keys.values()),
            step, decay, **{key: self.options[key] for key in
                           ['threshold', 'goal_threshold', 'top_k', 'max_strength']})


def validate_options(options):
    if options['device'] not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Branch embedding device must be auto, cpu or cuda')
    for key in ['threshold', 'goal_threshold']:
        if not math.isfinite(float(options[key])) or not 0 <= float(options[key]) < 1:
            raise ValueError(f'Invalid branch semantic {key}')
    for key in ['batch_size', 'cpu_threads', 'top_k', 'cache_entries']:
        if int(options[key]) != float(options[key]) or int(options[key]) < 1:
            raise ValueError(f'Invalid branch embedding {key}')
    if not 64 <= int(options['max_length']) <= 4096:
        raise ValueError('Branch embedding max_length must be in [64, 4096]')
    for key in ['gpu_memory_gib', 'gpu_reserve_gib', 'min_host_free_gib', 'timeout_seconds']:
        if not math.isfinite(float(options[key])) or float(options[key]) <= 0:
            raise ValueError(f'Invalid branch embedding {key}')
    if not math.isfinite(float(options['max_strength'])) or float(options['max_strength']) < 0:
        raise ValueError('Invalid branch semantic max_strength')


class SemanticDonors:
    def __init__(self, memory, vectors, step, decay, *, threshold=.95,
                 goal_threshold=.9, top_k=8, max_strength=2.):
        self.memory, self.vectors = memory, vectors
        self.step, self.decay = int(step), float(decay)
        self.threshold, self.goal_threshold = float(threshold), float(goal_threshold)
        self.top_k, self.max_strength = int(top_k), float(max_strength)
        self.by_depth = {}


        ordered = sorted(memory.pools.values(),
                         key=lambda p: (-max(int(s) for s in p['buckets']), p['state'], p['depth']))
        indexed = 0
        for pool in ordered:
            keys = memory.semantic_keys.get(pool['state'], {})
            if not keys or not all(k in vectors for k in keys.values()):
                continue
            if any(int(s) < self.step and rows for s, rows in pool['buckets'].items()):
                self.by_depth.setdefault(int(pool['depth']), []).append(pool)
                indexed += 1
                if indexed >= 20000:
                    break
        self.index = {}
        for depth, pools in self.by_depth.items():
            self.index[depth] = (
                np.stack([vectors[memory.semantic_keys[p['state']]['state']] for p in pools]),
                np.stack([vectors[memory.semantic_keys[p['state']]['goal']] for p in pools]))

    def donors(self, state, depth):
        keys = self.memory.semantic_keys.get(str(state), {})
        empty = dict(semantic_neighbors=0, semantic_mass=0., semantic_max_similarity=0.,
                     semantic_clusters=0)
        if not keys or not all(k in self.vectors for k in keys.values()) or depth not in self.index:
            return [], empty
        pools = self.by_depth[depth]
        sv, gv = self.index[depth]
        similarities = np.clip(sv @ self.vectors[keys['state']], -1., 1.)
        goals = np.clip(gv @ self.vectors[keys['goal']], -1., 1.)
        eligible = [i for i, pool in enumerate(pools) if pool['state'] != state
                    and similarities[i] > self.threshold and goals[i] > self.goal_threshold]
        eligible.sort(key=lambda i: (-float(similarities[i]), pools[i]['state']))
        own = self.memory.pools.get(f'{state}:depth={depth}', {})
        used = {c for s, rows in own.get('buckets', {}).items() if int(s) <= self.step for c in rows}
        result, selected, max_similarity, sources = [], set(), 0., {}
        for i in eligible[:self.top_k]:
            pool, cosine = pools[i], float(similarities[i])
            kernel = ((cosine-self.threshold)/(1-self.threshold))**2
            kernel *= ((float(goals[i])-self.goal_threshold)/(1-self.goal_threshold))**2
            for source, rows in sorted(pool['buckets'].items(), key=lambda x: int(x[0])):
                age = self.step-int(source)
                if age <= 0:
                    continue
                for cluster, row in sorted(rows.items()):
                    if cluster in used:
                        continue
                    used.add(cluster)
                    total = float(row['success']+row['failure'])
                    if total <= 0:
                        continue
                    factor = kernel * self.decay**age / max(1., total)
                    result.append((cluster, row, factor))
                    selected.add(pool['state'])
                    summary = sources.setdefault(pool['state'], dict(
                        state=pool['state'], task=pool['task'], depth=int(depth),
                        similarity=cosine, goal_similarity=float(goals[i]), mass=0., clusters=0))
                    summary['mass'] += factor * total
                    summary['clusters'] += 1
                    max_similarity = max(max_similarity, cosine)
        mass = sum(f * (r['success']+r['failure']) for _, r, f in result)
        scale = min(1., self.max_strength/mass) if mass else 0.
        result = [(c, r, f*scale) for c, r, f in result if f*scale > 0]
        for source in sources.values():
            source['mass'] *= scale
        return result, dict(semantic_neighbors=len(selected) if result else 0,
                            semantic_mass=mass*scale, semantic_max_similarity=max_similarity,
                            semantic_clusters=len(result), semantic_sources=list(sources.values()))

    def diagnostics(self):
        return dict(semantic_indexed_states=sum(len(p) for p in self.by_depth.values()),
                    semantic_cached_vectors=len(self.vectors), semantic_threshold=self.threshold,
                    semantic_goal_threshold=self.goal_threshold,
                    semantic_max_strength=self.max_strength)


def _minimum_cuda_budget(model_path, options):
    files = list(Path(model_path).glob('*.safetensors'))
    weights = 0
    for path in files:
        with path.open('rb') as handle:
            size = int.from_bytes(handle.read(8), 'little')
            if not 0 < size <= 16 * 2**20:
                raise ValueError('Invalid safetensors header size')
            header = json.loads(handle.read(size))
        for key, tensor in header.items():
            if key != '__metadata__':

                elements = math.prod(tensor['shape'])
                dtype = tensor['dtype']
                width = 2 if dtype.startswith(('F', 'BF')) else {
                    'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1
                }.get(dtype, 8)
                weights += elements * width

    if not files:
        return float(options['gpu_memory_gib']) * 2**30
    config = json.loads((Path(model_path) / 'config.json').read_text())
    activation = (int(options.get('batch_size', 4)) * int(options.get('max_length', 1024))
                  * int(config['hidden_size']) * 2 * 16)
    return weights + max(512 * 2**20, activation)


def _cuda_budget(options, free_bytes, minimum, *, before_context=False):
    context_margin = 384 * 2**20 if before_context else 0
    budget = min(float(options['gpu_memory_gib']) * 2**30,
                 free_bytes - float(options['gpu_reserve_gib']) * 2**30 - context_margin)
    return int(budget) if budget >= minimum else 0


def _encoder_event(request_path, info):
    local = request_path.parent / 'worker_status.json'
    previous = json.loads(local.read_text()) if local.exists() else {}
    event = dict(previous, pid=os.getpid(), time=time.time(), **info)
    for path in (local, request_path.parent.parent / 'encoder_last.json'):
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(event))
        os.replace(temporary, path)


def _encoder_candidate_devices():
    """The launcher lists physical training GPUs; Ray may mask its CPU collector."""
    visible = (os.environ.get('EL_TSBPO_BRANCH_EMBEDDING_CUDA_VISIBLE_DEVICES')
               or os.environ.get('EL_TSBPO_CUDA_VISIBLE_DEVICES')
               or os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    if visible.strip() == '-1':
        return []
    return list(dict.fromkeys(part.strip() for part in visible.split(',') if part.strip()))


def _select_encoder_gpu(model_path, options):
    minimum = _minimum_cuda_budget(model_path, options)
    candidates = []
    for gpu in _encoder_candidate_devices():
        try:
            probe = subprocess.run(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
                                    '--format=csv,noheader,nounits'],
                                   text=True, capture_output=True, check=False, timeout=5)
            if probe.returncode == 0:
                free = float(probe.stdout.strip()) * 2**20
                if _cuda_budget(options, free, minimum, before_context=True):
                    candidates.append((free, gpu))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    if not candidates:
        return None, 0., minimum
    free, gpu = max(candidates, key=lambda item: item[0])
    return gpu, free, minimum


def _encode_worker(request_path, requested_device):
    request_path = Path(request_path)
    request = json.loads(request_path.read_text())
    opts = request['options']
    device = 'cpu'
    info = dict(device='cpu', device_reason='explicit_cpu', cuda_free_before_mib=0.,
                cuda_budget_mib=0., cuda_oom_retries=0, status='selecting',
                texts=len(request['texts']))
    minimum = None
    if requested_device != 'cpu':
        info['cuda_visible_devices_inherited'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        visible, free, minimum = _select_encoder_gpu(request['model'], opts)
        info['cuda_visible_device_selected'] = visible
        if visible:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible
            info['cuda_free_before_mib'] = free / 2**20
            device = 'cuda'
            info['device_reason'] = 'gpu_fits_bounded_budget'
        else:
            info['device_reason'] = 'no_training_gpu_with_headroom'
    info['device'] = device
    _encoder_event(request_path, info)
    if requested_device != 'cpu' and device != 'cuda':
        raise RuntimeError('No training GPU has enough free memory for the semantic encoder')
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(int(opts['cpu_threads']))
    torch.set_num_interop_threads(1)
    if device == 'cuda':
        total = torch.cuda.get_device_properties(0).total_memory
        free, _ = torch.cuda.mem_get_info()
        budget = _cuda_budget(opts, free, minimum)
        if not budget:
            raise RuntimeError('GPU headroom changed after encoder CUDA context creation')
        torch.cuda.set_per_process_memory_fraction(min(.95, budget / total), 0)
        info['cuda_budget_mib'] = budget / 2**20
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(request['model'], local_files_only=True, padding_side='left')
    kwargs = dict(local_files_only=True, dtype=torch.bfloat16,
                  attn_implementation='sdpa', low_cpu_mem_usage=True)
    if device == 'cuda':

        kwargs['device_map'] = {'': 0}
    model = AutoModel.from_pretrained(request['model'], **kwargs)
    if device == 'cpu':
        model = model.to(device)
    model = model.eval().requires_grad_(False)
    vectors, truncated = None, 0
    limit = int(opts['max_length'])
    encoded = tokenizer(request['texts'], padding=False, truncation=False)['input_ids']
    clipped = []
    for ids in encoded:
        if len(ids) > limit:
            ids = ids[:limit//2] + ids[-(limit-limit//2):]
            truncated += 1
        clipped.append(dict(input_ids=ids, attention_mask=[1]*len(ids)))

    order = sorted(range(len(clipped)), key=lambda i: len(clipped[i]['input_ids']))
    batch_size = int(opts['batch_size'])
    start = 0
    with torch.inference_mode():
        while start < len(order):
            indices = order[start:start+batch_size]
            inputs = output = None
            try:
                inputs = tokenizer.pad([clipped[i] for i in indices], padding=True,
                                       return_tensors='pt').to(device)
                output = model(**inputs, use_cache=False).last_hidden_state[:, -1].float()
                batch = torch.nn.functional.normalize(output, dim=1).cpu().numpy()
            except torch.cuda.OutOfMemoryError:
                if device != 'cuda' or batch_size == 1:
                    raise
                inputs = output = None
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                info['cuda_oom_retries'] += 1
                continue
            if vectors is None:
                vectors = np.empty((len(order), batch.shape[1]), dtype=batch.dtype)
            vectors[indices] = batch
            del inputs, output, batch
            start += len(indices)
    np.save(request_path.parent / 'vectors.npy', vectors, allow_pickle=False)
    info.update(status='completed', truncated=truncated, batch_size=batch_size,
                cuda_peak_mib=torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else 0.,
                cuda_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20 if device == 'cuda' else 0.)
    (request_path.parent / 'result.json').write_text(json.dumps(info))
    _encoder_event(request_path, info)


    del model
    for path in Path(request['model']).glob('*.safetensors'):
        try:
            with path.open('rb') as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass


if __name__ == '__main__':
    import fcntl
    request_path = Path(sys.argv[1])


    with (request_path.parent.parent / 'encoder.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _encode_worker(request_path, sys.argv[2])
        except Exception as exc:
            _encoder_event(request_path, dict(status='failed', error=str(exc)))
            raise
