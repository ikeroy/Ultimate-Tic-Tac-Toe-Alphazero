import glob
import io
import json
import os
import signal
import shutil
import sys
import tempfile
from collections import deque, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import random
import time
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader, IterableDataset
import threading
import tqdm
import queue
import numpy as np
import gc
import zstandard as zstd
import alphazero
from neural_net import get_device, get_storage_dtype, to_half_safe
from config import load_config, find_latest_checkpoint
from tracy_utils import tracy_zone, tracy_thread, TracyZone, tracy_frame
import frozen_eval

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_project_root_importable():
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)


def _fmt_pct(x):
    pct = x * 100
    return f"{int(pct)}%" if pct == int(pct) else f"{pct:.1f}%"


def _make_game_instance(config, probs=None):
    """Create a fresh UTTT game state."""
    return config.Game()


def _is_unified_game(config):
    """UTTT has no runtime game variants."""
    return False


UNIFIED_VARIANT_NAMES = []


def _compute_unified_probs(config, prev_sample_counts=None):
    return None


def _compute_gating_probs(config):
    return None


def _count_variant_samples(paths, iteration):
    return None


def _collect_variant_metrics(pm, include_fast=False):
    return {}


def _track_variant_play_metrics(run, variant_metrics, *, epoch, step, vs_context):
    return None


def _analyze_iteration_variants(config, paths, experiment_name, iteration):
    return {}


def _generate_visualizations(config, paths, iteration, run, total_train_steps,
                             variant_stats=None):
    return None


def save_compressed(tensor, path, half_storage=True, zstd_level=1):
    """Save tensor with zstd compression, optionally as half-precision."""
    buffer = io.BytesIO()
    if half_storage:
        tensor = to_half_safe(tensor, get_storage_dtype())
    # Clone views/slices to avoid serializing the entire backing storage
    if tensor.storage_offset() or tensor.untyped_storage().size() != tensor.nelement() * tensor.element_size():
        tensor = tensor.clone()
    torch.save(tensor, buffer)
    with open(path, 'wb') as f:
        f.write(zstd.ZstdCompressor(level=zstd_level, threads=-1).compress(buffer.getvalue()))


def load_compressed(path):
    """Load a .ptz file, returning tensor in storage dtype (typically float16)."""
    with open(path, 'rb') as f:
        data = zstd.ZstdDecompressor().decompress(f.read())
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def _atomic_save_compressed(tensor, path, half_storage=True, zstd_level=1):
    """Save tensor to path atomically via temp file."""
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    os.close(fd)
    try:
        save_compressed(tensor, tmp_path, half_storage, zstd_level=zstd_level)
        os.replace(tmp_path, path)  # atomic on POSIX
    except:
        os.unlink(tmp_path)
        raise


def _load_hist_tensor(path):
    """Load a history tensor from .ptz (compressed) or .pt (legacy) file."""
    if path.endswith(".ptz"):
        return load_compressed(path)
    return torch.load(path, map_location="cpu", mmap=True)


def _sample_recent_history_batch(paths, batch_size):
    """Sample a batch of canonical positions from the most recent history file.

    Used by per-checkpoint diagnostics (effective rank). Returns a CPU float
    tensor of shape [<=batch_size, C, H, W], or None if no history exists.
    """
    hist_dir = paths["history"]
    files = sorted(glob.glob(os.path.join(hist_dir, "*-canonical-*.ptz")), reverse=True)
    if not files:
        files = sorted(glob.glob(os.path.join(hist_dir, "*-canonical-*.pt")), reverse=True)
    if not files:
        return None
    tensor = _load_hist_tensor(files[0]).float()
    n = tensor.size(0)
    if n == 0:
        return None
    take = min(batch_size, n)
    if take == n:
        return tensor
    idx = torch.randperm(n)[:take]
    return tensor[idx]


def prepare_inference_model(nn, config, compile=False):
    """Apply inference optimizations from config."""
    if hasattr(nn, 'enable_inference_optimizations'):
        nn.enable_inference_optimizations(
            amp=config.amp_inference,
            compile=compile,
        )


def _glob_hist_files(location, pattern):
    """Glob for both .ptz and .pt files, preferring .ptz."""
    files = sorted(glob.glob(os.path.join(location, pattern + ".ptz")))
    if not files:
        files = sorted(glob.glob(os.path.join(location, pattern + ".pt")))
    return files


def glob_file_triples(directory, pattern="*-canonical-*.ptz"):
    """Glob for file triples (canonical, v, pi) and return (c_path, v_path, pi_path, size) tuples."""
    c_files = sorted(glob.glob(os.path.join(directory, pattern)))
    triples = []
    for c_path in c_files:
        v_path = c_path.replace("-canonical-", "-v-")
        pi_path = c_path.replace("-canonical-", "-pi-")
        size = int(c_path.rsplit("-", 1)[-1].split(".")[0])
        triples.append((c_path, v_path, pi_path, size))
    return triples


def resolve_history_dir(dataset_dir):
    """Return a dataset's history directory, accepting either layout."""
    nested = os.path.join(dataset_dir, "history")
    return nested if os.path.isdir(nested) else dataset_dir


def select_training_work_items(file_triples, samples_needed, *, allow_replacement=False):
    """Choose compressed-history rows without materializing the source data.

    Normal self-play keeps the old behavior: when it needs more rows than are
    available, it returns every row and lets the training loop make additional
    passes. Anchor-mixed training instead samples with replacement so both
    sources retain the requested per-update ratio during early iterations.
    """
    if samples_needed < 0:
        raise ValueError(f"samples_needed must be non-negative, got {samples_needed}")
    if not file_triples:
        raise ValueError("cannot select training samples from an empty dataset")
    if samples_needed == 0:
        return []

    sizes = [size for _, _, _, size in file_triples]
    total_size = sum(sizes)
    if total_size <= 0:
        raise ValueError("cannot select training samples from a zero-size dataset")

    if samples_needed >= total_size and not allow_replacement:
        selected_per_file = [list(range(size)) for size in sizes]
    else:
        if samples_needed <= total_size:
            all_indices = sorted(random.sample(range(total_size), samples_needed))
        else:
            all_indices = sorted(random.choices(range(total_size), k=samples_needed))

        cumulative_sizes = []
        cumulative = 0
        for size in sizes:
            cumulative_sizes.append(cumulative)
            cumulative += size
        selected_per_file = [[] for _ in file_triples]
        file_index = 0
        for index in all_indices:
            while (
                file_index < len(cumulative_sizes) - 1
                and index >= cumulative_sizes[file_index + 1]
            ):
                file_index += 1
            selected_per_file[file_index].append(index - cumulative_sizes[file_index])

    return [
        (c_path, v_path, pi_path, selected_per_file[index])
        for index, (c_path, v_path, pi_path, _) in enumerate(file_triples)
        if selected_per_file[index]
    ]


def load_selected_training_samples(work_items, num_workers, desc):
    """Load selected history rows in parallel and concatenate them once."""
    if not work_items:
        raise ValueError("cannot load an empty training selection")

    selected = []
    if num_workers <= 1:
        for item in tqdm.tqdm(work_items, desc=desc, leave=False):
            selected.append(_load_and_select(item))
    else:
        results = [None] * len(work_items)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_index = {
                executor.submit(_load_and_select, item): index
                for index, item in enumerate(work_items)
            }
            for future in tqdm.tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc=desc,
                leave=False,
            ):
                results[future_to_index[future]] = future.result()
        selected = results

    canonical, values, policies = zip(*selected)
    return torch.cat(canonical), torch.cat(values), torch.cat(policies)


def _load_triple(args):
    """Load a single file triple. Runs in thread pool (GIL released by I/O + zstd)."""
    c_path, v_path, pi_path = args
    return (_load_hist_tensor(c_path), _load_hist_tensor(v_path), _load_hist_tensor(pi_path))


def _load_triple_float(triple):
    """Load a file triple and convert to float32. Used for prefetching in symmetry expansion."""
    c_path, v_path, pi_path = triple[:3]
    return (load_compressed(c_path).float(), load_compressed(v_path).float(), load_compressed(pi_path).float())


def _load_and_shuffle(c_path, v_path, pi_path, size, seed=None):
    """Load a file triple and randomly permute samples. Used for streaming dataset prefetch."""
    c = _load_hist_tensor(c_path)
    v = _load_hist_tensor(v_path)
    pi = _load_hist_tensor(pi_path)
    if seed is None:
        perm = torch.randperm(size)
    else:
        g = torch.Generator()
        g.manual_seed(int(seed))
        perm = torch.randperm(size, generator=g)
    return c[perm], v[perm], pi[perm]


def _parallel_load_triples(triples, num_workers, desc="Loading Data"):
    """Load file triples in parallel, preserving input order.

    Args:
        triples: list of (c_path, v_path, pi_path[, ...]) tuples
        num_workers: thread count (<=1 falls back to sequential)
        desc: tqdm progress bar label
    Returns:
        list of (c_tensor, v_tensor, pi_tensor) in input order
    """
    work = [(t[0], t[1], t[2]) for t in triples]
    if num_workers <= 1:
        return [_load_triple(w) for w in tqdm.tqdm(work, desc=desc, leave=False)]

    results = [None] * len(work)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(_load_triple, w): i for i, w in enumerate(work)}
        for future in tqdm.tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx), desc=desc, leave=False,
        ):
            results[future_to_idx[future]] = future.result()
    return results


def _load_and_select(args):
    """Load a file triple and select subset by indices. Keeps only selected rows in memory."""
    c_path, v_path, pi_path, indices = args
    c = _load_hist_tensor(c_path)
    v = _load_hist_tensor(v_path)
    pi = _load_hist_tensor(pi_path)
    idx = torch.tensor(indices, dtype=torch.long)
    result = (c[idx], v[idx], pi[idx])
    del c, v, pi
    return result



GRArgs = namedtuple(
    "GRArgs",
    [
        "title",
        "game",
        "max_batch_size",
        "iteration",
        "data_save_size",
        "mcts_workers",
        "record_batch_metrics",
        "data_folder",
        "progress_path",
    ],
    defaults=(0, 30_000, os.cpu_count() - 1, False, None, None),
)


SelfPlayResult = namedtuple(
    "SelfPlayResult",
    [
        "win_rates",
        "hit_rate",
        "game_length",
        "resign_win_rates",
        "resign_rate",
        "avg_depth",
        "avg_entropy",
        "fast_avg_depth",
        "fast_avg_entropy",
        "avg_mpt",
        "avg_vm",
        "saturation",
        "churn_rate",
        "avg_batch_size",
        "median_batch_size",
        "min_batch_size",
        "max_batch_size",
        "avg_inference_ms",
        "median_inference_ms",
        "min_inference_ms",
        "max_inference_ms",
        "theoretical_hr",
        "variant_game_counts",   # dict {vid: count} or {} for non-unified games
        "variant_win_rates",     # dict {vid: [rate_per_player...]} or {} for non-unified
        "variant_metrics",       # dict {vid: {metric_name: value}} or {} for non-unified
    ],
)


class GameRunner:
    def __init__(self, players, pm, args):
        self.players = players
        self.pm = pm
        self.args = args
        self.device = get_device()
        self.num_players = self.args.game.NUM_PLAYERS()
        if len(self.players) != self.num_players:
            raise Exception("There must be a player for each player")

        self.num_model_groups = self.pm.num_model_groups()

        # Build model_group → model mapping for GPU inference
        model_groups = list(self.pm.params().model_groups)
        self._group_models = [None] * self.num_model_groups
        for i, p in enumerate(self.players):
            if not isinstance(p, (RandPlayer, PlayoutPlayer)):
                g = model_groups[i]
                if self._group_models[g] is None:
                    self._group_models[g] = p

        # NN model groups (groups that actually need GPU inference)
        self._nn_groups = [g for g in range(self.num_model_groups)
                           if self._group_models[g] is not None]

        # Pipeline queues
        self.gpu_ready_queue = queue.SimpleQueue()
        self.gpu_result_queue = queue.SimpleQueue()
        self.buffer_pool = queue.SimpleQueue()
        self.monitor_queue = queue.SimpleQueue()
        self.saved_samples = 0
        self._batch_lock = threading.Lock()
        self._batch_sizes = []
        self._inference_times = []

        # Buffer pool: 1 per batcher thread + 1 GPU + 1 result worker
        cs = self.args.game.CANONICAL_SHAPE()
        num_buffers = len(self._nn_groups) + 2
        for _ in range(num_buffers):
            buf = torch.zeros(self.args.max_batch_size, cs[0], cs[1], cs[2])
            if str(self.device) == "cuda":
                buf = buf.pin_memory()
            self.buffer_pool.put(buf)

        # History buffers
        self.hist_canonical = torch.zeros(self.args.data_save_size, cs[0], cs[1], cs[2])
        self.hist_v = torch.zeros(self.args.data_save_size, self.num_players + 1)
        self.hist_pi = torch.zeros(self.args.data_save_size, self.args.game.NUM_MOVES())

    @tracy_zone
    def run(self):
        # Two-stage Ctrl+C:
        #   1st press: ack visibly + call pm.stop() so workers drain gracefully
        #              (preserves current iteration's data).
        #   2nd press: restore the default handler and re-raise immediately,
        #              so an unresponsive drain (long MCTS step, queued
        #              history saving, etc.) doesn't leave the user stuck
        #              mashing Ctrl+C with no feedback.
        interrupted = False
        prev_handler = signal.getsignal(signal.SIGINT)
        def _sigint_handler(sig, frame):
            nonlocal interrupted
            if interrupted:
                signal.signal(signal.SIGINT, prev_handler)
                tqdm.tqdm.write("\nSecond Ctrl+C — forcing exit.")
                raise KeyboardInterrupt
            interrupted = True
            tqdm.tqdm.write(
                "\nCtrl+C received — stopping selfplay; press again to force exit."
            )
            self.pm.stop()
        signal.signal(signal.SIGINT, _sigint_handler)
        try:
            self._run_inner()
        finally:
            signal.signal(signal.SIGINT, prev_handler)
            if interrupted:
                raise KeyboardInterrupt

    def _run_inner(self):
        self._batch_sizes.clear()
        self._inference_times.clear()
        # Check if any model group has an NN model
        has_nn = any(m is not None for m in self._group_models)

        if has_nn:
            # Pre-warm CUDA graphs before starting threads to avoid
            # capture race conditions (graph capture is device-wide).
            for g in self._nn_groups:
                model = self._group_models[g]
                if model is not None:
                    model.warmup_graphs(self.args.max_batch_size)

        # Block SIGINT in workers so it always lands on the main thread —
        # Python handlers only run on main, which parks in Thread.join's
        # sem_wait(-1) and only sees EINTR if the signal targets it.
        prev_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            batcher_threads = []
            if has_nn:
                for g in self._nn_groups:
                    t = threading.Thread(target=self.batcher, args=(g,))
                    t.start()
                    batcher_threads.append(t)
                gpu_thread = threading.Thread(target=self.gpu_loop)
                gpu_thread.start()
                result_thread = threading.Thread(target=self.result_worker)
                result_thread.start()

            mcts_workers = []
            for i in range(self.args.mcts_workers):
                mcts_workers.append(threading.Thread(target=self.pm.play))
                mcts_workers[i].start()

            monitor = threading.Thread(target=self.monitor)
            monitor.start()
            if self.pm.params().history_enabled:
                hist_saver = threading.Thread(target=self.hist_saver)
                hist_saver.start()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, prev_mask)

        for mw in mcts_workers:
            mw.join()
        if has_nn:
            for bt in batcher_threads:
                bt.join()
            gpu_thread.join()
            result_thread.join()
        monitor.join()
        if self.pm.params().history_enabled:
            hist_saver.join()

    @tracy_zone
    def monitor(self):
        tracy_thread("monitor")
        last_completed = 0
        last_update = time.time()
        n = self.pm.params().games_to_play
        pbar = tqdm.tqdm(total=n, unit="games", desc=self.args.title, leave=False)

        def _build_postfix():
            hr = 0
            hits = self.pm.cache_hits()
            misses = self.pm.cache_misses()
            total = hits + misses
            if total > 0:
                hr = hits / total
            max_cache = self.pm.cache_max_size()
            if max_cache > 0:
                cache_sz = self.pm.cache_size()
                sat = cache_sz / max_cache
                evictions = self.pm.cache_evictions()
                churn = evictions / hits if hits > 0 else 0
                reinserts = self.pm.cache_reinserts()
                thr = (hits + reinserts) / total if total > 0 else 0
                postfix = {"hr": _fmt_pct(hr), "thr": _fmt_pct(thr)}
                postfix["churn"] = f"{churn:.2f}"
            else:
                postfix = {"hr": "N/A"}
            if self.args.record_batch_metrics:
                with self._batch_lock:
                    sizes = list(self._batch_sizes)
                    inf_times = list(self._inference_times)
                if sizes:
                    sizes.sort()
                    n_bs = len(sizes)
                    postfix["bs"] = f"{sizes[0]}/{sizes[n_bs//2]}/{sizes[-1]}"
                if inf_times:
                    resolved = []
                    for t in inf_times:
                        if isinstance(t, tuple):
                            t[1].synchronize()
                            resolved.append(t[0].elapsed_time(t[1]) / 1000.0)
                        else:
                            resolved.append(t)
                    inf_times = resolved
                    inf_times.sort()
                    n_it = len(inf_times)
                    postfix["nn_ms"] = (
                        f"{inf_times[0]*1000:.1f}/"
                        f"{inf_times[n_it//2]*1000:.1f}/"
                        f"{inf_times[-1]*1000:.1f}"
                    )
            completed = self.pm.games_completed()
            num_perms = self.pm.num_seat_perms()
            if completed > 0 and num_perms > 1:
                player_wins = [0.0] * (self.num_players + 1)  # last = draws
                total_games = 0
                for perm_idx in range(num_perms):
                    ps = self.pm.perm_scores(perm_idx)
                    pg = self.pm.perm_games_completed(perm_idx)
                    if pg == 0:
                        continue
                    total_games += pg
                    for seat in range(self.num_players):
                        original_player = (seat + perm_idx) % self.num_players
                        player_wins[original_player] += ps[seat]
                    player_wins[-1] += ps[self.num_players]
                if total_games > 0:
                    win_rates = [pw / total_games for pw in player_wins]
                else:
                    win_rates = [0] * (self.num_players + 1)
            else:
                scores = self.pm.scores()
                win_rates = [0] * len(scores)
                if completed > 0:
                    for i in range(len(scores)):
                        win_rates[i] = scores[i] / completed
            postfix["wr"] = "/".join(_fmt_pct(x) for x in win_rates)
            return postfix, completed

        def _write_progress(postfix, completed):
            if not self.args.progress_path:
                return
            progress = {
                "title": self.args.title,
                "completed_games": int(completed),
                "total_games": int(n),
                "remaining_games": int(max(0, n - completed)),
                "postfix": postfix,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            tmp_path = self.args.progress_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(progress, f, indent=2)
                f.write("\n")
            os.replace(tmp_path, self.args.progress_path)

        while self.pm.remaining_games() > 0:
            try:
                self.monitor_queue.get(timeout=1)
            except queue.Empty:
                pass
            if time.time() - last_update > 1:
                postfix, completed = _build_postfix()
                _write_progress(postfix, completed)
                pbar.set_postfix(postfix)
                pbar.update(completed - last_completed)
                last_completed = completed
                last_update = time.time()

        postfix, completed = _build_postfix()
        _write_progress(postfix, completed)
        pbar.set_postfix(postfix)
        pbar.update(n - last_completed)
        pbar.close()

    @tracy_zone
    def batcher(self, group):
        tracy_thread(f"batcher_{group}")
        while self.pm.remaining_games() > 0:
            try:
                batch_tensor = self.buffer_pool.get(timeout=1)
            except queue.Empty:
                continue

            game_indices = self.pm.build_batch(group, batch_tensor)
            if not game_indices:
                self.buffer_pool.put(batch_tensor)
                continue

            if self.args.record_batch_metrics:
                with self._batch_lock:
                    self._batch_sizes.append(len(game_indices))

            gpu_tensor = batch_tensor[:len(game_indices)].to(
                self.device, non_blocking=True
            )
            self.gpu_ready_queue.put((gpu_tensor, batch_tensor, game_indices, group))

    @tracy_zone
    def gpu_loop(self):
        tracy_thread("gpu_loop")
        while self.pm.remaining_games() > 0:
            # Try non-blocking first — if batch already queued, grab immediately
            try:
                gpu_tensor, batch_tensor, game_indices, group = (
                    self.gpu_ready_queue.get_nowait()
                )
            except queue.Empty:
                # No batch ready — signal batcher to hand off partial batch
                self.pm.set_eager(True)
                try:
                    gpu_tensor, batch_tensor, game_indices, group = (
                        self.gpu_ready_queue.get(timeout=1)
                    )
                except queue.Empty:
                    self.pm.set_eager(False)
                    continue
                self.pm.set_eager(False)

            model = self._group_models[group]
            if self.args.record_batch_metrics and self.device.type == 'cuda':
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
            elif self.args.record_batch_metrics:
                t0 = time.perf_counter()
            v, pi = model.process(gpu_tensor)
            if self.args.record_batch_metrics:
                if self.device.type == 'cuda':
                    end_ev.record()
                    with self._batch_lock:
                        self._inference_times.append((start_ev, end_ev))
                else:
                    with self._batch_lock:
                        self._inference_times.append(time.perf_counter() - t0)
            self.gpu_result_queue.put((batch_tensor, game_indices, v, pi, group))

    @tracy_zone
    def result_worker(self):
        tracy_thread("result_worker")
        while self.pm.remaining_games() > 0:
            try:
                batch_tensor, game_indices, v, pi, group = (
                    self.gpu_result_queue.get(timeout=1)
                )
            except queue.Empty:
                continue
            v_np = v.cpu().numpy()
            pi_np = pi.cpu().numpy()
            self.pm.update_inferences(group, game_indices, v_np, pi_np)
            self.buffer_pool.put(batch_tensor)
            self.monitor_queue.put(len(game_indices))

    @tracy_zone
    def hist_saver(self):
        tracy_thread("hist_saver")
        batch = 0
        data_folder = self.args.data_folder
        if data_folder is None:
            raise ValueError("GRArgs.data_folder must be set when history saving is needed")
        os.makedirs(data_folder, exist_ok=True)
        while self.pm.remaining_games() > 0 or self.pm.hist_count() > 0:
            size = self.pm.build_history_batch(
                self.hist_canonical, self.hist_v, self.hist_pi
            )
            if size == 0:
                continue
            prefix = os.path.join(data_folder, f"{self.args.iteration:04d}-{batch:04d}")
            save_compressed(self.hist_canonical[:size], f"{prefix}-canonical-{size}.ptz")
            save_compressed(self.hist_v[:size], f"{prefix}-v-{size}.ptz")
            save_compressed(self.hist_pi[:size], f"{prefix}-pi-{size}.ptz")
            self.saved_samples += size
            batch += 1


class RandPlayer:
    """Marker for random evaluation. Actual eval runs inline in C++ MCTS threads."""
    pass


class PlayoutPlayer:
    """Marker for rollout-based evaluation. Actual eval runs inline in C++ MCTS threads."""
    pass


def set_eval_types(params, players):
    """Set eval_type on params based on player types."""
    eval_types = []
    for p in players:
        if isinstance(p, PlayoutPlayer):
            eval_types.append(alphazero.EvalType.PLAYOUT)
        elif isinstance(p, RandPlayer):
            eval_types.append(alphazero.EvalType.RANDOM)
        else:
            eval_types.append(alphazero.EvalType.NN)
    params.eval_type = eval_types


def set_model_groups(params, players):
    """Set model_groups on params based on model object identity.

    Players sharing the same model object get the same group.
    Non-NN groups get an unused inference queue (eval_type routes them
    inline in C++, so the queue is never consumed).
    """
    model_ids = {}
    groups = []
    for p in players:
        pid = id(p)
        if pid not in model_ids:
            model_ids[pid] = len(model_ids)
        groups.append(model_ids[pid])
    params.model_groups = groups


def base_params(config, start_temp, bs, cb):
    """Create PlayParams from config."""
    params = alphazero.PlayParams()
    params.cache_shards = config.resolved_cache_shards
    params.max_cache_size = config.max_cache_size
    params.cpuct = config.cpuct
    params.start_temp = start_temp
    params.temp_decay_half_life = float(config.temp_decay_half_life)
    params.final_temp = config.final_temp
    params.max_batch_size = bs
    params.concurrent_games = bs * cb
    params.fpu_reduction = config.fpu_reduction
    params.gumbel_enabled = config.gumbel_enabled
    params.gumbel_m = config.gumbel_m
    params.gumbel_c_visit = config.gumbel_c_visit
    params.gumbel_c_scale = config.gumbel_c_scale
    params.gumbel_full = config.gumbel_full
    params.fast_search_uses_gumbel = config.resolve_fast_search_uses_gumbel()
    return params


_ELO_ALPHA = math.log(10) / 400.0


def elo_prob(r1, r2):
    x = _ELO_ALPHA * (r2 - r1)
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _wr_from_counts(wins_ij, wins_ji, draws_ij):
    """Compute the symmetric win-rate pair (wr[i,j], wr[j,i]) from raw
    integer game counts using the half-draw convention.

    Returns (NaN, NaN) if there are zero games between the two agents.
    Correctly handles multi-player games (the comparison is always 2-sided
    -- a draw counts as 0.5 to each side regardless of NUM_PLAYERS).
    """
    n = float(wins_ij) + float(wins_ji) + float(draws_ij)
    if n <= 0:
        return float("nan"), float("nan")
    rate_ij = (float(wins_ij) + 0.5 * float(draws_ij)) / n
    return rate_ij, 1.0 - rate_ij


@tracy_zone
def get_elo(past_elo, win_rates, new_agent):
    if new_agent != 0:
        past_elo[new_agent] = past_elo[new_agent - 1]
    iters = 5000
    mask = ~np.isnan(win_rates[new_agent])
    for _ in tqdm.trange(iters, leave=False):
        rates = np.clip(win_rates[new_agent, mask], 0.001, 0.999)
        x = _ELO_ALPHA * (past_elo[new_agent] - past_elo[mask])
        # Clamp to avoid overflow in exp (sigmoid saturates beyond ~700)
        x_safe = np.clip(x, -500, 500)
        probs = np.where(x_safe >= 0, 1.0 / (1.0 + np.exp(-x_safe)),
                         np.exp(x_safe) / (1.0 + np.exp(x_safe)))
        past_elo[new_agent] += np.sum(rates - probs) * 32
    return past_elo


def whr_refit(wins, draws=None, max_sweeps=200, tol=0.1, anchor=0):
    """Whole-history Bradley-Terry MLE on integer game-count matrices.

    Each agent (row/col index) is treated as a fully independent player.
    No temporal Wiener prior, no smoothing across iters — every agent's
    rating is fixed once we have the game record. This is the natural
    formulation for our setting where each "iter" is a frozen network
    snapshot, not a time-evolving player.

    The likelihood for a pair (i, j) with `wins[i,j]` i-wins, `wins[j,i]`
    j-wins, and `draws[i,j]` draws is the Bradley-Terry product over all
    those games. Draws contribute as half-wins on each side (the standard
    elo/BT convention).

    Args:
        wins: (n, n) int/float matrix. wins[i, j] = games where i beat j.
            Asymmetric in general (one side wins per game).
        draws: optional (n, n) matrix of drawn games between i and j.
            Should be symmetric. Pass None for no draws.
        max_sweeps: maximum Gauss-Seidel passes over agents.
        tol: convergence threshold on max rating change across a pass.
        anchor: index pinned at rating 0 (sets the gauge). Auto-falls back
            to the first agent with games if `anchor` has none.

    Returns:
        np.ndarray of shape (n,) with elo ratings. Agents with no games
        stay at the anchor's value (typically 0). Saturated rows (all
        wins or all losses) get a large but finite rating from clipping
        the empirical rate to [0.001, 0.999].
    """
    wins = np.asarray(wins, dtype=np.float64)
    if draws is None:
        draws = np.zeros_like(wins)
    else:
        draws = np.asarray(draws, dtype=np.float64)
    if wins.shape != draws.shape or wins.ndim != 2 or wins.shape[0] != wins.shape[1]:
        raise ValueError("wins and draws must be square matrices of matching shape")

    n = wins.shape[0]
    elo = np.zeros(n)

    # n_games[i, j] = total games between i and j.
    # eff_wins[i, j] = i's effective wins vs j (draws count as half-wins).
    n_games = wins + wins.T + draws
    eff_wins = wins + 0.5 * draws

    valid = (n_games.sum(axis=1) > 0)
    if not valid.any():
        return elo
    if not valid[anchor]:
        anchor = int(np.argmax(valid))

    for _ in range(max_sweeps):
        max_delta = 0.0
        for i in range(n):
            if i == anchor or not valid[i]:
                continue
            mask = n_games[i] > 0
            if not mask.any():
                continue
            ng = n_games[i, mask]                       # game counts per opponent
            ew = eff_wins[i, mask]                      # effective i-wins per opponent
            rate = np.clip(ew / ng, 0.001, 0.999)       # empirical rate (clipped)
            x = np.clip(_ELO_ALPHA * (elo[i] - elo[mask]), -500, 500)
            # Numerically stable sigmoid (matches get_elo's pattern).
            p = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                         np.exp(x) / (1.0 + np.exp(x)))
            # grad_i  = α * Σ_j n_ij * (rate_ij - p_ij)
            # hess_ii = -α² * Σ_j n_ij * p_ij * (1 - p_ij)
            grad = _ELO_ALPHA * float(np.sum(ng * (rate - p)))
            hess = -(_ELO_ALPHA ** 2) * float(np.sum(ng * p * (1 - p)))
            if hess >= -1e-12:
                continue  # degenerate; skip
            delta = float(np.clip(-grad / hess, -200.0, 200.0))
            elo[i] += delta
            if abs(delta) > max_delta:
                max_delta = abs(delta)
        if max_delta < tol:
            break
    return elo


def _katago_window_curve(c, alpha, beta, ratio):
    """Shared window-growth curve: c * (1 + beta * (ratio^alpha - 1) / alpha).

    Used in both iterations and games units; caller picks `c` and `ratio` to
    match the unit. Returns a float; caller casts to the right type.
    """
    return c * (1 + beta * (ratio ** alpha - 1) / alpha)


def games_per_iter(config):
    """Total games produced per training iteration.

    Constant given the self-play config. By default this equals
    self_play_batch_size * NUM_PLAYERS * self_play_concurrent_batch_mult *
    self_play_chunks (see self_play() in this module). A positive
    self_play_games_per_iter overrides that product so GPU batch sizing can be
    tuned independently from generated data volume.
    """
    override = int(getattr(config, "self_play_games_per_iter", 0) or 0)
    if override > 0:
        return override
    return (
        config.self_play_batch_size
        * config.Game.NUM_PLAYERS()
        * config.self_play_concurrent_batch_mult
        * config.self_play_chunks
    )


def calc_hist_window_games(config, total_games):
    """Games-based window size (KataGo Nwindow formula, expressed in games).

    Until total_games <= c, returns total_games (use everything generated so
    far) — matches KataGo's "growing buffer" early phase.
    """
    c = config.window_size_scalar_games
    if total_games <= c:
        return int(total_games)
    return int(_katago_window_curve(
        c, config.window_size_alpha, config.window_size_beta, total_games / c))


def calc_hist_size(config, i, paths=None):
    """Training window size in iterations.

    config.window_size_unit selects the formula:
      "iterations" (default): ratio = (i+1) / window_size_scalar (legacy).
      "games":     target Nwindow in games via KataGo curve, divided by the
                   per-iter game count (constant) to convert back to iters.

    `paths` is currently unused but kept for forward compatibility with
    formulas that might want to inspect on-disk history.
    """
    unit = getattr(config, "window_size_unit", "iterations")
    if unit == "games":
        per_iter = max(1, games_per_iter(config))
        total_games = (i + 1) * per_iter
        target_games = calc_hist_window_games(config, total_games)
        # Round up so we always cover the target.
        return max(1, int(math.ceil(target_games / per_iter)))

    return int(_katago_window_curve(
        config.window_size_scalar,
        config.window_size_alpha,
        config.window_size_beta,
        (i + 1) / config.window_size_scalar,
    ))


def maybe_save(
    config,
    c,
    v,
    p,
    size,
    batch,
    iteration,
    location,
    name="",
    force=False,
    use_compression=False,
    half_storage=True,
):
    if size == config.hist_size or (force and size > 0):
        save_fn = (lambda t, path: save_compressed(t, path, half_storage, zstd_level=config.zstd_level)) if use_compression else torch.save
        ext = ".ptz" if use_compression else ".pt"
        save_fn(
            c[:size],
            os.path.join(
                location, f"{iteration:04d}-{batch:04d}{name}-canonical-{size}{ext}"
            ),
        )
        save_fn(
            v[:size],
            os.path.join(
                location, f"{iteration:04d}-{batch:04d}{name}-v-{size}{ext}"
            ),
        )
        save_fn(
            p[:size],
            os.path.join(
                location, f"{iteration:04d}-{batch:04d}{name}-pi-{size}{ext}"
            ),
        )
        return True
    return False


@tracy_zone
def exploit_symmetries(config, paths, iteration):
    Game = config.Game
    if Game.NUM_SYMMETRIES() <= 1:
        return

    tmp_hist = paths["tmp_history"]
    # Find raw files (exclude any already-symmetrized files)
    all_triples = glob_file_triples(tmp_hist, f"{iteration:04d}-*-canonical-*.ptz")
    raw_triples = [t for t in all_triples if "-syms-" not in t[0]]
    if not raw_triples:
        return

    # Delete stale syms files from a previous interrupted run
    stale_syms = glob_file_triples(tmp_hist, f"{iteration:04d}-*-syms-canonical-*.ptz")
    for c_path, v_path, pi_path, _ in stale_syms:
        for fp in (c_path, v_path, pi_path):
            if os.path.exists(fp):
                os.remove(fp)

    i_out = 0
    batch_out = 0
    cs = Game.CANONICAL_SHAPE()
    c_out = torch.zeros(config.hist_size, cs[0], cs[1], cs[2])
    v_out = torch.zeros(config.hist_size, Game.NUM_PLAYERS() + 1)
    p_out = torch.zeros(config.hist_size, Game.NUM_MOVES())

    max_workers = min(os.cpu_count() or 4, 8)
    _thread_local = threading.local()

    def _compute_symmetries(args):
        """Compute symmetries for a single sample. Runs in thread pool (GIL released by C++)."""
        c, v, pi = args
        if not hasattr(_thread_local, 'game'):
            _thread_local.game = Game()
        ph = alphazero.PlayHistory(c, v, pi)
        syms = _thread_local.game.symmetries(ph)
        result = []
        for sym in syms:
            result.append((
                torch.from_numpy(np.array(sym.canonical())),
                torch.from_numpy(np.array(sym.v())),
                torch.from_numpy(np.array(sym.pi())),
            ))
        return result

    total_samples = sum(size for _, _, _, size in raw_triples)

    with ThreadPoolExecutor(max_workers=1) as io_pool, \
         ThreadPoolExecutor(max_workers=max_workers) as executor:
        pbar = tqdm.tqdm(total=total_samples, desc="Creating Symmetric Samples", leave=False)
        next_future = io_pool.submit(_load_triple_float, raw_triples[0])
        for idx, (c_path, v_path, pi_path, size) in enumerate(raw_triples):
            c_data, v_data, pi_data = next_future.result()
            if idx + 1 < len(raw_triples):
                next_future = io_pool.submit(_load_triple_float, raw_triples[idx + 1])

            def _samples():
                for j in range(size):
                    yield (c_data[j], v_data[j], pi_data[j])

            futures = executor.map(
                _compute_symmetries,
                _samples(),
                chunksize=max(1, size // (max_workers * 4)),
            )
            for sym_list in futures:
                for c_sym, v_sym, p_sym in sym_list:
                    c_out[i_out] = c_sym
                    v_out[i_out] = v_sym
                    p_out[i_out] = p_sym
                    i_out += 1
                    if i_out == config.hist_size:
                        if maybe_save(config, c_out, v_out, p_out, i_out, batch_out, iteration,
                                      location=tmp_hist, name="-syms", use_compression=True,
                                      half_storage=config.half_storage):
                            i_out = 0
                            batch_out += 1
                pbar.update(1)

            del c_data, v_data, pi_data
        pbar.close()

    # Save remaining samples
    if i_out > 0:
        maybe_save(config, c_out, v_out, p_out, i_out, batch_out, iteration,
                   location=tmp_hist, name="-syms", use_compression=True, force=True,
                   half_storage=config.half_storage)

    del c_out, v_out, p_out

    # Delete the raw files now that symmetrized versions exist
    for c_path, v_path, pi_path, _ in raw_triples:
        for fp in (c_path, v_path, pi_path):
            if os.path.exists(fp):
                os.remove(fp)


@tracy_zone
def resample_by_surprise(config, paths, experiment_name, iteration):
    import neural_net

    Game = config.Game
    hist_location = paths["history"]
    tmp_hist = paths["tmp_history"]

    # Find files in tmp_history
    file_triples = glob_file_triples(tmp_hist, f"{iteration:04d}-*-canonical-*.ptz")
    if not file_triples:
        return np.zeros(0)

    # Pass 1: compute per-sample loss by loading one file at a time
    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
    )
    loss_arrays = []
    with ThreadPoolExecutor(max_workers=1) as io_pool:
        next_future = io_pool.submit(_load_triple, (file_triples[0][0], file_triples[0][1], file_triples[0][2]))
        for idx, (c_path, v_path, pi_path, size) in enumerate(tqdm.tqdm(file_triples, desc="Computing Loss", leave=False)):
            c_t, v_t, pi_t = next_future.result()
            if idx + 1 < len(file_triples):
                nxt = file_triples[idx + 1]
                next_future = io_pool.submit(_load_triple, (nxt[0], nxt[1], nxt[2]))
            ds = TensorDataset(c_t, v_t, pi_t)
            dl = DataLoader(ds, batch_size=config.train_batch_size, shuffle=False,
                            pin_memory=torch.cuda.is_available())
            file_loss = nn.sample_loss(dl, size)
            loss_arrays.append(file_loss)
            del c_t, v_t, pi_t, ds, dl
    del nn

    loss = np.concatenate(loss_arrays).astype(np.float64, copy=False)
    del loss_arrays
    sample_count = len(loss)
    valid_loss = np.isfinite(loss) & (loss >= 0)
    if not np.all(valid_loss):
        loss = np.where(valid_loss, loss, 0.0)
    total_loss = np.sum(loss)

    # Compute copy counts vectorized
    if not np.isfinite(total_loss) or total_loss <= 0:
        copies = np.ones(sample_count, dtype=np.int64)
    else:
        weights = 0.5 + (loss / total_loss) * 0.5 * sample_count
        int_w = np.floor(weights).astype(np.int64)
        frac_w = weights - int_w
        frac_mask = np.random.random(sample_count) < frac_w
        copies = int_w + frac_mask.astype(np.int64)
    loss_for_logging = loss
    del loss

    # Clear old history for iteration before saving new history
    os.makedirs(hist_location, exist_ok=True)
    for fn in glob.glob(os.path.join(hist_location, f"{iteration:04d}-*.pt*")):
        os.remove(fn)

    # Pass 2: resample and write, loading one file at a time
    i_out = 0
    batch_out = 0
    cs = Game.CANONICAL_SHAPE()
    c_out = torch.zeros(config.hist_size, cs[0], cs[1], cs[2])
    v_out = torch.zeros(config.hist_size, Game.NUM_PLAYERS() + 1)
    p_out = torch.zeros(config.hist_size, Game.NUM_MOVES())

    sample_idx = 0
    with ThreadPoolExecutor(max_workers=1) as io_pool:
        next_future = io_pool.submit(_load_triple, (file_triples[0][0], file_triples[0][1], file_triples[0][2]))
        for idx, (c_path, v_path, pi_path, size) in enumerate(tqdm.tqdm(file_triples, desc="Resampling Data", leave=False)):
            c_t, v_t, pi_t = next_future.result()
            if idx + 1 < len(file_triples):
                nxt = file_triples[idx + 1]
                next_future = io_pool.submit(_load_triple, (nxt[0], nxt[1], nxt[2]))

            ds_copies = copies[sample_idx:sample_idx + size]
            expanded_idx = np.repeat(np.arange(size), ds_copies)
            if len(expanded_idx) == 0:
                del c_t, v_t, pi_t
                sample_idx += size
                continue
            c_exp = c_t[expanded_idx]
            v_exp = v_t[expanded_idx]
            p_exp = pi_t[expanded_idx]
            del c_t, v_t, pi_t

            pos = 0
            while pos < len(expanded_idx):
                space = config.hist_size - i_out
                chunk = min(space, len(expanded_idx) - pos)
                c_out[i_out:i_out + chunk] = c_exp[pos:pos + chunk]
                v_out[i_out:i_out + chunk] = v_exp[pos:pos + chunk]
                p_out[i_out:i_out + chunk] = p_exp[pos:pos + chunk]
                i_out += chunk
                pos += chunk
                if maybe_save(config, c_out, v_out, p_out, i_out, batch_out, iteration, location=hist_location, use_compression=True, half_storage=config.half_storage):
                    i_out = 0
                    batch_out += 1
            del c_exp, v_exp, p_exp, expanded_idx
            sample_idx += size

    maybe_save(config, c_out, v_out, p_out, i_out, batch_out, iteration, location=hist_location, use_compression=True, force=True, half_storage=config.half_storage)

    del c_out, v_out, p_out

    # Clean up tmp_history files for this iteration
    for c_path, v_path, pi_path, _ in file_triples:
        for fp in (c_path, v_path, pi_path):
            if os.path.exists(fp):
                os.remove(fp)

    return loss_for_logging


@tracy_zone
def iteration_loss(config, paths, experiment_name, iteration):
    import neural_net

    Game = config.Game
    hist_location = paths["history"]
    c = _glob_hist_files(hist_location, f"{iteration:04d}-*-canonical-*")
    v = _glob_hist_files(hist_location, f"{iteration:04d}-*-v-*")
    p = _glob_hist_files(hist_location, f"{iteration:04d}-*-pi-*")
    loaded = _parallel_load_triples(
        list(zip(c, v, p)), config.resolved_loader_threads, desc="Loading Iteration Data"
    )
    datasets = [TensorDataset(ct, vt, pt) for ct, vt, pt in loaded]
    del loaded

    dataset = ConcatDataset(datasets)
    _dl_workers = config.resolved_loader_threads
    dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, pin_memory=torch.cuda.is_available(),
                            num_workers=_dl_workers)

    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
    )
    v_loss, pi_loss = nn.losses(dataloader)

    del datasets, dataset, dataloader, nn

    return v_loss, pi_loss


@tracy_zone
def train(config, paths, experiment_name, iteration, hist_size, run, total_train_steps,
          lr=None, parent_iteration=None):
    """Index-then-extract training: pre-select samples, load one file at a time.

    Memory bounded at ~1.5 GB regardless of window size.
    """
    import neural_net

    Game = config.Game
    hist_location = paths["history"]
    bs = config.train_batch_size

    # Phase 1: Discover files and sizes from filenames (no loading)
    file_triples = []
    iter_range = range(max(0, iteration - hist_size), iteration + 1)
    for i in tqdm.tqdm(iter_range, desc="Discovering Training Files", leave=False):
        triples = glob_file_triples(hist_location, f"{i:04d}-*-canonical-*.ptz")
        if not triples:
            triples = glob_file_triples(hist_location, f"{i:04d}-*-canonical-*.pt")
        file_triples.extend(triples)

    sizes = [size for _, _, _, size in file_triples]
    total_size = sum(sizes)

    num_iters_in_window = min(hist_size, iteration + 1)
    average_generation = total_size / num_iters_in_window
    steps_to_train = int(math.ceil(average_generation / bs * config.train_sample_rate))
    samples_needed = steps_to_train * bs

    # Phase 2: Select self-play and, optionally, fixed teacher-anchor rows.
    # Anchor mode samples with replacement when a small initial history would
    # otherwise be traversed repeatedly; this preserves the configured mix in
    # every optimization update instead of letting the anchor dominate early.
    anchor_fraction = float(getattr(config, "teacher_anchor_fraction", 0.0))
    anchor_triples = []
    anchor_samples = 0
    selfplay_samples = samples_needed
    if anchor_fraction > 0.0:
        anchor_dir = resolve_history_dir(config.teacher_anchor_dir)
        anchor_triples = glob_file_triples(anchor_dir, "*-canonical-*.ptz")
        if not anchor_triples:
            anchor_triples = glob_file_triples(anchor_dir, "*-canonical-*.pt")
        if not anchor_triples:
            raise RuntimeError(
                f"teacher anchor dataset has no history triples: {config.teacher_anchor_dir}"
            )
        if samples_needed < 2:
            raise RuntimeError("teacher anchor mixing requires at least two training samples")
        anchor_samples = max(1, min(samples_needed - 1, round(samples_needed * anchor_fraction)))
        selfplay_samples = samples_needed - anchor_samples

    selfplay_work_items = select_training_work_items(
        file_triples,
        selfplay_samples,
        allow_replacement=bool(anchor_triples),
    )
    num_workers = config.resolved_loader_threads
    c_data, v_data, pi_data = load_selected_training_samples(
        selfplay_work_items,
        num_workers,
        "Extracting Self-Play Samples",
    )

    if anchor_triples:
        anchor_work_items = select_training_work_items(
            anchor_triples,
            anchor_samples,
            allow_replacement=True,
        )
        anchor_c, anchor_v, anchor_pi = load_selected_training_samples(
            anchor_work_items,
            num_workers,
            "Extracting Teacher Anchor",
        )
        c_data = torch.cat((c_data, anchor_c))
        v_data = torch.cat((v_data, anchor_v))
        pi_data = torch.cat((pi_data, anchor_pi))
        del anchor_c, anchor_v, anchor_pi
        actual_fraction = anchor_samples / samples_needed
        run.track(
            anchor_samples,
            name="teacher_anchor_samples",
            epoch=iteration,
            step=total_train_steps,
        )
        run.track(
            actual_fraction,
            name="teacher_anchor_fraction",
            epoch=iteration,
            step=total_train_steps,
        )

    # Phase 4: Train
    dataset = TensorDataset(c_data, v_data, pi_data)
    _dl_workers = config.resolved_loader_threads
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, pin_memory=torch.cuda.is_available(),
                            num_workers=_dl_workers)
    del c_data, v_data, pi_data

    parent_iteration = iteration if parent_iteration is None else parent_iteration
    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{parent_iteration:04d}-{experiment_name}.pt"
    )
    if lr is not None:
        nn.set_lr(lr)
    v_loss, pi_loss = nn.train(
        dataloader, steps_to_train, run, iteration, total_train_steps,
        ema_averaging=config.ema_averaging,
    )
    total_train_steps += steps_to_train
    nn.save_checkpoint(paths["checkpoint"], f"{iteration + 1:04d}-{experiment_name}.pt",
                       zstd_level=config.zstd_level)
    del dataset, dataloader, nn
    return v_loss, pi_loss, total_train_steps


def _load_reservoir_meta(reservoir_dir):
    """Load reservoir metadata, or None if not found."""
    meta_path = os.path.join(reservoir_dir, "reservoir_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def _save_reservoir_meta(reservoir_dir, meta):
    """Save reservoir metadata atomically."""
    meta_path = os.path.join(reservoir_dir, "reservoir_meta.json")
    tmp_path = meta_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(meta, f)
    os.replace(tmp_path, meta_path)


def _load_compare_past_state(experiment_dir):
    """Return the set of anchor iters that have been retired due to win-rate
    saturation. Empty set if no state file exists."""
    path = os.path.join(experiment_dir, "compare_past_state.json")
    if not os.path.exists(path):
        return set()
    try:
        with open(path) as f:
            data = json.load(f)
        return set(int(x) for x in data.get("saturated_anchors", []))
    except Exception:
        return set()


def _save_compare_past_state(experiment_dir, saturated_set):
    """Persist the set of retired anchors atomically."""
    path = os.path.join(experiment_dir, "compare_past_state.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"saturated_anchors": sorted(int(x) for x in saturated_set)}, f)
    os.replace(tmp_path, path)


def _load_train_steps_per_iter(experiment_dir, total_agents):
    """Load per-iter train step counts (the `step=X` originally used when
    logging that iter's `elo` metric to aim). Used for WHR overwrite-logging
    to keep step alignment with the original elo curve.

    Returns an int array of length `total_agents`. Missing entries (or a
    missing file on resume) are filled with linear interpolation based on
    the latest `total_train_steps.txt`.
    """
    path = os.path.join(experiment_dir, "train_steps_per_iter.csv")
    if os.path.isfile(path):
        try:
            arr = np.loadtxt(path, delimiter=",").astype(np.int64)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            out = np.zeros(total_agents, dtype=np.int64)
            n = min(len(arr), total_agents)
            out[:n] = arr[:n]
            return out
        except Exception:
            pass
    # Fallback: linear-interpolate from the single saved total_train_steps.
    tts_path = os.path.join(experiment_dir, "total_train_steps.txt")
    out = np.zeros(total_agents, dtype=np.int64)
    if os.path.isfile(tts_path):
        try:
            tts = int(float(np.loadtxt(tts_path)))
            # Find which iter we're at by counting checkpoints.
            ckpt_dir = os.path.join(experiment_dir, "checkpoint")
            if os.path.isdir(ckpt_dir):
                ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
                latest = max((int(f.split("-", 1)[0]) for f in ckpts), default=0)
                if latest > 0:
                    # Linear interpolation across iters [0..latest].
                    for j in range(min(latest + 1, total_agents)):
                        out[j] = int(round(tts * (j + 1) / (latest + 1)))
        except Exception:
            pass
    return out


def _save_train_steps_per_iter(experiment_dir, arr):
    """Persist train_steps_per_iter (int array) to CSV atomically."""
    path = os.path.join(experiment_dir, "train_steps_per_iter.csv")
    tmp_path = path + ".tmp"
    np.savetxt(tmp_path, arr.astype(np.int64), delimiter=",", fmt="%d")
    os.replace(tmp_path, path)


def _load_counts(experiment_dir, variant_name, total_agents, wr_matrix,
                 assumed_batch=64):
    """Load wins[i,j] and draws[i,j] count matrices from disk.

    `variant_name` is "" for the overall matrices, or one of the variant
    names ("skirmish", etc.) for per-variant matrices.

    If the count files don't exist (legacy run), fall back to deriving
    approximate counts from the existing win-rate matrix and an assumed
    per-comparison batch size. This is imperfect (it loses the W/L/D
    decomposition for legacy entries, treating them as fully decisive)
    but keeps WHR functional for resumed legacy runs.

    Returns (wins, draws) as (n, n) int64 matrices.
    """
    suffix = f"_{variant_name}" if variant_name else ""
    wins_path = os.path.join(experiment_dir, f"wins{suffix}.csv")
    draws_path = os.path.join(experiment_dir, f"draws{suffix}.csv")

    wins = np.zeros((total_agents, total_agents), dtype=np.int64)
    draws = np.zeros((total_agents, total_agents), dtype=np.int64)

    if os.path.exists(wins_path) and os.path.exists(draws_path):
        try:
            tmp_w = np.loadtxt(wins_path, delimiter=",").astype(np.int64)
            tmp_d = np.loadtxt(draws_path, delimiter=",").astype(np.int64)
            sz = min(tmp_w.shape[0], total_agents)
            wins[:sz, :sz] = tmp_w[:sz, :sz]
            draws[:sz, :sz] = tmp_d[:sz, :sz]
            return wins, draws
        except Exception:
            pass

    # Fallback: derive counts from existing wr matrix using assumed batch size.
    # We treat each non-NaN wr[i,j] as `assumed_batch` decisive games with the
    # given win fraction (rounded). Draws are zero in this fallback because we
    # can't recover them from the half-draw-collapsed rate.
    if wr_matrix is not None:
        n = min(wr_matrix.shape[0], total_agents)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r = wr_matrix[i, j]
                if not np.isfinite(r):
                    continue
                wins[i, j] = int(round(r * assumed_batch))
                # The symmetric j-row entry will be set when we visit (j, i).
    return wins, draws


def _save_counts(experiment_dir, variant_name, wins, draws):
    """Persist wins and draws integer matrices atomically."""
    suffix = f"_{variant_name}" if variant_name else ""
    wins_path = os.path.join(experiment_dir, f"wins{suffix}.csv")
    draws_path = os.path.join(experiment_dir, f"draws{suffix}.csv")
    for path, arr in [(wins_path, wins), (draws_path, draws)]:
        tmp_path = path + ".tmp"
        np.savetxt(tmp_path, arr.astype(np.int64), delimiter=",", fmt="%d")
        os.replace(tmp_path, path)


def _load_chunk(reservoir_dir, chunk_idx):
    """Load a reservoir chunk, returning (c, v, pi, iters) tensors."""
    prefix = os.path.join(reservoir_dir, f"chunk_{chunk_idx:02d}")
    c = load_compressed(prefix + "_canonical.ptz")
    v = load_compressed(prefix + "_v.ptz")
    pi = load_compressed(prefix + "_pi.ptz")
    iters = load_compressed(prefix + "_iters.ptz")
    return c, v, pi, iters


def _save_chunk(reservoir_dir, chunk_idx, c, v, pi, iters, zstd_level=1):
    """Save a reservoir chunk atomically."""
    prefix = os.path.join(reservoir_dir, f"chunk_{chunk_idx:02d}")
    _atomic_save_compressed(c, prefix + "_canonical.ptz", zstd_level=zstd_level)
    _atomic_save_compressed(v, prefix + "_v.ptz", zstd_level=zstd_level)
    _atomic_save_compressed(pi, prefix + "_pi.ptz", zstd_level=zstd_level)
    _atomic_save_compressed(iters, prefix + "_iters.ptz", half_storage=False, zstd_level=zstd_level)


def glob_reservoir_chunks(reservoir_dir):
    """Glob for reservoir chunk files, returning (c_path, v_path, pi_path, size) tuples.

    Uses metadata chunk_size when available, falls back to loading canonical tensor.
    """
    c_files = sorted(glob.glob(os.path.join(reservoir_dir, "chunk_*_canonical.ptz")))
    if not c_files:
        return []

    # Try to get per-chunk sizes from metadata (avoids loading tensors)
    meta = _load_reservoir_meta(reservoir_dir)
    chunk_sizes = meta.get("chunk_sizes") if meta else None

    triples = []
    for i, c_path in enumerate(c_files):
        v_path = c_path.replace("_canonical.ptz", "_v.ptz")
        pi_path = c_path.replace("_canonical.ptz", "_pi.ptz")
        if os.path.exists(v_path) and os.path.exists(pi_path):
            if chunk_sizes and i < len(chunk_sizes):
                size = chunk_sizes[i]
            else:
                # Backward compat: no chunk_sizes in metadata, load tensor
                c_tensor = load_compressed(c_path)
                size = c_tensor.shape[0]
                del c_tensor
            triples.append((c_path, v_path, pi_path, size))
    return triples


def _index_staging(staging_dir):
    """Lightweight listing of staging files.

    Returns a list of (c_path, v_path, pi_path, size, iter_num) — no data
    loaded. The streaming reservoir code uses this so it never has to hold
    all of staging in memory at once.
    """
    staging_triples = glob_file_triples(staging_dir)
    out = []
    for c_path, v_path, pi_path, size in staging_triples:
        iter_num = int(os.path.basename(c_path).split("-")[0])
        out.append((c_path, v_path, pi_path, size, iter_num))
    return out


def _sample_staging_rows(file_index, n_samples, offset=0):
    """Random-sample rows from staging[offset:] across the indexed files.

    Loads only the files that contain selected rows; uses ``torch.randperm``
    on the *remaining* row count so RNG consumption matches the legacy
    ``staging_x[perm]`` path exactly. Returns (c, v, pi, iters) tensors in
    perm-order (i.e. the same row order ``staging[perm]`` would produce),
    or ``None`` if there are no remaining rows.
    """
    total = sum(sz for _, _, _, sz, _ in file_index)
    remaining = total - offset
    if remaining <= 0:
        return None

    if n_samples >= remaining:
        # All-rows path: matches legacy "use staging slice directly" (no RNG).
        perm = torch.arange(remaining, dtype=torch.long)
    else:
        perm = torch.randperm(remaining)[:n_samples]

    abs_perm = perm + offset
    sorted_abs, sort_back = abs_perm.sort()
    n_sp = sorted_abs.shape[0]

    cs, vs, pis, iters_list = [], [], [], []
    cum = 0
    sp_idx = 0
    for c_path, v_path, pi_path, sz, iter_num in file_index:
        end = sp_idx
        while end < n_sp and int(sorted_abs[end]) < cum + sz:
            end += 1
        if end > sp_idx:
            local = (sorted_abs[sp_idx:end] - cum).long()
            c = load_compressed(c_path)
            v = load_compressed(v_path)
            pi = load_compressed(pi_path)
            cs.append(c[local].clone())
            vs.append(v[local].clone())
            pis.append(pi[local].clone())
            iters_list.append(torch.full((local.numel(),), iter_num, dtype=torch.int16))
            del c, v, pi
        sp_idx = end
        cum += sz

    c_sorted = torch.cat(cs)
    v_sorted = torch.cat(vs)
    pi_sorted = torch.cat(pis)
    iters_sorted = torch.cat(iters_list)

    inv = sort_back.argsort()
    return c_sorted[inv], v_sorted[inv], pi_sorted[inv], iters_sorted[inv]


def _migrate_legacy_reservoir(config, reservoir_dir):
    """One-time migration from per-iteration reservoir files to chunked format."""
    legacy_triples = glob_file_triples(reservoir_dir)
    legacy_triples += glob_file_triples(reservoir_dir, "*-canonical-*.pt")
    if not legacy_triples:
        return

    tqdm.tqdm.write(f"Migrating {len(legacy_triples)} legacy reservoir files to chunked format...")
    chunk_size = config.reservoir_chunk_size
    chunks_filled = 0
    chunk_sizes_list = []
    buf_c, buf_v, buf_pi, buf_iters = [], [], [], []
    buf_len = 0

    for c_path, v_path, pi_path, size in tqdm.tqdm(legacy_triples, desc="Migration", leave=False):
        c = _load_hist_tensor(c_path)
        v = _load_hist_tensor(v_path)
        pi = _load_hist_tensor(pi_path)
        iter_num = int(os.path.basename(c_path).split("-")[0])
        buf_c.append(c)
        buf_v.append(v)
        buf_pi.append(pi)
        buf_iters.append(torch.full((size,), iter_num, dtype=torch.int16))
        buf_len += size

        while buf_len >= chunk_size:
            all_c = torch.cat(buf_c)
            all_v = torch.cat(buf_v)
            all_pi = torch.cat(buf_pi)
            all_iters = torch.cat(buf_iters)
            _save_chunk(reservoir_dir, chunks_filled,
                        all_c[:chunk_size], all_v[:chunk_size],
                        all_pi[:chunk_size], all_iters[:chunk_size],
                        zstd_level=config.zstd_level)
            # Keep remainder
            remainder = buf_len - chunk_size
            if remainder > 0:
                buf_c = [all_c[chunk_size:]]
                buf_v = [all_v[chunk_size:]]
                buf_pi = [all_pi[chunk_size:]]
                buf_iters = [all_iters[chunk_size:]]
            else:
                buf_c, buf_v, buf_pi, buf_iters = [], [], [], []
            buf_len = remainder
            chunk_sizes_list.append(chunk_size)
            chunks_filled += 1
            del all_c, all_v, all_pi, all_iters

    # Save final partial chunk
    if buf_len > 0:
        all_c = torch.cat(buf_c)
        all_v = torch.cat(buf_v)
        all_pi = torch.cat(buf_pi)
        all_iters = torch.cat(buf_iters)
        _save_chunk(reservoir_dir, chunks_filled,
                    all_c, all_v, all_pi, all_iters,
                    zstd_level=config.zstd_level)
        chunk_sizes_list.append(buf_len)
        chunks_filled += 1

    # Determine last_updated from iteration numbers in filenames
    last_updated = []
    for i in range(chunks_filled):
        _, _, _, iters = _load_chunk(reservoir_dir, i)
        last_updated.append(int(iters.max().item()))

    meta = {
        "version": 2,
        "n_chunks": config.reservoir_n_chunks,
        "chunk_size": chunk_size,
        "chunk_sizes": chunk_sizes_list,
        "chunks_filled": chunks_filled,
        "last_updated": last_updated,
    }
    _save_reservoir_meta(reservoir_dir, meta)

    # Delete legacy files
    for c_path, v_path, pi_path, _ in legacy_triples:
        for p in (c_path, v_path, pi_path):
            if os.path.exists(p):
                os.remove(p)

    tqdm.tqdm.write(f"Migration complete: {chunks_filled} chunks created")


def update_reservoir(config, paths, iteration, hist_size):
    """Move evicted window data to reservoir staging, then periodically merge into chunks.

    Streams staging file-by-file so peak memory is bounded by one chunk's
    worth of samples regardless of how many iterations of staging
    accumulated. RNG consumption matches the previous concat-then-sample
    implementation; see test_reservoir_streaming.py for the equivalence
    proof.
    """
    hist_location = paths["history"]
    reservoir_location = paths["reservoir"]
    staging_dir = os.path.join(reservoir_location, "staging")

    oldest_in_window = max(0, iteration - hist_size)
    prev_oldest = max(0, (iteration - 1) - calc_hist_size(config, iteration - 1))
    evicted_iters = list(range(prev_oldest, oldest_in_window))

    if evicted_iters:
        os.makedirs(staging_dir, exist_ok=True)
        for it in tqdm.tqdm(evicted_iters, desc="Moving Evicted History", leave=False):
            for pattern in (f"{it:04d}-*.ptz", f"{it:04d}-*.pt"):
                for src in glob.glob(os.path.join(hist_location, pattern)):
                    dst = os.path.join(staging_dir, os.path.basename(src))
                    os.rename(src, dst)

    if iteration % config.reservoir_update_interval != 0:
        return

    os.makedirs(reservoir_location, exist_ok=True)

    meta = _load_reservoir_meta(reservoir_location)
    if meta is None:
        legacy_triples = glob_file_triples(reservoir_location)
        legacy_triples += glob_file_triples(reservoir_location, "*-canonical-*.pt")
        if legacy_triples:
            _migrate_legacy_reservoir(config, reservoir_location)
            meta = _load_reservoir_meta(reservoir_location)

    file_index = _index_staging(staging_dir) if os.path.isdir(staging_dir) else []
    total_staging = sum(sz for _, _, _, sz, _ in file_index)
    if total_staging == 0:
        return

    if meta is None:
        meta = {
            "version": 2,
            "n_chunks": config.reservoir_n_chunks,
            "chunk_size": config.reservoir_chunk_size,
            "chunk_sizes": [],
            "chunks_filled": 0,
            "last_updated": [],
        }

    n_chunks = config.reservoir_n_chunks
    chunk_size = config.reservoir_chunk_size
    chunks_filled = meta["chunks_filled"]

    if "chunk_sizes" not in meta:
        meta["chunk_sizes"] = [chunk_size] * chunks_filled

    # FILLING PHASE — stream files sequentially into chunk-sized batches.
    # Peak memory: one staging file + one chunk in flight.
    consumed = 0
    if chunks_filled < n_chunks:
        buf_c, buf_v, buf_pi, buf_iters = [], [], [], []
        buf_n = 0
        file_iter = iter(file_index)
        cur = None  # [c, v, pi, iters, pos, sz]

        while chunks_filled < n_chunks and consumed < total_staging:
            # Refill buf up to chunk_size or until staging is exhausted.
            while buf_n < chunk_size and consumed < total_staging:
                if cur is None:
                    try:
                        c_path, v_path, pi_path, sz, iter_num = next(file_iter)
                    except StopIteration:
                        break
                    c = load_compressed(c_path)
                    v = load_compressed(v_path)
                    pi = load_compressed(pi_path)
                    iters = torch.full((sz,), iter_num, dtype=torch.int16)
                    cur = [c, v, pi, iters, 0, sz]
                c, v, pi, iters, pos, sz = cur
                take = min(chunk_size - buf_n, sz - pos)
                buf_c.append(c[pos:pos + take])
                buf_v.append(v[pos:pos + take])
                buf_pi.append(pi[pos:pos + take])
                buf_iters.append(iters[pos:pos + take])
                buf_n += take
                consumed += take
                cur[4] = pos + take
                if cur[4] >= sz:
                    cur = None

            if buf_n == 0:
                break

            out_c = torch.cat(buf_c)
            out_v = torch.cat(buf_v)
            out_pi = torch.cat(buf_pi)
            out_iters = torch.cat(buf_iters)
            _save_chunk(reservoir_location, chunks_filled,
                        out_c, out_v, out_pi, out_iters,
                        zstd_level=config.zstd_level)
            meta["last_updated"].append(iteration)
            meta["chunk_sizes"].append(buf_n)
            chunks_filled += 1
            buf_c, buf_v, buf_pi, buf_iters = [], [], [], []
            buf_n = 0
            del out_c, out_v, out_pi, out_iters

        meta["chunks_filled"] = chunks_filled
        del buf_c, buf_v, buf_pi, buf_iters, cur

    # MERGE PHASE — sample new_per_chunk rows per chunk via the streaming
    # sampler, materializing only the rows needed for this chunk.
    if chunks_filled >= n_chunks and consumed < total_staging:
        remaining = total_staging - consumed
        K = min(config.reservoir_chunks_per_update, chunks_filled)
        last_updated = meta["last_updated"]
        target_indices = sorted(range(chunks_filled),
                                key=lambda i: last_updated[i])[:K]

        C = (n_chunks / K) * config.reservoir_update_interval
        decay = config.reservoir_recency_decay
        target_rate = 1 - decay ** C
        w_old_est = decay ** (C / 2)
        new_per_chunk = int(target_rate * chunk_size * w_old_est / (1 - target_rate))
        new_per_chunk = max(1, min(new_per_chunk, remaining))

        for chunk_idx in tqdm.tqdm(target_indices, desc="Merging Reservoir Chunks", leave=False):
            sampled = _sample_staging_rows(file_index, new_per_chunk, offset=consumed)
            if sampled is None:
                break
            new_c, new_v, new_pi, new_iters = sampled
            del sampled

            old_c, old_v, old_pi, old_iters = _load_chunk(reservoir_location, chunk_idx)
            pool_c = torch.cat([old_c, new_c])
            pool_v = torch.cat([old_v, new_v])
            pool_pi = torch.cat([old_pi, new_pi])
            pool_iters = torch.cat([old_iters, new_iters])
            del old_c, old_v, old_pi, old_iters, new_c, new_v, new_pi, new_iters

            ages = (iteration - pool_iters.float()).clamp(min=0)
            weights = decay ** ages
            select_size = min(chunk_size, pool_c.shape[0])
            if select_size < pool_c.shape[0]:
                selected = torch.multinomial(weights, select_size, replacement=False)
            else:
                selected = torch.arange(pool_c.shape[0])

            _save_chunk(reservoir_location, chunk_idx,
                        pool_c[selected], pool_v[selected],
                        pool_pi[selected], pool_iters[selected],
                        zstd_level=config.zstd_level)
            meta["chunk_sizes"][chunk_idx] = select_size
            del pool_c, pool_v, pool_pi, pool_iters, weights, selected, ages
            last_updated[chunk_idx] = iteration
            gc.collect()

        meta["last_updated"] = last_updated

    _save_reservoir_meta(reservoir_location, meta)

    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)

    gc.collect()


def load_reservoir(paths, num_workers=1):
    """Load reservoir as a TensorDataset, or None if no reservoir exists.

    Supports chunked format (primary), per-iteration files (legacy), and
    monolithic files (oldest legacy).
    """
    reservoir_dir = paths["reservoir"]

    # Try chunked format first
    meta = _load_reservoir_meta(reservoir_dir)
    if meta is not None and meta.get("chunks_filled", 0) > 0:
        chunk_triples = glob_reservoir_chunks(reservoir_dir)
        if chunk_triples:
            loaded = _parallel_load_triples(chunk_triples, num_workers, desc="Loading Reservoir")
            cs = [ct for ct, _, _ in loaded]
            vs = [vt for _, vt, _ in loaded]
            pis = [pt for _, _, pt in loaded]
            return TensorDataset(torch.cat(cs), torch.cat(vs), torch.cat(pis))

    # Try per-iteration format
    triples = glob_file_triples(reservoir_dir)
    if triples:
        loaded = _parallel_load_triples(triples, num_workers, desc="Loading Reservoir")
        cs = [ct for ct, _, _ in loaded]
        vs = [vt for _, vt, _ in loaded]
        pis = [pt for _, _, pt in loaded]
        return TensorDataset(torch.cat(cs), torch.cat(vs), torch.cat(pis))

    # Fall back to old monolithic format
    res_c_path = os.path.join(reservoir_dir, "canonical.ptz")
    if not os.path.exists(res_c_path):
        return None
    c = load_compressed(res_c_path)
    v = load_compressed(os.path.join(reservoir_dir, "v.ptz"))
    pi = load_compressed(os.path.join(reservoir_dir, "pi.ptz"))
    return TensorDataset(c, v, pi)


def load_available_window(paths, start, end, num_workers=1):
    """Load whatever window history files exist across an iteration range."""
    hist_location = paths["history"]
    all_triples = []
    for i in range(start, end):
        c = _glob_hist_files(hist_location, f"{i:04d}-*-canonical-*")
        v = _glob_hist_files(hist_location, f"{i:04d}-*-v-*")
        p = _glob_hist_files(hist_location, f"{i:04d}-*-pi-*")
        all_triples.extend(zip(c, v, p))
    loaded = _parallel_load_triples(all_triples, num_workers, desc="Loading Window")
    return [TensorDataset(ct, vt, pt) for ct, vt, pt in loaded]


class StreamingCompressedDataset(IterableDataset):
    """Memory-bounded streaming dataset with cross-file batch mixing.

    Holds up to ``active_files`` loaded triples at once and builds each batch
    by round-robin slicing across them, so samples in a single batch come from
    multiple iters' history. When a slot drains, the prefetched next file
    takes its place. Yields (canonical, v, pi) batches; use with
    DataLoader(batch_size=None, num_workers=0).
    """

    def __init__(self, file_triples, batch_size, passes=1, active_files=8, seed=None):
        self.file_triples = file_triples  # [(c_path, v_path, pi_path, size), ...]
        self.batch_size = batch_size
        self.passes = passes
        self.active_files = max(1, active_files)
        self.seed = seed

    def __iter__(self):
        for p in range(self.passes):
            yield from self._one_pass(p)

    def _one_pass(self, pass_idx=0):
        order = list(range(len(self.file_triples)))
        if self.seed is None:
            random.shuffle(order)
        else:
            random.Random(self.seed + pass_idx).shuffle(order)
        if not order:
            return

        target = min(self.active_files, len(order))
        next_idx = 0
        F = len(self.file_triples)

        def _load(i):
            file_idx = order[i]
            sub_seed = None
            if self.seed is not None:
                sub_seed = (self.seed + pass_idx * F + file_idx) & ((1 << 63) - 1)
            return list(_load_and_shuffle(*self.file_triples[file_idx], seed=sub_seed)) + [0]
            # slot = [c, v, pi, cursor]

        n_workers = min(4, target)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            pending = deque()
            while next_idx < len(order) and len(pending) < target + n_workers:
                pending.append(executor.submit(_load, next_idx))
                next_idx += 1

            active = []
            while pending and len(active) < target:
                active.append(pending.popleft().result())

            while active:
                n = len(active)
                per = self.batch_size // n
                rem = self.batch_size - per * n

                cs, vs, pis = [], [], []
                for fi in range(n):
                    slot = active[fi]
                    want = per + (1 if fi < rem else 0)
                    avail = slot[0].size(0) - slot[3]
                    take = min(want, avail)
                    if take > 0:
                        end = slot[3] + take
                        cs.append(slot[0][slot[3]:end])
                        vs.append(slot[1][slot[3]:end])
                        pis.append(slot[2][slot[3]:end])
                        slot[3] = end

                if cs:
                    yield (
                        torch.cat(cs) if len(cs) > 1 else cs[0],
                        torch.cat(vs) if len(vs) > 1 else vs[0],
                        torch.cat(pis) if len(pis) > 1 else pis[0],
                    )

                # Evict drained slots, refill from prefetch queue
                drained = [i for i, s in enumerate(active) if s[3] >= s[0].size(0)]
                for i in reversed(drained):
                    active.pop(i)
                while len(active) < target and pending:
                    active.append(pending.popleft().result())
                    if next_idx < len(order):
                        pending.append(executor.submit(_load, next_idx))
                        next_idx += 1


@tracy_zone
def self_play(config, paths, experiment_name, best, iteration, depth, fast_depth,
              variant_probs=None, fallback_checkpoint_dir=None):
    Game = config.Game
    bs = config.self_play_batch_size
    cb = Game.NUM_PLAYERS() * config.self_play_concurrent_batch_mult
    n = games_per_iter(config)
    params = base_params(config, config.self_play_temp, bs, cb)
    params.games_to_play = n
    params.mcts_visits = [depth] * Game.NUM_PLAYERS()
    params.history_enabled = True
    params.history_value_target = (
        alphazero.HistoryValueTarget.ROOT
        if config.value_target == "root"
        else alphazero.HistoryValueTarget.FINAL
    )
    params.epsilon = 0.25
    params.playout_cap_randomization = config.playout_cap_percent > 0.0
    params.playout_cap_depth = fast_depth
    params.playout_cap_percent = config.playout_cap_percent
    params.resign_percent = config.resign_percent
    params.resign_playthrough_percent = config.resign_playthrough_percent
    params.mcts_root_temp = config.mcts_root_temp
    params.root_fpu_zero = config.root_fpu_zero
    params.shaped_dirichlet = config.shaped_dirichlet
    params.policy_target_pruning = config.policy_target_pruning

    use_rand = best == 0

    if use_rand:
        nn = RandPlayer()
        params.max_cache_size = 0
    else:
        nn = _resolve_checkpoint(Game, paths["checkpoint"], experiment_name,
                                 best, fallback_checkpoint_dir)
        prepare_inference_model(nn, config, compile=config.torch_compile)

    players = [nn] * Game.NUM_PLAYERS()
    set_model_groups(params, players)
    set_eval_types(params, players)

    # Clean tmp_history before self-play
    shutil.rmtree(paths["tmp_history"], ignore_errors=True)

    pm = alphazero.PlayManager(_make_game_instance(config, variant_probs), params)
    grargs = GRArgs(
        title="Self Play",
        game=Game,
        iteration=iteration,
        max_batch_size=bs,
        record_batch_metrics=True,
        data_folder=paths["tmp_history"],
    )

    gr = GameRunner(players, pm, grargs)
    gr.run()
    scores = pm.scores()
    win_rates = [0] * len(scores)
    sn = sum(scores)
    if sn > 0:
        for i in range(len(scores)):
            win_rates[i] = scores[i] / sn
    resign_scores = pm.resign_scores()
    resign_win_rates = [0] * len(resign_scores)
    rn = sum(resign_scores)
    if rn > 0:
        for i in range(len(resign_scores)):
            resign_win_rates[i] = resign_scores[i] / rn
    resign_rate = rn / sn if sn > 0 else 0
    hits = pm.cache_hits()
    misses = pm.cache_misses()
    total = hits + misses
    hr = 0
    if total > 0:
        hr = hits / total
    reinserts = pm.cache_reinserts()
    theoretical_hr = (hits + reinserts) / total if total > 0 else 0
    agl = pm.avg_game_length()
    avg_depth = pm.avg_leaf_depth()
    avg_entropy = pm.avg_search_entropy()
    fast_avg_depth = pm.fast_avg_leaf_depth()
    fast_avg_entropy = pm.fast_avg_search_entropy()
    avg_mpt = pm.avg_moves_per_turn()
    avg_vm = pm.avg_valid_moves()
    # Cache pressure metrics
    max_cache = pm.cache_max_size()
    cache_sz = pm.cache_size()
    saturation = cache_sz / max_cache if max_cache > 0 else 0
    evictions = pm.cache_evictions()
    churn_rate = evictions / hits if hits > 0 else 0
    # Batch size metrics
    with gr._batch_lock:
        sizes = gr._batch_sizes
    if sizes:
        sizes_sorted = sorted(sizes)
        avg_batch_size = sum(sizes) / len(sizes)
        min_batch_size = sizes_sorted[0]
        max_batch_size = sizes_sorted[-1]
        n_bs = len(sizes_sorted)
        median_batch_size = (sizes_sorted[n_bs // 2] + sizes_sorted[(n_bs - 1) // 2]) / 2
    else:
        avg_batch_size = min_batch_size = max_batch_size = median_batch_size = 0
    # Inference time metrics
    with gr._batch_lock:
        inf_times = gr._inference_times
    if inf_times:
        resolved = []
        for t in inf_times:
            if isinstance(t, tuple):
                t[1].synchronize()
                resolved.append(t[0].elapsed_time(t[1]) / 1000.0)
            else:
                resolved.append(t)
        inf_sorted = sorted(resolved)
        avg_inference_ms = (sum(inf_sorted) / len(inf_sorted)) * 1000
        min_inference_ms = inf_sorted[0] * 1000
        max_inference_ms = inf_sorted[-1] * 1000
        n_it = len(inf_sorted)
        median_inference_ms = ((inf_sorted[n_it // 2] + inf_sorted[(n_it - 1) // 2]) / 2) * 1000
    else:
        avg_inference_ms = min_inference_ms = max_inference_ms = median_inference_ms = 0
    variant_game_counts = {}
    variant_win_rates = {}
    for vid in range(pm.num_tracked_variants()):
        vgames = pm.variant_games_completed(vid)
        variant_game_counts[vid] = vgames
        if vgames > 0:
            vscores = pm.variant_scores(vid)
            vn = sum(vscores) or 1.0
            variant_win_rates[vid] = [float(vscores[j]) / vn for j in range(len(vscores))]
    variant_metrics = _collect_variant_metrics(pm, include_fast=True)
    del gr, pm, nn
    gc.collect()
    return SelfPlayResult(
        win_rates=win_rates, hit_rate=hr, game_length=agl,
        resign_win_rates=resign_win_rates, resign_rate=resign_rate,
        avg_depth=avg_depth, avg_entropy=avg_entropy,
        fast_avg_depth=fast_avg_depth, fast_avg_entropy=fast_avg_entropy,
        avg_mpt=avg_mpt, avg_vm=avg_vm,
        saturation=saturation, churn_rate=churn_rate,
        avg_batch_size=avg_batch_size, median_batch_size=median_batch_size,
        min_batch_size=min_batch_size, max_batch_size=max_batch_size,
        avg_inference_ms=avg_inference_ms, median_inference_ms=median_inference_ms,
        min_inference_ms=min_inference_ms, max_inference_ms=max_inference_ms,
        theoretical_hr=theoretical_hr,
        variant_game_counts=variant_game_counts,
        variant_win_rates=variant_win_rates,
        variant_metrics=variant_metrics,
    )


def _resolve_checkpoint(Game, primary_dir, experiment_name, iteration, fallback_dir=None):
    """Load a checkpoint, falling back to a secondary directory if not found locally."""
    import neural_net
    primary_path = os.path.join(primary_dir, f"{iteration:04d}-{experiment_name}.pt")
    if os.path.exists(primary_path):
        return neural_net.NNWrapper.load_checkpoint(Game, primary_dir, f"{iteration:04d}-{experiment_name}.pt")
    cps = glob.glob(os.path.join(primary_dir, f"{iteration:04d}-*.pt"))
    if cps:
        return neural_net.NNWrapper.load_checkpoint(Game, "", cps[0])
    if fallback_dir:
        fallback_path = os.path.join(fallback_dir, f"{iteration:04d}-{experiment_name}.pt")
        if os.path.exists(fallback_path):
            return neural_net.NNWrapper.load_checkpoint(Game, fallback_dir, f"{iteration:04d}-{experiment_name}.pt")
        cps = glob.glob(os.path.join(fallback_dir, f"{iteration:04d}-*.pt"))
        if cps:
            return neural_net.NNWrapper.load_checkpoint(Game, "", cps[0])
    raise FileNotFoundError(f"Checkpoint {iteration:04d} not found in {primary_dir}" +
                           (f" or {fallback_dir}" if fallback_dir else ""))


def _raw_policy_action(policy, valid_moves, temperature, rng):
    """Choose a legal action directly from policy probabilities, with no MCTS."""
    probs = np.asarray(policy, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid_moves, dtype=bool).reshape(-1)
    if probs.shape[0] != valid.shape[0]:
        raise ValueError(
            f"policy/valid size mismatch: {probs.shape[0]} vs {valid.shape[0]}"
        )
    legal = np.flatnonzero(valid)
    if legal.size == 0:
        raise ValueError("cannot choose a raw-policy action with no legal moves")
    masked = np.where(valid, probs, 0.0)
    masked = np.nan_to_num(masked, nan=0.0, posinf=0.0, neginf=0.0)
    masked = np.maximum(masked, 0.0)
    if temperature <= 0.0:
        legal_scores = masked[legal]
        if np.any(legal_scores > 0.0):
            return int(legal[int(np.argmax(legal_scores))])
        return int(legal[0])
    if masked.sum() <= 0.0:
        weights = np.ones(legal.size, dtype=np.float64) / legal.size
    else:
        # Temperature on already-normalized policy probabilities. This is
        # intentionally raw-policy sampling: no search values or visit counts.
        legal_scores = np.power(masked[legal], 1.0 / temperature)
        total = legal_scores.sum()
        weights = (
            legal_scores / total
            if total > 0.0 and np.isfinite(total)
            else np.ones(legal.size, dtype=np.float64) / legal.size
        )
    return int(rng.choice(legal, p=weights))


def _raw_policy_process(agent, canonical):
    batch = torch.from_numpy(np.asarray(canonical, dtype=np.float32)).unsqueeze(0)
    if hasattr(agent, "process"):
        _, pi = agent.process(batch)
    else:
        _, pi = agent.predict(batch)
    if hasattr(pi, "detach"):
        pi = pi.detach().cpu().numpy()
    return np.asarray(pi, dtype=np.float64)[0]


def selfplay_parent_iteration(config, iteration, current_best):
    """Choose the checkpoint that generates this iteration's self-play."""
    if bool(getattr(config, "selfplay_use_latest", False)):
        return int(iteration)
    return int(current_best)


def update_uttt_ai_champion(state, iteration, win_rate, draw_rate):
    """Update external champion state from a raw-policy UTTT.ai result."""
    score = float(win_rate) + 0.5 * float(draw_rate)
    previous = float(state.get("score", float("-inf")))
    improved = score > previous
    if improved:
        state = {
            "iteration": int(iteration),
            "score": score,
            "win_rate": float(win_rate),
            "draw_rate": float(draw_rate),
        }
    return state, improved


@tracy_zone
def play_raw_policy_match(Game, players, games, temperature=0.0, seed=0,
                          opening_plies=0):
    """Play raw policy-head games with alternating seats.

    Returns ``(player0_win_rate, draw_rate, avg_game_length)`` where player0 is
    ``players[0]`` in the supplied list. The game is played without MCTS.
    ``opening_plies`` random legal moves are applied before policy play to avoid
    replaying the same two deterministic games when temperature is zero.
    """
    if Game.NUM_PLAYERS() != 2:
        raise ValueError("raw external benchmark currently supports two-player games only")
    rng = np.random.default_rng(seed)
    wins = [0, 0]
    draws = 0
    total_moves = 0
    for game_idx in range(games):
        seat_agents = [players[game_idx % 2], players[(game_idx + 1) % 2]]
        seat_to_player = [game_idx % 2, (game_idx + 1) % 2]
        gs = Game()
        moves = 0
        for _ in range(opening_plies):
            if gs.scores() is not None:
                break
            valid = np.asarray(gs.valid_moves())
            legal = np.flatnonzero(valid)
            if len(legal) == 0:
                break
            gs.play_move(int(rng.choice(legal)))
            moves += 1
        while gs.scores() is None:
            seat = gs.current_player()
            policy = _raw_policy_process(seat_agents[seat], gs.canonicalized())
            action = _raw_policy_action(policy, gs.valid_moves(), temperature, rng)
            gs.play_move(action)
            moves += 1
        scores = np.asarray(gs.scores(), dtype=np.float64)
        total_moves += moves
        if scores[-1] > 0:
            draws += 1
        else:
            winner_seat = int(np.argmax(scores[:2]))
            wins[seat_to_player[winner_seat]] += 1
    denom = max(1, games)
    return wins[0] / denom, draws / denom, total_moves / denom


_UTTT_AI_EVAL_WARNED = False


def maybe_eval_uttt_ai(config, paths, experiment_name, iteration, run, total_train_steps):
    """Track raw-policy W/D rates versus the cached uttt.ai model if enabled."""
    global _UTTT_AI_EVAL_WARNED
    interval = int(getattr(config, "uttt_ai_eval_interval", 0))
    if interval <= 0 or config.game != "uttt" or iteration <= 0 or iteration % interval != 0:
        return None
    try:
        _ensure_project_root_importable()
        import neural_net
        from tools.uttt_ai_adapter import UtttAiPolicyValueNet, default_model_path

        model_path = getattr(config, "uttt_ai_eval_model", "") or default_model_path()
        if not os.path.exists(model_path):
            if not _UTTT_AI_EVAL_WARNED:
                tqdm.tqdm.write(
                    f"Skipping UTTT.ai eval: cached model not found at {model_path}"
                )
                _UTTT_AI_EVAL_WARNED = True
            return None
        nn = neural_net.NNWrapper.load_checkpoint(
            config.Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
        )
        prepare_inference_model(nn, config)
        uttt_ai = UtttAiPolicyValueNet(
            model_path,
            require_cuda=bool(getattr(config, "uttt_ai_eval_require_cuda", False)),
        )
        games = int(getattr(config, "uttt_ai_eval_games", 16))
        temp = float(getattr(config, "uttt_ai_eval_temperature", 0.0))
        opening_plies = int(getattr(config, "uttt_ai_eval_opening_plies", 0))
        opening_seed = int(getattr(config, "uttt_ai_eval_opening_seed", -1))
        eval_seed = iteration if opening_seed < 0 else opening_seed
        win_rate, draw_rate, avg_len = play_raw_policy_match(
            config.Game, [nn, uttt_ai], games, temperature=temp, seed=eval_seed,
            opening_plies=opening_plies
        )
        csv_path = os.path.join(paths["experiment"], "uttt_ai_eval.csv")
        needs_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        if not needs_header:
            with open(csv_path, "r", encoding="utf-8") as f:
                header = f.readline()
                rest = f.read()
            if "opening_plies" not in header or "opening_seed" not in header:
                with open(csv_path, "w", encoding="utf-8") as f:
                    fields = header.rstrip("\n").split(",")
                    if "opening_plies" not in fields:
                        fields.append("opening_plies")
                    if "opening_seed" not in fields:
                        fields.append("opening_seed")
                    f.write(",".join(fields) + "\n")
                    for line in rest.splitlines():
                        existing = line.split(",")
                        while len(existing) < len(fields):
                            existing.append("")
                        f.write(",".join(existing[:len(fields)]) + "\n")
        with open(csv_path, "a", encoding="utf-8") as f:
            if needs_header:
                f.write(
                    "iteration,win_rate,draw_rate,average_game_length,games,"
                    "total_train_steps,timestamp,opening_plies,opening_seed\n"
                )
            f.write(
                f"{iteration},{win_rate:.10g},{draw_rate:.10g},{avg_len:.10g},"
                f"{games},{total_train_steps},{time.time():.6f},"
                f"{opening_plies},{eval_seed}\n"
            )
        run.track(
            win_rate, name="win_rate", epoch=iteration, step=total_train_steps,
            context={"vs": "uttt.ai", "search": "raw_policy", "from": "all_games"}
        )
        run.track(
            win_rate, name="uttt_ai_win_rate", epoch=iteration, step=total_train_steps,
            context={"search": "raw_policy", "from": "all_games"}
        )
        run.track(
            draw_rate, name="draw_rate", epoch=iteration, step=total_train_steps,
            context={"vs": "uttt.ai", "search": "raw_policy", "from": "all_games"}
        )
        run.track(
            draw_rate, name="uttt_ai_draw_rate", epoch=iteration, step=total_train_steps,
            context={"search": "raw_policy", "from": "all_games"}
        )
        run.track(
            win_rate + 0.5 * draw_rate, name="uttt_ai_score",
            epoch=iteration, step=total_train_steps,
            context={"search": "raw_policy"}
        )
        run.track(
            avg_len, name="average_game_length", epoch=iteration, step=total_train_steps,
            context={"vs": "uttt.ai", "search": "raw_policy"}
        )
        run.track(
            avg_len, name="uttt_ai_average_game_length", epoch=iteration,
            step=total_train_steps, context={"search": "raw_policy"}
        )
        tqdm.tqdm.write(
            f"UTTT.ai raw eval [iter {iteration}]: "
            f"wr={_fmt_pct(win_rate)} draw={_fmt_pct(draw_rate)} "
            f"avg_len={avg_len:.1f} games={games} opening={opening_plies} "
            f"seed={eval_seed}"
        )
        return win_rate, draw_rate, avg_len
    except Exception as exc:
        tqdm.tqdm.write(f"Skipping UTTT.ai eval at iter {iteration}: {exc}")
        return None


@tracy_zone
def play_past(config, paths, experiment_name, depth, iteration, past_iter, batch_size=64,
              fallback_checkpoint_dir=None, variant_probs=None):
    import neural_net

    Game = config.Game
    num_players = Game.NUM_PLAYERS()
    nn = neural_net.NNWrapper.load_checkpoint(
        Game, paths["checkpoint"], f"{iteration:04d}-{experiment_name}.pt"
    )
    # Compare/pitting runs are more stable without torch.compile's capture path.
    prepare_inference_model(nn, config)
    if past_iter == 0:
        nn_past = RandPlayer()
    else:
        nn_past = _resolve_checkpoint(Game, paths["checkpoint"], experiment_name,
                                      past_iter, fallback_checkpoint_dir)
        prepare_inference_model(nn_past, config)

    bs = batch_size
    cb = num_players

    if num_players == 2:
        # 2 players, 2 unique models
        # players list must be length num_players
        players = [nn, nn_past]
        model_groups = [0, 1]
        seat_perms = [[0, 1], [1, 0]]
        n_perms = 2
    else:
        # >2 players: nn (group 0) and nn_past (group 1)
        # players/model_groups must be length num_players
        # Default mapping: seat 0 = nn, rest = nn_past
        players = [nn_past] * num_players
        players[0] = nn
        model_groups = [1] * num_players
        model_groups[0] = 0
        # Generate all seat permutations:
        # First N: nn in each seat, rest nn_past
        seat_perms = []
        for i in range(num_players):
            perm = [1] * num_players
            perm[i] = 0
            seat_perms.append(perm)
        # Next N: nn_past in each seat, rest nn
        for i in range(num_players):
            perm = [0] * num_players
            perm[i] = 1
            seat_perms.append(perm)
        n_perms = len(seat_perms)

    n = bs * cb * n_perms  # total games across all permutations
    params = base_params(config, config.eval_temp, bs, cb)
    params.games_to_play = n
    params.mcts_visits = [depth] * num_players
    params.model_groups = model_groups
    params.seat_perms = seat_perms
    set_eval_types(params, players)

    pm = alphazero.PlayManager(_make_game_instance(config, variant_probs), params)
    grargs = GRArgs(
        title=f"Bench {iteration} v {past_iter}",
        game=Game,
        iteration=iteration,
        max_batch_size=bs,
    )
    gr = GameRunner(players, pm, grargs)
    gr.run()

    # Extract per-perm results to compute nn win rate and draw rate
    nn_rate = 0
    draw_rate = 0
    hr = 0
    agl = pm.avg_game_length()
    avg_depth = pm.avg_leaf_depth()
    avg_entropy = pm.avg_search_entropy()
    avg_mpt = pm.avg_moves_per_turn()
    avg_vm = pm.avg_valid_moves()

    hits = pm.cache_hits()
    total = hits + pm.cache_misses()
    if total > 0:
        hr = hits / total

    # Raw integer counts: total games, games won by nn-side (any nn-seat won),
    # games won by past-side (any past-seat won), and drawn games. These are
    # the canonical "evidence" used by WHR. Sum unchanged-by-perm because
    # perm_scores stores integer win counts per seat.
    n_games = 0
    nn_wins = 0
    past_wins = 0
    n_draws = 0
    for perm_idx in range(pm.num_seat_perms()):
        perm_scores = pm.perm_scores(perm_idx)
        perm_games = pm.perm_games_completed(perm_idx)
        if perm_games == 0:
            continue
        perm = seat_perms[perm_idx]
        n_games += int(perm_games)
        # Accumulate per-side wins (nn-side: perm[seat]==0, past-side: perm[seat]==1).
        for seat in range(num_players):
            if perm[seat] == 0:
                nn_rate += perm_scores[seat] / perm_games
                nn_wins += int(perm_scores[seat])
            else:
                past_wins += int(perm_scores[seat])
        draw_rate += perm_scores[num_players] / perm_games
        n_draws += int(perm_scores[num_players])

    nn_rate /= n_perms
    draw_rate /= n_perms

    # Per-variant rates and counts.
    variant_nn_rates = {}
    variant_draw_rates = {}
    variant_counts = {}  # vid -> (n_games, nn_wins, past_wins, n_draws)
    for vid in range(pm.num_tracked_variants()):
        vgames = pm.variant_games_completed(vid)
        if vgames == 0:
            continue
        vnn = 0.0
        v_n_games = 0
        v_nn_wins = 0
        v_past_wins = 0
        v_draws = 0
        for perm_idx in range(pm.num_seat_perms()):
            vperm_scores = pm.variant_perm_scores(vid, perm_idx)
            vperm_games = pm.variant_perm_games_completed(vid, perm_idx)
            if vperm_games == 0:
                continue
            perm = seat_perms[perm_idx]
            v_n_games += int(vperm_games)
            for seat in range(num_players):
                if perm[seat] == 0:
                    vnn += vperm_scores[seat] / vperm_games
                    v_nn_wins += int(vperm_scores[seat])
                else:
                    v_past_wins += int(vperm_scores[seat])
            v_draws += int(vperm_scores[num_players])
        variant_nn_rates[vid] = vnn / n_perms
        vscores = pm.variant_scores(vid)
        variant_draw_rates[vid] = vscores[num_players] / vgames
        variant_counts[vid] = (v_n_games, v_nn_wins, v_past_wins, v_draws)

    variant_metrics = _collect_variant_metrics(pm)
    del gr, nn, nn_past, pm
    gc.collect()
    return (nn_rate, draw_rate, hr, agl, avg_depth, avg_entropy, avg_mpt, avg_vm,
            variant_nn_rates, variant_draw_rates, variant_metrics,
            n_games, nn_wins, past_wins, n_draws, variant_counts)


def get_lr(config, iteration, lr_state):
    """Compute learning rate for the current iteration."""
    if config.lr_schedule == "constant":
        lr = config.lr
    elif config.lr_schedule == "step":
        lr = config.lr_steps[0][1] if config.lr_steps else config.lr
        for step_iter, step_lr in config.lr_steps:
            if iteration >= step_iter:
                lr = step_lr
    elif config.lr_schedule == "adaptive":
        lr = lr_state['current_lr']
        can_drop = (
            iteration >= config.lr_min_iter
            and iteration - lr_state['last_drop_iter'] >= config.lr_min_between_drops
            and (config.lr_max_drops == 0 or lr_state['num_drops'] < config.lr_max_drops)
            and iteration - lr_state['last_best_iter'] >= config.lr_patience
        )
        if can_drop:
            lr *= config.lr_drop_factor
            lr_state['num_drops'] += 1
            lr_state['last_drop_iter'] = iteration
            lr_state['current_lr'] = lr
    elif config.lr_schedule == "cosine":
        cosine_start = (
            config.lr_cosine_start_iter
            if config.lr_cosine_start_iter >= 0
            else max(0, config.lr_warmup_target)
        )
        cosine_end = (
            config.lr_cosine_end_iter
            if config.lr_cosine_end_iter >= 0
            else config.iterations - 1
        )
        cosine_length = max(1, cosine_end - cosine_start)
        progress = min(1.0, max(0.0, (iteration - cosine_start) / cosine_length))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = config.lr * (
            config.lr_cosine_min_factor
            + (1.0 - config.lr_cosine_min_factor) * cosine_factor
        )
    elif config.lr_schedule == "exponential":
        exponential_start = (
            config.lr_exponential_start_iter
            if config.lr_exponential_start_iter >= 0
            else max(0, config.lr_warmup_target)
        )
        exponential_end = (
            config.lr_exponential_end_iter
            if config.lr_exponential_end_iter >= 0
            else config.iterations - 1
        )
        exponential_length = max(1, exponential_end - exponential_start)
        progress = min(
            1.0,
            max(0.0, (iteration - exponential_start) / exponential_length),
        )
        lr = config.lr * config.lr_exponential_final_factor ** progress
    else:
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")

    # Apply window-fill warmup
    if config.lr_warmup_target > 0 and iteration < config.lr_warmup_target:
        warmup_factor = config.lr_warmup_floor + (1.0 - config.lr_warmup_floor) * (iteration + 1) / config.lr_warmup_target
        lr *= warmup_factor

    return lr


def _default_lr_state(config):
    return {
        'current_lr': config.lr,
        'num_drops': 0,
        'last_drop_iter': -config.lr_min_between_drops,
        'last_best_iter': 0,
    }


def _bootstrap_train_phase(nn, files, config, run, source_n, total_train_steps, phase_name):
    """Run one epoch of bootstrap training with PyTorch-style ReduceLROnPlateau.

    Returns (steps_trained, early_stopped).
    """
    from lr_scheduler import PlateauLRScheduler, ema_update

    bbs = config.train_batch_size
    epochs = max(1, config.bootstrap_epochs)
    total_steps = sum(int(math.ceil(s / bbs)) for _, _, _, s in files) * epochs
    eval_interval = min(config.bootstrap_eval_interval, max(total_steps, 1))

    scheduler = PlateauLRScheduler(
        set_lr=nn.set_lr,
        initial_lr=config.bootstrap_lr,
        drop_factor=config.bootstrap_lr_drop_factor,
        max_drops=config.bootstrap_lr_max_drops,
        patience=eval_interval * config.bootstrap_lr_patience,
        conv_patience=eval_interval * config.bootstrap_convergence_patience,
        cooldown=eval_interval * config.bootstrap_lr_cooldown,
        threshold=config.bootstrap_convergence_threshold,
    )
    ema_loss = None

    streaming_ds = StreamingCompressedDataset(
        files, bbs, passes=epochs, active_files=config.streaming_active_files,
    )
    dl = DataLoader(streaming_ds, batch_size=None, num_workers=0)

    current_step = 0
    early_stopped = False

    nn.nnet.train()
    pbar = tqdm.tqdm(total=total_steps, desc=f"  {phase_name}", leave=False, unit="step")

    for batch in dl:
        canonical, target_vs, target_pis = batch
        canonical = canonical.float().contiguous().to(nn.device, non_blocking=nn._non_blocking)
        target_vs = target_vs.float().contiguous().to(nn.device, non_blocking=nn._non_blocking)
        target_pis = target_pis.float().contiguous().to(nn.device, non_blocking=nn._non_blocking)

        nn.optimizer.zero_grad()
        out_v, out_pi = nn.nnet(canonical)
        l_v = nn.loss_v(target_vs, out_v)
        l_pi = nn.loss_pi(target_pis, out_pi)
        total_loss = l_v + l_pi
        total_loss.backward()
        nn.optimizer.step()

        v_val = l_v.item()
        pi_val = l_pi.item()
        current_step += 1
        step_loss = v_val + pi_val
        ema_loss = ema_update(ema_loss, step_loss)

        run.track(v_val, name="loss", epoch=source_n,
                  step=total_train_steps + current_step, context={"type": "value"})
        run.track(pi_val, name="loss", epoch=source_n,
                  step=total_train_steps + current_step, context={"type": "policy"})
        run.track(step_loss, name="loss", epoch=source_n,
                  step=total_train_steps + current_step, context={"type": "total"})

        event, lr = scheduler.step(ema_loss)

        pbar.update()
        pbar.set_postfix(
            ema=f"{ema_loss:.4f}",
            best=f"{scheduler.best:.4f}",
            bad=f"{scheduler.num_bad}",
            lr=f"{lr:.6f}",
        )

        if event == "drop":
            pbar.write(f"  LR drop #{scheduler.drops} at step {current_step}: "
                       f"lr={lr:.6f} (best={scheduler.best:.4f}, cur={ema_loss:.4f})")
        elif event == "stop":
            pbar.write(f"  {phase_name} converged at step {current_step}/{total_steps} "
                       f"(ema={ema_loss:.4f}, best={scheduler.best:.4f}, lr={lr:.6f})")
            early_stopped = True
            break

    pbar.close()
    if not early_stopped:
        tqdm.tqdm.write(f"  {phase_name} completed {current_step} steps")

    return current_step, early_stopped


def _bootstrap_retrain(nn, reservoir_files, window_files, config, run, source_n, total_train_steps):
    """Bootstrap retrain: 1 epoch on reservoir, then 1 epoch on window.

    Each phase uses step-level LR patience with early stopping.
    LR resets to bootstrap_lr at the start of each phase.

    Returns updated total_train_steps.
    """
    phases = []
    if reservoir_files:
        phases.append(("Retrain: Reservoir", reservoir_files))
    if window_files:
        phases.append(("Retrain: Window", window_files))

    for phase_name, files in phases:
        steps, _ = _bootstrap_train_phase(
            nn, files, config, run, source_n, total_train_steps, phase_name,
        )
        total_train_steps += steps

    return total_train_steps


def _log_win_rate_matrix(paths, iteration, run, total_train_steps):
    """Log win rate matrix heatmap to aim. Works for any game."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import aim as aim_lib
    except ImportError:
        return
    try:
        wr_path = os.path.join(paths["experiment"], "win_rate.csv")
        if not os.path.exists(wr_path):
            return
        wr_mat = np.genfromtxt(wr_path, delimiter=",")
        n = min(iteration + 2, wr_mat.shape[0])
        # Mask unplayed matchups so they render distinctly from genuine 0.5
        # draws (previously they were painted yellow-green like a fair draw).
        wr_display = np.ma.masked_invalid(wr_mat[:n, :n])
        sz = min(max(4, n // 3), 12)
        fig, ax = plt.subplots(figsize=(sz, sz))
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#d0d0d0")
        # origin="lower" flips the y axis so the latest agent (highest index)
        # sits at the top — biggest-vs-biggest lands in the upper-right corner.
        im = ax.imshow(wr_display, cmap=cmap, vmin=0, vmax=1,
                       interpolation="nearest", origin="lower")
        ax.set_title(f"Win Rate Matrix (iteration {iteration})", fontsize=11)
        ax.set_xlabel("Opponent iteration"); ax.set_ylabel("Agent iteration")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Overlay numeric labels for small matrices.
        if n <= 12:
            for i in range(n):
                for j in range(n):
                    v = wr_mat[i, j]
                    if np.isnan(v):
                        continue
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="black")
        plt.tight_layout()
        run.track(aim_lib.Image(fig), name="win_rate_matrix",
                  epoch=iteration, step=total_train_steps)
    except Exception:
        pass
    finally:
        plt.close("all")


def _seed_initial_checkpoint(initial_checkpoint, checkpoint_dir, experiment_name):
    """Copy a caller-selected checkpoint into a new run's iteration-zero slot."""
    source = os.path.abspath(initial_checkpoint)
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Initial checkpoint does not exist: {source}")
    destination = os.path.join(checkpoint_dir, f"0000-{experiment_name}.pt")
    shutil.copy2(source, destination)
    return destination


def main(config, experiment_dir, start=0, aim_repo=None, bootstrap_from="", initial_checkpoint=""):
    """Main training loop.

    Args:
        config: TrainConfig instance with all training parameters.
        experiment_dir: Path to experiment directory (e.g. data/uttt/my-experiment/).
        start: Iteration to resume from (0 = fresh start).
        aim_repo: Path for aim logging directory. Default None uses project root.
        bootstrap_from: Path to source experiment dir to bootstrap from (empty = no bootstrap).
        initial_checkpoint: Checkpoint copied to iteration zero for a fresh run.
    """
    import neural_net

    Game = config.Game
    paths = config.resolve_paths(experiment_dir)
    experiment_name = os.path.basename(experiment_dir)

    if initial_checkpoint and start != 0:
        raise ValueError("initial_checkpoint is only valid for a fresh run")
    if initial_checkpoint and bootstrap_from:
        raise ValueError("initial_checkpoint and bootstrap_from cannot be used together")

    # Ensure all directories exist
    for key in ("checkpoint", "history", "tmp_history", "reservoir"):
        os.makedirs(paths[key], exist_ok=True)

    # Save config at start
    config.save(os.path.join(experiment_dir, "config.yaml"))

    @tracy_zone
    def create_init_net():
        nnargs = neural_net.nnargs_from_config(config)
        nn = neural_net.NNWrapper(Game, nnargs)
        nn.save_checkpoint(paths["checkpoint"], f"0000-{experiment_name}.pt",
                           zstd_level=config.zstd_level)

    try:
        import aim
        aim_hash_path = os.path.join(experiment_dir, ".aim_run_hash")
        if start > 0 and os.path.exists(aim_hash_path):
            run = aim.Run(
                run_hash=open(aim_hash_path).read().strip(),
                repo=aim_repo,
                force_resume=True,
            )
        else:
            run = aim.Run(experiment=experiment_name, repo=aim_repo)
            run.name = config.game
            run["hparams"] = {
                "network": config.network_name,
                "panel_size": config.gating_panel_size,
                "panel_win_rate": config.gating_panel_win_rate,
                "best_win_rate": config.gating_best_win_rate,
                "cpuct": config.cpuct,
                "fpu_reduction": config.fpu_reduction,
                "self_play_temp": config.self_play_temp,
                "eval_temp": config.eval_temp,
                "final_temp": config.final_temp,
                "training_sample_rate": config.train_sample_rate,
                "depth": config.depth,
                "channels": config.channels,
                "kernel_size": config.kernel_size,
                "lr_schedule": config.lr_schedule,
                "full_mcts_visits": config.selfplay_mcts_visits,
                "fast_mcts_visits": config.fast_mcts_visits,
                "teacher_anchor_fraction": config.teacher_anchor_fraction,
                "teacher_anchor_enabled": int(bool(config.teacher_anchor_dir)),
                "selfplay_use_latest": config.selfplay_use_latest,
                "gating_enabled": config.gating_enabled,
                "uttt_ai_select_best": config.uttt_ai_select_best,
            }
        with open(aim_hash_path, "w") as f:
            f.write(run.hash)
    except ImportError:
        print("aim is used for nice web logging with graphs. I would advise `pip install aim`.")
        print("Using a dummy logger for now that does nothing.\n")

        class DummyRun:
            def track(*args, **kwargs):
                pass

        run = DummyRun()

    import atexit
    atexit.register(lambda: hasattr(run, 'close') and run.close())

    # LR state for adaptive schedule
    lr_state_path = os.path.join(experiment_dir, "lr_state.json")

    fallback_checkpoint_dir = None

    if start == 0:
        if initial_checkpoint:
            seeded_path = _seed_initial_checkpoint(
                initial_checkpoint, paths["checkpoint"], experiment_name
            )
            with open(os.path.join(experiment_dir, "initial_checkpoint.json"), "w") as f:
                json.dump({"source": os.path.abspath(initial_checkpoint), "seeded": seeded_path}, f)
        else:
            create_init_net()

        # Read bootstrap source size from checkpoints (ground truth)
        source_n = 0
        if bootstrap_from:
            source_checkpoint_dir = os.path.join(bootstrap_from, "checkpoint")
            source_n = find_latest_checkpoint(source_checkpoint_dir)
            if source_n == 0:
                raise RuntimeError(f"No checkpoints found in bootstrap source: {bootstrap_from}")

        total_agents = max(config.iterations, source_n) + 1
        wr = np.empty((total_agents, total_agents))
        wr[:] = np.nan
        # Raw W/L/D count matrices (the canonical evidence for WHR).
        # wins[i,j] is asymmetric (games where i beat j); draws[i,j] is symmetric.
        wins = np.zeros((total_agents, total_agents), dtype=np.int64)
        draws = np.zeros((total_agents, total_agents), dtype=np.int64)
        elo = np.zeros(total_agents)
        whr = np.zeros(total_agents)
        if _is_unified_game(config):
            wr_v = [np.full((total_agents, total_agents), np.nan) for _ in range(4)]
            wins_v = [np.zeros((total_agents, total_agents), dtype=np.int64) for _ in range(4)]
            draws_v = [np.zeros((total_agents, total_agents), dtype=np.int64) for _ in range(4)]
            elo_v = [np.zeros(total_agents) for _ in range(4)]
            whr_v = [np.zeros(total_agents) for _ in range(4)]
        else:
            wins_v = None
            draws_v = None
            whr_v = None
        # Per-iter train-step count (parallel to elo) for WHR overwrite-logging.
        train_steps_per_iter = np.zeros(total_agents, dtype=np.int64)
        # Set of retired anchors (positive compare_past entries that saturated).
        saturated_anchors = _load_compare_past_state(experiment_dir)
        current_best = 0
        total_train_steps = 0
        lr_state = _default_lr_state(config)
        if bootstrap_from:
            lr_state['last_best_iter'] = source_n

        # Handle bootstrap from existing experiment
        if bootstrap_from:
            source_dir = bootstrap_from
            source_config_path = os.path.join(source_dir, "config.yaml")
            source_paths = config.resolve_paths(source_dir)
            source_elo_path = os.path.join(source_dir, "elo.csv")
            source_wr_path = os.path.join(source_dir, "win_rate.csv")

            # Copy elo/wr from source (if available)
            if os.path.exists(source_elo_path) and os.path.exists(source_wr_path):
                source_elo = np.genfromtxt(source_elo_path, delimiter=",")
                source_wr = np.genfromtxt(source_wr_path, delimiter=",")
                copy_size = min(len(source_elo), total_agents)
                wr[:copy_size, :copy_size] = source_wr[:copy_size, :copy_size]
                elo[:copy_size] = source_elo[:copy_size]

            # Copy per-variant elo/wr from source so variant elo stays anchored
            # across bootstrap (otherwise elo_v starts at 0 for all source slots
            # and the post-bootstrap variant elo collapses ~hundreds of points
            # below regular elo).
            if _is_unified_game(config):
                for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                    src_velo = os.path.join(source_dir, f"elo_{vname}.csv")
                    src_vwr = os.path.join(source_dir, f"win_rate_{vname}.csv")
                    if os.path.exists(src_velo) and os.path.exists(src_vwr):
                        s_velo = np.genfromtxt(src_velo, delimiter=",")
                        s_vwr = np.genfromtxt(src_vwr, delimiter=",")
                        copy_size = min(len(s_velo), total_agents)
                        wr_v[vid][:copy_size, :copy_size] = s_vwr[:copy_size, :copy_size]
                        elo_v[vid][:copy_size] = s_velo[:copy_size]

            # Detect architecture match before phase loop
            same_arch = True
            if os.path.exists(source_config_path):
                source_cfg = load_config(source_config_path, {}, warn=False)
                same_arch = (
                    source_cfg.depth == config.depth
                    and source_cfg.channels == config.channels
                    and source_cfg.kernel_size == config.kernel_size
                    and source_cfg.dense_net == config.dense_net
                    and source_cfg.head_channels == config.head_channels
                    and source_cfg.head_pool == config.head_pool
                )

            # Build phase list
            bootstrap_phases = [] if config.bootstrap_window_only else ["Copy Reservoir"]
            bootstrap_phases.append("Copy Window")
            if same_arch:
                bootstrap_phases.append("Copy Checkpoint")
            else:
                bootstrap_phases.append("Retrain")
            bootstrap_phases.append("Calibrate ELO")

            phase_bar = tqdm.tqdm(bootstrap_phases, desc="Bootstrap", leave=True)
            for phase in phase_bar:
                phase_bar.set_description(f"Bootstrap: {phase}")

                if phase == "Copy Reservoir":
                    os.makedirs(paths["reservoir"], exist_ok=True)
                    # Copy all reservoir files (chunks, metadata, and any legacy files)
                    source_res = source_paths["reservoir"]
                    res_files = (
                        glob.glob(os.path.join(source_res, "chunk_*.ptz"))
                        + glob.glob(os.path.join(source_res, "reservoir_meta.json"))
                        + [p for c, v, p2, _ in glob_file_triples(source_res)
                           for p in (c, v, p2)]
                    )
                    if res_files:
                        for src in tqdm.tqdm(res_files, desc="  Files", leave=False):
                            shutil.copy2(src, os.path.join(paths["reservoir"], os.path.basename(src)))

                elif phase == "Copy Window":
                    source_window_triples = []
                    for wi in range(0, source_n):
                        triples = glob_file_triples(source_paths["history"], f"{wi:04d}-*-canonical-*.ptz")
                        if not triples:
                            triples = glob_file_triples(source_paths["history"], f"{wi:04d}-*-canonical-*.pt")
                        source_window_triples.extend(triples)
                    if source_window_triples:
                        os.makedirs(paths["history"], exist_ok=True)
                        all_win_files = []
                        for c, v, p, _ in source_window_triples:
                            all_win_files.extend([c, v, p])
                        for src in tqdm.tqdm(all_win_files, desc="  Files", leave=False):
                            shutil.copy2(src, os.path.join(paths["history"], os.path.basename(src)))

                elif phase == "Copy Checkpoint":
                    best_source = source_n
                    if os.path.exists(source_elo_path):
                        best_source = int(np.argmax(source_elo[:source_n + 1]))
                    cps = sorted(glob.glob(os.path.join(source_paths["checkpoint"], f"{best_source:04d}-*.pt")))
                    if cps:
                        dest_name = f"{source_n:04d}-{experiment_name}.pt"
                        shutil.copy2(cps[-1], os.path.join(paths["checkpoint"], dest_name))

                elif phase == "Retrain":
                    # Try chunked reservoir first, fall back to per-iteration
                    reservoir_files = glob_reservoir_chunks(paths["reservoir"])
                    if not reservoir_files:
                        reservoir_files = glob_file_triples(paths["reservoir"])
                    window_files = glob_file_triples(paths["history"])

                    nn = neural_net.NNWrapper.load_checkpoint(
                        Game, paths["checkpoint"], f"0000-{experiment_name}.pt"
                    )

                    if reservoir_files or window_files:
                        total_train_steps = _bootstrap_retrain(
                            nn, reservoir_files, window_files, config, run, source_n, total_train_steps
                        )

                    nn.save_checkpoint(paths["checkpoint"], f"{source_n:04d}-{experiment_name}.pt",
                                       zstd_level=config.zstd_level)

                elif phase == "Calibrate ELO":
                    # Build a one-off list mirroring compare_past behavior but
                    # walking back from source_n. For the bootstrap calibration
                    # we still want a fresh comparison against the most-recent
                    # bootstrap_compare_past iters (legacy semantics).
                    compare_start = max(0, source_n - config.bootstrap_compare_past)
                    for past_iter in tqdm.tqdm(range(compare_start, source_n), desc="  ELO", leave=False):
                        (nn_rate, draw_rate, _, _, _, _, _, _,
                         variant_nn_rates, variant_draw_rates, _,
                         n_games, nn_wins_n, past_wins_n, n_draws_n,
                         variant_counts) = play_past(
                            config, paths, experiment_name, config.compare_mcts_visits,
                            source_n, past_iter, config.past_compare_batch_size,
                            fallback_checkpoint_dir=source_paths["checkpoint"],
                        )
                        # Store raw counts; derive wr from counts for live elo.
                        wins[source_n, past_iter] = nn_wins_n
                        wins[past_iter, source_n] = past_wins_n
                        draws[source_n, past_iter] = n_draws_n
                        draws[past_iter, source_n] = n_draws_n
                        wr[source_n, past_iter], wr[past_iter, source_n] = _wr_from_counts(
                            wins[source_n, past_iter], wins[past_iter, source_n],
                            draws[source_n, past_iter])
                        if _is_unified_game(config):
                            for vid, (vng, vnw, vpw, vnd) in variant_counts.items():
                                wins_v[vid][source_n, past_iter] = vnw
                                wins_v[vid][past_iter, source_n] = vpw
                                draws_v[vid][source_n, past_iter] = vnd
                                draws_v[vid][past_iter, source_n] = vnd
                                wr_v[vid][source_n, past_iter], wr_v[vid][past_iter, source_n] = _wr_from_counts(
                                    wins_v[vid][source_n, past_iter],
                                    wins_v[vid][past_iter, source_n],
                                    draws_v[vid][source_n, past_iter])
                        gc.collect()

                    elo = get_elo(elo, wr, source_n)
                    if _is_unified_game(config):
                        for vid in range(len(UNIFIED_VARIANT_NAMES)):
                            if not np.all(np.isnan(wr_v[vid][source_n])):
                                elo_v[vid] = get_elo(elo_v[vid], wr_v[vid], source_n)

            phase_bar.close()

            current_best = source_n
            start = source_n
            fallback_checkpoint_dir = source_paths["checkpoint"]

            # Save bootstrap metadata for resume
            bootstrap_meta_path = os.path.join(experiment_dir, "bootstrap_meta.json")
            with open(bootstrap_meta_path, "w") as f:
                json.dump({"source_checkpoint_dir": source_paths["checkpoint"]}, f)

            if os.path.exists(source_elo_path):
                best_source_iter = int(np.argmax(source_elo[:source_n + 1]))
                if elo[source_n] < elo[best_source_iter]:
                    tqdm.tqdm.write(f"Warning: bootstrapped network (ELO {elo[source_n]:.0f}) is weaker than "
                                    f"source best at iter {best_source_iter} (ELO {elo[best_source_iter]:.0f}). "
                                    f"Training will continue and should improve.")

        np.savetxt(os.path.join(experiment_dir, "elo.csv"), elo, delimiter=",")
        np.savetxt(os.path.join(experiment_dir, "win_rate.csv"), wr, delimiter=",")
        _save_counts(experiment_dir, "", wins, draws)
        if _is_unified_game(config):
            for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                np.savetxt(os.path.join(experiment_dir, f"elo_{vname}.csv"), elo_v[vid], delimiter=",")
                np.savetxt(os.path.join(experiment_dir, f"win_rate_{vname}.csv"), wr_v[vid], delimiter=",")
                _save_counts(experiment_dir, vname, wins_v[vid], draws_v[vid])
        np.savetxt(
            os.path.join(experiment_dir, "total_train_steps.txt"),
            [total_train_steps],
            delimiter=",",
        )
    else:
        total_agents = config.iterations + 1
        tmp_wr = np.genfromtxt(os.path.join(experiment_dir, "win_rate.csv"), delimiter=",")
        wr = np.empty((total_agents, total_agents))
        wr[:] = np.nan
        old_size = min(tmp_wr.shape[0], total_agents)
        wr[:old_size, :old_size] = tmp_wr[:old_size, :old_size]
        # Raw count matrices. Load from disk if present; else fall back to
        # deriving approximate counts from win_rate * batch_size assumption.
        wins = np.zeros((total_agents, total_agents), dtype=np.int64)
        draws = np.zeros((total_agents, total_agents), dtype=np.int64)
        wins, draws = _load_counts(
            experiment_dir, "", total_agents, wr,
            assumed_batch=int(config.past_compare_batch_size))
        tmp_elo = np.genfromtxt(os.path.join(experiment_dir, "elo.csv"), delimiter=",")
        elo = np.zeros(total_agents)
        old_size = min(tmp_elo.shape[0], total_agents)
        elo[:old_size] = tmp_elo[:old_size]
        whr = np.zeros(total_agents)
        whr_path = os.path.join(experiment_dir, "whr.csv")
        if os.path.exists(whr_path):
            tmp = np.genfromtxt(whr_path, delimiter=",")
            sz = min(len(tmp), total_agents)
            whr[:sz] = tmp[:sz]
        if _is_unified_game(config):
            wr_v = []
            wins_v = []
            draws_v = []
            elo_v = []
            whr_v = []
            for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                vwr_path = os.path.join(experiment_dir, f"win_rate_{vname}.csv")
                velo_path = os.path.join(experiment_dir, f"elo_{vname}.csv")
                vwhr_path = os.path.join(experiment_dir, f"whr_{vname}.csv")
                m = np.full((total_agents, total_agents), np.nan)
                if os.path.exists(vwr_path):
                    tmp = np.genfromtxt(vwr_path, delimiter=",")
                    sz = min(tmp.shape[0], total_agents)
                    m[:sz, :sz] = tmp[:sz, :sz]
                wr_v.append(m)
                wins_vid, draws_vid = _load_counts(
                    experiment_dir, vname, total_agents, m,
                    assumed_batch=int(config.past_compare_batch_size))
                wins_v.append(wins_vid)
                draws_v.append(draws_vid)
                ev = np.zeros(total_agents)
                if os.path.exists(velo_path):
                    tmp = np.genfromtxt(velo_path, delimiter=",")
                    sz = min(len(tmp), total_agents)
                    ev[:sz] = tmp[:sz]
                elo_v.append(ev)
                wv = np.zeros(total_agents)
                if os.path.exists(vwhr_path):
                    tmp = np.genfromtxt(vwhr_path, delimiter=",")
                    sz = min(len(tmp), total_agents)
                    wv[:sz] = tmp[:sz]
                whr_v.append(wv)
        else:
            wins_v = None
            draws_v = None
            whr_v = None
        train_steps_per_iter = _load_train_steps_per_iter(experiment_dir, total_agents)
        saturated_anchors = _load_compare_past_state(experiment_dir)
        total_train_steps = int(
            np.genfromtxt(os.path.join(experiment_dir, "total_train_steps.txt"))
        )
        if os.path.exists(lr_state_path):
            with open(lr_state_path) as f:
                lr_state = json.load(f)
        else:
            lr_state = _default_lr_state(config)

        # Recover bootstrap fallback checkpoint dir on resume
        bootstrap_meta_path = os.path.join(experiment_dir, "bootstrap_meta.json")
        if os.path.exists(bootstrap_meta_path):
            with open(bootstrap_meta_path) as f:
                fallback_checkpoint_dir = json.load(f).get("source_checkpoint_dir")

        # Use the gating-derived best (matches what was tracked to aim during
        # the prior run). argmax(elo) can point at a slot whose checkpoint
        # never existed locally — e.g. after bootstrap, the source's elo array
        # is copied verbatim but the source's best checkpoint is renamed into
        # slot `source_n` in the new dir.
        current_best = int(lr_state.get('last_best_iter', 0))
        if current_best > start or not (
            os.path.exists(os.path.join(paths["checkpoint"], f"{current_best:04d}-{experiment_name}.pt"))
            or (fallback_checkpoint_dir and glob.glob(
                os.path.join(fallback_checkpoint_dir, f"{current_best:04d}-*.pt")))
        ):
            # Legacy run with no lr_state, or stale value — fall back to
            # argmax restricted to slots whose checkpoint actually exists.
            candidates = [
                i for i in range(start + 1)
                if os.path.exists(os.path.join(paths["checkpoint"], f"{i:04d}-{experiment_name}.pt"))
            ]
            current_best = max(candidates, key=lambda i: elo[i]) if candidates else 0

    uttt_ai_champion_path = os.path.join(experiment_dir, "uttt_ai_champion.json")
    uttt_ai_champion = {"iteration": 0, "score": float("-inf")}
    if os.path.exists(uttt_ai_champion_path):
        with open(uttt_ai_champion_path, encoding="utf-8") as f:
            uttt_ai_champion = json.load(f)
        if config.uttt_ai_select_best:
            current_best = int(uttt_ai_champion["iteration"])

    postfix = {"best": current_best}
    panel = [current_best]

    # Legacy variant state remains inert for UTTT.
    unified_probs = None
    variant_sample_count_history = []  # rolling 5-iter window to smooth prob corrections
    if _is_unified_game(config) and start > 0:
        for past in range(max(0, start - 5), start):
            counts = _count_variant_samples(paths, past)
            if counts is not None:
                variant_sample_count_history.append(counts)
    if not _is_unified_game(config):
        wr_v = None
        elo_v = None

    # Rolling per-anchor history of (iter, mean_kl) for frozen-eval trajectory
    # analysis. Built up across iters as data arrives. Persisted to disk via
    # frozen_eval.append_kl_history, so resumes / restarts pick up the prior
    # trajectory instead of needing 3+ fresh iters to print a slope.
    _FROZEN_KL_HIST_LEN = 30
    frozen_kl_history = {a: [] for a in config.frozen_eval_anchor_iters}
    if config.frozen_eval_anchor_iters and start > 0:
        for anchor in config.frozen_eval_anchor_iters:
            prior = frozen_eval.load_kl_history(paths, anchor, max_iter=start)
            if prior:
                frozen_kl_history[anchor] = prior[-_FROZEN_KL_HIST_LEN:]

    with tqdm.tqdm(range(start, config.iterations), initial=start, total=config.iterations, desc="Build Amazing Network") as pbar:
        for i in pbar:
            stage_times = {}
            iteration_start = time.time()

            run.track(
                current_best, name="best_network", epoch=i, step=total_train_steps
            )

            stage_start = time.time()
            with TracyZone("stage_history"):
                sat_thr = config.compare_past_saturation_threshold
                for entry in config.compare_past:
                    if entry < 0:
                        past_iter = max(0, i + entry)
                        is_anchor = False
                        vs_tag = f"-{abs(entry)}"
                    else:
                        if entry in saturated_anchors or entry >= i:
                            continue
                        past_iter = entry
                        is_anchor = True
                        vs_tag = f"anchor{entry}"
                    if past_iter == i or not math.isnan(wr[i, past_iter]):
                        continue
                    (nn_rate, draw_rate, _, game_length, bench_depth, bench_entropy,
                     bench_mpt, bench_vm, variant_nn_rates, variant_draw_rates,
                     variant_play_metrics,
                     n_games_played, nn_wins_n, past_wins_n, n_draws_n,
                     variant_counts) = play_past(
                        config, paths, experiment_name,
                        config.compare_mcts_visits, i, past_iter,
                        config.past_compare_batch_size,
                        fallback_checkpoint_dir=fallback_checkpoint_dir,
                        variant_probs=unified_probs,
                    )
                    # Store raw counts. wins is asymmetric (i-wins vs j-wins),
                    # draws is symmetric. wr is derived from counts via the half-
                    # draw convention -- this fixes the existing 1/N-per-draw bug
                    # for multi-player games as a side effect. Uses assignment
                    # (not +=) to match the existing wr[i,j] = ... semantics:
                    # the NaN-guard above ensures we never overwrite real data.
                    wins[i, past_iter] = nn_wins_n
                    wins[past_iter, i] = past_wins_n
                    draws[i, past_iter] = n_draws_n
                    draws[past_iter, i] = n_draws_n
                    full_rate, inv_rate = _wr_from_counts(
                        wins[i, past_iter], wins[past_iter, i], draws[i, past_iter])
                    wr[i, past_iter] = full_rate
                    wr[past_iter, i] = inv_rate
                    run.track(nn_rate, name="win_rate", epoch=i, step=total_train_steps, context={"vs": vs_tag, "from": "all_games"})
                    run.track(draw_rate, name="draw_rate", epoch=i, step=total_train_steps, context={"vs": vs_tag, "from": "all_games"})
                    run.track(game_length, name="average_game_length", epoch=i, step=total_train_steps, context={"vs": vs_tag})
                    run.track(bench_depth, name="avg_leaf_depth", epoch=i, step=total_train_steps, context={"vs": vs_tag, "search": "full"})
                    run.track(bench_entropy, name="search_entropy", epoch=i, step=total_train_steps, context={"vs": vs_tag, "search": "full"})
                    run.track(bench_mpt, name="moves_per_turn", epoch=i, step=total_train_steps, context={"vs": vs_tag})
                    run.track(bench_vm, name="avg_valid_moves", epoch=i, step=total_train_steps, context={"vs": vs_tag})
                    for vid, vrate in variant_nn_rates.items():
                        run.track(vrate, name="win_rate", epoch=i, step=total_train_steps,
                                  context={"vs": vs_tag, "variant": UNIFIED_VARIANT_NAMES[vid]})
                        if wr_v is not None and vid in variant_counts:
                            _, vnw, vpw, vnd = variant_counts[vid]
                            wins_v[vid][i, past_iter] = vnw
                            wins_v[vid][past_iter, i] = vpw
                            draws_v[vid][i, past_iter] = vnd
                            draws_v[vid][past_iter, i] = vnd
                            v_full, v_inv = _wr_from_counts(
                                wins_v[vid][i, past_iter],
                                wins_v[vid][past_iter, i],
                                draws_v[vid][i, past_iter])
                            wr_v[vid][i, past_iter] = v_full
                            wr_v[vid][past_iter, i] = v_inv
                    for vid, drate in variant_draw_rates.items():
                        run.track(drate, name="draw_rate", epoch=i, step=total_train_steps,
                                  context={"vs": vs_tag, "variant": UNIFIED_VARIANT_NAMES[vid]})
                    _track_variant_play_metrics(run, variant_play_metrics, epoch=i,
                                                step=total_train_steps,
                                                vs_context=vs_tag)
                    postfix[f"vs {vs_tag}"] = _fmt_pct(full_rate)
                    # Retire anchor if it saturated (only for positive entries).
                    if is_anchor and (full_rate >= sat_thr or full_rate <= 1 - sat_thr):
                        saturated_anchors.add(entry)
                        _save_compare_past_state(experiment_dir, saturated_anchors)
                    gc.collect()
            stage_times["history"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_rating"):
                if config.elo_enabled or config.whr_enabled:
                    # Record the train step at which this iter's live rating is
                    # logged. WHR overwrite-logging reuses the same x-axis.
                    train_steps_per_iter[i] = total_train_steps

                if config.elo_enabled:
                    elo = get_elo(elo, wr, i)
                    run.track(elo[i], name="elo", epoch=i, step=total_train_steps, context={"type": "current"})
                    run.track(elo[current_best], name="elo", epoch=i, step=total_train_steps, context={"type": "best"})
                    if elo_v is not None:
                        for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                            if not np.all(np.isnan(wr_v[vid][i])):
                                elo_v[vid] = get_elo(elo_v[vid], wr_v[vid], i)
                            # Per-variant elo now lives on the same `elo` metric
                            # with a `variant` context tag (was `variant_elo`).
                            run.track(elo_v[vid][i], name="elo", epoch=i, step=total_train_steps,
                                      context={"type": "current", "variant": vname})
                    postfix["elo"] = int(elo[i])
                    np.savetxt(os.path.join(experiment_dir, "elo.csv"), elo, delimiter=",")

                if config.whr_enabled:
                    # WHR refit on raw W/L/D counts (no rate-based approximation,
                    # no Wiener prior -- each iter is a fully independent player).
                    # aim's overwrite semantics keep the chart at the latest refit.
                    whr = whr_refit(wins, draws, max_sweeps=200, tol=0.1)
                    if whr_v is not None:
                        for vid in range(len(UNIFIED_VARIANT_NAMES)):
                            whr_v[vid] = whr_refit(
                                wins_v[vid], draws_v[vid], max_sweeps=200, tol=0.1)
                    for j in range(i + 1):
                        step_j = int(train_steps_per_iter[j])
                        run.track(float(whr[j]), name="whr", epoch=j, step=step_j,
                                  context={"type": "current"})
                        if whr_v is not None:
                            for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                                run.track(float(whr_v[vid][j]), name="whr",
                                          epoch=j, step=step_j,
                                          context={"type": "current", "variant": vname})
                    postfix["whr"] = int(whr[i])
                    np.savetxt(os.path.join(experiment_dir, "whr.csv"), whr, delimiter=",")

                pbar.set_postfix(postfix)
                if config.elo_enabled or config.whr_enabled:
                    _save_train_steps_per_iter(experiment_dir, train_steps_per_iter)
                # Persist wr + raw counts. Without these, a crash between
                # training and gating's save would cause iter i to be skipped
                # on resume, leaving wr[i, i-compare_past] permanently NaN.
                np.savetxt(os.path.join(experiment_dir, "win_rate.csv"), wr, delimiter=",")
                _save_counts(experiment_dir, "", wins, draws)
                if wr_v is not None:
                    for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                        np.savetxt(os.path.join(experiment_dir, f"win_rate_{vname}.csv"), wr_v[vid], delimiter=",")
                        if config.elo_enabled:
                            np.savetxt(os.path.join(experiment_dir, f"elo_{vname}.csv"), elo_v[vid], delimiter=",")
                        if config.whr_enabled:
                            np.savetxt(os.path.join(experiment_dir, f"whr_{vname}.csv"), whr_v[vid], delimiter=",")
                        _save_counts(experiment_dir, vname, wins_v[vid], draws_v[vid])
            stage_times["elo"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_selfplay"):
                if _is_unified_game(config):
                    counts = _count_variant_samples(paths, i - 1) if i > 0 else None
                    if counts is not None:
                        variant_sample_count_history.append(counts)
                        if len(variant_sample_count_history) > 5:
                            variant_sample_count_history.pop(0)
                    rolled_counts = (
                        [sum(h[v] for h in variant_sample_count_history) for v in range(4)]
                        if variant_sample_count_history else None
                    )
                    unified_probs = _compute_unified_probs(config, rolled_counts)
                    for vi, vname in enumerate(UNIFIED_VARIANT_NAMES):
                        run.track(unified_probs[vi], name="variant_prob", epoch=i,
                                  step=total_train_steps, context={"variant": vname})
                selfplay_parent = selfplay_parent_iteration(config, i, current_best)
                run.track(
                    selfplay_parent, name="selfplay_network", epoch=i,
                    step=total_train_steps,
                )
                sp = self_play(
                    config, paths, experiment_name,
                    selfplay_parent, i,
                    config.selfplay_mcts_visits,
                    config.fast_mcts_visits,
                    variant_probs=unified_probs,
                    fallback_checkpoint_dir=fallback_checkpoint_dir,
                )
                for j in range(len(sp.win_rates) - 1):
                    run.track(sp.win_rates[j], name="win_rate", epoch=i, step=total_train_steps, context={"vs": "self", "player": j + 1, "from": "all_games"})
                for j in range(len(sp.resign_win_rates) - 1):
                    run.track(sp.resign_win_rates[j], name="win_rate", epoch=i, step=total_train_steps, context={"vs": "self", "player": j + 1, "from": "resignation"})
                run.track(sp.resign_rate, name="resignation_rate", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(sp.win_rates[-1], name="draw_rate", epoch=i, step=total_train_steps, context={"vs": "self", "from": "all_games"})
                run.track(sp.resign_win_rates[-1], name="draw_rate", epoch=i, step=total_train_steps, context={"vs": "self", "from": "resignation"})
                run.track(float(sp.hit_rate), name="cache_rate", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "hit_rate"})
                run.track(float(sp.theoretical_hr), name="cache_rate", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "theoretical_hit_rate"})
                run.track(sp.game_length, name="average_game_length", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(sp.avg_depth, name="avg_leaf_depth", epoch=i, step=total_train_steps, context={"vs": "self", "search": "full"})
                run.track(sp.avg_entropy, name="search_entropy", epoch=i, step=total_train_steps, context={"vs": "self", "search": "full"})
                if sp.fast_avg_depth > 0:
                    run.track(sp.fast_avg_depth, name="avg_leaf_depth", epoch=i, step=total_train_steps, context={"vs": "self", "search": "fast"})
                if sp.fast_avg_entropy > 0:
                    run.track(sp.fast_avg_entropy, name="search_entropy", epoch=i, step=total_train_steps, context={"vs": "self", "search": "fast"})
                run.track(sp.avg_mpt, name="moves_per_turn", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(sp.avg_vm, name="avg_valid_moves", epoch=i, step=total_train_steps, context={"vs": "self"})
                run.track(float(sp.saturation), name="cache_rate", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "saturation"})
                run.track(float(sp.churn_rate), name="cache_rate", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "churn"})
                run.track(float(sp.avg_batch_size), name="batch_size", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "avg"})
                run.track(float(sp.median_batch_size), name="batch_size", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "median"})
                run.track(float(sp.min_batch_size), name="batch_size", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "min"})
                run.track(float(sp.max_batch_size), name="batch_size", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "max"})
                run.track(float(sp.avg_inference_ms), name="inference_ms", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "avg"})
                run.track(float(sp.median_inference_ms), name="inference_ms", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "median"})
                run.track(float(sp.min_inference_ms), name="inference_ms", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "min"})
                run.track(float(sp.max_inference_ms), name="inference_ms", epoch=i, step=total_train_steps, context={"vs": "self", "stat": "max"})
                postfix["wr"] = "/".join(_fmt_pct(x) for x in sp.win_rates)
                pbar.set_postfix(postfix)
                if sp.variant_game_counts:
                    total_vg = sum(sp.variant_game_counts.values()) or 1
                    for vid, cnt in sp.variant_game_counts.items():
                        run.track(cnt / total_vg, name="variant_game_frac", epoch=i,
                                  step=total_train_steps,
                                  context={"variant": UNIFIED_VARIANT_NAMES[vid], "vs": "self"})
                if sp.variant_win_rates:
                    for vid, vrates in sp.variant_win_rates.items():
                        vname = UNIFIED_VARIANT_NAMES[vid]
                        for j in range(len(vrates) - 1):
                            run.track(vrates[j], name="win_rate", epoch=i,
                                      step=total_train_steps,
                                      context={"vs": "self", "player": j + 1,
                                               "variant": vname, "from": "all_games"})
                        run.track(vrates[-1], name="draw_rate", epoch=i,
                                  step=total_train_steps,
                                  context={"vs": "self", "variant": vname, "from": "all_games"})
                _track_variant_play_metrics(run, sp.variant_metrics, epoch=i,
                                            step=total_train_steps, vs_context="self")
            stage_times["selfplay"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_symmetries"):
                exploit_symmetries(config, paths, i)
            stage_times["symmetries"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_resampling"):
                surprise_loss_arr = resample_by_surprise(config, paths, experiment_name, i)
            stage_times["resampling"] = time.time() - stage_start
            if len(surprise_loss_arr) > 0:
                try:
                    import aim as _aim
                    run.track(_aim.Distribution(surprise_loss_arr), name="surprise_loss",
                              epoch=i, step=total_train_steps)
                except Exception:
                    pass

            stage_start = time.time()
            with TracyZone("stage_training"):
                hist_size = calc_hist_size(config, i)
                oldest_iteration = max(0, i - hist_size)
                run.track(hist_size, name="history", epoch=i, step=total_train_steps, context={"type": "window_size"})
                run.track(oldest_iteration, name="history", epoch=i, step=total_train_steps, context={"type": "oldest_iteration"})
                lr = get_lr(config, i, lr_state)
                run.track(lr, name="learning_rate", epoch=i, step=total_train_steps)
                parent_iter = current_best if config.train_from_best else i
                v_loss, pi_loss, total_train_steps = train(
                    config, paths, experiment_name, i, hist_size, run,
                    total_train_steps, lr=lr, parent_iteration=parent_iter
                )
                np.savetxt(
                    os.path.join(experiment_dir, "total_train_steps.txt"),
                    [total_train_steps],
                    delimiter=",",
                )
                with open(lr_state_path, 'w') as f:
                    json.dump(lr_state, f)
                postfix["vloss"] = v_loss
                postfix["ploss"] = pi_loss
                pbar.set_postfix(postfix)
                gc.collect()
            stage_times["training"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_variant_analysis"):
                variant_stats = {}
                if _is_unified_game(config):
                    variant_stats = _analyze_iteration_variants(config, paths, experiment_name, i)
                    for vname, stats in variant_stats.items():
                        # Mean scalars for time-series trend lines
                        run.track(float(stats["pi_loss"].mean()), name="loss", epoch=i,
                                  step=total_train_steps,
                                  context={"type": "policy", "variant": vname})
                        run.track(float(stats["v_loss"].mean()), name="loss", epoch=i,
                                  step=total_train_steps,
                                  context={"type": "value", "variant": vname})
                        run.track(float(stats["entropy"].mean()), name="loss", epoch=i,
                                  step=total_train_steps,
                                  context={"type": "target_entropy", "variant": vname})
                        run.track(float(stats["top1"].mean()), name="policy_top1_mass", epoch=i,
                                  step=total_train_steps, context={"variant": vname})
                        # Distributions for spread/skew visibility
                        try:
                            import aim as _aim
                            run.track(_aim.Distribution(stats["pi_loss"]), name="pi_loss_dist",
                                      epoch=i, step=total_train_steps, context={"variant": vname})
                            run.track(_aim.Distribution(stats["v_loss"]), name="v_loss_dist",
                                      epoch=i, step=total_train_steps, context={"variant": vname})
                            run.track(_aim.Distribution(stats["entropy"]), name="entropy_dist",
                                      epoch=i, step=total_train_steps, context={"variant": vname})
                            run.track(_aim.Distribution(stats["top1"]), name="pi_top1_dist",
                                      epoch=i, step=total_train_steps, context={"variant": vname})
                        except Exception:
                            pass
                    if variant_stats:
                        try:
                            import aim as _aim
                            for key, dist_name in (("pi_loss", "pi_loss_dist"),
                                                   ("v_loss", "v_loss_dist"),
                                                   ("entropy", "entropy_dist"),
                                                   ("top1", "pi_top1_dist")):
                                agg = np.concatenate([s[key] for s in variant_stats.values()])
                                run.track(_aim.Distribution(agg), name=dist_name,
                                          epoch=i, step=total_train_steps)
                        except Exception:
                            pass
            stage_times["variant_analysis"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_reservoir"):
                update_reservoir(config, paths, i, hist_size)
                gc.collect()
            stage_times["reservoir"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_uttt_ai_eval"):
                uttt_ai_metrics = maybe_eval_uttt_ai(
                    config, paths, experiment_name, i + 1, run, total_train_steps
                )
                if uttt_ai_metrics is not None:
                    uttt_ai_wr, uttt_ai_dr, _ = uttt_ai_metrics
                    if config.uttt_ai_select_best:
                        uttt_ai_champion, improved = update_uttt_ai_champion(
                            uttt_ai_champion, i + 1, uttt_ai_wr, uttt_ai_dr
                        )
                        with open(uttt_ai_champion_path, "w", encoding="utf-8") as f:
                            json.dump(uttt_ai_champion, f)
                        if improved:
                            current_best = int(uttt_ai_champion["iteration"])
                            lr_state["last_best_iter"] = current_best
                            postfix["best"] = current_best
                            with open(lr_state_path, "w", encoding="utf-8") as f:
                                json.dump(lr_state, f)
                        run.track(
                            float(uttt_ai_champion["score"]),
                            name="uttt_ai_score", epoch=i + 1,
                            step=total_train_steps,
                            context={"search": "raw_policy", "type": "best"},
                        )
                    postfix["vs uttt.ai"] = (
                        f"{_fmt_pct(uttt_ai_wr)}/{_fmt_pct(uttt_ai_dr)}D"
                    )
                    pbar.set_postfix(postfix)
            stage_times["uttt_ai_eval"] = time.time() - stage_start

            stage_start = time.time()
            with TracyZone("stage_gating"):
                next_net = i + 1
                panel_nn_rate = 0
                panel_draw_rate = 0
                panel_game_length = 0
                panel_depth = 0
                panel_entropy = 0
                panel_mpt = 0
                panel_vm = 0
                best_win_rate = 0
                gating_probs = _compute_gating_probs(config) if _is_unified_game(config) else None
                gate_panel = panel if config.gating_enabled else []
                for gate_net in tqdm.tqdm(
                    gate_panel, desc=f"Pitting against Panel {gate_panel}", leave=False
                ):
                    (nn_rate, draw_rate, _, game_length, gate_depth, gate_entropy,
                     gate_mpt, gate_vm, variant_nn_rates, variant_draw_rates,
                     variant_play_metrics,
                     n_games_played, nn_wins_n, past_wins_n, n_draws_n,
                     variant_counts) = play_past(
                        config, paths, experiment_name,
                        config.compare_mcts_visits, next_net, gate_net, config.gate_compare_batch_size,
                        fallback_checkpoint_dir=fallback_checkpoint_dir,
                        variant_probs=gating_probs,
                    )
                    panel_nn_rate += nn_rate
                    panel_draw_rate += draw_rate
                    panel_game_length += game_length
                    panel_depth += gate_depth
                    panel_entropy += gate_entropy
                    panel_mpt += gate_mpt
                    panel_vm += gate_vm
                    # Store raw counts (gating uses gate_compare_batch_size).
                    wins[next_net, gate_net] = nn_wins_n
                    wins[gate_net, next_net] = past_wins_n
                    draws[next_net, gate_net] = n_draws_n
                    draws[gate_net, next_net] = n_draws_n
                    g_full, g_inv = _wr_from_counts(
                        wins[next_net, gate_net], wins[gate_net, next_net],
                        draws[next_net, gate_net])
                    wr[next_net, gate_net] = g_full
                    wr[gate_net, next_net] = g_inv
                    gc.collect()
                    if gate_net == current_best:
                        run.track(nn_rate, name="win_rate", epoch=next_net, step=total_train_steps, context={"vs": "best", "from": "all_games"})
                        run.track(draw_rate, name="draw_rate", epoch=next_net, step=total_train_steps, context={"vs": "best", "from": "all_games"})
                        run.track(game_length, name="average_game_length", epoch=next_net, context={"vs": "best"})
                        run.track(gate_depth, name="avg_leaf_depth", epoch=next_net, step=total_train_steps, context={"vs": "best", "search": "full"})
                        run.track(gate_entropy, name="search_entropy", epoch=next_net, step=total_train_steps, context={"vs": "best", "search": "full"})
                        run.track(gate_mpt, name="moves_per_turn", epoch=next_net, step=total_train_steps, context={"vs": "best"})
                        run.track(gate_vm, name="avg_valid_moves", epoch=next_net, step=total_train_steps, context={"vs": "best"})
                        for vid, vrate in variant_nn_rates.items():
                            run.track(vrate, name="win_rate", epoch=next_net, step=total_train_steps,
                                      context={"vs": "best", "variant": UNIFIED_VARIANT_NAMES[vid]})
                            if wr_v is not None and vid in variant_counts:
                                _, vnw, vpw, vnd = variant_counts[vid]
                                wins_v[vid][next_net, gate_net] = vnw
                                wins_v[vid][gate_net, next_net] = vpw
                                draws_v[vid][next_net, gate_net] = vnd
                                draws_v[vid][gate_net, next_net] = vnd
                                gv_full, gv_inv = _wr_from_counts(
                                    wins_v[vid][next_net, gate_net],
                                    wins_v[vid][gate_net, next_net],
                                    draws_v[vid][next_net, gate_net])
                                wr_v[vid][next_net, gate_net] = gv_full
                                wr_v[vid][gate_net, next_net] = gv_inv
                        for vid, drate in variant_draw_rates.items():
                            run.track(drate, name="draw_rate", epoch=next_net, step=total_train_steps,
                                      context={"vs": "best", "variant": UNIFIED_VARIANT_NAMES[vid]})
                        _track_variant_play_metrics(run, variant_play_metrics, epoch=next_net,
                                                    step=total_train_steps, vs_context="best")
                        best_win_rate = g_full
                        postfix["vs best"] = _fmt_pct(best_win_rate)
                        pbar.set_postfix(postfix)
                if gate_panel:
                    panel_nn_rate /= len(gate_panel)
                    panel_draw_rate /= len(gate_panel)
                    panel_game_length /= len(gate_panel)
                    panel_depth /= len(gate_panel)
                    panel_entropy /= len(gate_panel)
                    panel_mpt /= len(gate_panel)
                    panel_vm /= len(gate_panel)
                    run.track(panel_nn_rate, name="win_rate", epoch=next_net, step=total_train_steps, context={"vs": "panel", "from": "all_games"})
                    run.track(panel_draw_rate, name="draw_rate", epoch=next_net, step=total_train_steps, context={"vs": "panel", "from": "all_games"})
                    run.track(panel_game_length, name="average_game_length", epoch=next_net, context={"vs": "panel"})
                    run.track(panel_depth, name="avg_leaf_depth", epoch=next_net, step=total_train_steps, context={"vs": "panel"})
                    run.track(panel_entropy, name="search_entropy", epoch=next_net, step=total_train_steps, context={"vs": "panel"})
                    run.track(panel_mpt, name="moves_per_turn", epoch=next_net, step=total_train_steps, context={"vs": "panel"})
                    run.track(panel_vm, name="avg_valid_moves", epoch=next_net, step=total_train_steps, context={"vs": "panel"})
                    panel_win_rate = panel_nn_rate + panel_draw_rate / Game.NUM_PLAYERS()
                    if len(gate_panel) > 1:
                        postfix["vs panel"] = _fmt_pct(panel_win_rate)
                    panel_ratio = len(gate_panel) / config.gating_panel_size
                    wanted_panel_win_rate = (config.gating_panel_win_rate * panel_ratio) + (
                        config.gating_best_win_rate * (1.0 - panel_ratio)
                    )
                    if (
                        panel_win_rate > wanted_panel_win_rate
                        and best_win_rate > config.gating_best_win_rate
                    ):
                        current_best = next_net
                        lr_state['last_best_iter'] = next_net
                        with open(lr_state_path, 'w') as f:
                            json.dump(lr_state, f)
                        postfix["best"] = current_best
                        pbar.set_postfix(postfix)
                        panel.append(current_best)
                        while len(panel) > config.gating_panel_size:
                            panel = panel[1:]
                    np.savetxt(os.path.join(experiment_dir, "win_rate.csv"), wr, delimiter=",")
                    _save_counts(experiment_dir, "", wins, draws)
                    if wr_v is not None:
                        for vid, vname in enumerate(UNIFIED_VARIANT_NAMES):
                            np.savetxt(os.path.join(experiment_dir, f"win_rate_{vname}.csv"), wr_v[vid], delimiter=",")
                            np.savetxt(os.path.join(experiment_dir, f"elo_{vname}.csv"), elo_v[vid], delimiter=",")
                            _save_counts(experiment_dir, vname, wins_v[vid], draws_v[vid])
            stage_times["gating"] = time.time() - stage_start

            # Per-checkpoint diagnostics (Features 1, 3).
            # Load the new checkpoint once and run all enabled diagnostics.
            wants_frozen = bool(config.frozen_eval_anchor_iters)
            wants_rank = config.effective_rank_enabled
            if wants_frozen or wants_rank:
                stage_start = time.time()
                with TracyZone("stage_diagnostics"):
                    diag_nn = None
                    try:
                        import neural_net
                        diag_nn = neural_net.NNWrapper.load_checkpoint(
                            Game, paths["checkpoint"],
                            f"{next_net:04d}-{experiment_name}.pt",
                        )
                    except Exception as exc:
                        tqdm.tqdm.write(f"  Diagnostics warning [iter {next_net}]: failed to "
                                        f"load checkpoint: {exc}")

                    if diag_nn is not None and wants_rank:
                        try:
                            sample = _sample_recent_history_batch(
                                paths, config.effective_rank_batch_size,
                            )
                            if sample is not None:
                                pr = diag_nn.effective_rank(sample)
                                run.track(pr, name="effective_rank",
                                          epoch=next_net, step=total_train_steps,
                                          context={})
                        except Exception as exc:
                            tqdm.tqdm.write(f"  Effective rank warning [iter {next_net}]: {exc}")

                    if diag_nn is not None and wants_frozen:
                        for anchor in config.frozen_eval_anchor_iters:
                            snap_ok = frozen_eval.ensure_snapshot(
                                config, paths, experiment_name, anchor,
                                fallback_checkpoint_dir=fallback_checkpoint_dir,
                            )
                            if not (snap_ok and (i % max(1, config.frozen_eval_interval) == 0)):
                                continue
                            try:
                                metrics = frozen_eval.evaluate_checkpoint(
                                    diag_nn, config, paths, anchor,
                                )
                                frozen_eval.log_metrics_to_aim(
                                    run, metrics, anchor_iter=anchor,
                                    epoch=next_net, step=total_train_steps,
                                )
                                kls = [vm["kl_mcts_net"] for vm in metrics.values()
                                       if "kl_mcts_net" in vm]
                                if kls:
                                    mean_kl = sum(kls) / len(kls)
                                    frozen_eval.append_kl_history(
                                        paths, anchor, next_net, mean_kl,
                                    )
                                    hist = frozen_kl_history.setdefault(anchor, [])
                                    hist.append((next_net, mean_kl))
                                    if len(hist) > _FROZEN_KL_HIST_LEN:
                                        frozen_kl_history[anchor] = hist[-_FROZEN_KL_HIST_LEN:]
                                    frozen_eval.print_kl_health(
                                        run, frozen_kl_history[anchor],
                                        epoch=next_net, step=total_train_steps,
                                        anchor_iter=anchor,
                                    )
                            except Exception as exc:
                                tqdm.tqdm.write(f"  Frozen eval warning [iter {next_net}, "
                                                f"anchor {anchor}]: skipped: {exc}")

                    if diag_nn is not None:
                        del diag_nn
                    gc.collect()
                stage_times["diagnostics"] = time.time() - stage_start

            # Log variant_sample_frac at end of iteration from this iteration's actual data.
            if _is_unified_game(config):
                current_counts = _count_variant_samples(paths, i)
                if current_counts:
                    total_sc = sum(current_counts) or 1
                    for vi, vname in enumerate(UNIFIED_VARIANT_NAMES):
                        run.track(current_counts[vi] / total_sc, name="variant_sample_frac",
                                  epoch=i, step=total_train_steps, context={"variant": vname})

            stage_start = time.time()
            _log_win_rate_matrix(paths, i, run, total_train_steps)
            if _is_unified_game(config):
                _generate_visualizations(config, paths, i, run, total_train_steps, variant_stats)
            stage_times["visualizations"] = time.time() - stage_start

            total_time = time.time() - iteration_start
            for stage_name, stage_time in stage_times.items():
                percentage = (stage_time / total_time) * 100.0
                run.track(percentage, name="time_percent", epoch=i, step=total_train_steps, context={"stage": stage_name})
                run.track(stage_time, name="time_seconds", epoch=i, step=total_train_steps, context={"stage": stage_name})
            run.track(total_time, name="time_seconds", epoch=i, step=total_train_steps, context={"stage": "total"})

            tracy_frame()

    if hasattr(run, 'close'):
        run.close()
