"""Training configuration system.

Provides TrainConfig dataclass with YAML loading, CLI overrides,
game registry, experiment directory resolution, and path management.
"""

import glob
import os
import re
from dataclasses import dataclass, field, asdict

import yaml
import alphazero


GAME_REGISTRY = {
    "uttt": "UltimateTicTacToeGS",
}


@dataclass
class TrainConfig:
    # Game
    game: str = "uttt"

    # Network
    depth: int = 4
    channels: int = 12
    kernel_size: int = 5
    dense_net: bool = True
    uttt_hierarchical: bool = False
    head_channels: int = 32
    head_pool: bool = True
    v_head_convs: int = 0
    pi_head_convs: int = 0
    v_fc_layers: int = 1
    pi_fc_layers: int = 0
    # UTTT hierarchical policy head:
    # "flat" = original 81-logit head.
    # "factorized_simple" = board logits + per-board cell logits.
    # "factorized_hybrid" = original flat logits + factorized logits.
    uttt_policy_head: str = "flat"
    # Add the macro embedding of the board that each cell sends the opponent to
    # into UTTT hierarchical policy features.
    uttt_destination_context: bool = False
    # Separate macro-board/transformer width for UTTT hierarchical networks.
    # -1 ties macro width to channels, preserving legacy behavior.
    uttt_macro_channels: int = -1
    # Optional transformer over the 3x3 macro-board token grid in the UTTT
    # hierarchical architecture. 0 layers preserves the pure-CNN baseline.
    uttt_macro_transformer_layers: int = 0
    uttt_macro_transformer_heads: int = 4
    uttt_macro_transformer_ff_mult: float = 4.0
    uttt_macro_transformer_dropout: float = 0.0
    # "batch" (default, current behavior) or "layer" (GroupNorm(1,C) — per-sample
    # LayerNorm across (C,H,W); avoids the batch-statistic propagation that
    # contributes to feature collapse in bootstrapped RL targets).
    trunk_norm: str = "batch"
    # AlphaZero standard. KataGo uses 3e-5 (even lower) and the AlphaZero
    # paper specifies 1e-4. Experiments validated the move from 1e-3 -> 1e-4;
    # 1e-3 was the old codebase default and
    # produced visibly worse plasticity / effective rank.
    weight_decay: float = 1e-4
    # Optimizer used for network training. Checkpoints persist this choice so
    # optimizer state can be reconstructed before loading.
    optimizer: str = "sgd"
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    # "relu" (default) or "crelu" (Concatenated ReLU, preserves negative-direction
    # features and mitigates dying-ReLU plasticity loss; doubles intermediate
    # channel count in blocks).
    trunk_act: str = "relu"
    # Orthogonal regularization strength on trunk Conv2d weights. 0 disables.
    # Penalizes filter correlation directly; promotes high effective rank.
    orth_reg_lambda: float = 0.0

    # MCTS
    cpuct: float = 1.25
    fpu_reduction: float = 0.25
    mcts_root_temp: float = 1.25
    root_fpu_zero: bool = True
    shaped_dirichlet: bool = True
    policy_target_pruning: bool = True
    selfplay_mcts_visits: int = 100
    fast_mcts_visits: int = 25
    compare_mcts_visits: int = 50
    # Playout cap randomization (KataGo). Fraction of self-play moves that
    # use fast_mcts_visits instead of selfplay_mcts_visits. Capped moves
    # don't enter training data. Set 0 to disable (every move is full search
    # and contributes data).
    playout_cap_percent: float = 0.75
    # Gumbel AlphaZero (Danihelka 2022). Replaces PUCT root + Dirichlet noise
    # with Gumbel-Top-K + Sequential Halving and trains on the improved-policy
    # target. Off by default. Capped searches always fall back to PUCT.
    gumbel_enabled: bool = False
    # Max considered actions at root. Auto-capped at runtime to
    # min(num_legal_actions, num_sims_target). Paper uses 16 universally
    # across Go/chess/Atari; ablation showed m in {8,16,32} equivalent.
    gumbel_m: int = 16
    gumbel_c_visit: float = 50.0  # sigma formula constant (paper default)
    gumbel_c_scale: float = 1.0   # sigma formula constant (paper default)
    gumbel_full: bool = False     # also use pi'-matching at non-root nodes
    # What algorithm to use for capped/fast self-play searches:
    #   "auto"   - same as the main search algo (Gumbel if gumbel_enabled, else PUCT)
    #   "puct"   - force PUCT for capped (even when main is Gumbel)
    #   "gumbel" - force Gumbel for capped (Gumbel-everywhere)
    # Retesting showed PUCT-for-capped trains a stronger network than
    # Gumbel-for-capped at low capped visit counts (capped Gumbel introduces
    # ~40% top-1 variance in trajectories, polluting training data).
    fast_search_algo: str = "auto"

    def resolve_fast_search_uses_gumbel(self) -> bool:
        """Return True iff capped self-play searches should run Gumbel."""
        if self.fast_search_algo == "auto":
            return self.gumbel_enabled
        if self.fast_search_algo == "puct":
            return False
        if self.fast_search_algo == "gumbel":
            return True
        raise ValueError(
            f"fast_search_algo must be 'auto'/'puct'/'gumbel', got {self.fast_search_algo!r}"
        )

    # Temperature
    self_play_temp: float = 1.0
    eval_temp: float = 0.5
    final_temp: float = 0.2
    # Temperature-decay half-life in turns; 0 disables decay.
    temp_decay_half_life: object = 10

    # Self-play
    self_play_batch_size: int = 1024
    self_play_concurrent_batch_mult: int = 2
    self_play_chunks: int = 1
    # 0 = derive from self_play_batch_size * NUM_PLAYERS *
    # self_play_concurrent_batch_mult * self_play_chunks.
    # Positive values decouple generated games/iter from GPU batch sizing.
    self_play_games_per_iter: int = 0
    # AlphaZero-style continuous self-play. When enabled, iteration i uses
    # checkpoint i even if that checkpoint has not passed the arena gate.
    selfplay_use_latest: bool = False

    # Training
    train_batch_size: int = 1024
    # If true, each new candidate starts gradient training from the current
    # gated best checkpoint. Failed candidates are discarded as parents.
    train_from_best: bool = False
    # KataGo's max-train-bucket-per-new-data. Each fresh sample gets 4
    # training passes through the network. This was validated empirically;
    # the previous default was 1.
    train_sample_rate: int = 4
    # Optional fixed supervised dataset mixed into every training update.
    # This is intended for a compact external-teacher anchor: normal self-play
    # remains the majority of samples, while the anchor prevents policy/value
    # drift on positions with stronger search targets.
    teacher_anchor_dir: str = ""
    teacher_anchor_fraction: float = 0.0
    lr: float = 0.01
    cv: float = 1.5

    # LR schedule: "constant", "step", "adaptive", "cosine", or "exponential"
    lr_schedule: str = "constant"
    lr_steps: list = field(default_factory=list)  # for step mode: [[0, 0.01], [250, 0.003]]
    lr_cosine_min_factor: float = 0.1
    lr_cosine_start_iter: int = -1  # -1 = start after warmup
    lr_cosine_end_iter: int = -1    # -1 = final configured iteration
    lr_exponential_final_factor: float = 0.1
    lr_exponential_start_iter: int = -1  # -1 = start after warmup
    lr_exponential_end_iter: int = -1    # -1 = final configured iteration
    lr_drop_factor: float = 0.3
    lr_patience: int = 8
    lr_min_iter: int = 50
    lr_min_between_drops: int = 30
    lr_max_drops: int = 3
    # Value target stored in self-play history:
    #   "final" = final game W/L/D result, current AlphaZero-style behavior
    #   "root"  = MCTS root-value estimate at the sampled position
    value_target: str = "final"
    # Policy target loss name. "cross_entropy" and "kl" have identical
    # gradients for soft policy targets; "kl" subtracts target entropy when
    # reported. Kept explicit so experiments can state the intended target.
    policy_loss: str = "cross_entropy"

    # Gating
    gating_enabled: bool = True
    gating_panel_size: int = 1
    gating_panel_win_rate: float = 0.52
    gating_best_win_rate: float = 0.52

    # History
    hist_size: int = 30_000
    # Window-growth formula: Nwindow = scalar * (1 + beta * (ratio^alpha - 1) / alpha)
    # window_size_unit determines what "scalar" means:
    #   "iterations" (legacy default): scalar in iters; ratio = (iter+1)/scalar
    #   "games":      scalar in games; ratio = total_games/scalar_games.
    #                 total_games is computed exactly from the self-play
    #                 product self_play_batch_size * NUM_PLAYERS *
    #                 self_play_concurrent_batch_mult * self_play_chunks
    #                 (constant per iter, so no history walking).
    # KataGo calibration: their reported numbers are 4.2M games -> 241M
    # samples (~57 samples/game after playout-cap-randomization keeps only
    # ~25% of moves). Their sample-space c=250_000 -> ~4400 games. Their
    # alpha=0.75, beta=0.4. To roughly match KataGo's growth shape on a
    # games basis, set scalar_games=4400, alpha=0.75, beta=0.4.
    # Games-based is preferable when different variants/games have very
    # different lengths, since each game counts equally regardless of how
    # many positions it contributes.
    window_size_unit: str = "iterations"
    window_size_alpha: float = 0.5
    window_size_beta: float = 0.7
    window_size_scalar: float = 6.0
    window_size_scalar_games: int = 4400

    # Cache
    max_cache_size: int = 200_000
    cache_shards: int = -1  # -1 = os.cpu_count()

    # Workers
    result_workers: int = 2
    data_workers: int = -1  # -1 = os.cpu_count() - 1

    # Training I/O
    streaming_active_files: int = 8

    # Resignation
    resign_percent: float = 0.02
    resign_playthrough_percent: float = 0.20

    # Eval game counts (total games = batch_size * NUM_PLAYERS per position)
    past_compare_batch_size: int = 64
    gate_compare_batch_size: int = 64

    # Iteration control
    iterations: int = 200
    # List of historical comparisons to run each iter for elo.
    # Negative entries = relative offset (-20 = "20 iters back").
    # Positive entries = absolute anchor iters (e.g. 50 = "always vs iter 50").
    # Anchors auto-retire (skip future iters) once their win rate saturates
    # past compare_past_saturation_threshold.
    compare_past: list = field(default_factory=lambda: [-20])
    compare_past_saturation_threshold: float = 0.97
    # Reporting-only rating metrics. Gating/model promotion still uses direct
    # win-rate comparisons vs best/panel. Disable these during architecture
    # probes to avoid extra Aim noise and WHR refit overhead.
    elo_enabled: bool = True
    whr_enabled: bool = True

    # Reservoir
    reservoir_recency_decay: float = 0.995
    reservoir_n_chunks: int = 100
    reservoir_chunk_size: int = 100_000
    reservoir_chunks_per_update: int = 10
    reservoir_update_interval: int = 10

    # Bootstrap
    bootstrap_window_only: bool = False
    bootstrap_compare_past: int = 5
    bootstrap_epochs: int = 1
    bootstrap_eval_interval: int = 500
    bootstrap_lr: float = 0.01
    bootstrap_lr_drop_factor: float = 0.3
    bootstrap_lr_patience: int = 3
    bootstrap_lr_cooldown: int = 1
    bootstrap_lr_max_drops: int = 3
    bootstrap_convergence_threshold: float = 0.002
    bootstrap_convergence_patience: int = 3

    # Inference optimization (AMP autocast + compile on GPU, skipped on CPU)
    amp_inference: bool = True
    half_storage: bool = True  # only affects history files
    torch_compile: bool = True

    # EMA weight averaging
    ema_averaging: bool = True

    # LR warmup
    lr_warmup_target: int = 15       # iterations to reach full LR
    lr_warmup_floor: float = 0.2     # minimum LR fraction during warmup

    # Compression
    zstd_level: int = 1

    # Parallel file loading
    loader_threads: int = -1  # -1 = os.cpu_count()

    # Frozen eval set (Feature 1, see plan i-want-the-vs-refactored-hippo.md)
    # Empty list = disabled. Multiple anchors give complementary windows
    # (e.g., [5, 30, 100, 300] for early/mid/late training signal).
    frozen_eval_anchor_iters: list = field(default_factory=list)
    frozen_eval_positions: int = 1024
    frozen_eval_interval: int = 1
    # Minimum complete games to play during snapshot. Guards against the
    # long-game edge case where pool fills in few games and one game dominates
    # the sample. Default is sane for any game; raise for more source diversity.
    frozen_eval_min_games: int = 20

    # Self-match at multiplied visits (Feature 2)
    # 0 = disabled. 2, 4, 8, ... = play current net at selfplay_mcts_visits vs
    # current net at selfplay_mcts_visits * multiplier.
    selfmatch_visit_multiplier: int = 0
    selfmatch_games: int = 100
    selfmatch_interval: int = 1

    # External UTTT.ai raw-policy benchmark.
    # Disabled by default. When enabled for game=uttt, every N iterations it
    # plays the freshly trained checkpoint directly from policy logits against
    # the cached uttt.ai ONNX policy/value net, with no MCTS on either side.
    uttt_ai_eval_interval: int = 10
    uttt_ai_eval_games: int = 256
    uttt_ai_eval_temperature: float = 0.0
    uttt_ai_eval_opening_plies: int = 6
    # Fixed opening seed makes checkpoint-to-checkpoint evals comparable.
    # Set to -1 to use the eval iteration as the seed instead.
    uttt_ai_eval_opening_seed: int = 0
    uttt_ai_eval_model: str = ""
    uttt_ai_eval_require_cuda: bool = False
    # Track the highest raw-policy UTTT.ai score as the saved champion. This
    # does not control self-play when selfplay_use_latest is enabled.
    uttt_ai_select_best: bool = False

    # Effective rank (Feature 3) — participation ratio of penultimate-layer
    # activations. Cheap; default ON.
    effective_rank_enabled: bool = True
    effective_rank_batch_size: int = 512

    def validate(self):
        if self.game not in GAME_REGISTRY:
            raise ValueError(f"Unknown game: {self.game}")
        valid_lr_schedules = {"constant", "step", "adaptive", "cosine", "exponential"}
        if self.lr_schedule not in valid_lr_schedules:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")
        valid_optimizers = {"sgd", "adamw"}
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        valid_value_targets = {"final", "root"}
        if self.value_target not in valid_value_targets:
            raise ValueError(f"Unknown value_target: {self.value_target}")
        valid_policy_losses = {"cross_entropy", "kl"}
        if self.policy_loss not in valid_policy_losses:
            raise ValueError(f"Unknown policy_loss: {self.policy_loss}")
        if not 0.0 <= self.adamw_beta1 < 1.0:
            raise ValueError(f"adamw_beta1 must be in [0, 1), got {self.adamw_beta1}")
        if not 0.0 <= self.adamw_beta2 < 1.0:
            raise ValueError(f"adamw_beta2 must be in [0, 1), got {self.adamw_beta2}")
        if self.adamw_eps <= 0.0:
            raise ValueError(f"adamw_eps must be positive, got {self.adamw_eps}")
        if not 0.0 <= self.lr_cosine_min_factor <= 1.0:
            raise ValueError(
                f"lr_cosine_min_factor must be between 0 and 1, got {self.lr_cosine_min_factor}"
            )
        if not 0.0 < self.lr_exponential_final_factor <= 1.0:
            raise ValueError(
                "lr_exponential_final_factor must be in (0, 1], "
                f"got {self.lr_exponential_final_factor}"
            )
        if self.gating_panel_size < 1:
            raise ValueError(f"gating_panel_size must be >= 1, got {self.gating_panel_size}")
        if not 0.0 <= self.teacher_anchor_fraction < 1.0:
            raise ValueError(
                "teacher_anchor_fraction must be in [0, 1), got "
                f"{self.teacher_anchor_fraction}"
            )
        if self.teacher_anchor_fraction > 0.0 and not self.teacher_anchor_dir:
            raise ValueError(
                "teacher_anchor_dir is required when teacher_anchor_fraction is positive"
            )
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.uttt_macro_transformer_layers < 0:
            raise ValueError(
                "uttt_macro_transformer_layers must be >= 0, "
                f"got {self.uttt_macro_transformer_layers}"
            )
        if self.uttt_macro_channels == 0 or self.uttt_macro_channels < -1:
            raise ValueError(
                "uttt_macro_channels must be -1 or a positive integer, "
                f"got {self.uttt_macro_channels}"
            )
        if self.uttt_ai_eval_interval < 0:
            raise ValueError(
                "uttt_ai_eval_interval must be >= 0, "
                f"got {self.uttt_ai_eval_interval}"
            )
        if self.uttt_ai_eval_games < 1:
            raise ValueError(
                "uttt_ai_eval_games must be >= 1, "
                f"got {self.uttt_ai_eval_games}"
            )
        if self.uttt_ai_eval_temperature < 0.0:
            raise ValueError(
                "uttt_ai_eval_temperature must be >= 0, "
                f"got {self.uttt_ai_eval_temperature}"
            )
        if self.uttt_ai_eval_opening_plies < 0:
            raise ValueError(
                "uttt_ai_eval_opening_plies must be >= 0, "
                f"got {self.uttt_ai_eval_opening_plies}"
            )
        if self.uttt_ai_eval_opening_seed < -1:
            raise ValueError(
                "uttt_ai_eval_opening_seed must be >= -1, "
                f"got {self.uttt_ai_eval_opening_seed}"
            )
        if self.uttt_ai_select_best and self.game != "uttt":
            raise ValueError("uttt_ai_select_best is only valid for game=uttt")
        if self.uttt_ai_select_best and self.uttt_ai_eval_interval <= 0:
            raise ValueError(
                "uttt_ai_eval_interval must be positive when uttt_ai_select_best is enabled"
            )
        if self.uttt_macro_transformer_heads < 1:
            raise ValueError(
                "uttt_macro_transformer_heads must be >= 1, "
                f"got {self.uttt_macro_transformer_heads}"
            )
        if self.uttt_macro_transformer_ff_mult <= 0.0:
            raise ValueError(
                "uttt_macro_transformer_ff_mult must be > 0, "
                f"got {self.uttt_macro_transformer_ff_mult}"
            )
        if not 0.0 <= self.uttt_macro_transformer_dropout < 1.0:
            raise ValueError(
                "uttt_macro_transformer_dropout must be in [0, 1), "
                f"got {self.uttt_macro_transformer_dropout}"
            )
        if not isinstance(self.temp_decay_half_life, (int, float)):
            raise ValueError(
                f"temp_decay_half_life must be a number, "
                f"got {type(self.temp_decay_half_life).__name__}"
            )
        if self.temp_decay_half_life < 0:
            raise ValueError("temp_decay_half_life must be non-negative")

    @property
    def network_name(self) -> str:
        return "densenet" if self.dense_net else "resnet"

    @property
    def auto_experiment_name(self) -> str:
        macro_suffix = ""
        if self.uttt_hierarchical and self.uttt_macro_channels > 0:
            macro_suffix = f"-{self.uttt_macro_channels}mc"
        return (
            f"{self.network_name}-{self.depth}d-{self.channels}c"
            f"{macro_suffix}-{self.kernel_size}k-{self.selfplay_mcts_visits}sims"
        )

    @property
    def Game(self):
        cls_name = GAME_REGISTRY[self.game]
        return getattr(alphazero, cls_name)

    @property
    def resolved_cache_shards(self) -> int:
        return os.cpu_count() if self.cache_shards == -1 else self.cache_shards

    @property
    def resolved_data_workers(self) -> int:
        return os.cpu_count() - 1 if self.data_workers == -1 else self.data_workers

    @property
    def resolved_loader_threads(self) -> int:
        return os.cpu_count() if self.loader_threads == -1 else self.loader_threads

    def resolve_experiment_dir(self, base="data", explicit_name=None) -> str:
        """Resolve experiment directory path.

        - If explicit_name given, use it directly: base/{game}/{explicit_name}/
        - Otherwise auto-derive from config and add integer suffix if collision:
          base/{game}/densenet-4d-16c-3k-300sims/
          base/{game}/densenet-4d-16c-3k-300sims-01/  (if first exists)
        """
        game_dir = os.path.join(base, self.game)

        if explicit_name:
            return os.path.join(game_dir, explicit_name)

        name = self.auto_experiment_name
        path = os.path.join(game_dir, name)

        if not os.path.exists(path):
            return path

        # Auto-suffix
        for i in range(1, 100):
            suffixed = os.path.join(game_dir, f"{name}-{i:02d}")
            if not os.path.exists(suffixed):
                return suffixed

        raise RuntimeError(f"Too many experiment directories for {name}")

    def resolve_paths(self, experiment_dir: str) -> dict:
        return {
            "experiment": experiment_dir,
            "checkpoint": os.path.join(experiment_dir, "checkpoint"),
            "history": os.path.join(experiment_dir, "history"),
            "tmp_history": os.path.join(experiment_dir, "tmp_history"),
            "reservoir": os.path.join(experiment_dir, "reservoir"),
        }

    def save(self, path: str):
        """Save config as YAML."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


def find_latest_checkpoint(checkpoint_dir):
    """Scan checkpoint dir for highest iteration number, return it (0 if none)."""
    pattern = os.path.join(checkpoint_dir, "*.pt")
    files = glob.glob(pattern)
    if not files:
        return 0
    best = 0
    for f in files:
        basename = os.path.basename(f)
        match = re.match(r"(\d+)-", basename)
        if match:
            best = max(best, int(match.group(1)))
    return best


def load_config(yaml_path: str, cli_overrides: dict, warn=True) -> TrainConfig:
    """Load config from YAML file with CLI overrides.

    yaml_path: path to YAML config file
    cli_overrides: dict of {key: string_value} from CLI --key val pairs
    """
    config = TrainConfig()

    if yaml_path:
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f) or {}
        for key, val in yaml_data.items():
            if not hasattr(config, key):
                if warn:
                    print(f"Warning: ignoring unknown config key in YAML: {key}")
                continue
            # YAML 1.1 (PyYAML default) parses unsigned scientific notation like
            # "1e-4" as a string, not a float. Coerce to the field's declared
            # type when the default is numeric and the YAML value is a string.
            field_type = type(getattr(config, key))
            if (
                field_type in (int, float)
                and isinstance(val, str)
                and key != "temp_decay_half_life"
            ):
                val = field_type(val)
            setattr(config, key, val)

    for key, val in cli_overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown config key: {key}")
        field_type = type(getattr(config, key))
        if field_type is list:
            if warn:
                print(f"Warning: cannot override list field '{key}' via CLI, use YAML instead")
            continue
        elif field_type is bool:
            setattr(config, key, str(val).lower() in ("true", "1", "yes"))
        else:
            setattr(config, key, field_type(val))

    config.validate()
    return config
