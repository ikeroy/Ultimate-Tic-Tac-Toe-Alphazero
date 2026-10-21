"""Unit tests for config.py -- Config loading, YAML, CLI overrides, paths, registry."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import alphazero
from config import TrainConfig, GAME_REGISTRY, load_config, find_latest_checkpoint
import neural_net


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_default_config():
    """TrainConfig() has all expected defaults."""
    config = TrainConfig()
    assert config.game == "uttt"
    assert config.depth == 4
    assert config.channels == 12
    assert config.kernel_size == 5
    assert config.dense_net is True
    assert config.uttt_hierarchical is False
    assert config.uttt_policy_head == "flat"
    assert config.uttt_destination_context is False
    assert config.uttt_macro_channels == -1
    assert config.uttt_macro_transformer_layers == 0
    assert config.uttt_macro_transformer_heads == 4
    assert config.uttt_macro_transformer_ff_mult == 4.0
    assert config.uttt_macro_transformer_dropout == 0.0
    assert config.cpuct == 1.25
    assert config.fpu_reduction == 0.25
    assert config.selfplay_mcts_visits == 100
    assert config.fast_mcts_visits == 25
    assert config.compare_mcts_visits == 50
    assert config.self_play_batch_size == 1024
    assert config.self_play_games_per_iter == 0
    assert config.selfplay_use_latest is False
    assert config.train_batch_size == 1024
    assert config.train_from_best is False
    assert config.gating_enabled is True
    assert config.iterations == 200
    assert config.lr_schedule == "constant"
    assert config.lr_steps == []
    assert config.lr_drop_factor == 0.3
    assert config.lr_patience == 8
    assert config.lr_min_iter == 50
    assert config.lr_min_between_drops == 30
    assert config.lr_max_drops == 3
    assert config.bootstrap_compare_past == 5
    assert config.elo_enabled is True
    assert config.whr_enabled is True
    assert config.uttt_ai_eval_interval == 10
    assert config.uttt_ai_eval_games == 256
    assert config.uttt_ai_eval_temperature == 0.0
    assert config.uttt_ai_eval_opening_plies == 6
    assert config.uttt_ai_eval_opening_seed == 0
    assert config.uttt_ai_select_best is False


def test_games_per_iter_default_product():
    """Default self-play games/iter follows the legacy batch product."""
    import game_runner

    config = TrainConfig()
    config.game = "uttt"
    config.self_play_batch_size = 16
    config.self_play_concurrent_batch_mult = 3
    config.self_play_chunks = 2
    config.self_play_games_per_iter = 0

    assert game_runner.games_per_iter(config) == 16 * config.Game.NUM_PLAYERS() * 3 * 2


def test_games_per_iter_explicit_override():
    """Explicit self-play games/iter decouples data volume from batch sizing."""
    import game_runner

    config = TrainConfig()
    config.game = "uttt"
    config.self_play_batch_size = 1024
    config.self_play_concurrent_batch_mult = 1
    config.self_play_chunks = 1
    config.self_play_games_per_iter = 1024

    assert game_runner.games_per_iter(config) == 1024


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs")


def test_yaml_loading_uttt():
    """Loading uttt.yaml sets game to uttt, preserves other defaults."""
    config = load_config(os.path.join(CONFIGS_DIR, "uttt.yaml"), {})
    assert config.game == "uttt"
    assert config.depth == 4  # default preserved
    assert config.channels == 12  # default preserved


def test_yaml_loading_uttt_transformer_enables_uttt_ai_eval():
    """Main UTTT transformer config tracks raw-policy eval vs uttt.ai."""
    config = load_config(os.path.join(CONFIGS_DIR, "uttt_transformer_g512_m32.yaml"), {})
    assert config.game == "uttt"
    assert config.uttt_ai_eval_interval == 10
    assert config.uttt_ai_eval_games == 256
    assert config.uttt_ai_eval_temperature == 0.0
    assert config.uttt_ai_eval_opening_plies == 6
    assert config.uttt_ai_eval_opening_seed == 0


# ---------------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------------


def test_cli_overrides():
    """CLI --key val overrides both defaults and YAML."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "uttt.yaml"),
        {"depth": "8", "cpuct": "2.0"},
    )
    assert config.depth == 8
    assert config.cpuct == 2.0


def test_bool_cli_override_false():
    """Boolean CLI overrides handle false."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "uttt.yaml"), {"dense_net": "false"}
    )
    assert config.dense_net is False


def test_bool_cli_override_true():
    """Boolean CLI overrides handle true."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "uttt.yaml"), {"dense_net": "true"}
    )
    assert config.dense_net is True


def test_rating_metric_cli_overrides():
    """Elo/WHR reporting can be disabled without list-based YAML changes."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "uttt.yaml"),
        {"elo_enabled": "false", "whr_enabled": "false"},
    )
    assert config.elo_enabled is False
    assert config.whr_enabled is False


def test_uttt_hierarchical_cli_override():
    """UTTT hierarchical architecture flag can be enabled from config overrides."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "uttt.yaml"), {"uttt_hierarchical": "true"}
    )
    assert config.game == "uttt"
    assert config.uttt_hierarchical is True


def test_uttt_policy_head_cli_override():
    """UTTT hierarchical policy head mode can be set from config overrides."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "uttt.yaml"),
        {"uttt_policy_head": "factorized_hybrid"},
    )
    assert config.uttt_policy_head == "factorized_hybrid"


def test_uttt_destination_context_cli_override():
    """UTTT destination-board context can be enabled from config overrides."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "uttt.yaml"),
        {"uttt_destination_context": "true"},
    )
    assert config.uttt_destination_context is True


def test_uttt_macro_transformer_cli_override():
    """UTTT macro transformer settings can be enabled from config overrides."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "uttt.yaml"),
        {
            "uttt_macro_transformer_layers": "2",
            "uttt_macro_channels": "32",
            "uttt_macro_transformer_heads": "4",
            "uttt_macro_transformer_ff_mult": "3.5",
            "uttt_macro_transformer_dropout": "0.1",
        },
    )
    assert config.uttt_macro_channels == 32
    assert config.uttt_macro_transformer_layers == 2
    assert config.uttt_macro_transformer_heads == 4
    assert config.uttt_macro_transformer_ff_mult == 3.5
    assert config.uttt_macro_transformer_dropout == 0.1


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------


def test_unknown_yaml_key_warns(capsys):
    """Unknown key in YAML prints warning and is skipped."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("game: uttt\nbogus_key: 42\n")
        f.flush()
        try:
            config = load_config(f.name, {})
            assert config.game == "uttt"
            captured = capsys.readouterr()
            assert "ignoring unknown config key in YAML: bogus_key" in captured.out
        finally:
            os.unlink(f.name)


def test_unknown_cli_key_raises():
    """Unknown --key in CLI raises ValueError."""
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(os.path.join(CONFIGS_DIR, "uttt.yaml"), {"nonexistent": "5"})


# ---------------------------------------------------------------------------
# Game registry
# ---------------------------------------------------------------------------


def test_game_registry_classes_exist():
    """All registry entries resolve to valid alphazero classes."""
    for name, cls_name in GAME_REGISTRY.items():
        assert hasattr(alphazero, cls_name), f"Missing class: {cls_name}"


def test_game_property():
    """Config.Game resolves to the correct alphazero class."""
    config = TrainConfig(game="uttt")
    assert config.Game == alphazero.UltimateTicTacToeGS


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------


def test_network_name():
    """network_name returns densenet or resnet."""
    config = TrainConfig(dense_net=True)
    assert config.network_name == "densenet"
    config.dense_net = False
    assert config.network_name == "resnet"


def test_auto_experiment_name():
    """auto_experiment_name formats correctly."""
    config = TrainConfig(
        dense_net=True, depth=4, channels=12, kernel_size=5, selfplay_mcts_visits=100
    )
    assert config.auto_experiment_name == "densenet-4d-12c-5k-100sims"

    config.dense_net = False
    assert config.auto_experiment_name == "resnet-4d-12c-5k-100sims"


def test_resolved_cache_shards():
    """resolved_cache_shards defaults to cpu_count."""
    config = TrainConfig(cache_shards=-1)
    assert config.resolved_cache_shards == os.cpu_count()

    config.cache_shards = 8
    assert config.resolved_cache_shards == 8


# ---------------------------------------------------------------------------
# Experiment directory resolution
# ---------------------------------------------------------------------------


def test_experiment_dir_new(tmp_path):
    """New experiment creates directory based on auto name."""
    config = TrainConfig()
    result = config.resolve_experiment_dir(base=str(tmp_path))
    expected = os.path.join(str(tmp_path), "uttt", "densenet-4d-12c-5k-100sims")
    assert result == expected


def test_experiment_dir_auto_suffix(tmp_path):
    """Auto-suffix -01 when experiment dir already exists."""
    base = tmp_path / "uttt" / "densenet-4d-12c-5k-100sims"
    base.mkdir(parents=True)
    config = TrainConfig()
    result = config.resolve_experiment_dir(base=str(tmp_path))
    assert result.endswith("-01")


def test_experiment_dir_auto_suffix_02(tmp_path):
    """Auto-suffix -02 when -01 also exists."""
    base_dir = tmp_path / "uttt"
    (base_dir / "densenet-4d-12c-5k-100sims").mkdir(parents=True)
    (base_dir / "densenet-4d-12c-5k-100sims-01").mkdir(parents=True)
    config = TrainConfig()
    result = config.resolve_experiment_dir(base=str(tmp_path))
    assert result.endswith("-02")


def test_experiment_dir_explicit_name(tmp_path):
    """Explicit experiment name used as-is."""
    config = TrainConfig()
    result = config.resolve_experiment_dir(
        base=str(tmp_path), explicit_name="my-run"
    )
    assert result.endswith("my-run")
    assert "uttt" in result


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_resolve_paths():
    """resolve_paths returns correct subdirectory structure."""
    config = TrainConfig()
    paths = config.resolve_paths("/data/uttt/exp1")
    assert paths["checkpoint"] == "/data/uttt/exp1/checkpoint"
    assert paths["history"] == "/data/uttt/exp1/history"
    assert paths["tmp_history"] == "/data/uttt/exp1/tmp_history"
    assert paths["reservoir"] == "/data/uttt/exp1/reservoir"
    assert paths["experiment"] == "/data/uttt/exp1"


# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------


def test_config_save_and_reload(tmp_path):
    """Save resolved config as YAML, reload, verify identical."""
    config = load_config(
        os.path.join(CONFIGS_DIR, "uttt.yaml"),
        {"iterations": "50"},
    )
    save_path = str(tmp_path / "config.yaml")
    config.save(save_path)

    reloaded = load_config(save_path, {})
    assert reloaded.game == config.game
    assert reloaded.iterations == 50
    assert reloaded.depth == config.depth
    assert reloaded.channels == config.channels


# ---------------------------------------------------------------------------
# LR schedule config
# ---------------------------------------------------------------------------


def test_lr_schedule_step_from_yaml(tmp_path):
    """Step LR schedule loads lr_steps from YAML."""
    yaml_content = (
        "game: uttt\n"
        "lr_schedule: step\n"
        "lr_steps:\n"
        "  - [0, 0.01]\n"
        "  - [250, 0.003]\n"
        "  - [750, 0.001]\n"
    )
    yaml_path = str(tmp_path / "step.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    config = load_config(yaml_path, {})
    assert config.lr_schedule == "step"
    assert config.lr_steps == [[0, 0.01], [250, 0.003], [750, 0.001]]


def test_lr_schedule_adaptive_from_yaml(tmp_path):
    """Adaptive LR schedule fields load from YAML."""
    yaml_content = (
        "game: uttt\n"
        "lr: 0.01\n"
        "lr_schedule: adaptive\n"
        "lr_drop_factor: 0.5\n"
        "lr_patience: 10\n"
        "lr_min_iter: 100\n"
        "lr_min_between_drops: 50\n"
        "lr_max_drops: 2\n"
    )
    yaml_path = str(tmp_path / "adaptive.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    config = load_config(yaml_path, {})
    assert config.lr_schedule == "adaptive"
    assert config.lr_drop_factor == 0.5
    assert config.lr_patience == 10
    assert config.lr_min_iter == 100
    assert config.lr_min_between_drops == 50
    assert config.lr_max_drops == 2


def test_lr_schedule_save_reload(tmp_path):
    """LR schedule fields survive save/reload cycle."""
    config = TrainConfig(
        lr_schedule="step",
        lr_steps=[[0, 0.01], [100, 0.003]],
    )
    save_path = str(tmp_path / "config.yaml")
    config.save(save_path)

    reloaded = load_config(save_path, {})
    assert reloaded.lr_schedule == "step"
    assert reloaded.lr_steps == [[0, 0.01], [100, 0.003]]


# ---------------------------------------------------------------------------
# Arg parsing (train.py parse_args)
# ---------------------------------------------------------------------------


from unittest.mock import patch
from train import parse_args


def _parse(argv):
    """Helper: call parse_args with given argv (excluding 'train.py')."""
    with patch("sys.argv", ["train.py"] + argv):
        return parse_args()


def test_parse_config_positional():
    """Positional config path is extracted."""
    args, overrides = _parse(["configs/uttt.yaml"])
    assert args.config == "configs/uttt.yaml"
    assert args.resume is None
    assert args.bootstrap is None
    assert overrides == {}


def test_parse_config_with_overrides():
    """Config + overrides parse correctly."""
    args, overrides = _parse(["configs/uttt.yaml", "--depth", "8", "--cpuct", "2.0"])
    assert args.config == "configs/uttt.yaml"
    assert overrides == {"depth": "8", "cpuct": "2.0"}


def test_parse_resume():
    """--resume is parsed, no config."""
    args, overrides = _parse(["--resume", "data/uttt/exp1"])
    assert args.config is None
    assert args.resume == "data/uttt/exp1"
    assert overrides == {}


def test_parse_resume_with_overrides():
    """--resume + overrides parse correctly."""
    args, overrides = _parse(["--resume", "data/uttt/exp1", "--iterations", "400"])
    assert args.resume == "data/uttt/exp1"
    assert args.config is None
    assert overrides == {"iterations": "400"}


def test_parse_bootstrap():
    """--bootstrap is parsed, no config."""
    args, overrides = _parse(["--bootstrap", "data/uttt/exp1"])
    assert args.config is None
    assert args.bootstrap == "data/uttt/exp1"
    assert overrides == {}


def test_parse_bootstrap_with_overrides():
    """--bootstrap + overrides don't consume values as config."""
    args, overrides = _parse([
        "--bootstrap", "data/uttt/exp1",
        "--selfplay_mcts_visits", "30",
        "--fast-mcts-visits", "1",
    ])
    assert args.config is None
    assert args.bootstrap == "data/uttt/exp1"
    assert overrides == {"selfplay_mcts_visits": "30", "fast_mcts_visits": "1"}


def test_parse_hyphen_underscore_normalization():
    """Override keys normalize hyphens to underscores."""
    args, overrides = _parse([
        "configs/uttt.yaml",
        "--self-play-batch-size", "128",
        "--temp-decay-half-life", "20",
        "--self-play-games-per-iter", "1024",
    ])
    assert overrides == {
        "self_play_batch_size": "128",
        "temp_decay_half_life": "20",
        "self_play_games_per_iter": "1024",
    }


def test_parse_bool_flag_no_value():
    """Boolean flag without value defaults to 'true'."""
    args, overrides = _parse(["configs/uttt.yaml", "--dense-net"])
    assert overrides == {"dense_net": "true"}


def test_parse_bootstrap_with_experiment():
    """--bootstrap + --experiment are both parsed."""
    args, overrides = _parse([
        "--bootstrap", "data/uttt/exp1",
        "--experiment", "my-new-run",
        "--channels", "32",
    ])
    assert args.bootstrap == "data/uttt/exp1"
    assert args.experiment == "my-new-run"
    assert args.config is None
    assert overrides == {"channels": "32"}


# ---------------------------------------------------------------------------
# find_latest_checkpoint
# ---------------------------------------------------------------------------


def test_find_latest_checkpoint(tmp_path):
    """find_latest_checkpoint returns highest iteration number from checkpoint files."""
    cp_dir = tmp_path / "checkpoint"
    cp_dir.mkdir()
    (cp_dir / "0005-net.pt").touch()
    (cp_dir / "0010-net.pt").touch()
    (cp_dir / "0003-net.pt").touch()
    assert find_latest_checkpoint(str(cp_dir)) == 10


def test_find_latest_checkpoint_empty(tmp_path):
    """find_latest_checkpoint returns 0 for empty directory."""
    cp_dir = tmp_path / "checkpoint"
    cp_dir.mkdir()
    assert find_latest_checkpoint(str(cp_dir)) == 0


# ---------------------------------------------------------------------------
# load_config warn parameter
# ---------------------------------------------------------------------------


def test_load_config_warn_false_suppresses_warnings(tmp_path, capsys):
    """warn=False suppresses unknown key warnings."""
    yaml_path = str(tmp_path / "stale.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\nlr_milestone: 100\nbootstrap_from: foo\n")
    load_config(yaml_path, {}, warn=False)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_stale_keys_do_not_corrupt_config(tmp_path):
    """Stale YAML keys are not set as attributes on the config object."""
    yaml_path = str(tmp_path / "stale.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\nlr_milestone: 100\nbootstrap_from: foo\n")
    config = load_config(yaml_path, {}, warn=False)
    assert config.game == "uttt"
    assert not hasattr(config, "lr_milestone")
    assert not hasattr(config, "bootstrap_from")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_invalid_lr_schedule(tmp_path):
    """Invalid lr_schedule raises ValueError."""
    yaml_path = str(tmp_path / "bad_lr.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\nlr_schedule: unknown\n")
    with pytest.raises(ValueError, match="Unknown lr_schedule"):
        load_config(yaml_path, {})


def test_validate_value_target_and_policy_loss(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("game: uttt\nvalue_target: root\npolicy_loss: kl\n")

    config = load_config(str(yaml_path), {})

    assert config.value_target == "root"
    assert config.policy_loss == "kl"


def test_validate_invalid_value_target(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("game: uttt\nvalue_target: roots\n")

    with pytest.raises(ValueError, match="Unknown value_target"):
        load_config(str(yaml_path), {})


def test_validate_invalid_policy_loss(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("game: uttt\npolicy_loss: wasserstein_or_something\n")

    with pytest.raises(ValueError, match="Unknown policy_loss"):
        load_config(str(yaml_path), {})


def test_validate_cosine_lr_schedule(tmp_path):
    """Cosine LR schedule and its minimum factor are accepted."""
    yaml_path = str(tmp_path / "cosine_lr.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\nlr_schedule: cosine\nlr_cosine_min_factor: 0.1\n")
    config = load_config(yaml_path, {})
    assert config.lr_schedule == "cosine"
    assert config.lr_cosine_min_factor == 0.1


def test_validate_exponential_lr_schedule(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "game: uttt\n"
        "lr_schedule: exponential\n"
        "lr_exponential_final_factor: 0.25\n"
        "lr_exponential_start_iter: 60\n"
        "lr_exponential_end_iter: 100\n"
    )

    config = load_config(yaml_path, {})

    assert config.lr_schedule == "exponential"
    assert config.lr_exponential_final_factor == 0.25
    assert config.lr_exponential_start_iter == 60
    assert config.lr_exponential_end_iter == 100


def test_validate_gating_panel_size_zero(tmp_path):
    """gating_panel_size: 0 raises ValueError."""
    yaml_path = str(tmp_path / "bad_panel.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\ngating_panel_size: 0\n")
    with pytest.raises(ValueError, match="gating_panel_size must be >= 1"):
        load_config(yaml_path, {})


def test_validate_iterations_zero(tmp_path):
    """iterations: 0 raises ValueError."""
    yaml_path = str(tmp_path / "bad_iter.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\niterations: 0\n")
    with pytest.raises(ValueError, match="iterations must be >= 1"):
        load_config(yaml_path, {})


def test_validate_unknown_game(tmp_path):
    """Unknown game raises ValueError."""
    yaml_path = str(tmp_path / "bad_game.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: nonexistent_game\n")
    with pytest.raises(ValueError, match="Unknown game"):
        load_config(yaml_path, {})


def test_temp_decay_half_life_int(tmp_path):
    """Scalar int half-life loads and validates."""
    yaml_path = str(tmp_path / "tdhl_int.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\ntemp_decay_half_life: 8\n")
    cfg = load_config(yaml_path, {})
    assert cfg.temp_decay_half_life == 8


def test_temp_decay_half_life_invalid_type(tmp_path):
    """String value raises ValueError."""
    yaml_path = str(tmp_path / "tdhl_str.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\ntemp_decay_half_life: not_a_number\n")
    with pytest.raises(ValueError, match="must be a number"):
        load_config(yaml_path, {})


def test_list_cli_override_warns(tmp_path, capsys):
    """CLI override of list field prints warning."""
    yaml_path = str(tmp_path / "config.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\n")
    load_config(yaml_path, {"lr_steps": "[[0, 0.01]]"})
    captured = capsys.readouterr()
    assert "cannot override list field 'lr_steps' via CLI" in captured.out


# ---------------------------------------------------------------------------
# nnargs_from_config
# ---------------------------------------------------------------------------


def test_nnargs_from_config_head_fields(tmp_path):
    """nnargs_from_config propagates head architecture fields from YAML."""
    yaml_path = str(tmp_path / "config.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            "game: uttt\n"
            "v_head_convs: 2\n"
            "pi_head_convs: 3\n"
            "v_fc_layers: 2\n"
            "pi_fc_layers: 1\n"
        )
    config = load_config(yaml_path, {})
    nnargs = neural_net.nnargs_from_config(config)
    assert nnargs.v_head_convs == 2
    assert nnargs.pi_head_convs == 3
    assert nnargs.v_fc_layers == 2
    assert nnargs.pi_fc_layers == 1


def test_nnargs_from_config_cli_override(tmp_path):
    """nnargs_from_config picks up head fields set via CLI overrides."""
    yaml_path = str(tmp_path / "config.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\n")
    config = load_config(yaml_path, {"v_head_convs": "1", "pi_fc_layers": "2"})
    nnargs = neural_net.nnargs_from_config(config)
    assert nnargs.v_head_convs == 1
    assert nnargs.pi_fc_layers == 2
    # defaults preserved for fields not overridden
    assert nnargs.pi_head_convs == 0
    assert nnargs.v_fc_layers == 1


def test_nnargs_from_config_optimizer(tmp_path):
    yaml_path = str(tmp_path / "config.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\noptimizer: adamw\nweight_decay: 0.0003\n")

    nnargs = neural_net.nnargs_from_config(load_config(yaml_path, {}))

    assert nnargs.optimizer == "adamw"
    assert nnargs.weight_decay == pytest.approx(0.0003)


def test_validate_invalid_optimizer(tmp_path):
    yaml_path = str(tmp_path / "config.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\noptimizer: rmsprop\n")

    with pytest.raises(ValueError, match="Unknown optimizer"):
        load_config(yaml_path, {})


def test_validate_adamw_parameters(tmp_path):
    yaml_path = str(tmp_path / "config.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\nadamw_beta2: 1.0\n")

    with pytest.raises(ValueError, match="adamw_beta2"):
        load_config(yaml_path, {})


def test_validate_invalid_uttt_ai_opening_plies(tmp_path):
    yaml_path = str(tmp_path / "config.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\nuttt_ai_eval_opening_plies: -1\n")

    with pytest.raises(ValueError, match="uttt_ai_eval_opening_plies"):
        load_config(yaml_path, {})


def test_validate_invalid_uttt_ai_opening_seed(tmp_path):
    yaml_path = str(tmp_path / "config.yaml")
    with open(yaml_path, "w") as f:
        f.write("game: uttt\nuttt_ai_eval_opening_seed: -2\n")

    with pytest.raises(ValueError, match="uttt_ai_eval_opening_seed"):
        load_config(yaml_path, {})
