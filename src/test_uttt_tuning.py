import random
import argparse
from pathlib import Path

import numpy as np
import pytest

from uttt_tuning import (
    GENETIC_KEYS,
    RunData,
    _elapsed_for_iter,
    _focused_base_params,
    _focused_startup_trials,
    _focused_trial_overrides,
    _hparam_metrics,
    _project_lr_series,
    discover_runs,
    genetic_fitness,
    initial_genetic_population,
    milestone_elo,
    mutate_genetic_hparams,
    seed_lr_only_study,
    seed_tpe_study,
    tpe_config_overrides,
    tpe_params_from_config,
)


def _run(elo):
    values = np.asarray(elo, dtype=float)
    games = np.ones_like(values)
    empty = np.zeros((0, 0))
    return RunData(
        name="test",
        path=Path("."),
        config={},
        latest_iter=len(values) - 1,
        checkpoint_count=len(values),
        elo=values,
        whr=values,
        train_steps=np.zeros_like(values),
        wins=empty,
        draws=empty,
        pair_games=empty,
        pair_scores=empty,
        per_iter_games=games,
        per_iter_score=np.zeros_like(values),
        per_iter_draw_rate=np.zeros_like(values),
        wall_times=np.zeros_like(values),
    )


def test_genetic_fitness_rewards_recent_progress():
    flat = _run([1000.0] * 20)
    improving = _run([900.0 + i * 5.0 for i in range(20)])

    assert genetic_fitness(improving) > genetic_fitness(flat)


def test_mutation_stays_in_search_bounds():
    parent = initial_genetic_population({"optimizer": "sgd", "lr": 0.05}, 1, 1)[0]

    for seed in range(100):
        child = mutate_genetic_hparams(parent, random.Random(seed))
        assert set(child).issubset(GENETIC_KEYS)
        assert child["optimizer"] in {"sgd", "adamw"}
        assert 4 <= child["train_sample_rate"] <= 20
        assert 0.50 <= child["gating_panel_win_rate"] <= 0.56
        assert child["gating_panel_win_rate"] == child["gating_best_win_rate"]
        assert 0.4 <= child["playout_cap_percent"] <= 0.9
        assert 0.9 <= child["adamw_beta2"] <= 0.9999


def test_initial_population_seeds_sgd_and_adamw():
    population = initial_genetic_population(
        {"optimizer": "sgd", "lr": 0.05, "weight_decay": 1e-4},
        4,
        7,
    )

    assert len(population) == 4
    assert population[0]["optimizer"] == "sgd"
    assert population[1]["optimizer"] == "adamw"
    assert population[1]["lr"] == 0.001


def test_elapsed_time_ignores_later_config_rewrite(tmp_path):
    run = _run([0.0, 100.0, 200.0])
    run.path = tmp_path
    run.wall_times = np.asarray([1000.0, 1010.0, 1020.0])
    config = tmp_path / "config.yaml"
    config.write_text("iterations: 60\n")
    config.touch()

    assert [_elapsed_for_iter(run, i) for i in range(3)] == [0.0, 10.0, 20.0]


def test_milestone_elo_averages_recent_observations():
    run = _run([float(i * 10) for i in range(25)])

    assert milestone_elo(run, iteration=20, window=4) == 185.0


def test_hparam_metrics_include_iter_20_objective_not_just_final_elo():
    run = _run([float(i * 10) for i in range(25)])

    metrics = _hparam_metrics(run)

    assert metrics["hparam/objective_elo_iter_20"] == 185.0
    assert metrics["hparam/final_elo"] == 240.0


def test_project_lr_series_supports_exponential_schedule():
    run = _run([0.0] * 61)
    run.config = {
        "lr": 0.04,
        "lr_schedule": "exponential",
        "lr_exponential_final_factor": 0.25,
        "lr_exponential_start_iter": 20,
        "lr_exponential_end_iter": 60,
        "lr_warmup_target": 0,
    }
    gating = {
        "accepted": np.zeros(61),
    }

    series = _project_lr_series(run, gating)["learning_rate"]

    assert series[20] == 0.04
    assert series[40] == pytest.approx(0.02)
    assert series[60] == pytest.approx(0.01)


def test_discover_runs_filters_before_parsing(tmp_path, monkeypatch):
    game_dir = tmp_path / "uttt"
    for name in ("wanted-01", "unrelated-01"):
        run_dir = game_dir / name
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text("game: uttt\n")
    parsed = []
    monkeypatch.setattr(
        "uttt_tuning.parse_run",
        lambda path: parsed.append(path.name) or path.name,
    )

    runs = discover_runs(tmp_path, "uttt", prefix="wanted-")

    assert runs == ["wanted-01"]
    assert parsed == ["wanted-01"]


def test_tpe_params_roundtrip_optimizer_specific_lr():
    params = tpe_params_from_config({
        "optimizer": "adamw",
        "lr": 0.0015,
        "weight_decay": 0.0002,
        "adamw_beta2": 0.98,
        "train_sample_rate": 11,
        "gating_best_win_rate": 0.53,
        "playout_cap_percent": 0.7,
    })
    overrides = tpe_config_overrides(params)

    assert "adamw_lr" in params
    assert "sgd_lr" not in params
    assert overrides["optimizer"] == "adamw"
    assert overrides["lr"] == 0.0015
    assert overrides["adamw_beta2"] == 0.98
    assert overrides["gating_panel_win_rate"] == overrides["gating_best_win_rate"]


def test_seed_tpe_study_filters_historical_run_prefixes(monkeypatch):
    import optuna
    import uttt_tuning

    safe = _run([float(i * 10) for i in range(25)])
    safe.name = "lrscan-safe"
    safe.config = {"optimizer": "sgd", "lr": 0.05}
    unsafe = _run([float(i * 20) for i in range(25)])
    unsafe.name = "densenet-unsafe"
    unsafe.config = {"optimizer": "sgd", "lr": 0.05}
    monkeypatch.setattr(uttt_tuning, "discover_runs", lambda *_: [safe, unsafe])
    study = optuna.create_study(direction="maximize")

    added = seed_tpe_study(
        study,
        Path("."),
        "uttt",
        iteration=20,
        include_prefixes=("lrscan-",),
    )

    assert added == 1
    assert study.trials[0].user_attrs["source_run"] == "lrscan-safe"
    assert study.trials[0].user_attrs["objective_iteration"] == 20


def test_seed_lr_only_study_imports_only_lr_baseline(tmp_path, monkeypatch):
    import optuna
    import uttt_tuning

    run_dir = tmp_path / "nineplane-baseline"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("optimizer: sgd\nlr: 0.05\n")
    run = _run([float(i * 10) for i in range(55)])
    run.name = run_dir.name
    run.path = run_dir
    run.config = {"optimizer": "sgd", "lr": 0.05, "train_sample_rate": 10}
    monkeypatch.setattr(uttt_tuning, "parse_run", lambda *_: run)
    study = optuna.create_study(direction="maximize")

    added = seed_lr_only_study(study, run_dir, 50, (0.005, 0.05))

    assert added == 1
    assert study.trials[0].params == {"lr": 0.05}
    assert study.trials[0].value == milestone_elo(run, 50)
    assert study.trials[0].user_attrs["source_run"] == "nineplane-baseline"
    assert seed_lr_only_study(study, run_dir, 50, (0.005, 0.05)) == 0


class _FixedTrial:
    def suggest_float(self, name, low, high, log=False):
        return (low + high) / 2

    def suggest_int(self, name, low, high):
        return (low + high) // 2


def test_focused_sgd_only_overrides_lr_and_gating():
    overrides = _focused_trial_overrides("sgd", _FixedTrial())

    assert set(overrides) == {"lr", "gating_panel_win_rate", "gating_best_win_rate"}
    assert overrides["gating_panel_win_rate"] == overrides["gating_best_win_rate"]


def test_focused_adamw_only_overrides_selected_parameters():
    overrides = _focused_trial_overrides("adamw", _FixedTrial())

    assert set(overrides) == {"playout_cap_percent", "adamw_beta2", "train_sample_rate"}


def test_focused_base_params_are_optimizer_specific():
    config = {
        "lr": 0.05,
        "gating_best_win_rate": 0.52,
        "playout_cap_percent": 0.75,
        "adamw_beta2": 0.999,
        "train_sample_rate": 10,
    }

    assert _focused_base_params("sgd", config) == {"lr": 0.05, "gating_threshold": 0.52}
    assert _focused_base_params("adamw", config) == {
        "playout_cap_percent": 0.75,
        "adamw_beta2": 0.999,
        "train_sample_rate": 10,
    }


def test_focused_startup_trials_are_optimizer_specific():
    args = argparse.Namespace(sgd_startup_trials=4, adamw_startup_trials=8)

    assert _focused_startup_trials("sgd", args) == 4
    assert _focused_startup_trials("adamw", args) == 8
