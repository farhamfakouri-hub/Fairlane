from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ree.config import (
    Config,
    apply_overrides,
    clock_to_minutes,
    config_hash,
    dump_config,
    load_config,
    make_rng,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_code_defaults() -> None:
    assert load_config(ROOT / "config" / "default.yaml") == Config()


def test_dump_and_load_roundtrip(tmp_path: Path) -> None:
    config = Config().with_overrides({"simulator.n_weeks": 5, "engine.speed.method": "v2"})
    target = tmp_path / "config.yaml"
    dump_config(config, target)
    assert load_config(target) == config


def test_overrides_reach_nested_fields_and_reject_unknown_keys() -> None:
    config = apply_overrides(Config(), {"simulator.population.n_cashiers": 7})
    assert config.simulator.population.n_cashiers == 7
    with pytest.raises(KeyError):
        apply_overrides(Config(), {"simulator.population.nonsense": 1})
    with pytest.raises(KeyError):
        apply_overrides(Config(), {"nowhere.n_weeks": 1})


def test_start_date_must_be_monday() -> None:
    with pytest.raises(ValidationError):
        Config().with_overrides({"simulator.start_date": "2025-11-25"})


def test_probability_vectors_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        Config().with_overrides({"simulator.timing.alpha_split": (0.5, 0.5, 0.5)})
    with pytest.raises(ValidationError):
        Config().with_overrides({"simulator.baskets.mixture_zone_a": (0.9, 0.9, 0.9)})


def test_pause_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        Config().with_overrides({"engine.pause.t_idle_seconds": 90.0})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"unexpected": 1})


def test_clock_parsing() -> None:
    assert clock_to_minutes("08:30") == 510
    assert clock_to_minutes("20:00") == 1200
    with pytest.raises(ValidationError):
        Config().with_overrides(
            {"store.rush_windows": [{"days": ["mon"], "start": "25:00", "end": "26:00"}]}
        )


def test_rng_is_reproducible_and_streams_are_independent() -> None:
    first = make_rng(5, "workers").random(4)
    again = make_rng(5, "workers").random(4)
    other_stream = make_rng(5, "baskets").random(4)
    other_seed = make_rng(6, "workers").random(4)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other_stream)
    assert not np.array_equal(first, other_seed)


def test_config_hash_changes_with_any_parameter() -> None:
    base = Config()
    assert config_hash(base) == config_hash(Config())
    changed = base.with_overrides({"engine.speed.shrinkage_k": 151.0})
    assert config_hash(changed) != config_hash(base)
    assert config_hash(base.engine) != config_hash(changed.engine)
