from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DEFAULT_TURNS_PER_DAY = 24
DEFAULT_EPISODE_STEPS = 720
PHASE_NAMES = ("early", "growth", "mid", "harvest", "liquidation")
LEGACY_CLOCK_FEATURES = (
    "step_norm", "day_norm", "hour_norm", "hour_sin", "hour_cos",
)
V4_CLOCK_EXTRA_FEATURES = (
    "remaining_steps_norm", "turns_to_day_end_norm",
    "season_sin", "season_cos",
    *(f"phase:{name}" for name in PHASE_NAMES),
)
CLOCK_FEATURES = LEGACY_CLOCK_FEATURES + V4_CLOCK_EXTRA_FEATURES


def _get(value: Any, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _config_int(configuration: Any, key: str, default: int) -> int:
    raw = _get(configuration, key, None) if configuration is not None else None
    if raw is None:
        return int(default)
    return max(1, int(raw))


@dataclass(frozen=True)
class GameClock:
    step: int
    day: int
    hour: int
    turns_per_day: int
    episode_steps: int

    @property
    def remaining_steps(self) -> int:
        return max(0, self.episode_steps - 1 - self.step)

    @property
    def phase_index(self) -> int:
        ratio = max(0.0, min(1.0, self.step / max(1, self.episode_steps)))
        for index, boundary in enumerate((0.20, 0.50, 0.80, 0.90)):
            if ratio < boundary:
                return index
        return 4

    def features(self) -> tuple[float, ...]:
        last_step = max(1, self.episode_steps - 1)
        day_count = max(1, math.ceil(self.episode_steps / self.turns_per_day))
        last_day = max(1, day_count - 1)
        last_hour = max(1, self.turns_per_day - 1)
        step = max(0, min(self.step, last_step))
        hour_angle = 2.0 * math.pi * (self.hour % self.turns_per_day) / self.turns_per_day
        season_angle = 2.0 * math.pi * step / max(1, self.episode_steps)
        phase = [0.0] * len(PHASE_NAMES)
        phase[self.phase_index] = 1.0
        return (
            step / last_step,
            max(0, min(self.day, last_day)) / last_day,
            max(0, min(self.hour, last_hour)) / last_hour,
            math.sin(hour_angle), math.cos(hour_angle),
            max(0, last_step - step) / last_step,
            max(0, last_hour - self.hour) / last_hour,
            math.sin(season_angle), math.cos(season_angle),
            *phase,
        )


def resolve_clock(observation: Any, configuration: Any = None) -> GameClock:
    turns_per_day = _config_int(
        configuration, "turnsPerDay", DEFAULT_TURNS_PER_DAY
    )
    episode_steps = _config_int(
        configuration, "episodeSteps", DEFAULT_EPISODE_STEPS
    )
    raw_step = _get(observation, "step", None)
    if raw_step is None:
        raw_day = _get(observation, "day", 0)
        raw_hour = _get(observation, "hour", 0)
        step = int(raw_day or 0) * turns_per_day + int(raw_hour or 0)
    else:
        step = int(raw_step)
    step = max(0, step)
    return GameClock(
        step=step,
        day=step // turns_per_day,
        hour=step % turns_per_day,
        turns_per_day=turns_per_day,
        episode_steps=episode_steps,
    )


def resolve_step(observation: Any, configuration: Any = None) -> int:
    return resolve_clock(observation, configuration).step
