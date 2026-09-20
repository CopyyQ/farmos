import json

from scripts.promote_v4_margin import compare_reports


def _write(path, margins):
    games = []
    for index, margin in enumerate(margins):
        games.append({
            "seed": 100 + index // 2,
            "seat": index % 2,
            "margin": margin,
            "statuses": ["DONE", "DONE"],
        })
    payload = {
        "summary": {
            "runtime_ok": True,
            "clock_ok": True,
            "strategy_steps_ok": True,
        },
        "games": games,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_margin_promotion_requires_closed_loop_improvement(tmp_path):
    baseline = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _write(baseline, [-5000, -2000, 1000, 3000])
    _write(candidate, [-4000, -1000, 2000, 3500])
    result = compare_reports(baseline, candidate)
    assert result["passed"] is True
    assert result["mean_margin_improvement"] > 0


def test_margin_promotion_rejects_single_game_collapse(tmp_path):
    baseline = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    _write(baseline, [-1000, -1000, -1000, -1000])
    _write(candidate, [10000, 10000, 10000, -10000])
    result = compare_reports(
        baseline, candidate, max_single_game_regression=5000
    )
    assert result["passed"] is False
    assert result["checks"]["no_single_game_collapse"] is False
