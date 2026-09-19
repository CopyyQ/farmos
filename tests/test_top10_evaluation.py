import json

from kaggrl.top10_evaluation import summarize_metrics


TEAMS = [f"Team {i}" for i in range(10)]


def test_summary_requires_and_reports_all_ten_teams(tmp_path):
    paths = []
    for i, team in enumerate(TEAMS):
        doc = {
            "team_name": team,
            "best_val_score": 0.1 + i / 100,
            "gate": {"pass": i % 2 == 0},
            "val": {"farmer_op_acc": 0.2 + i / 100, "hand_op_acc": 0.3, "market_op_acc": 0.5},
            "test": {"farmer_op_acc": 0.25, "hand_op_acc": 0.35, "market_op_acc": 0.55},
        }
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps(doc))
        paths.append(path)
    summary = summarize_metrics(paths)
    assert summary["team_count"] == 10
    assert set(summary["teams"]) == set(TEAMS)
    assert summary["gate_pass_count"] == 5
    assert summary["best_validation_expert"] == "Team 9"
