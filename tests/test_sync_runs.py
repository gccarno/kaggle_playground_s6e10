"""runs.csv merge: append missing run_ids, fill blank score columns, never overwrite."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from sync_runs import plan_merge  # noqa: E402


def test_appends_missing_rows_only():
    local = [{"run_id": "a", "final_oof_auc": "0.9"}]
    other = [{"run_id": "a", "final_oof_auc": "0.1"}, {"run_id": "b"}]
    append, fills = plan_merge(local, other)
    assert [r["run_id"] for r in append] == ["b"]
    assert fills == {}            # a set-once column is never overwritten


def test_fills_blank_scores_but_keeps_existing():
    local = [{"run_id": "a", "public_lb_score": "", "submission_msg": "[a] mine"}]
    other = [{"run_id": "a", "public_lb_score": "0.96", "submission_msg": "[a] theirs"}]
    assert plan_merge(local, other)[1] == {"a": {"public_lb_score": "0.96"}}


def test_only_restricts():
    append, _ = plan_merge([], [{"run_id": "a"}, {"run_id": "b"}], only={"b"})
    assert [r["run_id"] for r in append] == ["b"]
