"""Pick logic of the slot scheduler. Pure functions only, so no Kaggle traffic."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from submit_run import eligible  # noqa: E402
from submit_scheduler import pick  # noqa: E402


def make_row(root, run_id, ts, frac="1.0", notes="", with_sub=True):
    d = root / "experiments" / "preds" / run_id
    d.mkdir(parents=True)
    if with_sub:
        (d / "submission.csv").write_text("id,satisfaction\n1,0.5\n")
    return {"run_id": run_id, "run_tag": f"T_{run_id}", "timestamp_utc": ts,
            "sample_frac": frac, "notes": notes,
            "preds_dir": f"experiments/preds/{run_id}"}


def test_eligible_reasons(tmp_path):
    ok = make_row(tmp_path, "a", "2026-10-03T01:00:00+00:00")
    assert eligible(ok, set(), tmp_path) == (True, "")
    assert not eligible(ok, {"a"}, tmp_path)[0]
    assert not eligible(make_row(tmp_path, "b", "t", frac="0.05"), set(), tmp_path)[0]
    assert not eligible(make_row(tmp_path, "c", "t", with_sub=False), set(), tmp_path)[0]
    assert not eligible(make_row(tmp_path, "d", "t", notes="x NOSUBMIT y"), set(), tmp_path)[0]


def test_queue_first_then_newest_autofill(tmp_path):
    rows = [make_row(tmp_path, "old", "2026-10-01T00:00:00+00:00"),
            make_row(tmp_path, "new", "2026-10-03T00:00:00+00:00"),
            make_row(tmp_path, "q", "2026-09-30T00:00:00+00:00")]
    assert pick(["q"], rows, set(), 10, tmp_path) == ["q", "new", "old"]


def test_skips_submitted_and_caps_at_free(tmp_path):
    rows = [make_row(tmp_path, r, f"2026-10-0{i}T00:00:00+00:00")
            for i, r in enumerate(["a", "b", "c", "d"], 1)]
    assert pick([], rows, {"d"}, 2, tmp_path) == ["c", "b"]
    assert pick(["a"], rows, set(), 0, tmp_path) == []


def test_queue_dedup_and_ineligible_queue_entry_dropped(tmp_path):
    rows = [make_row(tmp_path, "a", "2026-10-01T00:00:00+00:00"),
            make_row(tmp_path, "smoke", "2026-10-02T00:00:00+00:00", frac="0.05")]
    assert pick(["a", "a", "smoke", "missing"], rows, set(), 10, tmp_path) == ["a"]


def test_nosubmit_blocks_autofill_but_not_explicit_queue(tmp_path):
    rows = [make_row(tmp_path, "a", "2026-10-01T00:00:00+00:00", notes="NOSUBMIT"),
            make_row(tmp_path, "b", "2026-10-02T00:00:00+00:00", notes="NOSUBMIT")]
    assert pick(["a"], rows, set(), 10, tmp_path) == ["a"]


def test_autofill_off(tmp_path):
    rows = [make_row(tmp_path, "a", "2026-10-01T00:00:00+00:00"),
            make_row(tmp_path, "b", "2026-10-02T00:00:00+00:00")]
    assert pick(["a"], rows, set(), 10, tmp_path, autofill=False) == ["a"]
