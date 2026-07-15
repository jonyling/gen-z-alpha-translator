import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sdg.attributes import Recipe, sample_recipes


def test_sample_is_deterministic():
    a = sample_recipes(50, seed=42)
    b = sample_recipes(50, seed=42)
    assert [r.__dict__ for r in a] == [r.__dict__ for r in b]


def test_sample_respects_count_and_fields():
    recs = sample_recipes(30, seed=1)
    assert len(recs) == 30
    assert all(isinstance(r, Recipe) for r in recs)
    assert all(r.direction in ("to_english", "to_slang") for r in recs)
    assert all(r.term for r in recs)


def test_direction_weighting_favors_to_english():
    recs = sample_recipes(400, seed=7)
    to_eng = sum(1 for r in recs if r.direction == "to_english")
    assert to_eng > len(recs) * 0.5  # weighted 0.65
