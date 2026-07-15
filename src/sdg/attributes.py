"""Constraint-First: sample the example RECIPE (attributes) before any teacher
call. Labels come from here (the spec), so the teacher can't mislabel them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from config import (DICT_DIR, DICT_SOURCES, RANDOM_SEED, SDG_CONTEXTS,
                    SDG_DIFFICULTY_WEIGHTS, SDG_DIRECTION_WEIGHTS,
                    SDG_HARD_NEG_FRAC, SDG_TONES)


@dataclass
class Recipe:
    direction: str
    term: str
    tone: str
    difficulty: str
    context: str
    is_hard_negative: bool


def load_term_pool() -> list[str]:
    terms: set[str] = set()
    for src in DICT_SOURCES:
        if src.get("emoji"):
            continue
        p = DICT_DIR / src["file"]
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if src["term_col"] in df.columns:
            terms |= {str(t).strip() for t in df[src["term_col"]].dropna()}
    return sorted({t for t in terms if t and 1 <= len(t) <= 40})


def _weighted(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    vals = [v for v, _ in pairs]
    weights = [w for _, w in pairs]
    return rng.choices(vals, weights=weights, k=1)[0]


def sample_recipes(n: int, seed: int = RANDOM_SEED) -> list[Recipe]:
    rng = random.Random(seed)
    pool = load_term_pool()
    if not pool:
        raise RuntimeError("Empty term pool — check DICT_SOURCES / data/dictionaries.")
    out: list[Recipe] = []
    for _ in range(n):
        out.append(Recipe(
            direction=_weighted(rng, SDG_DIRECTION_WEIGHTS),
            term=rng.choice(pool),
            tone=rng.choice(SDG_TONES),
            difficulty=_weighted(rng, SDG_DIFFICULTY_WEIGHTS),
            context=rng.choice(SDG_CONTEXTS),
            is_hard_negative=(rng.random() < SDG_HARD_NEG_FRAC),
        ))
    return out
