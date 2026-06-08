"""
Load and code the behavioural data for the precision-RSA project.

Each CSV row is one speaker-trial.  We code the free-text `expression` into a
binary vector of which set-A / set-B feature dimensions were mentioned
(number, size, colour, shape), attach the trial's condition (occlusion x
distinguishing-feature) and the distinguishing dimension, and tag each row
with group (ASD/NT), experiment (minimal/accuracy), and SRS.

The coded feature-mention vector is the model's dependent variable.
"""

import csv
import re
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FILES = {
    ("accuracy", "ASD"): "accuracy_ASD.csv",
    ("accuracy", "NT"):  "accuracy_NT.csv",
    ("minimal",  "ASD"): "minimal_ASD.csv",
    ("minimal",  "NT"):  "minimal_NT.csv",
}

DIMS = ["number", "size", "colour", "shape"]

# vocab / synonyms for coding free text -> feature values
NUM   = {"one": "one", "two": "two", "1": "one", "2": "two", "single": "one"}
SIZE  = {"big": "big", "large": "big", "larger": "big", "biggest": "big",
         "small": "small", "smaller": "small", "little": "small", "tiny": "small"}
COLOR = {"green", "blue", "black", "pink", "red", "yellow", "purple",
         "orange", "white", "grey", "gray", "brown"}
SHAPE = {"square": "square", "rectangle": "square",
         "circle": "circle", "round": "circle",
         "triangle": "triangle", "star": "star", "pentagon": "pentagon"}


def _toks(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _parse_set(s):
    """'two, big, blue, square' -> dict(number,size,colour,shape)."""
    p = [x.strip().lower() for x in s.split(",")]
    return dict(number=p[0], size=p[1], colour=p[2], shape=p[3])


def _mentions(value, dim, toks):
    """Does the token set mention `value` on dimension `dim` (with synonyms)?"""
    if dim == "number":
        cands = {value} | {k for k, v in NUM.items() if v == value}
    elif dim == "size":
        cands = {value} | {k for k, v in SIZE.items() if v == value}
    elif dim == "colour":
        cands = {value}                       # colours have no common synonyms here
    else:  # shape
        cands = {value, value + "s"} | {k for k, v in SHAPE.items() if v == value}
    cands |= {c + "s" for c in list(cands)}   # allow plurals throughout
    return int(bool(cands & toks))


def code_expression(expr, set_a, set_b):
    """Return (setA_mentions, setB_mentions) dicts of 0/1 per dimension."""
    toks = _toks(expr)
    fa, fb = _parse_set(set_a), _parse_set(set_b)
    setA = {d: _mentions(fa[d], d, toks) for d in DIMS}
    setB = {d: _mentions(fb[d], d, toks) for d in DIMS}
    return setA, setB


def _dist_dims(unique_property, dist_feature):
    """Which dimension(s) distinguish the target from the competitor."""
    if dist_feature == "type":
        return ["shape"]
    dims = []
    for tok in _toks(unique_property):
        if tok in SIZE:
            dims.append("size")
        elif tok in COLOR:
            dims.append("colour")
        elif tok in NUM:
            dims.append("number")
        elif tok in SHAPE:
            dims.append("shape")
    return sorted(set(dims)) or ["colour"]


# SRS exclusion cutoff agreed with collaborator (NT <= 67, ASD >= 68)
def _included(group, srs):
    try:
        s = float(srs)
    except (TypeError, ValueError):
        return True
    return s <= 67 if group == "NT" else s >= 68


def load_all():
    """Return a flat list of coded trial dicts across all four datasets."""
    trials = []
    for (experiment, group), fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        for r in csv.DictReader(open(path)):
            setA, setB = code_expression(r["expression"], r["set_a"], r["set_b"])
            trials.append(dict(
                experiment=experiment, group=group,
                pid=r["prolific_pid"], srs=r.get("srs_total", ""),
                included=_included(group, r.get("srs_total", "")),
                occlusion=r["occlusion"],                 # 'yes' / 'no'
                dist_feature=r["set_a_dist_feature"],     # 'type' / 'non_type'
                dist_dims=_dist_dims(r["unique_property"], r["set_a_dist_feature"]),
                set_a=_parse_set(r["set_a"]), set_b=_parse_set(r["set_b"]),
                setA=setA, setB=setB,
                expression=r["expression"].strip(),
            ))
    return trials


def empirical_rates(trials, experiment, included_only=True):
    """Mean set-A mention rate per (occlusion, dist_feature, group, dimension)."""
    acc = defaultdict(lambda: [0, 0])   # key -> [sum, n]
    for t in trials:
        if t["experiment"] != experiment:
            continue
        if included_only and not t["included"]:
            continue
        for d in DIMS:
            key = (t["occlusion"], t["dist_feature"], t["group"], d)
            acc[key][0] += t["setA"][d]
            acc[key][1] += 1
    return {k: (s / n if n else float("nan")) for k, (s, n) in acc.items()}


if __name__ == "__main__":
    trials = load_all()
    print(f"loaded {len(trials)} trials; "
          f"{sum(t['included'] for t in trials)} after SRS exclusion")

    for experiment in ("accuracy", "minimal"):
        rates = empirical_rates(trials, experiment)
        print(f"\n[{experiment}, SRS-included]")
        print("set-A mention rate, ASD vs NT, by condition")
        print(f"{'occ':>4} {'dist':>9} | " +
              " | ".join(f"{d:>14}" for d in DIMS))
        for occ in ("no", "yes"):
            for dist in ("type", "non_type"):
                cells = []
                for d in DIMS:
                    a = rates.get((occ, dist, "ASD", d), float("nan"))
                    n = rates.get((occ, dist, "NT", d), float("nan"))
                    cells.append(f"{a:.2f}/{n:.2f} ({a-n:+.2f})")
                print(f"{occ:>4} {dist:>9} | " + " | ".join(f"{c:>14}" for c in cells))
        print("  (cells: ASD/NT (ASD−NT);  shape is the type-distinguishing feature)")
