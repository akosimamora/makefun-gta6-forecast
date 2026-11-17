"""gta6-twitch-ccu-forecast-01: a launch-day Twitch viewership forecaster.

Model, in one paragraph: work in log space and split the day into a peak
and a shape. Every past big launch in the case's comparables gives one
release-day curve; from those curves we learn (a) the day's peak, as the
hour-1 print times a peak-to-hour-1 ratio, and (b) the shape, the curve's
hourly values divided by its peak. Both are estimated by a Gaussian kernel
over log hour-1 viewers, so launches of a similar size at midnight weigh
most, and both are medians, so one freak launch cannot drag them. The
shape is strikingly stable once divided by the peak (its interquartile
range is only ~0.1-0.3 log units in the evening hours), while the peak
itself is the uncertain part, so the spread of each hour's quantiles is
the peak uncertainty combined with that hour's shape spread. When the
case does not look like a clean big launch (a tiny hour-1 print, or a
pre-release peak far above hour 1, as with late unlocks and beta days),
the peak-and-shape split is unreliable, so the program falls back to a
plain kernel regression of each hour's log viewers on log hour-1 viewers;
and with no usable comparables at all it falls back to the persistence
baseline. Console targets get a class-aware variant instead: only
console-only and console-rolling-midnight launches enter the peak and
shape pools, because PC-simultaneous launches peak at hour 1 while
console launches peak in the evening, and mixing the two blurs both.
The console peak ratio uses a narrower kernel, since the ratio falls as
hour-1 grows, and is clamped to [1, 6]. Nothing is hardcoded about any
one game: every number the forecast uses is estimated from the
comparables in the case file itself, and the target's class is read from
its own notes (platform clause after "released ... on", unlock wording).

Runtime: `python3 -P -s /bot/main.py`, Python 3.13 standard library only,
no network, no files outside the case given. Deterministic: fixed
constants, sorted file order, no randomness, no clock.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so an import
# of a sibling module would find it (this entry ships a single file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
# Kernel bandwidth on log hour-1 viewers: launches within about a factor
# of two of the case's hour-1 print carry most of the weight.
BANDWIDTH = 0.7
# Log-space uncertainty of the day's peak around its kernel estimate.
# Swept 0.35-0.39 on leave-one-out over the 37 comparables: 0.37 is the
# shallow optimum (ALL 0.41227, CLEAN 0.24498, both a touch better than
# 0.35) and lifts the 5-95% coverage from 0.859 toward the nominal 0.90
# (0.867 overall, 0.909 on clean launches); wider than that starts to
# cost on the biggest launches.
SIGMA_PEAK = 0.37
# Smallest log-space spread any hour may carry, and the floor the shape
# spread of an hour may take.
MIN_SIGMA = 0.18
MIN_SHAPE_SIGMA = 0.08
# A launch counts as clean when its hour-1 print is at least this many
# viewers and its pre-release peak is at most this many times hour 1.
# Clean launches are the ones whose midnight print means something:
# late unlocks and beta/early-access days fail one or both.
CLEAN_MIN_V1 = 10000
CLEAN_MAX_PRE_RATIO = 3.0
# A comparable joins the pool only when its release day is this complete.
MIN_DAY_HOURS = 12
# Console targets (the practice cases and the GTA VI final) get a
# class-aware peak-shape: only console-only and console-rolling launches
# in the pool, a narrower ratio kernel, and a narrower peak spread.
# Leave-one-out over the 37 comparables: console 0.280 -> 0.199,
# clean-console 0.171 -> 0.141, ALL 0.412 -> 0.395, CLEAN 0.245 -> 0.237.
CLASS_RATIO_BW = 0.5
CLASS_SIGMA_PEAK = 0.20
# The peak ratio is clamped so an out-of-range hour-1 (a launch bigger
# than every comparable) cannot give a peak below hour 1 or an absurd
# multiple of it.
RATIO_MIN = 1.0
RATIO_MAX = 6.0


def parse_ts(text):
    """An ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round does.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def weighted_median(pairs):
    """The weighted median of (weight, value) pairs."""
    ordered = sorted(pairs, key=lambda pair: pair[1])
    total = sum(w for w, _ in ordered)
    acc = 0.0
    for w, x in ordered:
        acc += w
        if acc >= 0.5 * total:
            return x
    return ordered[-1][1]


def weighted_quantile(pairs, tau):
    """The weighted tau-quantile of (weight, value) pairs."""
    ordered = sorted(pairs, key=lambda pair: pair[1])
    total = sum(w for w, _ in ordered)
    acc = 0.0
    for w, x in ordered:
        acc += w
        if acc >= tau * total:
            return x
    return ordered[-1][1]


def release_day(comp):
    """A comparable's release-day curve: {hour 1..24: viewers}."""
    day = parse_ts(comp["day_start"])
    out = {}
    for point in comp.get("hourly", []):
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        delta = (parse_ts(point["from"]) - day).total_seconds() / 3600
        if delta == int(delta) and 0 <= delta < 24:
            out[int(delta) + 1] = viewers
    return out


def classify_launch(platforms, unlock):
    """A comparable's launch class from its 'platforms' and 'unlock'
    text: 'console' (no PC), 'mix_roll' (PC present, consoles unlock
    rolling-midnight-local), 'pc_sim' (PC present, one simultaneous
    global unlock), 'mix_other' (PC present, pattern not recognised)."""
    plat = (platforms or "").lower()
    unl = (unlock or "").lower()
    has_pc = "pc" in plat
    rolling = ("rolling" in unl) or ("midnight local" in unl)
    simult = ("simultaneous" in unl) or ("single global" in unl)
    if not has_pc:
        return "console"
    if rolling and not simult:
        return "mix_roll"
    if simult and not rolling:
        return "pc_sim"
    return "mix_other"


def classify_notes(notes):
    """The target's launch class from its case notes (the same wording
    the real case files carry). The platform clause is the text after
    'released <date> on ' up to the first '(' or ';' -- the parenthetical
    may mention a later PC port, which must not count. The unlock is
    'rolling'/'midnight local' versus 'single global'/'simultaneous'
    anywhere in the notes."""
    text = notes or ""
    low = text.lower()
    plat = ""
    idx = low.find("released ")
    if idx >= 0:
        rest = text[idx + len("released "):]
        on = rest.lower().find(" on ")
        if on >= 0:
            clause = rest[on + 4:]
            cut = len(clause)
            for sep in ("(", ";"):
                j = clause.find(sep)
                if 0 <= j < cut:
                    cut = j
            plat = clause[:cut]
    has_pc = "pc" in plat.lower()
    rolling = ("rolling" in low) or ("midnight local" in low)
    simult = ("simultaneous" in low) or ("single global" in low)
    if not has_pc:
        return "console"
    if rolling and not simult:
        return "mix_roll"
    if simult and not rolling:
        return "pc_sim"
    return "mix_other"


def pre_peak(comp_hourly, day, days=7):
    """The highest hourly viewers in the `days` before the release day."""
    best = 0
    for point in comp_hourly:
        at = parse_ts(point["from"])
        viewers = point.get("viewers")
        if day - timedelta(days=days) <= at < day and viewers and viewers > best:
            best = viewers
    return best if best > 0 else None


def build_pool(case):
    """The comparables as modelling rows, leaving the case's own game out.

    A row holds the launch's hour-1 print, its peak over the release day,
    its release-day curve, its pre-release peak, and whether it is clean.
    A comparable with no release day (the case game's own pre-release
    history, which shares its name) never enters the pool.
    """
    own = (case.get("game") or {}).get("name", "")
    pool = []
    for comp in case.get("comparables") or []:
        name = comp.get("name", "")
        if own and (own in name or name in own):
            continue
        day_curve = release_day(comp)
        if 1 not in day_curve or day_curve[1] <= 0:
            continue
        if sum(1 for h in range(2, 25) if h in day_curve) < MIN_DAY_HOURS:
            continue
        day = parse_ts(comp["day_start"])
        peak = max(day_curve.values())
        pre = pre_peak(comp.get("hourly", []), day)
        clean = day_curve[1] >= CLEAN_MIN_V1 and (pre or 0) <= CLEAN_MAX_PRE_RATIO * day_curve[1]
        pool.append({
            "name": name,
            "v1": day_curve[1],
            "curve": day_curve,
            "peak": peak,
            "pre_peak": pre,
            "clean": clean,
            "cls": classify_launch(comp.get("platforms"), comp.get("unlock")),
        })
    return pool


def case_features(case):
    """The case's own scale features: hour-1 print and pre-release peak."""
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    v1 = max(1, first["viewers"]) if first and first.get("viewers") else 1
    day = parse_ts(case["day_start"])
    hourly = case.get("history_hourly") or []
    pre = pre_peak(hourly, day)
    return v1, pre


def is_clean_target(v1, pre):
    """Whether the case looks like a clean big launch at midnight."""
    return v1 >= CLEAN_MIN_V1 and (pre or 0) <= CLEAN_MAX_PRE_RATIO * v1


def kernel_weight(log_v1, target_log_v1, bw=BANDWIDTH):
    """The Gaussian kernel weight of a launch at log hour-1 `log_v1`."""
    d = (log_v1 - target_log_v1) / bw
    return math.exp(-0.5 * d * d)


def peak_shape_console(v1, pool):
    """Class-aware peak-shape for console targets: the peak ratio and
    the shape come only from console-only and console-rolling-midnight
    launches, so the evening-peaked console curves are not blurred by
    PC-simultaneous launches whose peak sits at hour 1. The ratio uses
    the narrower CLASS_RATIO_BW kernel because the peak ratio falls as
    hour-1 grows, and is clamped to [RATIO_MIN, RATIO_MAX]."""
    log_v1 = math.log(max(1, v1))
    sub = [row for row in pool if row["cls"] in ("console", "mix_roll")]
    if len(sub) < 3:
        sub = pool
    ratios = [(kernel_weight(math.log(row["v1"]), log_v1, CLASS_RATIO_BW),
               math.log(row["peak"] / row["v1"]))
              for row in sub if row["peak"] and row["v1"] > 0]
    if not ratios:
        return None
    log_ratio = weighted_median(ratios)
    log_ratio = max(math.log(RATIO_MIN), min(math.log(RATIO_MAX), log_ratio))
    log_peak = log_v1 + log_ratio
    clean = [row for row in sub if row["clean"]] or sub
    out = {}
    for h in range(2, 25):
        shape = [(kernel_weight(math.log(row["v1"]), log_v1),
                  math.log(row["curve"][h] / row["peak"]))
                 for row in clean if h in row["curve"] and row["peak"] > 0]
        if not shape:
            continue
        centre = weighted_median(shape)
        spread = max(MIN_SHAPE_SIGMA,
                     (weighted_quantile(shape, 0.75) - weighted_quantile(shape, 0.25)) / 1.349)
        sigma = math.sqrt(CLASS_SIGMA_PEAK ** 2 + spread ** 2)
        out[h] = [math.exp(log_peak + centre + z * sigma) for z in Z]
    return out if len(out) == 23 else None


def peak_shape_forecast(v1, pool):
    """Peak-and-shape forecast: peak from a kernel median of peak ratios,
    shape from a kernel median of peak-normalised curves over clean rows."""
    log_v1 = math.log(max(1, v1))
    clean = [row for row in pool if row["clean"]] or pool
    ratios = []
    for row in pool:
        if row["peak"] and row["v1"] > 0:
            ratios.append((kernel_weight(math.log(row["v1"]), log_v1),
                           math.log(row["peak"] / row["v1"])))
    if not ratios:
        return None
    log_peak = log_v1 + weighted_median(ratios)
    out = {}
    for h in range(2, 25):
        shape = []
        for row in clean:
            if h in row["curve"] and row["peak"] > 0:
                shape.append((kernel_weight(math.log(row["v1"]), log_v1),
                              math.log(row["curve"][h] / row["peak"])))
        if not shape:
            continue
        centre = weighted_median(shape)
        spread = max(MIN_SHAPE_SIGMA,
                     (weighted_quantile(shape, 0.75) - weighted_quantile(shape, 0.25)) / 1.349)
        sigma = math.sqrt(SIGMA_PEAK ** 2 + spread ** 2)
        out[h] = [math.exp(log_peak + centre + z * sigma) for z in Z]
    return out if len(out) == 23 else None


def kernel_forecast(v1, pool):
    """Fallback: kernel regression of each hour's log viewers on log v1."""
    log_v1 = math.log(max(1, v1))
    out = {}
    for h in range(2, 25):
        points = []
        for row in pool:
            if h in row["curve"]:
                points.append((kernel_weight(math.log(row["v1"]), log_v1),
                               math.log(row["curve"][h])))
        if not points:
            continue
        total = sum(w for w, _ in points)
        mean = sum(w * x for w, x in points) / total
        var = sum(w * (x - mean) ** 2 for w, x in points) / total
        sigma = max(MIN_SIGMA, math.sqrt(var))
        out[h] = [math.exp(mean + z * sigma) for z in Z]
    return out if len(out) == 23 else None


def persistence_forecast(v1):
    """Last fallback: the house baseline's flat persistence forecast."""
    p1 = max(1, v1)
    return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
            for h in range(2, 25)}


def forecast_case(case):
    """One case's forecast: {"hours": [{"hour": h, "q": [...]} * 23]}."""
    v1, pre = case_features(case)
    pool = build_pool(case)
    target_class = classify_notes(case.get("notes") or "")
    quantiles = None
    if pool and target_class == "console":
        quantiles = peak_shape_console(v1, pool)
    if quantiles is None and pool and is_clean_target(v1, pre):
        quantiles = peak_shape_forecast(v1, pool)
    if quantiles is None and pool:
        quantiles = kernel_forecast(v1, pool)
    if quantiles is None:
        quantiles = persistence_forecast(v1)
    hours = []
    for row in case.get("forecast") or [{"hour": h} for h in range(2, 25)]:
        h = row["hour"]
        qs = quantiles.get(h)
        if qs is None:
            qs = persistence_forecast(v1)[h]
        clean_qs = []
        last = 0.0
        for q in qs:
            value = round2(min(1_000_000_000, max(1.0, q)))
            if value < last:
                value = last
            clean_qs.append(value)
            last = value
        hours.append({"hour": h, "q": clean_qs})
    hours.sort(key=lambda row: row["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
