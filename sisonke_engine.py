"""
sisonke_engine.py
==================

All the math for the Sisonke Football Predictive Terminal, kept
completely separate from the Streamlit UI so it can be tested directly
with plain Python (no Streamlit needed to verify the numbers are right).

DESIGN DECISIONS WORTH KNOWING ABOUT (documented up front because this
is a real-money-adjacent tool and every assumption should be visible,
not buried):

1. HOME ADVANTAGE IS NOT DOUBLE-COUNTED. Host stats are computed ONLY
   from a team's home matches, and visitor stats ONLY from their away
   matches - so whatever real home boost a team gets is already baked
   into their host numbers. No separate home-advantage multiplier is
   applied anywhere in this file.

2. TEAM STRENGTH NEVER COMES FROM A TEAM'S OWN RAW GOALS. The
   attack/defense STRENGTH RATIOS that differentiate one team from
   another are built entirely from territory metrics (big chances,
   shots on target, box touches) - never from a team's own win/loss
   record or their own average goals, which is exactly the "luck"
   signal the spec says to avoid. The one place raw goals appear at all
   is as a single LEAGUE-WIDE average goals figure used as a shared
   baseline unit (the same number for every team in that league) to
   convert relative territory strength into an actual expected-goals
   scale for the Poisson math - since "big chances" has no absolute
   goals unit on its own. That's a scale anchor, not a team signal.

3. PROBABILITIES ARE ALWAYS COMPUTED, NEVER FIXED. Every tactical
   multiplier in Section 6 only ever adjusts the INPUT attack/defense
   rates that feed the Dixon-Coles and Monte Carlo engines. The actual
   market probabilities always come out of real Poisson math or a real
   10,000-run simulation - a slider never directly sets a probability
   or an EV number by formula shortcut.

4. RHO (the Dixon-Coles low-score correlation parameter) is fitted from
   the league's own historical low-score frequencies, not a fixed
   constant - see fit_rho().

5. The half-life for time-decay weighting is chosen by actually
   backtesting candidate half-lives against real past results with a
   Brier score (see optimize_half_life()) - not asserted as a fixed
   number, unless you tick "Freeze Decay" for a fixed 45-day window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import poisson as scipy_poisson
from scipy.optimize import minimize as _minimize

try:
    import requests as _requests
except ImportError:  # Telegram sending degrades gracefully if requests isn't installed
    _requests = None

MIN_SAMPLE_ROWS = 5          # the "5-match sample safety rail"
FROZEN_HALF_LIFE_DAYS = 45   # fallback when "Freeze Decay" is ticked
HALF_LIFE_CANDIDATES = list(range(15, 181, 15))  # 15, 30, ..., 180
GOAL_CAP = 10                # max goals per side considered in the Poisson grid
MC_ITERATIONS = 10_000
DISPERSION_ADJUST_TRIGGER = 0.85
DISPERSION_ADJUST_FACTOR = 1.15

TERRITORY_STATS = ["big_chances", "shots_on_target", "box_touches"]

# ---------------------------------------------------------------------------
# Section 3: Column standardisation, division/fixture parsing
# ---------------------------------------------------------------------------

REQUIRED_BASE_COLUMNS = [
    "date", "home_team", "away_team", "home_goals", "away_goals",
    "home_shots_on_target", "away_shots_on_target",
    "home_big_chances", "away_big_chances",
    "home_box_touches", "away_box_touches",
]
DIVISION_COLUMN_CANDIDATES = ["league_country", "league", "competition"]


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strips whitespace, lowercases, and replaces spaces with
    underscores in every column name (e.g. 'Home Box Touches' becomes
    'home_box_touches')."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def find_division_column(df: pd.DataFrame) -> str | None:
    for candidate in DIVISION_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


# Same flexible-detection pattern as the division column, for the same
# reason: different CSV exports call this column different things, and
# blindly assuming "date" exists crashes anything downstream that reads
# it (half-life decay, backtesting, the fixture picker) with a cryptic
# AttributeError rather than a clear, actionable message.
DATE_COLUMN_CANDIDATES = [
    "date", "match_date", "fixture_date", "game_date", "kickoff",
    "kickoff_date", "kickoff_time", "date_time", "match_datetime",
    "match_time", "played_on", "utc_date",
]


def find_date_column(df: pd.DataFrame) -> str | None:
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


# Keyword fragments that flag a competition as a CUP/TOURNAMENT rather
# than a standard home-and-away league - this model's whole points/table
# framework (xPts, season simulation, "played N times") assumes a
# standard league fixture list, which doesn't hold for single/double-leg
# knockout cup football (extra time, penalties, one-off ties, no table).
CUP_TOURNAMENT_KEYWORDS = [
    "cup", "trophy", "shield", "playoff", "play-off", "knockout",
    "copa", "coupe", "pokal", "taça", "taca", "supercup", "super cup",
    "champions league", "europa league", "conference league", "libertadores",
    "sudamericana", "afcon", "world cup", "euros", "european championship",
    "nations league", "friendly", "friendlies", "qualifier", "qualifying",
]


def is_cup_or_tournament(division_text: str) -> bool:
    """Keyword-based flag, not a guarantee - a league that happens to
    have 'Cup' in its sponsor name would still need a manual override,
    but this catches the overwhelming majority of real knockout
    competitions without needing per-competition metadata the CSV
    doesn't provide."""
    if not division_text:
        return False
    text_lower = str(division_text).lower()
    return any(kw in text_lower for kw in CUP_TOURNAMENT_KEYWORDS)


def filter_to_standard_leagues(divisions: list[str]) -> tuple[list[str], list[str]]:
    """Splits a list of division names into (standard_leagues, excluded).
    Use this to keep cup/tournament competitions out of the workspace
    dropdown entirely, per the model being strictly for standard league
    play."""
    standard = [d for d in divisions if not is_cup_or_tournament(d)]
    excluded = [d for d in divisions if is_cup_or_tournament(d)]
    return standard, excluded


def normalize_name_casing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Makes 'Chelsea' and 'chelsea' resolve to ONE team instead of two
    different ones. Builds a SINGLE canonical-casing map from the
    combined values across ALL the given columns together (not one map
    per column) - otherwise 'Chelsea' could end up resolving to a
    different casing in home_team than in away_team, which would still
    silently split the same real team into two. Each variant is mapped to
    whichever ORIGINAL casing appears most often across all those columns
    combined (ties broken by first-seen), which preserves correct
    capitalization for names like 'PSV' or 'AS Roma' that a blind
    .title() would mangle into 'Psv' or 'As Roma'. Columns that don't
    exist are skipped."""
    df = df.copy()
    existing_columns = [c for c in columns if c in df.columns]
    if not existing_columns:
        return df

    combined = pd.concat([df[c].dropna().astype(str) for c in existing_columns], ignore_index=True)
    if combined.empty:
        return df

    canonical_map = combined.groupby(combined.str.lower()).agg(
        lambda s: s.value_counts().idxmax()
    ).to_dict()

    for col in existing_columns:
        df[col] = df[col].apply(
            lambda v: canonical_map.get(str(v).lower(), v) if pd.notna(v) else v
        )
    return df


def is_unplayed(home_goals_val, away_goals_val) -> bool:
    """A match counts as unplayed if either goals cell is blank/NaN, or
    contains a comma (covers both conventions described in the spec -
    a genuinely empty cell, or a combined 'x,y' placeholder string some
    spreadsheets use for an unplayed fixture)."""
    for val in (home_goals_val, away_goals_val):
        if val is None:
            return True
        if isinstance(val, float) and math.isnan(val):
            return True
        text = str(val).strip()
        if text == "" or text.lower() in {"nan", "none"}:
            return True
        if "," in text:
            return True
        try:
            float(text)
        except (TypeError, ValueError):
            return True
    return False


def split_played_unplayed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (settled_df, upcoming_df) for a division's fixtures."""
    unplayed_mask = df.apply(
        lambda r: is_unplayed(r.get("home_goals"), r.get("away_goals")), axis=1
    )
    return df[~unplayed_mask].copy(), df[unplayed_mask].copy()


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Parses the date column - and, critically, GUARANTEES the returned
    dataframe always has a `col` column afterward, even if the source CSV
    used a different name (or no date column at all). Everything
    downstream (decay weighting, backtesting, the fixture picker) reads
    this column directly off dataframe rows, so silently having no such
    column crashes deep in an unrelated tab with a cryptic
    'Pandas object has no attribute date' AttributeError instead of a
    clear message - this is what that bug looked like in practice."""
    df = df.copy()
    if col not in df.columns:
        detected = find_date_column(df)
        if detected is not None:
            df = df.rename(columns={detected: col})
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        # No date-like column found anywhere - fill with NaT rather than
        # leaving the column missing, so `row.date` always resolves to
        # something (safely treated as "unknown date") instead of raising.
        df[col] = pd.NaT
    return df


# ---------------------------------------------------------------------------
# Section 4 & Core Parameter A: time-decay weighted territory vectors
# ---------------------------------------------------------------------------

def decay_weights(dates: pd.Series, reference_date: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """Weight = exp(-ln(2) * days_elapsed / half_life). More recent
    matches (smaller days_elapsed) get a weight closer to 1.0."""
    days_elapsed = (reference_date - dates).dt.days.clip(lower=0).to_numpy(dtype=float)
    return np.exp(-math.log(2) * days_elapsed / half_life_days)


# Used only when "Freeze Decay" is on AND index-based weighting is
# requested - see decay_weights_by_index below for why this exists.
FROZEN_HALF_LIFE_MATCHES = 8.0  # a reasonable, editable index-based analog
                                 # to the 45 calendar-day freeze setting -
                                 # NOT a precise conversion (that depends on
                                 # fixture density, which varies by league)


def decay_weights_by_index(n_matches: int, half_life_matches: float) -> np.ndarray:
    """Weight = exp(-ln(2) * matches_ago / half_life_matches), where
    matches_ago counts backward from the most recent match (0) regardless
    of the ACTUAL CALENDAR GAP between matches.

    Why this exists: calendar-day decay has a blind spot. If a team's
    last domestic match was right before a long international break or
    the summer off-season, EVERY one of their matches - including the
    handful right before the break, which are still the most relevant
    form reference available - ends up heavily time-decayed just because
    a lot of calendar days happened to pass, not because the team's form
    is actually stale. Counting by match INDEX instead of days sidesteps
    that: the team's most recent match is always weight 1.0, their
    second-most-recent is next, and so on, regardless of how many
    calendar days sit between them and the upcoming fixture."""
    if n_matches <= 0:
        return np.array([])
    matches_ago = np.arange(n_matches, dtype=float)  # 0 = most recent (rows must be sorted newest-first)
    return np.exp(-math.log(2) * matches_ago / max(half_life_matches, 1e-6))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total_weight = weights.sum()
    if total_weight <= 0:
        return float(np.mean(values)) if len(values) else 0.0
    return float(np.sum(values * weights) / total_weight)


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = (sum(w))^2 / sum(w^2) - how many EQUALLY-weighted matches
    this decay-weighted sample is actually worth, not just how many
    matches exist. A team with 30 matches on file but a short half-life
    might have an ESS of only ~8 if the weighting is concentrated on
    the most recent few - ESS is the number that should drive "is this
    enough data" checks, not raw match count. This is the same
    "effective sample size, not total match count" caution that applies
    when a half-life search hits the edge of its candidate range (see
    half_life_at_search_boundary below) - a real fix needs to check ESS
    per team per venue, since that's the sample the decay curve is
    actually working with.

    With uniform weights of 1.0, ESS reduces to len(weights) exactly -
    e.g. a rolling-window profile where every in-window match counts
    equally has ESS == the number of matches in the window."""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    sq_total = np.sum(weights ** 2)
    if sq_total <= 0:
        return 0.0
    return float((total ** 2) / sq_total)


@dataclass
class TerritoryProfile:
    """The three advanced territory metrics for one team, one venue role
    (host or visitor), for both what they generate ('for') and what they
    concede ('against')."""
    n_matches: int
    effective_sample_size: float
    big_chances_for: float
    big_chances_against: float
    shots_on_target_for: float
    shots_on_target_against: float
    box_touches_for: float
    box_touches_against: float


def team_territory_profile(
    league_df: pd.DataFrame, team: str, venue: str, half_life_days: float, reference_date: pd.Timestamp,
    use_match_index: bool = False, half_life_matches: float = FROZEN_HALF_LIFE_MATCHES,
) -> TerritoryProfile | None:
    """venue is 'home' or 'away'. Only ever looks at rows where the team
    played in that exact venue role - this is the "strict venue-isolated
    split" from Section 3. Returns None if there's no data at all for
    this team/venue (caller applies the 5-match safety rail).

    use_match_index=True switches the decay basis from calendar days to
    match recency index (see decay_weights_by_index) - this is what
    "Freeze Decay" now uses, specifically to avoid a summer break or
    international window fictitiously flattening a team's recent form
    just because a lot of calendar days happened to pass."""
    if venue == "home":
        rows = league_df[league_df["home_team"] == team]
        for_prefix, against_prefix = "home_", "away_"
    else:
        rows = league_df[league_df["away_team"] == team]
        for_prefix, against_prefix = "away_", "home_"

    if rows.empty:
        return None

    if use_match_index:
        rows = rows.sort_values("date", ascending=False)  # index 0 = most recent
        w = decay_weights_by_index(len(rows), half_life_matches)
    else:
        w = decay_weights(rows["date"], reference_date, half_life_days)

    def wmean(col):
        vals = rows[col].fillna(0.0).to_numpy(dtype=float)
        return weighted_mean(vals, w)

    return TerritoryProfile(
        n_matches=len(rows),
        effective_sample_size=round(effective_sample_size(w), 1),
        big_chances_for=wmean(f"{for_prefix}big_chances"),
        big_chances_against=wmean(f"{against_prefix}big_chances"),
        shots_on_target_for=wmean(f"{for_prefix}shots_on_target"),
        shots_on_target_against=wmean(f"{against_prefix}shots_on_target"),
        box_touches_for=wmean(f"{for_prefix}box_touches"),
        box_touches_against=wmean(f"{against_prefix}box_touches"),
    )


def rolling_window_weights(days_elapsed: np.ndarray, window_days: float) -> np.ndarray:
    """The classic alternative to exponential time-decay: a HARD cutoff.
    Full weight (1.0) for any match within the window, ZERO outside it -
    unlike decay_weights, which never fully zeroes out an old match, just
    fades it smoothly toward (but never reaching) zero. A rolling window
    is a step function; exponential decay is a smooth curve that a
    rolling window is a special, discontinuous case of."""
    days_elapsed = np.asarray(days_elapsed, dtype=float)
    return np.where(days_elapsed <= window_days, 1.0, 0.0)


def team_territory_profile_rolling(
    league_df: pd.DataFrame, team: str, venue: str, window_days: float, reference_date: pd.Timestamp,
) -> "TerritoryProfile | None":
    """Rolling-window counterpart to team_territory_profile - same venue
    isolation and aggregation, but a hard cutoff instead of exponential
    decay. Deliberately kept as a fully separate function (not a mode
    flag on the live one) so running this comparison can never accidentally
    change what the live prediction path actually uses."""
    if venue == "home":
        rows = league_df[league_df["home_team"] == team]
        for_prefix, against_prefix = "home_", "away_"
    else:
        rows = league_df[league_df["away_team"] == team]
        for_prefix, against_prefix = "away_", "home_"

    if rows.empty:
        return None

    days_elapsed = (reference_date - rows["date"]).dt.days.clip(lower=0).to_numpy(dtype=float)
    w = rolling_window_weights(days_elapsed, window_days)
    if w.sum() <= 0:
        return None  # every match for this team/venue fell outside the window

    def wmean(col):
        vals = rows[col].fillna(0.0).to_numpy(dtype=float)
        return weighted_mean(vals, w)

    return TerritoryProfile(
        n_matches=int((w > 0).sum()),  # only count matches actually inside the window
        effective_sample_size=round(effective_sample_size(w), 1),  # == n_matches here (0/1 weights), kept for API consistency with the decay-weighted profile
        big_chances_for=wmean(f"{for_prefix}big_chances"),
        big_chances_against=wmean(f"{against_prefix}big_chances"),
        shots_on_target_for=wmean(f"{for_prefix}shots_on_target"),
        shots_on_target_against=wmean(f"{against_prefix}shots_on_target"),
        box_touches_for=wmean(f"{for_prefix}box_touches"),
        box_touches_against=wmean(f"{against_prefix}box_touches"),
    )


SOT_CONVERSION_SHRINKAGE_K = 15.0  # "pseudo-matches" of league-average prior
                                    # strength - same empirical-Bayes shrinkage
                                    # spirit used elsewhere for thin samples:
                                    # shrink_weight = ESS / (ESS + K), so a team
                                    # with ESS well below K sits close to the
                                    # league average, and one with ESS well
                                    # above K is trusted almost entirely on its
                                    # own data.


@dataclass
class SOTConversionProfile:
    """How efficiently a team turns shots on target into goals (FOR),
    and how efficiently opponents turn shots on target into goals
    AGAINST this team (i.e. this team's own defending/goalkeeping at
    preventing SOT from becoming goals) - separate from raw SOT volume,
    which team_territory_profile already covers. Both conversion rates
    are shrunk toward the league average based on effective sample size,
    since a handful of matches can produce a wildly noisy conversion
    rate (a single deflected shot or an outstanding goalkeeping display
    swings it hard) - the raw, un-shrunk numbers are kept alongside for
    transparency."""
    n_matches: int
    effective_sample_size: float
    sot_conversion_for: float
    sot_conversion_against: float
    raw_sot_conversion_for: float
    raw_sot_conversion_against: float
    league_avg_conversion: float
    shrinkage_weight: float  # 0 = entirely league average, 1 = entirely this team's own data


def league_average_sot_conversion(settled_df: pd.DataFrame) -> float:
    """Aggregate goals / aggregate shots-on-target across the WHOLE
    league (home and away pooled together, since conversion efficiency
    isn't a venue-specific concept the way attack/defense strength is) -
    the shrinkage target every team's own conversion rate is pulled
    toward when its own sample is thin."""
    if settled_df.empty:
        return 0.3  # a reasonable generic football fallback - roughly matches typical SOT conversion rates
    total_goals = settled_df["home_goals"].fillna(0.0).sum() + settled_df["away_goals"].fillna(0.0).sum()
    total_sot = settled_df["home_shots_on_target"].fillna(0.0).sum() + settled_df["away_shots_on_target"].fillna(0.0).sum()
    return _safe_ratio(float(total_goals), float(total_sot)) if total_sot > 0 else 0.3


def team_sot_conversion_profile(
    league_df: pd.DataFrame, team: str, venue: str, half_life_days: float, reference_date: pd.Timestamp,
    league_avg_conversion: float, use_match_index: bool = False, half_life_matches: float = FROZEN_HALF_LIFE_MATCHES,
    shrinkage_k: float = SOT_CONVERSION_SHRINKAGE_K,
) -> "SOTConversionProfile | None":
    """Same venue-isolated, decay-weighted approach as
    team_territory_profile, but for goals-per-SOT (for) and
    goals-conceded-per-SOT-faced (against). Uses the ratio of
    decay-weighted SUMS (sum(goals*w) / sum(sot*w)) rather than
    averaging each match's own ratio - this avoids a single low-SOT,
    lucky-goal match producing a wild per-match ratio that then gets
    treated as equally informative as a normal match; it's the same
    "aggregate the raw counts, then divide" principle odds_calibration_table
    uses when it aggregates predictions into bands rather than averaging
    per-match ratios."""
    if venue == "home":
        rows = league_df[league_df["home_team"] == team]
        for_prefix, against_prefix = "home_", "away_"
    else:
        rows = league_df[league_df["away_team"] == team]
        for_prefix, against_prefix = "away_", "home_"

    if rows.empty:
        return None

    if use_match_index:
        rows = rows.sort_values("date", ascending=False)
        w = decay_weights_by_index(len(rows), half_life_matches)
    else:
        w = decay_weights(rows["date"], reference_date, half_life_days)

    ess = effective_sample_size(w)

    goals_for = rows[f"{for_prefix}goals"].fillna(0.0).to_numpy(dtype=float)
    sot_for = rows[f"{for_prefix}shots_on_target"].fillna(0.0).to_numpy(dtype=float)
    goals_against = rows[f"{against_prefix}goals"].fillna(0.0).to_numpy(dtype=float)
    sot_against = rows[f"{against_prefix}shots_on_target"].fillna(0.0).to_numpy(dtype=float)

    raw_for = _safe_ratio(float(np.sum(goals_for * w)), float(np.sum(sot_for * w)))
    raw_against = _safe_ratio(float(np.sum(goals_against * w)), float(np.sum(sot_against * w)))

    shrink_weight = ess / (ess + shrinkage_k) if (ess + shrinkage_k) > 0 else 0.0
    shrunk_for = shrink_weight * raw_for + (1 - shrink_weight) * league_avg_conversion
    shrunk_against = shrink_weight * raw_against + (1 - shrink_weight) * league_avg_conversion

    return SOTConversionProfile(
        n_matches=len(rows),
        effective_sample_size=round(ess, 1),
        sot_conversion_for=round(shrunk_for, 3),
        sot_conversion_against=round(shrunk_against, 3),
        raw_sot_conversion_for=round(raw_for, 3),
        raw_sot_conversion_against=round(raw_against, 3),
        league_avg_conversion=round(league_avg_conversion, 3),
        shrinkage_weight=round(shrink_weight, 3),
    )


@dataclass
class LeagueBaseline:
    """League-wide averages used as the shared, non-team-specific
    scale anchor - see design note #2 at the top of this file."""
    avg_home_goals: float
    avg_away_goals: float
    home_big_chances_for: float
    home_big_chances_against: float
    home_sot_for: float
    home_sot_against: float
    home_box_for: float
    home_box_against: float
    away_big_chances_for: float
    away_big_chances_against: float
    away_sot_for: float
    away_sot_against: float
    away_box_for: float
    away_box_against: float


def compute_league_baseline(settled_df: pd.DataFrame) -> LeagueBaseline:
    def col_mean(col):
        return float(settled_df[col].fillna(0.0).mean()) if col in settled_df.columns and len(settled_df) else 0.0

    return LeagueBaseline(
        avg_home_goals=col_mean("home_goals") or 1.0,
        avg_away_goals=col_mean("away_goals") or 1.0,
        home_big_chances_for=col_mean("home_big_chances") or 1.0,
        home_big_chances_against=col_mean("away_big_chances") or 1.0,
        home_sot_for=col_mean("home_shots_on_target") or 1.0,
        home_sot_against=col_mean("away_shots_on_target") or 1.0,
        home_box_for=col_mean("home_box_touches") or 1.0,
        home_box_against=col_mean("away_box_touches") or 1.0,
        away_big_chances_for=col_mean("away_big_chances") or 1.0,
        away_big_chances_against=col_mean("home_big_chances") or 1.0,
        away_sot_for=col_mean("away_shots_on_target") or 1.0,
        away_sot_against=col_mean("home_shots_on_target") or 1.0,
        away_box_for=col_mean("away_box_touches") or 1.0,
        away_box_against=col_mean("home_box_touches") or 1.0,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 1.0
    return numerator / denominator


# The 3 territory ratios are (big_chances, shots_on_target, box_touches),
# in that fixed order, wherever a `weights` tuple appears. Equal weighting
# is the historical default (matches the model's original behavior
# unchanged); optimize_territory_weights() below can find a better-fitting
# blend per league instead.
TERRITORY_WEIGHTS_DEFAULT = (1 / 3, 1 / 3, 1 / 3)

# A modest, deliberately small grid for the weight-calibration search -
# each candidate here means re-running a walk-forward backtest, so this
# stays a manageable size rather than an exhaustive continuous search.
_TERRITORY_WEIGHT_STEPS = [0.15, 0.25, 1 / 3, 0.45, 0.55]


def _territory_weight_candidates():
    candidates = []
    for big_chances_w in _TERRITORY_WEIGHT_STEPS:
        for sot_w in _TERRITORY_WEIGHT_STEPS:
            box_w = 1.0 - big_chances_w - sot_w
            if 0.05 <= box_w <= 0.6:
                candidates.append((round(big_chances_w, 4), round(sot_w, 4), round(box_w, 4)))
    return candidates


def attack_strength(profile, baseline: LeagueBaseline, venue: str, weights: tuple = TERRITORY_WEIGHTS_DEFAULT) -> float:
    """Composite attack strength = weighted average of the 3 territory
    ratios vs the league baseline for that venue role (big chances, shots
    on target, box touches - in that order in `weights`). Defaults to
    equal weighting (1/3 each), matching the original behavior; a
    per-league calibrated weighting can be passed in instead - see
    optimize_territory_weights(). Falls back to a neutral 1.0 if the
    sample is too small (Section 3's 5-match safety rail)."""
    if profile is None or profile.n_matches < MIN_SAMPLE_ROWS:
        return 1.0
    if venue == "home":
        ratios = [
            _safe_ratio(profile.big_chances_for, baseline.home_big_chances_for),
            _safe_ratio(profile.shots_on_target_for, baseline.home_sot_for),
            _safe_ratio(profile.box_touches_for, baseline.home_box_for),
        ]
    else:
        ratios = [
            _safe_ratio(profile.big_chances_for, baseline.away_big_chances_for),
            _safe_ratio(profile.shots_on_target_for, baseline.away_sot_for),
            _safe_ratio(profile.box_touches_for, baseline.away_box_for),
        ]
    return float(np.average(ratios, weights=weights))


def defense_strength(profile, baseline: LeagueBaseline, venue: str, weights: tuple = TERRITORY_WEIGHTS_DEFAULT) -> float:
    """Composite defense strength (how much this team ALLOWS relative to
    league baseline, in the same venue role) - values BELOW 1.0 mean a
    better-than-average defense. Same weighting contract as
    attack_strength() above. Falls back to neutral 1.0 on a too-small
    sample."""
    if profile is None or profile.n_matches < MIN_SAMPLE_ROWS:
        return 1.0
    if venue == "home":
        ratios = [
            _safe_ratio(profile.big_chances_against, baseline.home_big_chances_against),
            _safe_ratio(profile.shots_on_target_against, baseline.home_sot_against),
            _safe_ratio(profile.box_touches_against, baseline.home_box_against),
        ]
    else:
        ratios = [
            _safe_ratio(profile.big_chances_against, baseline.away_big_chances_against),
            _safe_ratio(profile.shots_on_target_against, baseline.away_sot_against),
            _safe_ratio(profile.box_touches_against, baseline.away_box_against),
        ]
    return float(np.average(ratios, weights=weights))


def expected_goals(
    home_attack: float, away_defense: float, away_attack: float, home_defense: float, baseline: LeagueBaseline,
) -> tuple[float, float]:
    """Home advantage is NOT re-applied here - it's already inside
    home_attack/home_defense because those numbers only ever came from
    the team's own home-position rows (see design note #1)."""
    lambda_home = baseline.avg_home_goals * home_attack * away_defense
    lambda_away = baseline.avg_away_goals * away_attack * home_defense
    return max(lambda_home, 0.05), max(lambda_away, 0.05)


def team_expected_goals_against(defense: float, baseline: LeagueBaseline, venue: str) -> float:
    """A team's own STANDALONE 'expected goals against' rating - how
    many goals they'd be expected to concede against a LEAGUE-AVERAGE
    opponent (attack_strength held at the neutral 1.0), using only
    their own defense_strength rating for that venue. This is exactly
    the same formula expected_goals() uses to build a specific
    fixture's lambda, just with the opponent's attack fixed at neutral
    instead of a real opponent's rating - so the number reflects THIS
    team's own defense alone, not muddied by whoever they're about to
    play. Useful as a standalone team-profile stat (same spirit as the
    SOT conversion against rating above), independent of any one
    fixture - e.g. for comparing two teams' defensive quality directly,
    or for a league-wide "best defense" ranking.

    venue is the TEAM's own venue role ('home' or 'away') - a home
    team's expected goals against uses the AWAY-goals league baseline
    (what an average away side scores), since that's who they're
    notionally defending against; an away team's uses the HOME-goals
    baseline, mirroring exactly how expected_goals() pairs lambda_away
    with home_defense and lambda_home with away_defense."""
    opponent_baseline_goals = baseline.avg_away_goals if venue == "home" else baseline.avg_home_goals
    return max(opponent_baseline_goals * defense, 0.05)


# ---------------------------------------------------------------------------
# Core Parameter A: dynamic half-life optimisation via Brier score backtest
# ---------------------------------------------------------------------------

def _quick_lambda_for_backtest(
    history_df: pd.DataFrame, home_team: str, away_team: str, half_life_days: float, as_of: pd.Timestamp,
    weights: tuple = TERRITORY_WEIGHTS_DEFAULT,
):
    if len(history_df) < MIN_SAMPLE_ROWS * 2:
        return None
    baseline = compute_league_baseline(history_df)
    home_profile = team_territory_profile(history_df, home_team, "home", half_life_days, as_of)
    away_profile = team_territory_profile(history_df, away_team, "away", half_life_days, as_of)
    ha = attack_strength(home_profile, baseline, "home", weights)
    hd = defense_strength(home_profile, baseline, "home", weights)
    aa = attack_strength(away_profile, baseline, "away", weights)
    ad = defense_strength(away_profile, baseline, "away", weights)
    return expected_goals(ha, ad, aa, hd, baseline)


def _outcome_probs_from_lambdas(lam_home: float, lam_away: float) -> tuple[float, float, float]:
    home_win = draw = away_win = 0.0
    for x in range(GOAL_CAP + 1):
        px = scipy_poisson.pmf(x, lam_home)
        for y in range(GOAL_CAP + 1):
            py = scipy_poisson.pmf(y, lam_away)
            p = px * py
            if x > y:
                home_win += p
            elif x == y:
                draw += p
            else:
                away_win += p
    total = home_win + draw + away_win
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return home_win / total, draw / total, away_win / total


def optimize_half_life(settled_df: pd.DataFrame, max_matches_evaluated: int = 150, weights: tuple = TERRITORY_WEIGHTS_DEFAULT, candidates: list = None):
    """Backtests each candidate half-life on real past results using a
    Brier score, and returns the one with the lowest average error. Only
    evaluates the most recent `max_matches_evaluated` matches for
    performance - this is a real backtest, not a fixed guess, but a
    league's full season history doesn't need to be replayed dozens of
    times over to get a stable answer.

    `candidates` defaults to the normal HALF_LIFE_CANDIDATES (15-180 days)
    - the live app always uses this default. It exists as a parameter so
    a diagnostic can pass in an EXTENDED range (e.g. up to 365 days) to
    check whether a boundary result at 180 reflects the search genuinely
    plateauing there, or being artificially capped - without changing
    what the live app actually searches by default."""
    candidates = candidates if candidates is not None else HALF_LIFE_CANDIDATES
    df = settled_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < MIN_SAMPLE_ROWS * 3:
        return FROZEN_HALF_LIFE_DAYS, {"reason": "not enough settled matches to backtest - using frozen default"}

    eval_start = max(MIN_SAMPLE_ROWS * 2, len(df) - max_matches_evaluated)
    scores = {}
    for half_life in candidates:
        brier_terms = []
        for i in range(eval_start, len(df)):
            row = df.iloc[i]
            history = df.iloc[:i]
            lambdas = _quick_lambda_for_backtest(
                history, row["home_team"], row["away_team"], half_life, row["date"], weights
            )
            if lambdas is None:
                continue
            p_home, p_draw, p_away = _outcome_probs_from_lambdas(*lambdas)
            actual = (
                (1, 0, 0) if row["home_goals"] > row["away_goals"]
                else (0, 1, 0) if row["home_goals"] == row["away_goals"]
                else (0, 0, 1)
            )
            brier_terms.append(
                (p_home - actual[0]) ** 2 + (p_draw - actual[1]) ** 2 + (p_away - actual[2]) ** 2
            )
        if brier_terms:
            scores[half_life] = float(np.mean(brier_terms))

    if not scores:
        return FROZEN_HALF_LIFE_DAYS, {"reason": "backtest produced no evaluable matches - using frozen default"}

    best = min(scores, key=scores.get)
    return best, {"brier_scores_by_half_life": scores, "chosen": best}


EXTENDED_HALF_LIFE_CANDIDATES = HALF_LIFE_CANDIDATES + [210, 240, 270, 300, 330, 365]


def extended_half_life_diagnostic(settled_df: pd.DataFrame, max_matches_evaluated: int = 150, weights: tuple = TERRITORY_WEIGHTS_DEFAULT) -> dict:
    """Diagnostic-only: re-runs the half-life search with candidates
    extending well past the normal 180-day ceiling (up to 365 days). Does
    NOT change what the live app uses - optimize_half_life's default
    candidate list is untouched, this is purely for checking whether a
    180-day boundary result reflects the score genuinely still improving
    past 180 (the normal grid really was capping something real), or
    whether it plateaus/worsens past 180 (180 was already close to a
    genuine optimum, and hitting the boundary was closer to noise between
    neighboring candidates than a sign of anything being cut off)."""
    best, info = optimize_half_life(
        settled_df, max_matches_evaluated=max_matches_evaluated, weights=weights,
        candidates=EXTENDED_HALF_LIFE_CANDIDATES,
    )
    if "brier_scores_by_half_life" not in info:
        return {"chosen": best, "still_improving_past_180": None, "reason": info.get("reason")}

    scores = info["brier_scores_by_half_life"]
    within_180 = {hl: s for hl, s in scores.items() if hl <= 180}
    beyond_180 = {hl: s for hl, s in scores.items() if hl > 180}
    still_improving = None
    if within_180 and beyond_180:
        best_within = min(within_180.values())
        best_beyond = min(beyond_180.values())
        still_improving = best_beyond < best_within
    return {
        "chosen": best,
        "brier_scores_by_half_life": scores,
        "still_improving_past_180": still_improving,
    }


def half_life_at_search_boundary(half_life: float, candidates=HALF_LIFE_CANDIDATES) -> str | None:
    """Returns a warning string if the optimizer picked the exact minimum
    or maximum of the tested half-life range - not necessarily wrong, but
    a real diagnostic flag: it means either the TRUE optimum lies outside
    the range that was actually searched (the grid artificially capped
    it), or the backtest sample was too thin/noisy for the Brier search to
    meaningfully discriminate between candidates, and it landed on an edge
    somewhat arbitrarily. Returns None when the chosen half-life is
    comfortably inside the range (no warning needed)."""
    if half_life is None or (isinstance(half_life, float) and math.isnan(half_life)):
        return None
    lo, hi = min(candidates), max(candidates)
    if half_life <= lo:
        return (
            f"⚠️ The optimizer selected {half_life:.0f} days - the SHORTEST option tested "
            f"({lo}-{hi} day range). The true optimum may be even faster-decaying than what "
            "was searched, or the backtest sample may be too thin to distinguish clearly "
            "between short half-lives yet."
        )
    if half_life >= hi:
        return (
            f"⚠️ The optimizer selected {half_life:.0f} days - the LONGEST option tested "
            f"({lo}-{hi} day range). The true optimum may need MORE memory than what was "
            "searched (the grid capped it here), or the backtest sample may be too thin/noisy "
            "for the search to confidently distinguish between longer half-lives. Worth "
            "checking how many settled matches this league actually has, especially per team "
            "per venue, since that's the more relevant sample size than the total match count."
        )
    return None


def safety_rail_dilution_report(settled_df: pd.DataFrame, max_matches_evaluated: int = 150) -> dict:
    """Reports what share of the matches the half-life search actually
    evaluates have AT LEAST ONE side under the 5-match safety rail at
    that point in time - meaning that side contributes a flat, neutral
    1.0 (see attack_strength/defense_strength) rather than real signal.
    This directly explains a half-life search landing on a grid boundary
    somewhat arbitrarily: a "diluted" match scores IDENTICALLY no matter
    which half-life candidate is tested (the neutral side's contribution
    doesn't change with decay speed), so it adds noise to the search
    without adding any real discriminating signal. Uses the exact same
    walk-forward window (eval_start, per-match eligibility) as
    optimize_half_life, so the percentage reported here reflects what
    that search actually saw - not a separately-defined approximation.

    Deliberately independent of which half-life is chosen: n_matches is
    just a raw row count, unaffected by decay weighting, so this doesn't
    need to be recomputed per half-life candidate."""
    df = settled_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < MIN_SAMPLE_ROWS * 3:
        return {"total_evaluated": 0, "diluted_count": 0, "diluted_pct": float("nan"),
                "reason": "not enough settled matches to backtest"}

    eval_start = max(MIN_SAMPLE_ROWS * 2, len(df) - max_matches_evaluated)
    total = 0
    diluted = 0
    for i in range(eval_start, len(df)):
        row = df.iloc[i]
        history = df.iloc[:i]
        if len(history) < MIN_SAMPLE_ROWS * 2:
            continue  # matches _quick_lambda_for_backtest's own eligibility check exactly
        home_profile = team_territory_profile(history, row["home_team"], "home", FROZEN_HALF_LIFE_DAYS, row["date"])
        away_profile = team_territory_profile(history, row["away_team"], "away", FROZEN_HALF_LIFE_DAYS, row["date"])
        total += 1
        home_thin = home_profile is None or home_profile.n_matches < MIN_SAMPLE_ROWS
        away_thin = away_profile is None or away_profile.n_matches < MIN_SAMPLE_ROWS
        if home_thin or away_thin:
            diluted += 1

    if total == 0:
        return {"total_evaluated": 0, "diluted_count": 0, "diluted_pct": float("nan"),
                "reason": "no evaluable matches"}
    return {
        "total_evaluated": total,
        "diluted_count": diluted,
        "diluted_pct": round(100 * diluted / total, 1),
    }


def optimize_territory_weights(settled_df: pd.DataFrame, half_life_days: float, max_matches_evaluated: int = 150):
    """Same idea as optimize_half_life(), but searches how much weight to
    put on big chances vs shots on target vs box touches (instead of
    searching decay speed), holding half-life FIXED at whatever was
    already chosen for this league. Different leagues' playing styles can
    genuinely differ in which territory metric best predicts goals - a
    possession-heavy league might lean more on big chances, a
    counter-attacking one more on shots on target - so a single universal
    equal-weighted blend isn't guaranteed to fit every league equally
    well. This is a real backtest search, not a guess: each candidate
    weighting is scored with the same walk-forward Brier score as
    everywhere else in this app, and whichever scores best is returned.

    Deliberately NOT run automatically on every prediction (that would
    nest an expensive grid-search backtest inside every single fixture
    load) - call this explicitly (e.g. from a 'Calibrate weights for this
    league' button) and cache the result per league instead."""
    df = settled_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < MIN_SAMPLE_ROWS * 3:
        return TERRITORY_WEIGHTS_DEFAULT, {"reason": "not enough settled matches to calibrate - using equal weights"}

    eval_start = max(MIN_SAMPLE_ROWS * 2, len(df) - max_matches_evaluated)
    candidates = _territory_weight_candidates()
    scores = {}
    for weights in candidates:
        brier_terms = []
        for i in range(eval_start, len(df)):
            row = df.iloc[i]
            history = df.iloc[:i]
            lambdas = _quick_lambda_for_backtest(
                history, row["home_team"], row["away_team"], half_life_days, row["date"], weights
            )
            if lambdas is None:
                continue
            p_home, p_draw, p_away = _outcome_probs_from_lambdas(*lambdas)
            actual = (
                (1, 0, 0) if row["home_goals"] > row["away_goals"]
                else (0, 1, 0) if row["home_goals"] == row["away_goals"]
                else (0, 0, 1)
            )
            brier_terms.append(
                (p_home - actual[0]) ** 2 + (p_draw - actual[1]) ** 2 + (p_away - actual[2]) ** 2
            )
        if brier_terms:
            scores[weights] = float(np.mean(brier_terms))

    if not scores:
        return TERRITORY_WEIGHTS_DEFAULT, {"reason": "backtest produced no evaluable matches - using equal weights"}

    best = min(scores, key=scores.get)
    # Don't look up the equal-weight baseline by exact dict-key equality -
    # TERRITORY_WEIGHTS_DEFAULT uses unrounded 1/3 floats while candidates
    # are rounded to 4dp, so they'd never be bit-identical even though one
    # of the candidates IS effectively equal weighting. Find the closest
    # candidate instead.
    equal_weight_key = min(
        scores.keys(),
        key=lambda w: sum(abs(a - b) for a, b in zip(w, TERRITORY_WEIGHTS_DEFAULT)),
    )
    equal_weight_score = scores[equal_weight_key]
    return best, {
        "brier_scores_by_weights": scores,
        "chosen": best,
        "equal_weight_score": equal_weight_score,
        "best_score": scores[best],
        "improved_over_equal_weight": scores[best] < equal_weight_score,
    }


# ---------------------------------------------------------------------------
# Full-dataset walk-forward backtest, Brier Skill Score, and accuracy
# ---------------------------------------------------------------------------

CORRECT_SCORE_GRID = [
    (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2),
    (2, 2), (3, 0), (0, 3), (3, 1), (1, 3), (3, 3),
]
CORRECT_SCORE_MARKETS = [f"Correct Score {h}-{a}" for h, a in CORRECT_SCORE_GRID] + ["Correct Score Other"]


def actual_market_outcomes(home_goals: float, away_goals: float) -> dict:
    """The real yes/no result for each market, given a match's actual
    final score - thresholds mirror market_probs_from_matrix exactly, so
    "did the model's predicted side for THIS market actually happen"
    comparisons are apples-to-apples with what the live sheet shows.

    Draw No Bet markets return None (not True/False) when the match was
    actually a draw - a Draw No Bet stake is VOID on a draw, not lost,
    so it must never be silently scored as a wrong prediction. Every
    consumer of this dict (walk_forward_backtest, per_market_backtest_
    accuracy, the calibration fitters) treats None as "exclude this row
    from this market's sample" rather than "incorrect"."""
    total = home_goals + away_goals
    home_win = home_goals > away_goals
    draw = home_goals == away_goals
    away_win = home_goals < away_goals
    margin = home_goals - away_goals
    result = {
        "Home Win": home_win, "Draw": draw, "Away Win": away_win,
        "Double Chance 1X": home_win or draw,
        "Double Chance 12": home_win or away_win,
        "Double Chance X2": draw or away_win,
        "Draw No Bet Home": (home_win if not draw else None),
        "Draw No Bet Away": (away_win if not draw else None),
        "Over 1.5 Goals": total > 1.5, "Under 1.5 Goals": total < 1.5,
        "Over 2.5 Goals": total > 2.5, "Under 2.5 Goals": total < 2.5,
        "Over 3.5 Goals": total > 3.5, "Under 3.5 Goals": total < 3.5,
        "BTTS - Yes": home_goals >= 1 and away_goals >= 1,
        "BTTS - No": not (home_goals >= 1 and away_goals >= 1),
        "Home Clean Sheet": away_goals == 0, "Away Clean Sheet": home_goals == 0,
        "Home Win to Nil": home_win and away_goals == 0,
        "Away Win to Nil": away_win and home_goals == 0,
        "Asian Handicap Home -1.5": margin > 1.5, "Asian Handicap Away +1.5": not (margin > 1.5),
        "Asian Handicap Home +1.5": not ((-margin) > 1.5), "Asian Handicap Away -1.5": (-margin) > 1.5,
    }
    named_hit = False
    for h, a in CORRECT_SCORE_GRID:
        hit = (home_goals == h and away_goals == a)
        result[f"Correct Score {h}-{a}"] = hit
        named_hit = named_hit or hit
    result["Correct Score Other"] = not named_hit
    return result


def walk_forward_backtest(
    settled_df: pd.DataFrame, half_life_days: float = FROZEN_HALF_LIFE_DAYS,
    weights: tuple = TERRITORY_WEIGHTS_DEFAULT, include_all_markets: bool = False,
) -> pd.DataFrame:
    """Walks through EVERY settled match in chronological order (not just
    the last 10) and, for each one with enough PRIOR history, predicts it
    using ONLY data from before that match's date - a genuine
    out-of-sample backtest across the whole dataset, no lookahead bias.

    Uses a single fixed half-life for the whole sweep rather than
    re-running the expensive half-life optimizer separately for every
    match (that would nest one already-expensive backtest inside
    another - pass in whichever half-life the live projection is
    currently using so this reflects the same settings). Same idea for
    `weights` - pass in a per-league calibrated territory weighting (see
    optimize_territory_weights()) to backtest THAT weighting choice,
    rather than always assuming equal weights.

    The core Home/Draw/Away probabilities now fit rho and build the real
    Dixon-Coles-adjusted score matrix for every historical match, then
    read Home Win/Draw/Away Win straight off it via
    market_probs_from_matrix() - matching EXACTLY what the live
    prediction path does (build_score_matrix + market_probs_from_matrix),
    not an approximation of it. This replaced an earlier version that
    used a plain rho=0 Poisson shortcut for speed; that shortcut is what
    optimize_half_life() below still uses internally (a much
    higher-multiplicity search - up to 12 half-life candidates times up
    to 150 matches each - where fitting rho at every single step would
    be prohibitively slow for comparatively little gain in THAT specific
    search), so the two are now intentionally NOT the same computation -
    this function mirrors live prediction exactly; the half-life search
    still uses a fast approximation purely to pick a half-life, which is
    a coarser decision than a full probability estimate.

    include_all_markets=False (the default, used everywhere BSS/accuracy/
    the reliability badge/dilution report/half-life search already
    depend on this function) only keeps the 1X2 outcome, still on the
    real rho-adjusted matrix - the matrix is now always built regardless
    of this flag, since Home/Draw/Away needs it anyway. Passing True
    additionally stores every OTHER market's own predicted probability
    and ground-truth outcome (needed for per-market accuracy and
    per-market calibration fitting) - a real extra cost per row (mostly
    the market_probs_from_matrix() call itself and storing 2×len(MARKET_LIST)
    extra columns), so it's opt-in rather than always paid."""
    df = settled_df.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        history = df.iloc[:i]
        lambdas = _quick_lambda_for_backtest(history, row["home_team"], row["away_team"], half_life_days, row["date"], weights)
        if lambdas is None:
            continue
        rho = fit_rho(history) if len(history) >= MIN_SAMPLE_ROWS else 0.0
        matrix = build_score_matrix(lambdas[0], lambdas[1], rho)
        predicted_probs = market_probs_from_matrix(matrix)
        p_home, p_draw, p_away = predicted_probs["Home Win"], predicted_probs["Draw"], predicted_probs["Away Win"]
        actual = "H" if row["home_goals"] > row["away_goals"] else ("D" if row["home_goals"] == row["away_goals"] else "A")
        predicted_pick = max([("H", p_home), ("D", p_draw), ("A", p_away)], key=lambda kv: kv[1])[0]
        result_row = {
            "date": row["date"], "home_team": row["home_team"], "away_team": row["away_team"],
            "home_goals": row["home_goals"], "away_goals": row["away_goals"],
            "goal_difference": row["home_goals"] - row["away_goals"],
            "p_home": round(p_home, 4), "p_draw": round(p_draw, 4), "p_away": round(p_away, 4),
            "actual": actual, "predicted_pick": predicted_pick,
            "correct": predicted_pick == actual,
        }
        if include_all_markets:
            actual_outcomes = actual_market_outcomes(row["home_goals"], row["away_goals"])
            for market in MARKET_LIST:
                actual_val = actual_outcomes[market]
                result_row[f"prob__{market}"] = round(predicted_probs[market], 4)
                # Ground truth for THIS market, independent of any 50% threshold - this is
                # what per-market CALIBRATION fitting needs (see fit_all_market_calibrators),
                # since a market's calibration should be checked against "did this actually
                # happen", not "did the model's own >=50% pick happen to be right".
                result_row[f"actual__{market}"] = actual_val
                if actual_val is None:
                    # Not applicable for this match (e.g. Draw No Bet when the match itself
                    # WAS a draw - void, not a loss) - excluded from accuracy/calibration,
                    # never silently counted as wrong.
                    result_row[f"correct__{market}"] = None
                else:
                    predicted_side = predicted_probs[market] >= 0.5
                    result_row[f"correct__{market}"] = predicted_side == actual_val
        rows.append(result_row)
    return pd.DataFrame(rows)


def per_market_backtest_accuracy(backtest_df: pd.DataFrame, min_confidence_pct: float = 0.0) -> dict:
    """Accuracy % AND sample size (N) for each of the 22 markets
    individually, aggregated across the same walk-forward sweep.
    Requires backtest_df to have been built with
    walk_forward_backtest(..., include_all_markets=True) - returns an
    empty dict otherwise rather than guessing or crashing.

    min_confidence_pct=0 (the default) matches the ORIGINAL behaviour:
    every settled match counts, regardless of how confident the model
    was in that specific market - so N is the same for every market
    (the full backtest length) unless a higher floor is set here.
    min_confidence_pct=X only counts matches where the model's own
    probability for that market's named outcome was >= X% - the same
    kind of gate the live valuation sheet's confidence floor applies,
    so N will genuinely differ market to market once this is raised
    above 0 (some markets clear high confidence far more often than
    others) - this is what makes a high accuracy % on a market like
    Draw or Win-to-Nil auditable rather than a number with no visible N
    behind it.

    Returns {market: {"accuracy_pct": float, "n": int}} - callers that
    only want the accuracy value can do `result[market]["accuracy_pct"]`."""
    if backtest_df is None or backtest_df.empty:
        return {}
    result = {}
    for market in MARKET_LIST:
        correct_col = f"correct__{market}"
        prob_col = f"prob__{market}"
        if correct_col not in backtest_df.columns:
            continue
        subset = backtest_df[backtest_df[correct_col].notna()]  # drop not-applicable rows (e.g. Draw No Bet on a drawn match) rather than counting them as wrong
        if min_confidence_pct > 0 and prob_col in subset.columns:
            subset = subset[subset[prob_col] * 100 >= min_confidence_pct]
        n = int(len(subset))
        accuracy = round(subset[correct_col].astype(bool).mean() * 100, 1) if n > 0 else float("nan")
        result[market] = {"accuracy_pct": accuracy, "n": n}
    return result


def _actual_vector(actual: str) -> tuple[int, int, int]:
    return (1, 0, 0) if actual == "H" else ((0, 1, 0) if actual == "D" else (0, 0, 1))


def multiclass_brier_score(backtest_df: pd.DataFrame) -> float:
    """Standard 3-class Brier score: mean squared error between the
    predicted probability vector and the one-hot actual outcome, summed
    across the 3 classes. 0 is a perfect forecaster, 2 is the worst
    possible (fully confident and always wrong)."""
    if backtest_df.empty:
        return float("nan")
    terms = []
    for _, r in backtest_df.iterrows():
        a0, a1, a2 = _actual_vector(r["actual"])
        terms.append((r["p_home"] - a0) ** 2 + (r["p_draw"] - a1) ** 2 + (r["p_away"] - a2) ** 2)
    return float(np.mean(terms))


def climatology_brier_score(backtest_df: pd.DataFrame) -> float:
    """The 'no-skill' reference forecast Brier Skill Score is measured
    against: always predicting the dataset's OVERALL historical
    home/draw/away frequency for every match, regardless of who's
    playing."""
    if backtest_df.empty:
        return float("nan")
    freq_home = float((backtest_df["actual"] == "H").mean())
    freq_draw = float((backtest_df["actual"] == "D").mean())
    freq_away = float((backtest_df["actual"] == "A").mean())
    terms = []
    for _, r in backtest_df.iterrows():
        a0, a1, a2 = _actual_vector(r["actual"])
        terms.append((freq_home - a0) ** 2 + (freq_draw - a1) ** 2 + (freq_away - a2) ** 2)
    return float(np.mean(terms))


def brier_skill_score(backtest_df: pd.DataFrame) -> float:
    """BSS = 1 - (model Brier / climatology Brier). Positive means the
    model beats blindly guessing the league's overall historical outcome
    split; 0 means no better than that naive baseline; negative means
    WORSE than just guessing the league average."""
    model_bs = multiclass_brier_score(backtest_df)
    ref_bs = climatology_brier_score(backtest_df)
    if not ref_bs or math.isnan(ref_bs) or ref_bs == 0:
        return float("nan")
    return 1 - (model_bs / ref_bs)


# Thresholds are deliberately conservative: football is genuinely
# high-variance at the match level, so even a real, usable edge tends to
# show up as a modest positive BSS rather than a dramatic one - this
# isn't a "grade the model out of 100" scale, it's a rough traffic light
# for "is there evidence this model beats guessing this league's own
# historical average, given what's been backtested so far."
def league_reliability_badge(bss: float) -> tuple[str, str]:
    """Returns (emoji, label) for a league's overall Brier Skill Score.
    NaN/None (not enough backtest data yet) gets its own neutral badge -
    it is NOT the same thing as a negative BSS and shouldn't look like a
    warning."""
    if bss is None or (isinstance(bss, float) and math.isnan(bss)):
        return "⚪", "Insufficient Backtest Data"
    if bss > 0.02:
        return "🟢", "Reliable"
    if bss >= -0.02:
        return "🟡", "Marginal"
    return "🔴", "Unreliable"


# ---------------------------------------------------------------------------
# Continuous league reliability score: replaces a hand-curated "which
# leagues are good/bad for territorial metrics" list with a score derived
# directly from THIS league's own data. Combines three independently-
# computable signals - see compute_league_reliability_score() docstring
# for exactly what each one means and, importantly, what this does NOT
# capture (set-piece share, direct-play index, referee tendencies, and
# fixture congestion aren't in this dataset, so they aren't in the score;
# extending it later if that data becomes available is straightforward).
# ---------------------------------------------------------------------------

def compute_dispersion_index(settled_df: pd.DataFrame) -> float:
    """Ratio of actual combined-goals variance to what a true Poisson
    process with the same mean would produce (variance == mean, for a
    real Poisson variable). 1.0 = perfectly Poisson-behaved; above 1.0
    means this league's scorelines are MORE spread out than Poisson
    predicts (blowouts and 0-0s both more common than a tidy Poisson
    model expects - a "chaotic" league); below 1.0 means tighter,
    more predictable scorelines than Poisson expects.

    This is NOT the same thing as rho (fit_rho below), which corrects
    specifically for LOW-SCORE correlation (0-0/1-0/0-1/1-1 being more
    common than independent Poisson implies) - dispersion_index looks at
    overall scoreline variance, a different and complementary signal."""
    goals = pd.concat([settled_df["home_goals"], settled_df["away_goals"]]).dropna().to_numpy(dtype=float)
    if len(goals) < MIN_SAMPLE_ROWS or goals.mean() <= 0:
        return 1.0
    return float(goals.var(ddof=1) / goals.mean())


def xg_proxy_correlation(settled_df: pd.DataFrame, weights: tuple = TERRITORY_WEIGHTS_DEFAULT) -> float:
    """Pearson correlation between a simple match-level territorial
    differential (big chances / shots on target / box touches, home
    minus away, each standardised within this league so the three raw
    metrics are on a comparable scale before blending with `weights`)
    and the ACTUAL goal difference. This is a direct, model-independent
    check of "does dominating these territorial metrics actually
    translate into goals in THIS league" - the exact question this
    engine's whole design has been chasing since the very first
    La-Liga-vs-Championship discussion, now computed straight from the
    data instead of argued from football-culture priors.

    Returns NaN (not 0) when there isn't enough data or no variance to
    correlate against - NaN reads as "unknown", 0 would wrongly read as
    "measured and found to be zero relationship"."""
    needed = ["home_big_chances", "away_big_chances", "home_shots_on_target",
              "away_shots_on_target", "home_box_touches", "away_box_touches",
              "home_goals", "away_goals"]
    present = [c for c in needed if c in settled_df.columns]
    df = settled_df.dropna(subset=present).copy()
    if len(df) < MIN_SAMPLE_ROWS or len(present) < len(needed):
        return float("nan")

    def z(series):
        s = series.std(ddof=0)
        return (series - series.mean()) / s if s > 0 else series * 0.0

    bc_diff = z(df["home_big_chances"] - df["away_big_chances"])
    sot_diff = z(df["home_shots_on_target"] - df["away_shots_on_target"])
    box_diff = z(df["home_box_touches"] - df["away_box_touches"])
    territorial_diff = weights[0] * bc_diff + weights[1] * sot_diff + weights[2] * box_diff

    goal_diff = df["home_goals"] - df["away_goals"]
    if territorial_diff.std(ddof=0) == 0 or goal_diff.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(territorial_diff, goal_diff)[0, 1])


@dataclass
class LeagueReliabilityScore:
    bss: float
    dispersion_index: float
    xg_proxy_correlation: float
    composite_score: float  # 0-1, higher = territorial/xG-proxy signal is more trustworthy for this league
    n_matches: int


def compute_league_reliability_score(
    settled_df: pd.DataFrame, backtest_df: pd.DataFrame = None, weights: tuple = TERRITORY_WEIGHTS_DEFAULT,
) -> LeagueReliabilityScore:
    """ONE continuous 0-1 score for how much this league's territorial/
    xG-proxy metrics can be trusted, built entirely from this league's
    own data:

      - BSS (if a walk-forward backtest is supplied) - the most direct
        empirical evidence: does the model actually predict results
        well here, full stop.
      - xg_proxy_correlation - does the territorial differential line
        up with the real goal difference in this league.
      - dispersion_index - how close this league's goal-scoring
        variance sits to a well-behaved Poisson process; heavily
        over- or under-dispersed leagues are the harder-to-model ones.

    Each component is normalised to roughly [0, 1] before averaging, so
    no single component with a wider natural range can silently dominate
    the composite. HONESTY NOTE: this uses only what's actually
    computable from goals/big-chances/SOT/box-touches - it does not
    (and cannot, from this dataset) capture set-piece share, direct-play
    style, referee tendencies, or fixture congestion, all of which came
    up as relevant during the original league-by-league discussion this
    score is meant to replace. If that data becomes available later,
    this is the function to extend with more components."""
    n = len(settled_df)
    bss = brier_skill_score(backtest_df) if backtest_df is not None and not backtest_df.empty else float("nan")
    dispersion = compute_dispersion_index(settled_df)
    xg_corr = xg_proxy_correlation(settled_df, weights)

    # 0 BSS -> ~0.33 normalised, 0.10 BSS (a genuinely strong league) -> 1.0.
    # Missing BSS scores as a neutral 0.5 rather than dragging the composite down for a reason unrelated to reliability.
    bss_norm = float(np.clip((bss + 0.05) / 0.15, 0, 1)) if not math.isnan(bss) else 0.5
    # |r|=0.1 (weak/no real relationship) -> 0, |r|=0.5 (strong relationship) -> 1.0
    xg_corr_norm = float(np.clip((abs(xg_corr) - 0.1) / 0.4, 0, 1)) if not math.isnan(xg_corr) else 0.5
    # dispersion == 1.0 (perfectly Poisson) -> 1.0, degrading as it drifts either direction
    dispersion_norm = float(np.clip(1.0 - abs(dispersion - 1.0), 0, 1))

    composite = float(np.mean([bss_norm, xg_corr_norm, dispersion_norm]))

    return LeagueReliabilityScore(
        bss=round(bss, 4) if not math.isnan(bss) else float("nan"),
        dispersion_index=round(dispersion, 3),
        xg_proxy_correlation=round(xg_corr, 3) if not math.isnan(xg_corr) else float("nan"),
        composite_score=round(composite, 3),
        n_matches=n,
    )


# ---------------------------------------------------------------------------
# Odds/probability calibration: "2.00 odds means an implied 50% chance -
# does the team the model backs at that price actually win about half the
# time?" Uses the walk-forward backtest's own historical predictions, so
# it needs no real bookmaker odds data at all - it's checking whether the
# MODEL's own probabilities are trustworthy, not comparing against a
# specific bookmaker's pricing.
# ---------------------------------------------------------------------------

ODDS_BAND_EDGES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, float("inf")]
ODDS_BAND_LABELS = ["1.01-1.50", "1.51-2.00", "2.01-2.50", "2.51-3.00", "3.01-4.00", "4.01-6.00", "6.01+"]


def implied_probability(odds: float) -> float:
    """The plain reverse of fair_odds: 2.00 odds implies a 50% chance,
    3.00 implies 33.3%, etc. - exactly the relationship the odds
    calibration table below checks against real outcomes."""
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def odds_calibration_table(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """For every historical walk-forward prediction, takes the model's
    own probability for whichever side it actually picked, converts that
    to an implied odds price (1 / probability), buckets those into
    standard odds bands, and checks: within each band, how often did the
    picked side actually win? A well-calibrated model should show an
    actual win rate close to the band's implied probability - e.g.
    picks priced around 2.00 (50% implied) should win roughly half the
    time, not 30% or 70% of the time."""
    if backtest_df.empty:
        return pd.DataFrame(columns=[
            "Odds Band", "Predictions", "Avg Model-Implied Odds",
            "Model-Implied Win Rate %", "Actual Win Rate %", "Calibration Gap (pts)",
        ])

    pick_prob = backtest_df[["p_home", "p_draw", "p_away"]].max(axis=1)
    implied_odds = pick_prob.apply(lambda p: (1.0 / p) if p > 0 else float("inf"))
    bands = pd.cut(implied_odds, bins=ODDS_BAND_EDGES, labels=ODDS_BAND_LABELS, right=True, include_lowest=True)

    working = pd.DataFrame({
        "band": bands, "pick_prob": pick_prob, "implied_odds": implied_odds, "correct": backtest_df["correct"],
    })

    rows = []
    for label in ODDS_BAND_LABELS:
        bucket = working[working["band"] == label]
        if bucket.empty:
            continue
        avg_implied_odds = float(bucket["implied_odds"].replace(np.inf, np.nan).mean())
        model_implied_win_rate = float(bucket["pick_prob"].mean() * 100)
        actual_win_rate = float(bucket["correct"].mean() * 100)
        rows.append({
            "Odds Band": label,
            "Predictions": int(len(bucket)),
            "Avg Model-Implied Odds": round(avg_implied_odds, 2) if not math.isnan(avg_implied_odds) else None,
            "Model-Implied Win Rate %": round(model_implied_win_rate, 1),
            "Actual Win Rate %": round(actual_win_rate, 1),
            "Calibration Gap (pts)": round(actual_win_rate - model_implied_win_rate, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 7.5: Probability calibration (Platt scaling / isotonic regression)
#
# odds_calibration_table() above can reveal the raw model probabilities are
# systematically off in one of two different shapes:
#   - a smooth, ONE-DIRECTIONAL drift (e.g. increasingly overconfident as
#     odds lengthen) - Platt scaling fits this well with just 2 parameters.
#   - a NON-MONOTONIC, zig-zagging gap (well-calibrated in some bands, off
#     in others, no consistent direction) - isotonic regression fits this
#     better, since it makes no assumption about the correction's shape
#     beyond "a higher raw probability must still map to a higher (or
#     equal) corrected probability".
#
# Rather than a human eyeballing the calibration table to guess which shape
# a league has, select_calibrator() fits BOTH methods on the same historical
# rows and keeps whichever one actually scores better on genuinely held-out
# folds (k-fold cross-validation - never scored on the same rows it was fit
# on). "No correction at all" is included as a third candidate, so a league
# whose calibration gaps are just noise doesn't get a correction fit to
# that noise - see the Bundesliga vs. La Liga gap shapes discussed when this
# was designed: one drifted in a single direction, the other zig-zagged
# with no gap ever really clearing its own sampling noise.
#
# Fit PER OUTCOME CLASS (Home/Draw/Away each get their own independently-
# selected method), not on whichever side the model happened to pick - this
# is what lets "is a 35%-probability draw actually ~35%" be checked
# directly against every match where 35% draw probability occurred, not
# just the rare matches where draw happened to be the model's top pick
# (recall predicted_pick is H or A in the overwhelming majority of
# matches - draw is almost never the argmax - so a pick-only calibration
# would have almost no draw data to learn from).
#
# Isotonic regression is implemented directly via the Pool-Adjacent-
# Violators Algorithm (PAVA) rather than pulling in scikit-learn as a new
# dependency just for this one function - same "no shortcuts" spirit as
# market_probs_from_matrix() above, which sums the real Dixon-Coles grid
# cells instead of approximating.
# ---------------------------------------------------------------------------

CALIBRATION_MIN_SAMPLES = 40   # below this, select_calibrator() returns "none" outright - too few rows for either method to be trustworthy, same safety-rail spirit as MIN_SAMPLE_ROWS elsewhere in this file
CALIBRATION_CV_FOLDS = 5
CALIBRATION_EPS = 1e-6         # keeps logit()/log-loss finite at p=0 or p=1


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), CALIBRATION_EPS, 1 - CALIBRATION_EPS)
    return np.log(p / (1 - p))


def _sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _binary_log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), CALIBRATION_EPS, 1 - CALIBRATION_EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass
class PlattCalibrator:
    """sigmoid(a * logit(p) + b) - the standard Platt scaling
    formulation applied to logit(p) rather than a raw classifier score,
    since our 'raw score' already IS a probability. a=1, b=0 is a
    no-op (returns the input unchanged)."""
    a: float
    b: float

    def predict(self, probs) -> np.ndarray:
        return _sigmoid(self.a * _logit(probs) + self.b)


def fit_platt(probs, outcomes) -> PlattCalibrator:
    """Fits (a, b) by maximum likelihood - minimising binary log-loss -
    via Nelder-Mead (dependency-free: no gradient needed, and this is a
    tiny 2-parameter problem where a derivative-free optimiser is more
    than fast enough)."""
    x = _logit(probs)
    y = np.asarray(outcomes, dtype=float)

    def loss(params):
        a, b = params
        return _binary_log_loss(_sigmoid(a * x + b), y)

    result = _minimize(loss, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    a, b = result.x
    return PlattCalibrator(a=float(a), b=float(b))


@dataclass
class IsotonicCalibrator:
    """A monotonically non-decreasing step function fit by PAVA, stored
    as (x, fitted_y) breakpoints and evaluated on new inputs via linear
    interpolation between them (queries outside the training range clip
    to the nearest endpoint - numpy's default np.interp behaviour, and
    the standard way isotonic regression handles out-of-range inputs)."""
    thresholds: np.ndarray
    values: np.ndarray

    def predict(self, probs) -> np.ndarray:
        return np.interp(np.asarray(probs, dtype=float), self.thresholds, self.values)


def fit_isotonic(probs, outcomes) -> IsotonicCalibrator:
    """Pool-Adjacent-Violators Algorithm: sort by raw probability, then
    repeatedly merge any adjacent block whose average outcome rate would
    otherwise be LOWER than the block before it, until the whole
    sequence is non-decreasing. Every original point keeps the fitted
    value of whichever final block it belongs to."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    order = np.argsort(probs, kind="mergesort")
    x_sorted = probs[order]
    y_sorted = outcomes[order]

    block_sum = list(y_sorted)
    block_count = [1] * len(y_sorted)
    block_start = list(range(len(y_sorted)))
    block_end = list(range(len(y_sorted)))

    i = 0
    while i < len(block_sum) - 1:
        avg_i = block_sum[i] / block_count[i]
        avg_next = block_sum[i + 1] / block_count[i + 1]
        if avg_i > avg_next:
            block_sum[i] += block_sum[i + 1]
            block_count[i] += block_count[i + 1]
            block_end[i] = block_end[i + 1]
            del block_sum[i + 1]
            del block_count[i + 1]
            del block_start[i + 1]
            del block_end[i + 1]
            if i > 0:
                i -= 1  # merging may have created a new violation with the PREVIOUS block too
        else:
            i += 1

    fitted = np.empty(len(x_sorted))
    for s, c, start, end in zip(block_sum, block_count, block_start, block_end):
        fitted[start:end + 1] = s / c

    return IsotonicCalibrator(thresholds=x_sorted, values=fitted)


@dataclass
class CalibrationResult:
    """method is 'platt', 'isotonic', or 'none' - whichever won the
    cross-validated comparison. cv_log_loss holds ALL candidates' scores
    (not just the winner's) so the choice is auditable rather than a
    black box - see calibration_summary_table() below for a display-
    ready version of this."""
    method: str
    calibrator: object | None
    cv_log_loss: dict
    n_samples: int


def _kfold_indices(n: int, k: int, seed: int = 42) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    return np.array_split(rng.permutation(n), k)


def select_calibrator(
    probs, outcomes, n_folds: int = CALIBRATION_CV_FOLDS, seed: int = 42,
) -> CalibrationResult:
    """Fits Platt scaling AND isotonic regression on the same (raw
    probability, actual 0/1 outcome) pairs, scores BOTH via k-fold
    cross-validation - never evaluated on the rows used to fit it - and
    returns whichever wins on held-out log-loss, alongside the
    uncorrected baseline ('none'), so a league whose gaps are just noise
    isn't force-fit a correction that would only add variance. The
    winning method is then refit on the FULL dataset (not just the
    folds) for the calibrator actually returned - cross-validation is
    only used to CHOOSE the method, not to build the deployed one, so
    the deployed calibrator gets the benefit of every available row."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(probs)

    if n < CALIBRATION_MIN_SAMPLES:
        return CalibrationResult(method="none", calibrator=None, cv_log_loss={}, n_samples=n)

    folds = _kfold_indices(n, n_folds, seed=seed)
    losses = {"none": [], "platt": [], "isotonic": []}

    for k in range(n_folds):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        if len(train_idx) < 10 or len(test_idx) < 5:
            continue  # a fold this thin would just add noise to the comparison rather than resolve it

        p_train, y_train = probs[train_idx], outcomes[train_idx]
        p_test, y_test = probs[test_idx], outcomes[test_idx]

        losses["none"].append(_binary_log_loss(p_test, y_test))
        losses["platt"].append(_binary_log_loss(fit_platt(p_train, y_train).predict(p_test), y_test))
        losses["isotonic"].append(_binary_log_loss(fit_isotonic(p_train, y_train).predict(p_test), y_test))

    mean_losses = {method: float(np.mean(vals)) for method, vals in losses.items() if vals}
    if not mean_losses:
        return CalibrationResult(method="none", calibrator=None, cv_log_loss={}, n_samples=n)

    best_method = min(mean_losses, key=mean_losses.get)

    if best_method == "platt":
        calibrator = fit_platt(probs, outcomes)
    elif best_method == "isotonic":
        calibrator = fit_isotonic(probs, outcomes)
    else:
        calibrator = None

    return CalibrationResult(method=best_method, calibrator=calibrator, cv_log_loss=mean_losses, n_samples=n)


def build_calibration_training_data(backtest_df: pd.DataFrame, outcome_class: str) -> tuple[np.ndarray, np.ndarray]:
    """outcome_class is 'H', 'D', or 'A'. Returns that class's raw
    walk-forward-backtested probability and a binary 0/1 array for
    whether that class was the actual result, across every backtested
    match - the (probs, outcomes) pair select_calibrator() expects."""
    col = {"H": "p_home", "D": "p_draw", "A": "p_away"}[outcome_class]
    probs = backtest_df[col].to_numpy(dtype=float)
    outcomes = (backtest_df["actual"] == outcome_class).to_numpy(dtype=float)
    return probs, outcomes


def fit_1x2_calibrators(backtest_df: pd.DataFrame) -> dict:
    """{'H': CalibrationResult, 'D': CalibrationResult, 'A': CalibrationResult}
    - an independently-selected calibrator per outcome class, since
    there's no reason the home-win, draw, and away-win probabilities
    would need the same correction shape. Requires walk_forward_backtest()
    output (any include_all_markets setting works - only p_home/p_draw/
    p_away/actual are used here)."""
    return {cls: select_calibrator(*build_calibration_training_data(backtest_df, cls)) for cls in ("H", "D", "A")}


def apply_1x2_calibration(p_home: float, p_draw: float, p_away: float, calibrators: dict) -> tuple[float, float, float]:
    """Applies each class's own selected calibrator (if any) to the raw
    live probability, then renormalises the three corrected numbers so
    they still sum to 1. Independent per-class correction followed by
    renormalisation is the standard way to calibrate a multiclass
    probability vector without needing one joint 3-way model. Classes
    with method='none' (not enough backtest data, or no correction beat
    the uncorrected baseline) pass straight through unchanged."""
    raw = {"H": p_home, "D": p_draw, "A": p_away}
    corrected = {}
    for cls, p in raw.items():
        result = calibrators.get(cls)
        if result is None or result.calibrator is None:
            corrected[cls] = p
        else:
            corrected[cls] = float(result.calibrator.predict(np.array([p]))[0])

    total = sum(corrected.values())
    if total <= 0:
        return p_home, p_draw, p_away  # degenerate fallback - never divide by zero
    return corrected["H"] / total, corrected["D"] / total, corrected["A"] / total


def calibration_summary_table(calibrators: dict) -> pd.DataFrame:
    """Display-ready audit of what fit_1x2_calibrators() actually chose
    and why - which method won each class, its held-out log-loss versus
    the alternatives it beat, and the sample size behind the decision.
    Meant to be shown in the UI next to the odds calibration table itself,
    so which correction (if any) is currently active per league is
    always visible rather than a silent, unlogged choice."""
    rows = []
    for cls, label in (("H", "Home Win"), ("D", "Draw"), ("A", "Away Win")):
        result = calibrators.get(cls)
        if result is None:
            continue
        rows.append({
            "Outcome": label,
            "Method Selected": result.method,
            "N (backtest rows)": result.n_samples,
            "CV Log-Loss (none)": round(result.cv_log_loss.get("none"), 4) if "none" in result.cv_log_loss else None,
            "CV Log-Loss (platt)": round(result.cv_log_loss.get("platt"), 4) if "platt" in result.cv_log_loss else None,
            "CV Log-Loss (isotonic)": round(result.cv_log_loss.get("isotonic"), 4) if "isotonic" in result.cv_log_loss else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration persistence - JSON-safe serialisation so a league's fitted
# calibrators (and the other expensive-to-backtest parameters alongside
# them) can be saved to disk once and reloaded instantly on every later
# run, instead of being refit from a fresh walk-forward backtest every
# time a fixture or league is selected in the UI.
# ---------------------------------------------------------------------------

def serialize_calibration_result(result: CalibrationResult) -> dict:
    """Plain-dict, JSON-safe representation of a fitted CalibrationResult."""
    out = {"method": result.method, "cv_log_loss": result.cv_log_loss, "n_samples": result.n_samples}
    if result.method == "platt" and result.calibrator is not None:
        out["params"] = {"a": result.calibrator.a, "b": result.calibrator.b}
    elif result.method == "isotonic" and result.calibrator is not None:
        out["params"] = {
            "thresholds": result.calibrator.thresholds.tolist(),
            "values": result.calibrator.values.tolist(),
        }
    else:
        out["params"] = None
    return out


def deserialize_calibration_result(data: dict) -> CalibrationResult:
    method = data.get("method", "none")
    params = data.get("params")
    calibrator = None
    if method == "platt" and params:
        calibrator = PlattCalibrator(a=params["a"], b=params["b"])
    elif method == "isotonic" and params:
        calibrator = IsotonicCalibrator(
            thresholds=np.array(params["thresholds"], dtype=float),
            values=np.array(params["values"], dtype=float),
        )
    return CalibrationResult(
        method=method, calibrator=calibrator,
        cv_log_loss=data.get("cv_log_loss", {}), n_samples=data.get("n_samples", 0),
    )


def serialize_1x2_calibrators(calibrators: dict) -> dict:
    return {cls: serialize_calibration_result(result) for cls, result in calibrators.items()}


def deserialize_1x2_calibrators(data: dict) -> dict:
    return {cls: deserialize_calibration_result(d) for cls, d in data.items()}


# ---------------------------------------------------------------------------
# ALL-MARKET calibration - the same select_calibrator() machinery above
# (fit Platt AND isotonic, cross-validate, keep whichever wins, or "none"
# if neither beats the raw baseline), extended from just Home/Draw/Away to
# every one of the 39 markets in MARKET_LIST. Requires backtest_df built
# with walk_forward_backtest(..., include_all_markets=True), which stores
# prob__{market} (the model's raw probability) and actual__{market} (the
# real ground truth, independent of any 50% threshold - see that
# function's docstring) for exactly this purpose.
# ---------------------------------------------------------------------------

def build_market_calibration_training_data(backtest_df: pd.DataFrame, market: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs, outcomes) for ONE market, dropping any row where
    the market didn't apply to that match (actual__{market} is None -
    e.g. Draw No Bet on a match that was actually a draw). This is the
    market-general version of build_calibration_training_data() above."""
    prob_col, actual_col = f"prob__{market}", f"actual__{market}"
    if prob_col not in backtest_df.columns or actual_col not in backtest_df.columns:
        return np.array([]), np.array([])
    subset = backtest_df[[prob_col, actual_col]].dropna()
    if subset.empty:
        return np.array([]), np.array([])
    probs = subset[prob_col].to_numpy(dtype=float)
    outcomes = subset[actual_col].astype(bool).to_numpy(dtype=float)
    return probs, outcomes


def fit_all_market_calibrators(backtest_df: pd.DataFrame) -> dict:
    """{market_name: CalibrationResult} for EVERY market in MARKET_LIST,
    each independently cross-validated (Platt vs. isotonic vs. none) -
    the market-general version of fit_1x2_calibrators() above. Requires
    walk_forward_backtest(..., include_all_markets=True) output."""
    return {market: select_calibrator(*build_market_calibration_training_data(backtest_df, market)) for market in MARKET_LIST}


# Groups of markets that must stay internally consistent (sum to 1, or to
# each other) AFTER calibration - each market inside a group is calibrated
# independently first, then the WHOLE group is renormalised together, so
# calibration can correct each market's own bias without breaking the
# logical relationship between markets in the same group. Markets with no
# complementary partner (Clean Sheet, Win to Nil) are simply left as
# independently-calibrated, standalone probabilities.
CALIBRATION_COMPLEMENTARY_GROUPS = [
    ("Home Win", "Draw", "Away Win"),
    ("Over 1.5 Goals", "Under 1.5 Goals"),
    ("Over 2.5 Goals", "Under 2.5 Goals"),
    ("Over 3.5 Goals", "Under 3.5 Goals"),
    ("BTTS - Yes", "BTTS - No"),
    ("Draw No Bet Home", "Draw No Bet Away"),
    ("Asian Handicap Home -1.5", "Asian Handicap Away +1.5"),
    ("Asian Handicap Home +1.5", "Asian Handicap Away -1.5"),
    tuple(CORRECT_SCORE_MARKETS),
]

# Markets that are never independently calibrated - always re-derived from
# the (already-calibrated) markets they're built from, since they're just
# sums of other rows on the sheet, not their own independent read of the
# match.
DERIVED_MARKETS = {
    "Double Chance 1X": ("Home Win", "Draw"),
    "Double Chance 12": ("Home Win", "Away Win"),
    "Double Chance X2": ("Draw", "Away Win"),
}


def apply_all_market_calibration(predicted_probs: dict, calibrators: dict) -> dict:
    """Applies every market's own independently-selected calibrator
    (Platt/isotonic/none) to a live prediction's raw probabilities, then:
      1. Re-normalises every CALIBRATION_COMPLEMENTARY_GROUPS group so
         calibration never breaks a pair/set that has to sum to 1 (Over/
         Under, BTTS Yes/No, Draw No Bet, the two AH line-pairs, the full
         Home/Draw/Away split, and the 15-way Correct Score set).
      2. Re-derives the 3 Double Chance markets from the now-calibrated
         Home/Draw/Away, since they were never independently calibrated
         in the first place (see DERIVED_MARKETS).

    This is the SINGLE function that must run on dc_probs/mc_probs before
    either reaches build_valuation_sheet() - calibration has to be
    applied and resolved before the summary table is built, not after,
    so every EV/convergence/verdict calculation downstream is already
    working from calibrated numbers rather than needing a second pass."""
    corrected = dict(predicted_probs)
    for market in MARKET_LIST:
        if market in DERIVED_MARKETS:
            continue  # filled in at the end, from the calibrated values of what they're derived from
        result = calibrators.get(market)
        if result is not None and result.calibrator is not None and market in predicted_probs:
            corrected[market] = float(result.calibrator.predict(np.array([predicted_probs[market]]))[0])

    for group in CALIBRATION_COMPLEMENTARY_GROUPS:
        present = [m for m in group if m in corrected]
        total = sum(corrected[m] for m in present)
        if total > 0:
            for m in present:
                corrected[m] = corrected[m] / total

    for derived_market, components in DERIVED_MARKETS.items():
        if all(c in corrected for c in components):
            corrected[derived_market] = sum(corrected[c] for c in components)

    return corrected


def all_market_calibration_summary_table(calibrators: dict) -> pd.DataFrame:
    """Display-ready audit of fit_all_market_calibrators() - one row per
    market, which method won, and its held-out log-loss versus the
    alternatives it beat. The market-general version of
    calibration_summary_table() above."""
    rows = []
    for market in MARKET_LIST:
        result = calibrators.get(market)
        if result is None:
            continue
        rows.append({
            "Market": market,
            "Method Selected": result.method,
            "N (backtest rows)": result.n_samples,
            "CV Log-Loss (none)": round(result.cv_log_loss.get("none"), 4) if "none" in result.cv_log_loss else None,
            "CV Log-Loss (platt)": round(result.cv_log_loss.get("platt"), 4) if "platt" in result.cv_log_loss else None,
            "CV Log-Loss (isotonic)": round(result.cv_log_loss.get("isotonic"), 4) if "isotonic" in result.cv_log_loss else None,
        })
    return pd.DataFrame(rows)


def serialize_all_market_calibrators(calibrators: dict) -> dict:
    return {market: serialize_calibration_result(result) for market, result in calibrators.items()}


def deserialize_all_market_calibrators(data: dict) -> dict:
    return {market: deserialize_calibration_result(d) for market, d in data.items()}


def backtest_accuracy_pct(backtest_df: pd.DataFrame) -> float:
    """% of matches where the model's highest-probability pick (H/D/A)
    matched the actual result."""
    if backtest_df.empty:
        return float("nan")
    return float(100 * backtest_df["correct"].mean())


def apply_manual_override(p_home: float, p_draw: float, p_away: float, override_pct: float) -> tuple[float, float, float]:
    """A manual calibration nudge for sensitivity testing: shifts the
    home-win probability by override_pct percentage points (+/-) and
    rebalances draw/away proportionally so all three still sum to 1.
    override_pct=0 returns the inputs completely unchanged."""
    if override_pct == 0:
        return p_home, p_draw, p_away
    shift = override_pct / 100.0
    new_home = min(max(p_home + shift, 0.0001), 0.9999)
    remaining = 1 - new_home
    old_remaining = p_draw + p_away
    if old_remaining <= 0:
        new_draw = new_away = remaining / 2
    else:
        new_draw = remaining * (p_draw / old_remaining)
        new_away = remaining * (p_away / old_remaining)
    return new_home, new_draw, new_away


def meets_accuracy_floor(accuracy_pct: float, floor_pct: float) -> bool:
    if accuracy_pct is None or (isinstance(accuracy_pct, float) and math.isnan(accuracy_pct)):
        return False
    return accuracy_pct >= floor_pct


def passes_market_gate(confidence_pct: float, ev_pct: float, confidence_floor_pct: float, min_ev_pct: float) -> bool:
    """A market on the 22-row valuation sheet is only 'allowed' if BOTH:
      - its own confidence (the higher of its Dixon-Coles and Monte Carlo
        probability for THAT SPECIFIC market, not the match's outright
        1X2 confidence) clears the floor, AND
      - its EV edge clears the minimum EV gate.
    This is an AND, not an OR - a high-EV pick the model isn't actually
    confident about is exactly the risky case this feature exists to
    catch, so either condition failing blocks the market."""
    return confidence_pct >= confidence_floor_pct and ev_pct >= min_ev_pct


def market_gate_status(confidence_pct: float, ev_pct: float, confidence_floor_pct: float, min_ev_pct: float) -> str:
    """A short, human-readable reason a market is or isn't currently
    allowed - used directly as a displayed column, not just for styling."""
    below_confidence = confidence_pct < confidence_floor_pct
    below_ev = ev_pct < min_ev_pct
    if not below_confidence and not below_ev:
        return "✅ Allowed"
    if below_confidence and below_ev:
        return "⛔ Below Confidence & EV"
    if below_confidence:
        return "⛔ Below Confidence Floor"
    return "⛔ Below EV Gate"


# ---------------------------------------------------------------------------
# Core Parameter B: volatility auto-calibrator
# ---------------------------------------------------------------------------

@dataclass
class VolatilityProfile:
    dispersion_ratio: float
    squad_turnover_index: float
    vol_dampener: float
    adjusted: bool


def compute_volatility_profile(settled_df: pd.DataFrame) -> VolatilityProfile:
    if settled_df.empty:
        return VolatilityProfile(1.0, 0.0, 1.0, False)
    total_goals = (settled_df["home_goals"].fillna(0) + settled_df["away_goals"].fillna(0)).to_numpy(dtype=float)
    mean_goals = float(np.mean(total_goals)) if len(total_goals) else 1.0
    variance_goals = float(np.var(total_goals)) if len(total_goals) else 0.0
    std_goals = float(np.std(total_goals)) if len(total_goals) else 0.0

    dispersion_ratio = variance_goals / mean_goals if mean_goals > 0 else 1.0
    squad_turnover_index = std_goals / max(0.1, mean_goals)

    vol_dampener = 1.0
    adjusted = False
    if squad_turnover_index > DISPERSION_ADJUST_TRIGGER:
        vol_dampener = dispersion_ratio * DISPERSION_ADJUST_FACTOR
        adjusted = True
    else:
        vol_dampener = dispersion_ratio

    return VolatilityProfile(dispersion_ratio, squad_turnover_index, vol_dampener, adjusted)


# ---------------------------------------------------------------------------
# Squad Streak Momentum Tracker
# ---------------------------------------------------------------------------

OPPONENT_QUALITY_MIN_RATIO = 0.5   # floor - even a very weak opponent still earns SOME credit toward a win streak, never zero
OPPONENT_QUALITY_MAX_RATIO = 1.8   # ceiling - guards a single extreme PPG outlier (thin sample) from dominating the whole streak's effect
OPPONENT_QUALITY_MIN_MATCHES = 3   # an opponent's own sample below this is too thin to trust - treated as league-average (ratio 1.0) instead


def _league_average_ppg(all_matches_df: pd.DataFrame) -> float:
    """Points-per-game averaged across every team with a settled match in
    all_matches_df - the baseline every individual opponent's own PPG is
    compared against, to turn 'beat a team' into 'beat a team that was
    X% stronger/weaker than this league's average side'."""
    settled = all_matches_df.dropna(subset=["home_goals", "away_goals"])
    if settled.empty:
        return 1.3  # generic football fallback - close to a typical league-wide mean PPG
    standings = compute_standings_table(settled)
    if standings.empty or standings["Played"].sum() == 0:
        return 1.3
    ppg_per_team = standings["Points"] / standings["Played"].replace(0, np.nan)
    avg = ppg_per_team.mean(skipna=True)
    return float(avg) if pd.notna(avg) and avg > 0 else 1.3


def _team_ppg_asof(all_matches_df: pd.DataFrame, team: str, as_of_date) -> tuple[float | None, int]:
    """This team's points-per-game using only matches strictly BEFORE
    as_of_date - so when this is used to grade a STREAK opponent's
    quality, it reflects what that opponent had actually shown up to
    that point, never hindsight from results that happened afterward."""
    rows = all_matches_df[
        (all_matches_df["home_team"] == team) | (all_matches_df["away_team"] == team)
    ].dropna(subset=["date", "home_goals", "away_goals"])
    if as_of_date is not None:
        rows = rows[rows["date"] < as_of_date]
    if rows.empty:
        return None, 0
    pts = 0
    for _, r in rows.iterrows():
        gf, ga = (r["home_goals"], r["away_goals"]) if r["home_team"] == team else (r["away_goals"], r["home_goals"])
        if gf > ga:
            pts += 3
        elif gf == ga:
            pts += 1
    return pts / len(rows), len(rows)


def team_streak_multiplier(all_matches_df: pd.DataFrame, team: str):
    """Looks at ALL of a team's matches (home and away combined, sorted
    chronologically) to find their CURRENT streak, and returns
    (multiplier, description).

    Each match in the streak is weighted by the OPPONENT'S OWN STRENGTH
    at the time (points-per-game up to that date, relative to the
    league-average PPG), instead of every win/loss counting as the same
    flat +2%/-3% regardless of who it came against. A mid-table team's
    win streak built entirely against bottom-table sides now produces a
    smaller boost than the same length streak that included wins over
    top-table opposition - the model no longer treats "5 wins" as
    interchangeable regardless of opposition quality, which is exactly
    the gap flat per-match increments had: a streak's real predictive
    value depends on WHO it came against, not just how long it is.

    Opponents with too thin a sample of their own (< OPPONENT_QUALITY_MIN_MATCHES)
    are treated as league-average (ratio 1.0) rather than trusted on a
    noisy PPG - same shrink-to-baseline instinct used everywhere else in
    this engine for small samples. The quality ratio is also clamped to
    [OPPONENT_QUALITY_MIN_RATIO, OPPONENT_QUALITY_MAX_RATIO] so one
    extreme opponent can't single-handedly dominate a multi-match streak's
    effect.

    Base per-match rates (2% for a win, 3% for a loss, at league-average
    opponent quality) and the overall 0.88-1.12 cap are unchanged from
    the previous flat version - so a streak built entirely against
    average opposition produces almost exactly the same multiplier as
    before; what changes is everything AWAY from average opposition."""
    rows = all_matches_df[
        (all_matches_df["home_team"] == team) | (all_matches_df["away_team"] == team)
    ].dropna(subset=["date"]).sort_values("date")
    if rows.empty:
        return 1.0, "no data"

    results = []
    for _, r in rows.iterrows():
        if r["home_team"] == team:
            gf, ga, opponent = r["home_goals"], r["away_goals"], r["away_team"]
        else:
            gf, ga, opponent = r["away_goals"], r["home_goals"], r["home_team"]
        if pd.isna(gf) or pd.isna(ga):
            continue
        outcome = "W" if gf > ga else ("L" if gf < ga else "D")
        results.append({"result": outcome, "opponent": opponent, "date": r["date"]})

    if not results:
        return 1.0, "no settled matches"

    last = results[-1]["result"]
    if last not in ("W", "L"):
        return 1.0, "last match was a draw"

    streak_matches = []
    for r in reversed(results):
        if r["result"] == last:
            streak_matches.append(r)
        else:
            break

    streak = len(streak_matches)
    if streak < 2:
        return 1.0, f"streak of {streak} - below the 2-match trigger"

    league_avg_ppg = _league_average_ppg(all_matches_df)
    quality_ratios = []
    for match in streak_matches:
        opp_ppg, opp_n = _team_ppg_asof(all_matches_df, match["opponent"], match["date"])
        if opp_ppg is None or opp_n < OPPONENT_QUALITY_MIN_MATCHES or league_avg_ppg <= 0:
            ratio = 1.0
        else:
            ratio = np.clip(opp_ppg / league_avg_ppg, OPPONENT_QUALITY_MIN_RATIO, OPPONENT_QUALITY_MAX_RATIO)
        quality_ratios.append(float(ratio))
    avg_quality_ratio = float(np.mean(quality_ratios))

    if last == "W":
        # Beating stronger-than-average sides scales the boost UP;
        # beating weaker-than-average sides scales it DOWN.
        total_effect = sum(0.02 * ratio for ratio in quality_ratios)
        mult = min(1.0 + total_effect, 1.12)
        quality_note = "above-average" if avg_quality_ratio > 1.05 else ("below-average" if avg_quality_ratio < 0.95 else "average")
        return mult, (
            f"{streak}-match win streak vs. {quality_note} opposition "
            f"(avg opponent strength {avg_quality_ratio:.2f}x league PPG, +{(mult - 1) * 100:.1f}%)"
        )
    else:
        # Losing to stronger-than-average sides is less damning (smaller
        # penalty); losing to weaker-than-average sides is scaled UP
        # (bigger penalty) - a losing streak against bottom-table teams
        # should hurt more than the same streak against the league's best.
        total_effect = sum(0.03 * (2.0 - ratio) for ratio in quality_ratios)
        mult = max(1.0 - total_effect, 0.88)
        quality_note = "above-average" if avg_quality_ratio > 1.05 else ("below-average" if avg_quality_ratio < 0.95 else "average")
        return mult, (
            f"{streak}-match losing streak vs. {quality_note} opposition "
            f"(avg opponent strength {avg_quality_ratio:.2f}x league PPG, {(mult - 1) * 100:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Section 6: tactical & environmental multipliers
# ---------------------------------------------------------------------------

@dataclass
class TeamAdjustment:
    attack: float = 1.0
    defense: float = 1.0  # multiplicative on the "allowed" ratio - >1 makes defense WORSE


@dataclass
class TacticalInputs:
    home_newly_relegated: bool = False
    away_newly_relegated: bool = False
    home_relegation_threat: bool = False
    away_relegation_threat: bool = False
    # Injuries are now split by role - see the realism note above
    # apply_tactical_multipliers for why a striker injury and a defender
    # injury are no longer the same checkbox.
    home_striker_injury: bool = False
    away_striker_injury: bool = False
    home_defender_injury: bool = False
    away_defender_injury: bool = False
    home_bogey: bool = False
    away_bogey: bool = False
    home_new_manager: bool = False
    away_new_manager: bool = False
    home_boardroom_crisis: bool = False
    away_boardroom_crisis: bool = False
    home_dead_rubber: bool = False
    away_dead_rubber: bool = False
    home_travel_fatigue_units: int = 0  # 0-3, applies to the AWAY team traveling to home team's ground
    host_travel_fatigue_units: int = 0  # 0-3, the HOST's own midweek travel fatigue (e.g. a midweek away European leg) carried into THIS home fixture
    coastal_shock: bool = False  # applies to the traveling (away) team
    home_cup_distraction: bool = False
    away_cup_distraction: bool = False
    # New selectors
    home_tactical_setup: str = "Standard Open Play"   # or "Deep Ultra-Defensive Low-Block" / "High-Intensity Counter-Pressing Style"
    away_tactical_setup: str = "Standard Open Play"
    pitch_surface: str = "Standard Optimized Turf"     # or "Waterlogged Mud" / "Dry Uneven Grass, short and narrow"
    weather: str = "Clear Sky / Ideal Climate"          # or "Torrential Rain Storm" / "Gale-Force Wind Interference"
    referee_strictness: str = "Standard Average"        # or "Lenient (Flow Enforcer)" / "Hyper-Strict (Card Trigger)"
    pre_season_fixture: bool = False
    # Manually-set squad transfer impact - see realism note in
    # apply_tactical_multipliers for why this is a slider, not a fixed
    # constant like the other toggles.
    home_transfer_impact_pct: float = 0.0   # positive = a quality signing arrived; negative = a key player departed
    away_transfer_impact_pct: float = 0.0


def apply_tactical_multipliers(
    home_attack: float, home_defense: float, away_attack: float, away_defense: float,
    base_volatility: float, tactics: TacticalInputs,
):
    """Returns (home_adjustment, away_adjustment, adjusted_volatility, log).
    Every multiplier here only scales the INPUT rates - see design note
    #3 at the top of the file; nothing here sets a probability directly.

    REALISM NOTE (checked against actual research, not assumed):
    - New manager bounce: Premier League data (2021/22-2025/26, 35
      mid-season appointments) shows clubs jumping from ~0.90 to ~1.27
      points per game in the first 5 games under a new manager - a ~41%
      relative swing. That number is intentionally NOT applied at full
      strength here: most of it is regression to the mean (clubs sack
      managers exactly when results are at their worst, so ANY manager
      would see some bounce-back), plus small-sample noise that fades by
      games 11-20. A conservative 10% attack/defense bump is kept as a
      defensible middle estimate of the sustained part of the effect,
      not the full raw PPG swing.
    - Striker vs defender injuries are now DIFFERENT, not the same
      checkbox: losing a primary goal-scorer is a fairly predictable,
      attack-specific quality loss. Losing a first-choice center-back
      tends to show up more as increased defensive VARIANCE (makeshift
      back-lines make more individual errors) on top of a quality loss -
      this matches the general direction of the injury-performance
      literature (Hägglund et al. 2013, BJSM; and follow-up Bundesliga
      cost studies) even though no single published number cleanly
      separates "striker" vs "defender" effect size - these remain
      reasoned estimates, not a proven precise coefficient.
    - Key player transfer impact (signing arrived / departed) is
      DELIBERATELY a manual slider, not a fixed checkbox constant like
      the others above. Unlike a manager change (which has multi-season
      league-wide PPG data to check against), a transfer's real impact
      depends enormously on the specific player's quality, position, and
      how good their replacement is - there's no single defensible
      universal percentage the way there arguably is for the other
      toggles. Forcing a fixed number here would be less honest, not
      more precise, so this is left to your own judgement per case.
    """
    home = TeamAdjustment(home_attack, home_defense)
    away = TeamAdjustment(away_attack, away_defense)
    vol = base_volatility
    log = []

    def general_decline_or_boost(adj: TeamAdjustment, factor: float, label: str, side: str):
        adj.attack *= factor
        adj.defense /= factor  # factor<1 (decline) -> defense worsens; factor>1 (boost) -> defense improves
        log.append(f"{side}: {label} -> attack x{factor:.2f}, defense x{1/factor:.2f}")

    if tactics.home_newly_relegated:
        general_decline_or_boost(home, 0.90, "Newly relegated", "Home")
    if tactics.away_newly_relegated:
        general_decline_or_boost(away, 0.90, "Newly relegated", "Away")

    if tactics.home_relegation_threat:
        home.defense /= 1.08
        log.append("Home: Live relegation threat -> defense x0.926 (+8% grit)")
    if tactics.away_relegation_threat:
        away.defense /= 1.08
        log.append("Away: Live relegation threat -> defense x0.926 (+8% grit)")

    # --- Injuries, now split by role (see realism note above) ---
    if tactics.home_striker_injury:
        home.attack *= 0.88
        vol *= 1.03
        log.append("Home: Key striker/attacker out -> attack x0.88, volatility x1.03")
    if tactics.away_striker_injury:
        away.attack *= 0.88
        vol *= 1.03
        log.append("Away: Key striker/attacker out -> attack x0.88, volatility x1.03")
    if tactics.home_defender_injury:
        home.defense /= 0.90  # defense WORSENS ~11% - a makeshift back-line concedes more
        vol *= 1.08
        log.append("Home: Key defender out -> defense x1.11 (worse), volatility x1.08")
    if tactics.away_defender_injury:
        away.defense /= 0.90
        vol *= 1.08
        log.append("Away: Key defender out -> defense x1.11 (worse), volatility x1.08")

    # --- Key player transfer impact (manual slider - see realism note above) ---
    if tactics.home_transfer_impact_pct != 0:
        factor = max(0.01, 1 + tactics.home_transfer_impact_pct / 100)
        general_decline_or_boost(
            home, factor,
            f"Squad transfer impact ({tactics.home_transfer_impact_pct:+.0f}%, manually set)",
            "Home",
        )
    if tactics.away_transfer_impact_pct != 0:
        factor = max(0.01, 1 + tactics.away_transfer_impact_pct / 100)
        general_decline_or_boost(
            away, factor,
            f"Squad transfer impact ({tactics.away_transfer_impact_pct:+.0f}%, manually set)",
            "Away",
        )

    if tactics.home_bogey:
        general_decline_or_boost(home, 0.95, "Historical bogey hex", "Home")
    if tactics.away_bogey:
        general_decline_or_boost(away, 0.95, "Historical bogey hex", "Away")

    if tactics.home_new_manager:
        general_decline_or_boost(home, 1.10, "New manager bounce", "Home")
    if tactics.away_new_manager:
        general_decline_or_boost(away, 1.10, "New manager bounce", "Away")

    if tactics.home_boardroom_crisis:
        general_decline_or_boost(home, 0.85, "Boardroom crisis", "Home")
    if tactics.away_boardroom_crisis:
        general_decline_or_boost(away, 0.85, "Boardroom crisis", "Away")

    if tactics.home_dead_rubber:
        general_decline_or_boost(home, 0.90, "Dead rubber / beach mode", "Home")
        vol *= 0.90
    if tactics.away_dead_rubber:
        general_decline_or_boost(away, 0.90, "Dead rubber / beach mode", "Away")
        vol *= 0.90

    if tactics.home_travel_fatigue_units:
        factor = max(0.01, 1 - 0.04 * tactics.home_travel_fatigue_units)
        away.attack *= factor  # the AWAY team is the one traveling to the home team's ground
        log.append(f"Away: Travel fatigue x{tactics.home_travel_fatigue_units} unit(s) -> attack x{factor:.2f}")

    if tactics.host_travel_fatigue_units:
        factor = max(0.01, 1 - 0.04 * tactics.host_travel_fatigue_units)
        home.attack *= factor  # the HOST's own midweek travel, independent of this match's venue
        log.append(f"Home: Travel fatigue x{tactics.host_travel_fatigue_units} unit(s) -> attack x{factor:.2f}")

    if tactics.coastal_shock:
        away.attack *= 0.95
        vol *= 0.92
        log.append("Away: Coastal humidity shock -> attack x0.95, volatility x0.92")

    # --- Tactical setup selector (replaces the old low-block-only checkbox) ---
    if tactics.home_tactical_setup == "Deep Ultra-Defensive Low-Block":
        home.attack *= 0.85
        vol *= 0.82
        log.append("Home: Deep low-block -> attack x0.85, volatility x0.82")
    elif tactics.home_tactical_setup == "High-Intensity Counter-Pressing Style":
        # Not specified in the original spec - a reasoned estimate: winning
        # the ball back higher up creates more transition chances (attack
        # up) but also more end-to-end chaos (volatility up).
        home.attack *= 1.06
        vol *= 1.05
        log.append("Home: High-intensity counter-press -> attack x1.06, volatility x1.05")
    if tactics.away_tactical_setup == "Deep Ultra-Defensive Low-Block":
        away.attack *= 0.85
        vol *= 0.82
        log.append("Away: Deep low-block -> attack x0.85, volatility x0.82")
    elif tactics.away_tactical_setup == "High-Intensity Counter-Pressing Style":
        away.attack *= 1.06
        vol *= 1.05
        log.append("Away: High-intensity counter-press -> attack x1.06, volatility x1.05")

    # --- Pitch surface (affects BOTH teams equally - it's the same pitch) ---
    if tactics.pitch_surface == "Waterlogged Mud":
        home.attack *= 0.90
        away.attack *= 0.90
        vol *= 1.10
        log.append("Both: Waterlogged mud pitch -> attack x0.90 each, volatility x1.10")
    elif tactics.pitch_surface == "Dry Uneven Grass, short and narrow":
        home.attack *= 0.95
        away.attack *= 0.95
        vol *= 1.05
        log.append("Both: Dry uneven/narrow pitch -> attack x0.95 each, volatility x1.05")

    # --- Weather (affects BOTH teams equally) ---
    if tactics.weather == "Torrential Rain Storm":
        home.attack *= 0.92
        away.attack *= 0.92
        vol *= 1.12
        log.append("Both: Torrential rain -> attack x0.92 each, volatility x1.12")
    elif tactics.weather == "Gale-Force Wind Interference":
        home.attack *= 0.88
        away.attack *= 0.88
        vol *= 1.15
        log.append("Both: Gale-force wind -> attack x0.88 each, volatility x1.15")

    # --- Referee strictness (affects match chaos/variance, not attack directly) ---
    if tactics.referee_strictness == "Hyper-Strict (Card Trigger)":
        vol *= 1.15
        log.append("Match: Hyper-strict referee -> volatility x1.15 (per spec)")
    elif tactics.referee_strictness == "Lenient (Flow Enforcer)":
        vol *= 0.90
        log.append("Match: Lenient referee -> volatility x0.90 (reasoned estimate - not specified in original spec)")

    # --- Pre-season fixture (universal, both teams) ---
    if tactics.pre_season_fixture:
        home.attack *= 0.90
        away.attack *= 0.90
        log.append("Both: Pre-season fixture -> attack x0.90 each (stamina/sharpness penalty)")

    if tactics.home_cup_distraction:
        general_decline_or_boost(home, 0.88, "Look-ahead cup penalty", "Home")
    if tactics.away_cup_distraction:
        general_decline_or_boost(away, 0.88, "Look-ahead cup penalty", "Away")

    # --- Stacking realism clamp ---
    # With this many independent toggles, an extreme (if unlikely) combo -
    # e.g. relegated + striker injury + bogey + boardroom crisis + dead
    # rubber + low-block + cup distraction + a bad pitch + bad weather +
    # pre-season, all at once - multiplies out to roughly a 0.31x combined
    # attack factor (verified directly: 0.90*0.88*0.95*0.85*0.90*0.85*
    # 0.88*0.90*0.88*0.90 ≈ 0.307). No real single match swings a team's
    # underlying quality by ~70%, no matter how many bad things coincide -
    # that's several standard deviations beyond anything in the injury/
    # situational-factors literature this file's realism notes are based
    # on. This clamps the COMBINED RATIO applied by tactical multipliers
    # (relative to the real, data-derived baseline that came in), not the
    # baseline itself - so a genuinely strong team can still clearly
    # outperform a genuinely weak one; this only stops the SITUATIONAL
    # layer from compounding into an implausible extreme.
    COMBINED_RATIO_FLOOR = 0.55
    COMBINED_RATIO_CEILING = 1.55

    def clamp_ratio(adjusted_value, original_value, side_label, metric_label):
        if original_value <= 0:
            return adjusted_value
        ratio = adjusted_value / original_value
        clamped_ratio = float(np.clip(ratio, COMBINED_RATIO_FLOOR, COMBINED_RATIO_CEILING))
        if abs(clamped_ratio - ratio) > 1e-9:
            log.append(
                f"{side_label}: combined {metric_label} multiplier of x{ratio:.2f} "
                f"clamped to x{clamped_ratio:.2f} (realism cap - see stacking note)"
            )
            return original_value * clamped_ratio
        return adjusted_value

    home.attack = clamp_ratio(home.attack, home_attack, "Home", "attack")
    home.defense = clamp_ratio(home.defense, home_defense, "Home", "defense")
    away.attack = clamp_ratio(away.attack, away_attack, "Away", "attack")
    away.defense = clamp_ratio(away.defense, away_defense, "Away", "defense")
    vol = float(np.clip(vol, base_volatility * 0.5, base_volatility * 1.8))

    return home, away, vol, log


# ---------------------------------------------------------------------------
# Section 5, Engine A: Dixon-Coles Poisson model with data-fitted rho
# ---------------------------------------------------------------------------

def fit_rho(settled_df: pd.DataFrame) -> float:
    """Fits rho by grid search: pick the rho that makes the tau-adjusted
    Poisson-Poisson grid's predicted frequency of the four low-score
    cells (0-0, 1-0, 0-1, 1-1) match what's ACTUALLY observed in this
    league's settled matches, rather than assuming a fixed constant."""
    if settled_df.empty or len(settled_df) < MIN_SAMPLE_ROWS:
        return 0.0

    lam_h = float(settled_df["home_goals"].mean())
    lam_a = float(settled_df["away_goals"].mean())
    if lam_h <= 0 or lam_a <= 0:
        return 0.0

    total = len(settled_df)
    observed = {
        (0, 0): float(((settled_df["home_goals"] == 0) & (settled_df["away_goals"] == 0)).sum()) / total,
        (1, 0): float(((settled_df["home_goals"] == 1) & (settled_df["away_goals"] == 0)).sum()) / total,
        (0, 1): float(((settled_df["home_goals"] == 0) & (settled_df["away_goals"] == 1)).sum()) / total,
        (1, 1): float(((settled_df["home_goals"] == 1) & (settled_df["away_goals"] == 1)).sum()) / total,
    }

    best_rho, best_error = 0.0, float("inf")
    for rho_candidate in np.arange(-0.25, 0.251, 0.01):
        error = 0.0
        for (x, y), obs_p in observed.items():
            base = scipy_poisson.pmf(x, lam_h) * scipy_poisson.pmf(y, lam_a)
            pred_p = base * dixon_coles_tau(x, y, lam_h, lam_a, rho_candidate)
            error += (pred_p - obs_p) ** 2
        if error < best_error:
            best_error = error
            best_rho = float(rho_candidate)
    return best_rho


def dixon_coles_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def build_score_matrix(lam_h: float, lam_a: float, rho: float, goal_cap: int = GOAL_CAP) -> np.ndarray:
    """Builds the Dixon-Coles-adjusted score matrix and normalizes it so
    the whole grid sums to exactly 1.0 (100%)."""
    matrix = np.zeros((goal_cap + 1, goal_cap + 1))
    for x in range(goal_cap + 1):
        px = scipy_poisson.pmf(x, lam_h)
        for y in range(goal_cap + 1):
            py = scipy_poisson.pmf(y, lam_a)
            matrix[x, y] = px * py * dixon_coles_tau(x, y, lam_h, lam_a, rho)
    matrix = np.clip(matrix, 0, None)
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total
    return matrix


# ---------------------------------------------------------------------------
# Section 5, Engine B: 10,000-iteration Monte Carlo simulator
# ---------------------------------------------------------------------------

def monte_carlo_simulate(
    lam_h: float, lam_a: float, volatility_dampener: float = 1.0, iterations: int = MC_ITERATIONS,
    rng=None,
):
    """Draws directly from the raw expected-goal rates via
    np.random.poisson for `iterations` mock matches - deliberately
    bypassing the Dixon-Coles matrix entirely (per spec: "bypasses the
    post-processed matrix entirely to prevent flattening errors"). A
    volatility_dampener != 1.0 jitters the lambda per-simulation (rather
    than the goal draw itself) via a clipped normal multiplier, which is
    how the Section 6 volatility-affecting toggles and Core Parameter B's
    auto-calibrated dampener actually widen or narrow the simulated
    spread of results."""
    rng = rng or np.random.default_rng()
    noise_std = max(0.0, (volatility_dampener - 1.0)) + 0.10  # always some baseline match-to-match noise
    home_jitter = np.clip(rng.normal(1.0, noise_std, size=iterations), 0.05, None)
    away_jitter = np.clip(rng.normal(1.0, noise_std, size=iterations), 0.05, None)
    home_goals = rng.poisson(lam_h * home_jitter)
    away_goals = rng.poisson(lam_a * away_jitter)
    return home_goals, away_goals


# ---------------------------------------------------------------------------
# Section 8: 22-market probability extraction
# ---------------------------------------------------------------------------

MARKET_LIST = [
    "Home Win", "Draw", "Away Win",
    "Double Chance 1X", "Double Chance 12", "Double Chance X2",
    "Draw No Bet Home", "Draw No Bet Away",
    "Over 1.5 Goals", "Under 1.5 Goals",
    "Over 2.5 Goals", "Under 2.5 Goals",
    "Over 3.5 Goals", "Under 3.5 Goals",
    "BTTS - Yes", "BTTS - No",
    "Home Clean Sheet", "Away Clean Sheet",
    "Home Win to Nil", "Away Win to Nil",
    "Asian Handicap Home -1.5", "Asian Handicap Away +1.5",
    "Asian Handicap Home +1.5", "Asian Handicap Away -1.5",
] + CORRECT_SCORE_MARKETS
assert len(MARKET_LIST) == 22 + 2 + 15  # original 22 + Draw No Bet (2) + Correct Score (14 named + Other)


def market_probs_from_matrix(matrix: np.ndarray) -> dict:
    """Every market probability computed by literally summing the
    correct cells of the real Dixon-Coles grid - no shortcuts."""
    size = matrix.shape[0]
    idx = np.arange(size)
    home_grid, away_grid = np.meshgrid(idx, idx, indexing="ij")

    home_win = matrix[home_grid > away_grid].sum()
    draw = matrix[home_grid == away_grid].sum()
    away_win = matrix[home_grid < away_grid].sum()

    total_goals = home_grid + away_grid
    over = {t: matrix[total_goals > t].sum() for t in (1.5, 2.5, 3.5)}
    under = {t: matrix[total_goals < t].sum() for t in (1.5, 2.5, 3.5)}

    btts_yes = matrix[(home_grid >= 1) & (away_grid >= 1)].sum()
    btts_no = 1 - btts_yes

    home_clean_sheet = matrix[away_grid == 0].sum()
    away_clean_sheet = matrix[home_grid == 0].sum()

    home_win_to_nil = matrix[(home_grid > away_grid) & (away_grid == 0)].sum()
    away_win_to_nil = matrix[(away_grid > home_grid) & (home_grid == 0)].sum()

    ah_home_minus_1_5 = matrix[(home_grid - away_grid) > 1.5].sum()
    ah_away_plus_1_5 = 1 - ah_home_minus_1_5
    ah_away_minus_1_5 = matrix[(away_grid - home_grid) > 1.5].sum()
    ah_home_plus_1_5 = 1 - ah_away_minus_1_5

    # Draw No Bet - conditional on the match NOT being a draw (the stake
    # is void, not lost, on an actual draw), so these are home_win/away_win
    # renormalised over just the non-draw probability mass.
    decisive_mass = home_win + away_win
    dnb_home = home_win / decisive_mass if decisive_mass > 0 else 0.5
    dnb_away = away_win / decisive_mass if decisive_mass > 0 else 0.5

    # Correct Score - literal cells of the real grid for a fixed set of
    # common scorelines, plus "Other" catching every scoreline outside
    # that set (still sums to exactly 1 across the whole group).
    correct_score_probs = {}
    named_total = 0.0
    for h, a in CORRECT_SCORE_GRID:
        p = float(matrix[h, a]) if h < size and a < size else 0.0
        correct_score_probs[f"Correct Score {h}-{a}"] = p
        named_total += p
    correct_score_probs["Correct Score Other"] = max(0.0, 1.0 - named_total)

    return {
        "Home Win": home_win, "Draw": draw, "Away Win": away_win,
        "Double Chance 1X": home_win + draw,
        "Double Chance 12": home_win + away_win,
        "Double Chance X2": draw + away_win,
        "Draw No Bet Home": dnb_home, "Draw No Bet Away": dnb_away,
        "Over 1.5 Goals": over[1.5], "Under 1.5 Goals": under[1.5],
        "Over 2.5 Goals": over[2.5], "Under 2.5 Goals": under[2.5],
        "Over 3.5 Goals": over[3.5], "Under 3.5 Goals": under[3.5],
        "BTTS - Yes": btts_yes, "BTTS - No": btts_no,
        "Home Clean Sheet": home_clean_sheet, "Away Clean Sheet": away_clean_sheet,
        "Home Win to Nil": home_win_to_nil, "Away Win to Nil": away_win_to_nil,
        "Asian Handicap Home -1.5": ah_home_minus_1_5, "Asian Handicap Away +1.5": ah_away_plus_1_5,
        "Asian Handicap Home +1.5": ah_home_plus_1_5, "Asian Handicap Away -1.5": ah_away_minus_1_5,
        **correct_score_probs,
    }


def market_probs_from_simulation(home_goals: np.ndarray, away_goals: np.ndarray) -> dict:
    """Same 22 markets, computed as raw frequencies across the Monte
    Carlo simulation array - an entirely independent probability path
    from the matrix above, which is what makes the convergence score a
    genuine cross-check rather than comparing a method to itself."""
    home_win = np.mean(home_goals > away_goals)
    draw = np.mean(home_goals == away_goals)
    away_win = np.mean(home_goals < away_goals)
    total_goals = home_goals + away_goals

    over = {t: np.mean(total_goals > t) for t in (1.5, 2.5, 3.5)}
    under = {t: np.mean(total_goals < t) for t in (1.5, 2.5, 3.5)}

    btts_yes = np.mean((home_goals >= 1) & (away_goals >= 1))
    btts_no = 1 - btts_yes

    home_clean_sheet = np.mean(away_goals == 0)
    away_clean_sheet = np.mean(home_goals == 0)

    home_win_to_nil = np.mean((home_goals > away_goals) & (away_goals == 0))
    away_win_to_nil = np.mean((away_goals > home_goals) & (home_goals == 0))

    ah_home_minus_1_5 = np.mean((home_goals - away_goals) > 1.5)
    ah_away_plus_1_5 = 1 - ah_home_minus_1_5
    ah_away_minus_1_5 = np.mean((away_goals - home_goals) > 1.5)
    ah_home_plus_1_5 = 1 - ah_away_minus_1_5

    decisive_mask = home_goals != away_goals
    decisive_count = int(np.sum(decisive_mask))
    dnb_home = float(np.mean(home_goals[decisive_mask] > away_goals[decisive_mask])) if decisive_count > 0 else 0.5
    dnb_away = 1 - dnb_home if decisive_count > 0 else 0.5

    correct_score_probs = {}
    named_total = 0.0
    for h, a in CORRECT_SCORE_GRID:
        p = float(np.mean((home_goals == h) & (away_goals == a)))
        correct_score_probs[f"Correct Score {h}-{a}"] = p
        named_total += p
    correct_score_probs["Correct Score Other"] = max(0.0, 1.0 - named_total)

    return {
        "Home Win": home_win, "Draw": draw, "Away Win": away_win,
        "Double Chance 1X": home_win + draw,
        "Double Chance 12": home_win + away_win,
        "Double Chance X2": draw + away_win,
        "Draw No Bet Home": dnb_home, "Draw No Bet Away": dnb_away,
        "Over 1.5 Goals": over[1.5], "Under 1.5 Goals": under[1.5],
        "Over 2.5 Goals": over[2.5], "Under 2.5 Goals": under[2.5],
        "Over 3.5 Goals": over[3.5], "Under 3.5 Goals": under[3.5],
        "BTTS - Yes": btts_yes, "BTTS - No": btts_no,
        "Home Clean Sheet": home_clean_sheet, "Away Clean Sheet": away_clean_sheet,
        "Home Win to Nil": home_win_to_nil, "Away Win to Nil": away_win_to_nil,
        "Asian Handicap Home -1.5": ah_home_minus_1_5, "Asian Handicap Away +1.5": ah_away_plus_1_5,
        "Asian Handicap Home +1.5": ah_home_plus_1_5, "Asian Handicap Away -1.5": ah_away_minus_1_5,
        **correct_score_probs,
    }


# ---------------------------------------------------------------------------
# Section 8: EV, convergence, fair odds, verdicts, recommended action
# ---------------------------------------------------------------------------

VOLATILITY_LOW, VOLATILITY_HIGH = 0.85, 1.15
EV_ELITE_THRESHOLD = 0.03  # +3.0%


def convergence_score(p_dc: float, p_mc: float) -> float:
    """1.0 = the two engines agree perfectly, 0.0 = maximally apart."""
    return max(0.0, 1.0 - abs(p_dc - p_mc) * 2)


def fair_odds(p_dc: float, p_mc: float) -> float:
    best_p = max(p_dc, p_mc)
    if best_p <= 0:
        return float("inf")
    return 1.0 / best_p


def expected_value(model_prob: float, bookmaker_odds: float) -> float:
    return model_prob * bookmaker_odds - 1.0


def volatility_tier(vol_dampener: float) -> str:
    if vol_dampener < VOLATILITY_LOW:
        return "Low Chaos"
    if vol_dampener > VOLATILITY_HIGH:
        return "High Chaos"
    return "Balanced"


def value_verdict(ev: float) -> str:
    return "🔥 ELITE VALUE" if ev > EV_ELITE_THRESHOLD else "⚠️ HIGH-JUICE TRAP"


def recommended_action(ev: float, convergence: float, tier: str) -> str:
    """A rule-based synthesis of edge size, engine agreement, and
    volatility - never a market probability itself, just a plain-English
    action label built on top of numbers that were already computed."""
    if ev <= 0:
        return "🚫 AVOID - NO EDGE"
    if convergence < 0.5:
        return "❓ LOW CONFIDENCE - ENGINES DISAGREE"
    if ev > EV_ELITE_THRESHOLD and convergence >= 0.75:
        if tier == "High Chaos":
            return "🔥 STRONG BET (small stake - high chaos)"
        return "🔥 STRONG BET"
    if ev > EV_ELITE_THRESHOLD:
        return "✅ VALUE BET"
    return "🟡 MARGINAL - SMALL STAKE ONLY"


@dataclass
class MarketRow:
    market: str
    bookmaker_odds: float
    dc_prob: float
    mc_prob: float
    convergence: float
    fair_odds: float
    ev: float
    volatility_tier: str
    verdict: str
    recommended_action: str


def build_valuation_sheet(
    dc_probs: dict, mc_probs: dict, bookmaker_odds: dict, vol_dampener: float,
):
    tier = volatility_tier(vol_dampener)
    rows = []
    for market in MARKET_LIST:
        p_dc = dc_probs.get(market, 0.0)
        p_mc = mc_probs.get(market, 0.0)
        odds = bookmaker_odds.get(market, 0.0)
        conv = convergence_score(p_dc, p_mc)
        f_odds = fair_odds(p_dc, p_mc)
        ev = expected_value(max(p_dc, p_mc), odds) if odds > 0 else float("-inf")
        rows.append(MarketRow(
            market=market, bookmaker_odds=odds, dc_prob=p_dc, mc_prob=p_mc,
            convergence=conv, fair_odds=f_odds, ev=ev, volatility_tier=tier,
            verdict=value_verdict(ev) if odds > 0 else "-",
            recommended_action=recommended_action(ev, conv, tier) if odds > 0 else "ENTER ODDS",
        ))
    return rows


# ---------------------------------------------------------------------------
# Kelly Criterion & parlay combination
# ---------------------------------------------------------------------------

def kelly_stake_fraction(model_prob: float, bookmaker_odds: float, kelly_multiplier: float) -> float:
    b = bookmaker_odds - 1.0
    if b <= 0:
        return 0.0
    edge = model_prob * bookmaker_odds - 1.0
    full_kelly = edge / b
    return max(0.0, full_kelly) * kelly_multiplier


def round_to_nearest(amount: float, denomination: float = 10.0) -> float:
    return round(amount / denomination) * denomination


def _leg_value(leg, name):
    """Supports either a MarketRow object (attribute access) or a plain
    dict with the same field names - the cross-league bet slip stores
    legs as dicts (since they need to persist after you've moved on to a
    different fixture/league, long after the original MarketRow objects
    for that match are gone), while the single-fixture parlay builder
    still uses MarketRow objects directly."""
    return getattr(leg, name) if hasattr(leg, name) else leg[name]


def combine_parlay_legs(legs):
    """Returns (combined_odds, combined_model_probability). Works with a
    mix of MarketRow objects and/or plain dicts, in any combination."""
    combined_odds = 1.0
    combined_prob = 1.0
    for leg in legs:
        combined_odds *= _leg_value(leg, "bookmaker_odds")
        combined_prob *= max(_leg_value(leg, "dc_prob"), _leg_value(leg, "mc_prob"))
    return combined_odds, combined_prob


# ---------------------------------------------------------------------------
# Section 10: Deserved Points (xPts) and season Monte Carlo forecast
# ---------------------------------------------------------------------------

BOX_TOUCH_WEIGHT = 0.015


def compute_standings_table(settled_df: pd.DataFrame) -> pd.DataFrame:
    """The classic, literal league table - Played/Won/Drawn/Lost/GF/GA/
    GD/Points, computed directly from real match results. NOT the same
    thing as compute_xpts_table() below, which is model-based (deserved
    points from territory dominance) - this is the actual, real
    standings, with the standard tiebreak order (Points, then GD, then
    GF). This is the table that gets archived at end of season - see
    the app's season-archive functions, which snapshot exactly this
    table's output alongside a season label and date range, so a
    league's final table stays retrievable even after new-season
    matches start accumulating in the same underlying dataset."""
    columns = ["Position", "Team", "Played", "Won", "Drawn", "Lost", "GF", "GA", "GD", "Points"]
    if settled_df.empty:
        return pd.DataFrame(columns=columns)

    records: dict[str, dict] = {}

    def ensure(team: str) -> dict:
        if team not in records:
            records[team] = {"Team": team, "Played": 0, "Won": 0, "Drawn": 0, "Lost": 0, "GF": 0, "GA": 0}
        return records[team]

    for _, row in settled_df.iterrows():
        home, away = row.get("home_team"), row.get("away_team")
        hg, ag = row.get("home_goals"), row.get("away_goals")
        if pd.isna(hg) or pd.isna(ag) or not isinstance(home, str) or not isinstance(away, str):
            continue
        hg, ag = int(hg), int(ag)
        h, a = ensure(home), ensure(away)
        h["Played"] += 1
        a["Played"] += 1
        h["GF"] += hg
        h["GA"] += ag
        a["GF"] += ag
        a["GA"] += hg
        if hg > ag:
            h["Won"] += 1
            a["Lost"] += 1
        elif hg < ag:
            a["Won"] += 1
            h["Lost"] += 1
        else:
            h["Drawn"] += 1
            a["Drawn"] += 1

    if not records:
        return pd.DataFrame(columns=columns)

    rows = []
    for r in records.values():
        r["GD"] = r["GF"] - r["GA"]
        r["Points"] = r["Won"] * 3 + r["Drawn"]
        rows.append(r)

    table = pd.DataFrame(rows).sort_values(
        ["Points", "GD", "GF"], ascending=[False, False, False]
    ).reset_index(drop=True)
    table.insert(0, "Position", range(1, len(table) + 1))
    return table[columns]


def compute_xpts_table(settled_df: pd.DataFrame) -> pd.DataFrame:
    """Loops through every finished fixture and works out a "deserved"
    result from box-touch territory dominance (using a fixed 0.015 box
    touch weight per the spec), rather than the real final score. Also
    tracks the real GP/W/D/L/GD alongside it for context."""
    teams = pd.unique(settled_df[["home_team", "away_team"]].values.ravel("K"))
    records = {
        t: {"played": 0, "wins": 0, "draws": 0, "losses": 0, "goals_for": 0.0,
            "goals_against": 0.0, "actual_pts": 0.0, "xpts": 0.0}
        for t in teams
    }

    for _, row in settled_df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hbt = float(row.get("home_box_touches", 0) or 0)
        abt = float(row.get("away_box_touches", 0) or 0)
        dominance_diff = (hbt - abt) * BOX_TOUCH_WEIGHT

        if dominance_diff > 0.05:
            home_xpts, away_xpts = 3.0, 0.0
        elif dominance_diff < -0.05:
            home_xpts, away_xpts = 0.0, 3.0
        else:
            home_xpts, away_xpts = 1.0, 1.0

        hg, ag = row.get("home_goals", np.nan), row.get("away_goals", np.nan)
        if pd.notna(hg) and pd.notna(ag):
            home_actual = 3.0 if hg > ag else (1.0 if hg == ag else 0.0)
            away_actual = 3.0 if ag > hg else (1.0 if hg == ag else 0.0)
        else:
            home_actual = away_actual = 0.0
            hg = ag = 0.0

        records[home]["played"] += 1
        records[home]["actual_pts"] += home_actual
        records[home]["xpts"] += home_xpts
        records[home]["goals_for"] += hg
        records[home]["goals_against"] += ag
        records[away]["played"] += 1
        records[away]["actual_pts"] += away_actual
        records[away]["xpts"] += away_xpts
        records[away]["goals_for"] += ag
        records[away]["goals_against"] += hg

        if home_actual == 3.0:
            records[home]["wins"] += 1
            records[away]["losses"] += 1
        elif away_actual == 3.0:
            records[away]["wins"] += 1
            records[home]["losses"] += 1
        else:
            records[home]["draws"] += 1
            records[away]["draws"] += 1

    out = pd.DataFrame([
        {
            "team": t, "played": r["played"], "wins": r["wins"], "draws": r["draws"],
            "losses": r["losses"], "goal_difference": round(r["goals_for"] - r["goals_against"], 1),
            "actual_points": r["actual_pts"], "expected_points": round(r["xpts"], 2),
            "points_difference": round(r["actual_pts"] - r["xpts"], 2),
        }
        for t, r in records.items()
    ])
    # Sorted by ACTUAL points first (real league position), highest on top -
    # per the request to sort by current/actual points rather than xPts.
    return out.sort_values(
        ["actual_points", "expected_points"], ascending=[False, False]
    ).reset_index(drop=True)


def simulate_season(
    settled_df: pd.DataFrame, upcoming_df: pd.DataFrame, iterations: int = MC_ITERATIONS,
    rng=None, relegation_spots: int = 3, title_odds: dict | None = None,
    weights: tuple = TERRITORY_WEIGHTS_DEFAULT,
) -> pd.DataFrame:
    """Runs the full remaining-season Monte Carlo forecast. The
    crash-proof safety shield: any team with no historical matches (a
    brand-new call-up, or a data gap) gets its capability vector clamped
    to a safe floor of 0.01 rather than letting a NaN reach
    np.random.poisson.

    Now also tracks the FULL final-position distribution (not just
    "won it" / "finished last"), so a real "chance of finishing exactly
    Nth" percentage is available for every position, not just the two
    extremes. relegation_spots controls how many bottom places count as
    "relegated" (default 3, the most common real-world convention) -
    this replaces the old "only counts literally finishing dead last"
    behavior, which understated relegation risk for teams that are
    likely-but-not-certain to finish bottom.

    title_odds, if supplied (dict of team -> bookmaker odds to win the
    league), adds an "edge" column: (title_win_pct/100 * odds) - 1, the
    same EV formula used everywhere else in this app.
    """
    rng = rng or np.random.default_rng()
    baseline = compute_league_baseline(settled_df)
    all_teams = pd.unique(
        pd.concat([settled_df[["home_team", "away_team"]], upcoming_df[["home_team", "away_team"]]])
        .values.ravel("K")
    )
    all_teams = [t for t in all_teams if isinstance(t, str)]

    reference_date = settled_df["date"].max() if not settled_df.empty and settled_df["date"].notna().any() else pd.Timestamp.now()

    attack_home, defense_home, attack_away, defense_away = {}, {}, {}, {}
    for team in all_teams:
        hp = team_territory_profile(settled_df, team, "home", FROZEN_HALF_LIFE_DAYS, reference_date)
        ap = team_territory_profile(settled_df, team, "away", FROZEN_HALF_LIFE_DAYS, reference_date)
        ha = attack_strength(hp, baseline, "home", weights)
        hd = defense_strength(hp, baseline, "home", weights)
        aa = attack_strength(ap, baseline, "away", weights)
        ad = defense_strength(ap, baseline, "away", weights)
        attack_home[team] = ha if not math.isnan(ha) else 0.01
        defense_home[team] = hd if not math.isnan(hd) else 0.01
        attack_away[team] = aa if not math.isnan(aa) else 0.01
        defense_away[team] = ad if not math.isnan(ad) else 0.01

    current_points = {t: 0 for t in all_teams}
    for _, row in settled_df.iterrows():
        hg, ag = row.get("home_goals"), row.get("away_goals")
        if pd.isna(hg) or pd.isna(ag):
            continue
        if hg > ag:
            current_points[row["home_team"]] = current_points.get(row["home_team"], 0) + 3
        elif hg < ag:
            current_points[row["away_team"]] = current_points.get(row["away_team"], 0) + 3
        else:
            current_points[row["home_team"]] = current_points.get(row["home_team"], 0) + 1
            current_points[row["away_team"]] = current_points.get(row["away_team"], 0) + 1

    n_teams = len(all_teams)
    title_wins = {t: 0 for t in all_teams}
    relegation_finishes = {t: 0 for t in all_teams}
    position_counts = {t: {p: 0 for p in range(1, n_teams + 1)} for t in all_teams}
    fixtures = list(upcoming_df[["home_team", "away_team"]].itertuples(index=False, name=None))
    relegation_spots = max(1, min(relegation_spots, n_teams))

    for _ in range(iterations):
        sim_points = dict(current_points)
        for home, away in fixtures:
            if home not in attack_home or away not in attack_away:
                continue
            lam_h, lam_a = expected_goals(
                attack_home[home], defense_away[away], attack_away[away], defense_home[home], baseline
            )
            hg = rng.poisson(max(lam_h, 0.01))
            ag = rng.poisson(max(lam_a, 0.01))
            if hg > ag:
                sim_points[home] = sim_points.get(home, 0) + 3
            elif hg < ag:
                sim_points[away] = sim_points.get(away, 0) + 3
            else:
                sim_points[home] = sim_points.get(home, 0) + 1
                sim_points[away] = sim_points.get(away, 0) + 1

        if not sim_points:
            continue
        # Full final ranking this iteration, ties broken by team name (a
        # neutral, deterministic tiebreak - real tables use goal
        # difference, but that's a whole extra simulated dimension not
        # tracked per-iteration here, so this stays a reasonable
        # approximation rather than pretending precision it doesn't have).
        ranked = sorted(sim_points.items(), key=lambda kv: (-kv[1], kv[0]))
        for position, (team, _pts) in enumerate(ranked, start=1):
            position_counts[team][position] += 1
        champion = ranked[0][0]
        title_wins[champion] += 1
        for team, _pts in ranked[-relegation_spots:]:
            relegation_finishes[team] += 1

    def risk_flag(pct: float) -> str:
        if pct >= 40:
            return "🚨"
        if pct >= 15:
            return "⚠️"
        return "🟢"

    records = []
    for t in all_teams:
        title_pct = round(100 * title_wins[t] / iterations, 2)
        row = {
            "team": t,
            "current_points": current_points.get(t, 0),
            "title_win_pct": title_pct,
            "relegation_risk_pct": round(100 * relegation_finishes[t] / iterations, 2),
            "relegation_flag": risk_flag(100 * relegation_finishes[t] / iterations),
        }
        # Chance of finishing in each exact position - only worth showing
        # for a reasonably small league table, but computed for all sizes.
        for position in range(1, n_teams + 1):
            row[f"finish_pos_{position}_pct"] = round(100 * position_counts[t][position] / iterations, 2)
        if title_odds and t in title_odds and title_odds[t] and title_odds[t] > 0:
            row["title_odds"] = title_odds[t]
            row["title_edge_pct"] = round(expected_value(title_pct / 100, title_odds[t]) * 100, 2)
        records.append(row)

    out = pd.DataFrame(records)
    # Sorted by current (real) points first, highest on top - per the
    # request to sort by current standing rather than simulated title %.
    return out.sort_values(["current_points", "title_win_pct"], ascending=[False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 7: Sisonke Gold Mine Strategy Panel
# ---------------------------------------------------------------------------
# NOTE ON HONESTY: these are general football-knowledge-based starting
# hints (e.g. leagues broadly known for open, high-scoring football vs
# ones known for being cagey/defensive), NOT the output of a rigorous
# statistical backtest of each specific league. Treat them as an
# editable starting point, not verified fact - swap any of these out
# once you've backtested your own data and have real numbers to trust
# instead of a general reputation.

GOLD_MINE_STRATEGY = {
    ("Premier Division", "South Africa"): "OVER 1.5 GOALS / DOUBLE CHANCE",
    ("Championship", "England"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("League One", "England"): "OVER 2.5 GOALS / HOME DOUBLE CHANCE",
    ("Premier League", "England"): "BOTH TEAMS TO SCORE (YES) / OVER 2.5 GOALS",
    ("LaLiga", "Spain"): "UNDER 2.5 GOALS / HOME OR DRAW",
    ("LaLiga 2", "Spain"): "UNDER 2.5 GOALS",
    ("Bundesliga", "Germany"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("2. Bundesliga", "Germany"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("Serie A", "Italy"): "UNDER 2.5 GOALS",
    ("Serie B", "Italy"): "UNDER 2.5 GOALS / DOUBLE CHANCE",
    ("Ligue 1", "France"): "UNDER 2.5 GOALS",
    ("Ligue 2", "France"): "UNDER 2.5 GOALS",
    ("Bundesliga", "Austria"): "OVER 2.5 GOALS",
    ("Pro League", "Belgium"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("Challenger Pro League", "Belgium"): "OVER 1.5 GOALS",
    ("Brasileirão Série A", "Brazil"): "UNDER 2.5 GOALS / DOUBLE CHANCE",
    ("Brasileirão Série B", "Brazil"): "UNDER 2.5 GOALS",
    ("Premier League", "Canada"): "OVER 2.5 GOALS",
    ("MLS", "USA"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("USL Championship", "USA"): "OVER 2.5 GOALS",
    ("HNL", "Croatia"): "HOME DOUBLE CHANCE",
    ("Danish Superliga", "Denmark"): "OVER 2.5 GOALS",
    ("Premier Division", "Ireland"): "OVER 1.5 GOALS",
    ("J1 League", "Japan"): "BOTH TEAMS TO SCORE (YES)",
    ("Eredivisie", "Netherlands"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("Eerste Divisie", "Netherlands"): "OVER 2.5 GOALS",
    ("Eliteserien", "Norway"): "OVER 2.5 GOALS",
    ("Ekstraklasa", "Poland"): "UNDER 2.5 GOALS",
    ("Premier League", "Russia"): "UNDER 2.5 GOALS",
    ("Liga Portugal 2", "Portugal"): "UNDER 2.5 GOALS",
    ("Allsvenskan", "Sweden"): "OVER 2.5 GOALS",
    ("Super League", "Switzerland"): "OVER 2.5 GOALS",
    ("Challenge League", "Switzerland"): "OVER 1.5 GOALS",
    ("Super League", "China"): "UNDER 2.5 GOALS",
    ("Chilean Primera División", "Chile"): "UNDER 2.5 GOALS",
    ("Serie A", "Ecuador"): "HOME DOUBLE CHANCE",
    ("Liga 1", "Peru"): "HOME DOUBLE CHANCE",
    ("Primera A", "Colombia"): "UNDER 2.5 GOALS",
    ("First League", "Czech Republic"): "OVER 2.5 GOALS",
    ("A-League Men", "Australia"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("Botola Pro", "Morocco"): "UNDER 2.5 GOALS",
    ("Egyptian Premier League", "Egypt"): "UNDER 2.5 GOALS",
    ("Trendyol Süper Lig", "Turkey"): "OVER 2.5 GOALS / BOTH TEAMS TO SCORE (YES)",
    ("Besta deild karla", "Iceland"): "OVER 2.5 GOALS",
    ("División Profesional", "Bolivia"): "OVER 2.5 GOALS (altitude factor)",
    ("Premiership", "Scotland"): "HOME DOUBLE CHANCE",
    ("Indian Super League", "India"): "UNDER 2.5 GOALS",
    ("Liga MX Apertura", "Mexico"): "OVER 2.5 GOALS",
    ("Super Liga", "Romania"): "UNDER 2.5 GOALS",
}


def gold_mine_hint(division_text: str) -> str:
    """Direct lookup first, then a keyword-matching fallback loop so a
    slightly different bracket/formatting in the CSV (e.g. 'England
    Premier League' vs 'Premier League (England)') still locks onto the
    right entry.

    IMPORTANT: candidates are tried LONGEST-LEAGUE-NAME-FIRST. Several
    real league names are literal substrings of another real league name
    in this same dictionary (e.g. 'Bundesliga' is a substring of
    '2. Bundesliga'; 'LaLiga' is a substring of 'LaLiga 2') - matching in
    dictionary insertion order would let the shorter, WRONG division win
    just because its name happens to appear inside the correct one's
    text. Sorting by length descending means the more specific name is
    always tried before the shorter one it's contained in."""
    if not division_text:
        return "No Gold Mine data for this division yet."
    text_lower = division_text.lower()

    candidates_by_length = sorted(
        GOLD_MINE_STRATEGY.items(), key=lambda item: len(item[0][0]), reverse=True
    )

    for (league, country), hint in candidates_by_length:
        if league.lower() in text_lower and country.lower() in text_lower:
            return f"SISONKE GOLD MINE MARKET ({league}, {country}): Target {hint}"

    for (league, country), hint in candidates_by_length:
        if league.lower() in text_lower or country.lower() in text_lower:
            return f"SISONKE GOLD MINE MARKET (closest match: {league}, {country}): Target {hint}"

    return "No Gold Mine data matched this division - showing raw model output only."


# ---------------------------------------------------------------------------
# League playstyle profile banner - qualitative reputation tags per league,
# same honesty caveat as GOLD_MINE_STRATEGY above: general football
# knowledge, not a backtested statistical fingerprint of each league.
# ---------------------------------------------------------------------------

LEAGUE_PLAYSTYLE_PROFILE = {
    ("Premier Division", "South Africa"): "Physical duels, moderate tempo, set-piece reliant",
    ("Championship", "England"): "High box-touch intensity, direct/transition-heavy, congested fixture list",
    ("League One", "England"): "Direct play, high pressing, moderate technical quality",
    ("Premier League", "England"): "Fast transitions, high pressing, open end-to-end play",
    ("LaLiga", "Spain"): "Possession-based, patient build-up, low box-touch chaos",
    ("LaLiga 2", "Spain"): "Cagey, low-tempo, defensively organised",
    ("Bundesliga", "Germany"): "Fast transition attack, high pressing, open play",
    ("2. Bundesliga", "Germany"): "High-intensity transitions, aggressive pressing",
    ("Serie A", "Italy"): "Tactically disciplined, low box-touch intensity, defensively structured",
    ("Serie B", "Italy"): "Cagey, set-piece reliant, moderate tempo",
    ("Ligue 1", "France"): "Counter-attacking, uneven quality gaps, moderate tempo",
    ("Ligue 2", "France"): "Physical, direct, low technical intensity",
    ("Bundesliga", "Austria"): "Open play, fast transitions",
    ("Pro League", "Belgium"): "Technical, open play, high box-touch intensity",
    ("Challenger Pro League", "Belgium"): "Direct, moderate intensity",
    ("Brasileirão Série A", "Brazil"): "Technical, congested calendar fatigue, set-piece reliant",
    ("Brasileirão Série B", "Brazil"): "Physical, direct, low technical polish",
    ("Premier League", "Canada"): "Open play, moderate intensity",
    ("MLS", "USA"): "High tempo, open play, travel-fatigue heavy (large geography)",
    ("USL Championship", "USA"): "Direct, physical, moderate intensity",
    ("HNL", "Croatia"): "Home-dominant, technical, low away scoring",
    ("Danish Superliga", "Denmark"): "High pressing, fast transitions",
    ("Premier Division", "Ireland"): "Physical, direct, set-piece reliant",
    ("J1 League", "Japan"): "Technical, disciplined pressing, open play",
    ("Eredivisie", "Netherlands"): "Fast transition attack, high box-touch intensity, open play",
    ("Eerste Divisie", "Netherlands"): "Open, high-tempo, developmental squads",
    ("Eliteserien", "Norway"): "Direct, physical, weather-affected variance",
    ("Ekstraklasa", "Poland"): "Cagey, defensively structured",
    ("Premier League", "Russia"): "Low-tempo, defensively disciplined",
    ("Liga Portugal 2", "Portugal"): "Technical, low-scoring, patient build-up",
    ("Allsvenskan", "Sweden"): "Direct, physical, weather-affected variance",
    ("Super League", "Switzerland"): "High-tempo, open play",
    ("Challenge League", "Switzerland"): "Cagey, moderate intensity",
    ("Super League", "China"): "Cagey, defensively structured",
    ("Chilean Primera División", "Chile"): "Technical, low-scoring, patient build-up",
    ("Serie A", "Ecuador"): "Home-dominant (altitude factor in some venues), physical",
    ("Liga 1", "Peru"): "Home-dominant (altitude factor in some venues), physical",
    ("Primera A", "Colombia"): "Technical, low-scoring, patient build-up",
    ("First League", "Czech Republic"): "High-tempo, open play",
    ("A-League Men", "Australia"): "High tempo, open play, travel-fatigue heavy (large geography)",
    ("Botola Pro", "Morocco"): "Cagey, defensively structured",
    ("Egyptian Premier League", "Egypt"): "Cagey, low-scoring, physical",
    ("Trendyol Süper Lig", "Turkey"): "High-tempo, open play, high chaos/card variance",
    ("Besta deild karla", "Iceland"): "Weather-affected variance, direct play",
    ("División Profesional", "Bolivia"): "High-scoring (altitude factor), open play",
    ("Premiership", "Scotland"): "Physical duels, high box-touch intensity, direct play",
    ("Indian Super League", "India"): "Cagey, low-scoring, physical",
    ("Liga MX Apertura", "Mexico"): "Technical, high-altitude variance in some venues, open play",
    ("Super Liga", "Romania"): "Cagey, defensively structured",
}


def league_playstyle_profile(division_text: str) -> str:
    """Same longest-name-first matching logic as gold_mine_hint, kept as
    a separate lookup since the playstyle tag and the market hint are
    conceptually different things a user might want independently."""
    if not division_text:
        return "No playstyle profile for this division yet."
    text_lower = division_text.lower()
    candidates_by_length = sorted(
        LEAGUE_PLAYSTYLE_PROFILE.items(), key=lambda item: len(item[0][0]), reverse=True
    )
    for (league, country), tags in candidates_by_length:
        if league.lower() in text_lower and country.lower() in text_lower:
            return tags
    for (league, country), tags in candidates_by_length:
        if league.lower() in text_lower or country.lower() in text_lower:
            return tags
    return "No playstyle profile matched this division."


# ---------------------------------------------------------------------------
# Dynamic prediction explanation - a plain-language readout of everything
# that fed into one specific projection, generated fresh each time rather
# than a canned template string.
# ---------------------------------------------------------------------------

def generate_prediction_explanation(
    home_team: str, away_team: str,
    half_life_days: float, half_life_frozen: bool,
    home_attack_raw: float, away_attack_raw: float,
    home_momentum_mult: float, home_momentum_desc: str,
    away_momentum_mult: float, away_momentum_desc: str,
    tactic_log: list[str],
    rho: float,
    lam_home: float, lam_away: float,
    dc_probs: dict, mc_probs: dict,
    vol_dampener_adjusted: float,
) -> str:
    """Builds a fresh, specific explanation from the ACTUAL numbers that
    went into this one projection - not a fixed template that just fills
    in team names. Every section only appears if it was actually
    relevant (e.g. the momentum section is skipped entirely if neither
    team has a real streak), so the explanation reads differently for
    different fixtures rather than always listing the same boilerplate."""
    lines = [f"### Why the model predicts what it does: {home_team} vs {away_team}", ""]

    stronger = home_team if home_attack_raw > away_attack_raw else away_team
    lines.append(
        f"**Territory data**: based on venue-isolated big chances, shots on target, and box "
        f"touches, {stronger} shows the stronger underlying attacking territory profile "
        f"(host rating {home_attack_raw:.2f} vs visitor rating {away_attack_raw:.2f}). These "
        f"ratings are each team's own numbers divided by THIS LEAGUE'S real, separately-"
        f"computed average - 1.00 always means 'exactly average for this specific league' by "
        f"definition (a number divided by itself is always 1), so it looks the same across "
        f"every league on purpose. The actual underlying average - e.g. average goals, big "
        f"chances, shots on target, box touches for this exact league - genuinely differs "
        f"league to league and is shown in the 'This League's Real Baseline' panel above."
    )

    decay_note = (
        f"a frozen {half_life_days:.0f}-day window (manually locked)" if half_life_frozen
        else f"an auto-optimised {half_life_days:.0f}-day half-life, chosen by backtesting "
             f"candidate windows against this division's own real results"
    )
    lines.append(f"**Recency weighting**: recent matches are weighted more heavily using {decay_note}.")

    momentum_bits = []
    if abs(home_momentum_mult - 1.0) > 1e-6:
        momentum_bits.append(f"{home_team}: {home_momentum_desc} (x{home_momentum_mult:.2f} to attack)")
    if abs(away_momentum_mult - 1.0) > 1e-6:
        momentum_bits.append(f"{away_team}: {away_momentum_desc} (x{away_momentum_mult:.2f} to attack)")
    if momentum_bits:
        lines.append(f"**Streak/momentum**: {'; '.join(momentum_bits)}.")
    else:
        lines.append("**Streak/momentum**: neither team is currently on a qualifying win or loss streak, so no adjustment applied.")

    if tactic_log:
        lines.append("**Manual tactical/environmental adjustments applied**:")
        for entry in tactic_log:
            lines.append(f"  - {entry}")
    else:
        lines.append("**Manual tactical/environmental adjustments applied**: none - this is the model's baseline data-only projection.")

    lines.append(
        f"**Low-score correlation**: a Dixon-Coles rho of {rho:+.3f} was fitted from this "
        f"division's own historical 0-0/1-1/1-0/0-1 frequency (0 means no measurable "
        f"correlation was found in the data)."
    )
    lines.append(f"**Final expected goals**: {home_team} {lam_home:.2f} - {lam_away:.2f} {away_team}.")

    dc_pick = max([("Home", dc_probs.get("Home Win", 0)), ("Draw", dc_probs.get("Draw", 0)), ("Away", dc_probs.get("Away Win", 0))], key=lambda kv: kv[1])
    mc_pick = max([("Home", mc_probs.get("Home Win", 0)), ("Draw", mc_probs.get("Draw", 0)), ("Away", mc_probs.get("Away Win", 0))], key=lambda kv: kv[1])
    agree_note = "agree" if dc_pick[0] == mc_pick[0] else "DISAGREE"
    lines.append(
        f"**Engine cross-check**: Dixon-Coles favors **{dc_pick[0]}** "
        f"({dc_pick[1]*100:.1f}%), Monte Carlo favors **{mc_pick[0]}** ({mc_pick[1]*100:.1f}%) - "
        f"the two engines {agree_note}."
    )
    lines.append(f"**Match volatility dampener** (used by the Monte Carlo engine): {vol_dampener_adjusted:.3f}.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional Telegram notification - the ONE piece of this app that needs
# real internet access (sending a message inherently requires it), unlike
# everything else which stays fully local/offline. Only ever called when
# the user explicitly clicks "Send" - never runs automatically.
# ---------------------------------------------------------------------------

def send_telegram_message(bot_token: str, chat_id: str, text: str) -> tuple[bool, str]:
    if not bot_token or not chat_id:
        return False, "Bot token and chat ID are both required."
    if _requests is None:
        return False, "The 'requests' package isn't installed - run: pip install requests"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = _requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=10)
        if resp.status_code == 200:
            return True, "Sent."
        return False, f"Telegram API returned HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Request failed: {exc}"


