"""
sisonke_app.py
================

The Streamlit UI for the Sisonke Football Predictive Terminal. All the
actual math lives in sisonke_engine.py (tested independently - see
test_engine.py, integration_dry_run.py, and bias_stress_test.py) - this
file is purely the dashboard wiring: tabs, sidebar, inputs, charts.

RUN:
    streamlit run sisonke_app.py

HONESTY NOTES:
- The Gold Mine + League Playstyle panels are general football-reputation
  starting points, NOT the output of a rigorous statistical backtest of
  each specific league - see sisonke_engine.py's own notes on these.
- Tactical multiplier percentages have been reviewed against general
  football-analytics literature (see apply_tactical_multipliers'
  docstring for specifics on what was adjusted and why), but several -
  especially the counter-press style, pitch/weather effects, and referee
  strictness beyond the one spec'd value - remain reasoned estimates,
  not values fitted to your own data. They're editable in the sidebar for
  exactly that reason.
- This model is built for standard home/away league play (round-robin
  points tables) - NOT cup/knockout/tournament competitions, which follow
  different incentive and squad-rotation patterns entirely. Divisions
  that look like cups (by name) are filtered out of the workspace
  selector automatically.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import sisonke_engine as E

st.set_page_config(page_title="⚽ Sisonke Football Predictive Terminal", page_icon="⚽", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_FILE = DATA_DIR / "master_sisonke_database.csv"
LEAGUE_PARAMS_FILE = DATA_DIR / "league_params_cache.json"


# ---------------------------------------------------------------------------
# Per-league parameter cache (disk-persisted) - half-life, rho, territory
# weights, and the 1X2 calibrators are all backtest-derived and genuinely
# expensive to recompute (walk_forward_backtest walks every settled match
# in the division). Without this, Streamlit reruns the ENTIRE script on
# every single widget interaction - selecting a different fixture, ticking
# a checkbox, anything - which was re-running that full backtest sweep each
# time even though nothing about the LEAGUE actually changed. This cache is
# per-league, keyed by how many settled matches the league had when it was
# computed: as long as that count hasn't grown (no new results have come
# in), the cached parameters are reused instantly instead of being refit.
# Saved to disk (not just st.session_state) so it also survives an app
# restart, not just staying warm within one running session.
# ---------------------------------------------------------------------------

def _atomic_write_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_league_params_cache() -> dict:
    if LEAGUE_PARAMS_FILE.exists():
        try:
            return json.loads(LEAGUE_PARAMS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_league_params_cache(cache: dict):
    _atomic_write_json(cache, LEAGUE_PARAMS_FILE)


def get_cached_league_params(division: str, n_settled: int, half_life_frozen: bool) -> dict | None:
    """Returns the cached param dict for this division ONLY if it was
    computed at the same settled-match count, in the same half-life
    mode, as right now - a growing match count means new results have
    come in since the cache was built, so it's treated as stale rather
    than silently served."""
    cache = st.session_state.get("_league_params_disk_cache")
    if cache is None:
        cache = load_league_params_cache()
        st.session_state["_league_params_disk_cache"] = cache
    entry = cache.get(division)
    if entry and entry.get("based_on_n_settled") == n_settled and entry.get("half_life_frozen") == half_life_frozen:
        return entry
    return None


def set_cached_league_params(division: str, n_settled: int, half_life_frozen: bool, **params):
    cache = st.session_state.get("_league_params_disk_cache")
    if cache is None:
        cache = load_league_params_cache()
    cache[division] = {
        "based_on_n_settled": n_settled,
        "half_life_frozen": half_life_frozen,
        "computed_at": pd.Timestamp.now().isoformat(),
        **params,
    }
    st.session_state["_league_params_disk_cache"] = cache
    save_league_params_cache(cache)


def get_cached_territory_weights(division: str) -> tuple | None:
    """Territory weights are a separate, manually-triggered calibration
    (not tied to settled-match staleness the way half-life/rho/
    calibrators are) - persisted under their own key so a calibrated
    weighting survives an app restart instead of living only in
    st.session_state for the current session."""
    cache = st.session_state.get("_league_params_disk_cache")
    if cache is None:
        cache = load_league_params_cache()
        st.session_state["_league_params_disk_cache"] = cache
    weights = cache.get("_territory_weights", {}).get(division)
    return tuple(weights) if weights else None


def set_cached_territory_weights(division: str, weights: tuple):
    cache = st.session_state.get("_league_params_disk_cache")
    if cache is None:
        cache = load_league_params_cache()
    cache.setdefault("_territory_weights", {})[division] = list(weights)
    st.session_state["_league_params_disk_cache"] = cache
    save_league_params_cache(cache)


# ---------------------------------------------------------------------------
# Local storage (Section 1 upload port + requested download/clear controls)
# ---------------------------------------------------------------------------
def _atomic_write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def load_database_from_disk():
    if DB_FILE.exists():
        try:
            return E.standardise_columns(pd.read_csv(DB_FILE, dtype=str))
        except Exception:
            return None
    return None


def save_database_to_disk(df: pd.DataFrame):
    _atomic_write_csv(df, DB_FILE)


def prepare_raw_upload(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Standardises columns AND normalises team-name casing so 'Chelsea'
    and 'chelsea' resolve to one team, right at load time - before
    anything else touches the data."""
    df = E.standardise_columns(raw_df)
    df = E.normalize_name_casing(df, ["home_team", "away_team"])
    return df


def coerce_working_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_raw_upload(raw_df)
    numeric_cols = [
        "home_goals", "away_goals",
        "home_shots_on_target", "away_shots_on_target",
        "home_big_chances", "away_big_chances",
        "home_box_touches", "away_box_touches",
    ]
    df = E.coerce_numeric(df, numeric_cols)
    had_date_column = E.find_date_column(df) is not None or "date" in df.columns
    df = E.parse_dates(df, "date")
    df.attrs["had_date_column"] = had_date_column
    return df


if "raw_db" not in st.session_state:
    st.session_state.raw_db = load_database_from_disk()
if "bookmaker_odds" not in st.session_state:
    st.session_state.bookmaker_odds = {m: 2.00 for m in E.MARKET_LIST}
if "title_odds" not in st.session_state:
    st.session_state.title_odds = {}
if "multi_bet_slip" not in st.session_state:
    st.session_state.multi_bet_slip = []  # list of dicts - persists across different fixtures/leagues


def find_division_series(df: pd.DataFrame):
    col = E.find_division_column(df)
    if col is None:
        return None, None, None
    all_divisions = sorted(df[col].dropna().unique().tolist())
    standard, excluded = E.filter_to_standard_leagues(all_divisions)
    return col, standard, excluded


# ---------------------------------------------------------------------------
# Sidebar (Section 1 + local storage controls)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚽ Sisonke Control Deck")
    active_tab = st.radio(
        "Active workspace",
        ["📁 Research & Sentiment Tracker", "📊 Active Projections Matrix"],
        key="active_workspace",
    )

    st.divider()
    st.subheader("📤 Historical Matchday Upload")
    st.caption(
        "Upload one league's CSV at a time - each upload adds/updates just that league "
        "and leaves every other previously loaded league untouched. Re-uploading a file "
        "for a league you've already loaded replaces its old rows with the new ones "
        "(no duplicates); a genuinely new league is simply added alongside the rest."
    )
    uploaded = st.file_uploader("Upload a league's database (.csv)", type=["csv"], key="db_uploader")
    if uploaded is not None:
        try:
            new_raw = pd.read_csv(uploaded, dtype=str)
            new_raw = prepare_raw_upload(new_raw)

            if st.session_state.raw_db is None:
                merged = new_raw
                merge_note = f"✅ First upload - {len(new_raw)} row(s) loaded."
            else:
                new_div_col = E.find_division_column(new_raw)
                if new_div_col is None:
                    merged = pd.concat([st.session_state.raw_db, new_raw], ignore_index=True)
                    merge_note = (
                        f"⚠️ Couldn't find a division column in this file, so its {len(new_raw)} "
                        "row(s) were appended rather than replacing a specific league - if this is "
                        "an updated file for a league you already loaded, clear that league first "
                        "(below) to avoid duplicate rows."
                    )
                else:
                    new_divisions = new_raw[new_div_col].dropna().unique().tolist()
                    existing_div_col = E.find_division_column(st.session_state.raw_db)
                    if existing_div_col:
                        kept = st.session_state.raw_db[~st.session_state.raw_db[existing_div_col].isin(new_divisions)]
                    else:
                        kept = st.session_state.raw_db
                    merged = pd.concat([kept, new_raw], ignore_index=True)
                    shown_divs = ", ".join(str(d) for d in new_divisions[:3]) + ("..." if len(new_divisions) > 3 else "")
                    merge_note = (
                        f"✅ Added/updated {len(new_divisions)} league(s) ({shown_divs}) - "
                        f"{len(new_raw)} row(s). All other previously loaded leagues were kept "
                        f"untouched ({len(kept)} existing row(s) preserved)."
                    )

            st.session_state.raw_db = merged
            save_database_to_disk(merged)
            st.success(merge_note)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't read that file: {exc}")

    if st.session_state.raw_db is not None:
        st.caption(f"💾 Database loaded: {len(st.session_state.raw_db)} row(s).")

        st.download_button(
            "⬇️ Download current database (.csv)",
            data=st.session_state.raw_db.to_csv(index=False).encode("utf-8"),
            file_name="master_sisonke_database.csv",
            mime="text/csv",
            key="db_download_btn",
        )

        with st.expander("🧹 Clear data"):
            _dcol, _divs, _ = find_division_series(st.session_state.raw_db)
            if _dcol:
                clear_division = st.selectbox("League to clear", ["(pick one)"] + _divs, key="clear_division_pick")
                if st.button("Clear this league only", key="clear_one_league"):
                    if clear_division != "(pick one)":
                        remaining = st.session_state.raw_db[st.session_state.raw_db[_dcol] != clear_division]
                        st.session_state.raw_db = remaining
                        save_database_to_disk(remaining)
                        st.success(f"Cleared '{clear_division}'.")
                        st.rerun()
            confirm_wipe = st.checkbox("I understand this deletes the ENTIRE database", key="confirm_wipe_db")
            if st.button("🗑️ Clear ALL data", disabled=not confirm_wipe, key="clear_all_db"):
                st.session_state.raw_db = None
                if DB_FILE.exists():
                    DB_FILE.unlink()
                st.success("Database cleared.")
                st.rerun()
    else:
        st.caption("No database loaded yet.")

    st.divider()
    with st.expander("📱 Telegram Notifications"):
        st.caption("Sends the current prediction on demand - never runs automatically, and is the only part of this app that needs internet access.")
        tg_token = st.text_input("Bot Token", type="password", key="tg_token")
        tg_chat_id = st.text_input("Chat ID", key="tg_chat_id")
        # NOTE: st.text_input(..., key="tg_token") already writes its value
        # into st.session_state["tg_token"] automatically - Streamlit
        # forbids writing to that same key again by hand once the widget
        # has run this turn (raises StreamlitWidgetAlreadyInstantiatedError).
        # The local variables tg_token/tg_chat_id above already hold the
        # current values if this expander needs them directly.


if st.session_state.raw_db is None:
    st.title("⚽ SISONKE FOOTBALL HUB")
    st.info("👋 Upload your historical matchday database in the sidebar to get started.")
    st.stop()

working_df = coerce_working_frame(st.session_state.raw_db)
division_col, divisions, excluded_divisions = find_division_series(working_df)

if division_col is None:
    st.error(
        "⚠️ Couldn't find a division column - your CSV needs one named "
        "`league_country`, `league`, or `competition`."
    )
    st.stop()

if not divisions:
    st.error("⚠️ Every division in this file looks like a cup/tournament competition - this model is built strictly for standard league play.")
    st.stop()

if excluded_divisions:
    st.sidebar.caption(
        f"🚫 {len(excluded_divisions)} cup/tournament competition(s) hidden "
        f"(this model is for standard league play only): {', '.join(excluded_divisions[:3])}"
        + ("..." if len(excluded_divisions) > 3 else "")
    )

if not working_df.attrs.get("had_date_column", True):
    st.warning(
        "📅 No date column was found in your CSV (checked for `date`, `match_date`, "
        "`fixture_date`, `kickoff`, and a few other common names). Everything will "
        "still run, but time-decay weighting, the backtest, and fixture dates in the "
        "Sentiment Tracker will all be blind to recency until you add one - add a "
        "column with one of those names and re-upload."
    )


def league_profile_banner(division: str):
    hint = E.gold_mine_hint(division)
    style = E.league_playstyle_profile(division)
    st.info(f"💡 **{hint}**")
    st.caption(f"🎨 League profile: {style}")


# ---------------------------------------------------------------------------
# Section 2: Research & Sentiment Tracker (offline tab)
# ---------------------------------------------------------------------------
def render_sentiment_tracker():
    st.title("📁 Research & Sentiment Tracker")
    st.caption("🔒 An isolated, offline screening workspace - nothing here blocks the main analytics hub downstream.")

    division = st.selectbox("🏆 League workspace", divisions, key="sentiment_division")
    league_profile_banner(division)
    division_df = working_df[working_df[division_col] == division]
    settled_div, upcoming_div = E.split_played_unplayed(division_df)

    if upcoming_div.empty:
        st.warning("No unplayed fixtures detected for this division (no blank/comma goal cells found).")
        return

    fixture_labels = [
        f"{r.home_team} vs {r.away_team}"
        + (f" ({r.date.date()})" if pd.notna(getattr(r, "date", None)) else "")
        for r in upcoming_div.itertuples()
    ]
    fixture_choice = st.selectbox("🎯 Select Target Upcoming Fixture", fixture_labels, key="sentiment_fixture")

    st.subheader("📋 The 7-Day Diary Checklist")
    c1, c2, c3, c4 = st.columns(4)
    d7 = c1.checkbox("📅 7 Days Out - initial team news scanned", key="diary_7d")
    d72 = c2.checkbox("🕐 72 Hours Out - press conference checked", key="diary_72h")
    d24 = c3.checkbox("⏰ 24 Hours Out - confirmed absentees noted", key="diary_24h")
    d60 = c4.checkbox("⏱️ 60 Mins Out - final lineup confirmed", key="diary_60m")
    ticked = sum([d7, d72, d24, d60])

    st.subheader("🎭 Sentiment")
    sentiment = st.selectbox(
        "Current season motivation", ["🏖️ Beach Mode", "📉 Relegation Battle", "📈 Promotion Race", "🔥 Derby"],
        key="sentiment_choice",
    )

    confidence = round((ticked / 4) * 10)
    st.subheader("🎯 Confidence Rating")
    st.metric("Confidence Score (out of 10)", confidence)
    if confidence <= 3:
        st.error("🔴 PASS / NO BET - insufficient information gathered for this fixture.")
    else:
        st.success(f"🟢 Sufficient research depth logged ({ticked}/4 checklist items).")
    st.caption("ℹ️ This rating is advisory only - it never locks or breaks the Active Projections Matrix downstream, even at a PASS rating.")


# ---------------------------------------------------------------------------
# Active Projections Matrix tab (the main calculator)
# ---------------------------------------------------------------------------
def render_projections_matrix():
    division = st.selectbox("🏆 League workspace", divisions, key="matrix_division")
    league_profile_banner(division)
    division_df = working_df[working_df[division_col] == division]
    settled_div, upcoming_div = E.split_played_unplayed(division_df)

    # One shared disk-cache lookup for everything backtest-derived for this
    # league (half-life search, rho, reliability BSS, 1X2 calibrators) -
    # keyed on settled-match count + half-life mode, so it's reused across
    # every fixture click / checkbox toggle within the SAME league state,
    # and only invalidated when new results actually arrive or the user
    # explicitly hits "Recalibrate now" / recalibrates territory weights.
    n_settled = len(settled_div)
    force_recalibrate = st.session_state.pop("_force_recalibrate_flag", False)
    early_freeze_decay = st.session_state.get("freeze_decay", False)
    cached_params = None if force_recalibrate else get_cached_league_params(division, n_settled, half_life_frozen=early_freeze_decay)

    if upcoming_div.empty:
        st.warning("⚠️ No unplayed fixtures detected for this division.")
        return
    if len(settled_div) < E.MIN_SAMPLE_ROWS:
        st.warning(
            f"⚠️ Only {len(settled_div)} settled match(es) in this division - below the "
            f"{E.MIN_SAMPLE_ROWS}-match safety rail. Calculations will fall back to "
            "neutral baselines wherever a team's own sample is too small."
        )

    fixture_labels = [f"{r.home_team} vs {r.away_team}" for r in upcoming_div.itertuples()]
    fixture_choice = st.selectbox("🎯 Select fixture to project", fixture_labels, key="matrix_fixture")
    home_team, away_team = fixture_choice.split(" vs ")

    # --- Core Parameter A: time decay ---
    st.subheader("⏳ Time-Decay Half-Life")
    st.caption("Automatically calibrated by default (Brier-score backtest across 15-180 days) - this is the main calibrator and should be left alone for normal use.")
    freeze_decay = st.checkbox(
        "🧊 Freeze Decay (special cases only - e.g. right after a summer break or international window)",
        key="freeze_decay",
    )
    if freeze_decay:
        half_life = E.FROZEN_HALF_LIFE_DAYS
        use_match_index = True
        st.caption(
            f"Frozen at {half_life} calendar days - BUT weighting is based on **match recency, not calendar "
            "days elapsed**, specifically so a summer break or international window doesn't fictitiously flatten "
            "a team's recent form just because a lot of calendar time happened to pass. A team's actual last "
            f"{E.FROZEN_HALF_LIFE_MATCHES:.0f} matches still carry the normal weight regardless of the gap."
        )
    else:
        use_match_index = False
        if cached_params:
            half_life = cached_params["half_life_days"]
            st.caption(
                f"📦 Using cached optimal half-life: **{half_life} days** (from a backtest run "
                f"when this league had {n_settled} settled matches)."
            )
        else:
            with st.spinner("Backtesting half-life candidates against real results..."):
                half_life, hl_info = E.optimize_half_life(settled_div)
            st.caption(f"⚙️ Optimal half-life selected via Brier-score backtest: **{half_life} days**.")
        boundary_warning = E.half_life_at_search_boundary(half_life)
        if boundary_warning:
            st.warning(boundary_warning)
            dilution_cache_key = (division, len(settled_div))
            if st.session_state.get("_dilution_cache_key") != dilution_cache_key:
                with st.spinner("Checking how much of the search window is diluted by thin per-team-venue samples..."):
                    dilution_report = E.safety_rail_dilution_report(settled_div)
                st.session_state["_dilution_cache_key"] = dilution_cache_key
                st.session_state["_dilution_report_cached"] = dilution_report
            else:
                dilution_report = st.session_state["_dilution_report_cached"]

            if dilution_report.get("total_evaluated", 0) > 0:
                st.caption(
                    f"🔬 Dilution check: of the {dilution_report['total_evaluated']} matches this "
                    f"search actually evaluated, **{dilution_report['diluted_count']} "
                    f"({dilution_report['diluted_pct']}%)** had at least one side under the 5-match "
                    "safety rail at that point in time - contributing a flat neutral value instead "
                    "of real signal, regardless of which half-life was being tested. A high "
                    "percentage here is a real, concrete explanation for a boundary result; a low "
                    "one means something else is driving it (genuinely longer/shorter true memory "
                    "for this league, or a data quality issue worth checking)."
                )
            else:
                st.caption(f"🔬 Dilution check: {dilution_report.get('reason', 'not enough data to check')}.")

            if half_life >= max(E.HALF_LIFE_CANDIDATES):
                st.caption(
                    "Since dilution alone may not fully explain this, you can directly check "
                    "whether the true optimum lies beyond the normal 180-day search ceiling:"
                )
                if st.button("📈 Test half-life candidates beyond 180 days (up to 365)", key="extended_hl_btn"):
                    with st.spinner("Extending the search up to 365 days..."):
                        extended_result = E.extended_half_life_diagnostic(settled_div)
                    if extended_result.get("still_improving_past_180") is True:
                        st.warning(
                            f"📈 Still improving past 180: the score kept getting better all the way "
                            f"to **{extended_result['chosen']} days**. The normal 180-day ceiling is "
                            "genuinely capping something real for this league - worth treating this "
                            "league's memory as long-term, or considering the extended range."
                        )
                    elif extended_result.get("still_improving_past_180") is False:
                        st.success(
                            "✅ The score plateaus or worsens past 180 days - 180 was already close "
                            "to a genuine optimum here. The boundary hit likely reflects noise between "
                            "neighboring candidates (e.g. 165 vs 180) rather than a meaningfully capped "
                            "result - not something to worry about."
                        )
                    else:
                        st.caption(extended_result.get("reason", "Not enough data to run the extended check."))

    reference_date = settled_div["date"].max() if settled_div["date"].notna().any() else pd.Timestamp.now()

    # --- Core Parameter B: volatility ---
    st.subheader("🎛️ Volatility Auto-Calibrator")
    vol_profile = E.compute_volatility_profile(settled_div)
    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("📊 Dispersion Ratio", f"{vol_profile.dispersion_ratio:.3f}")
    vc2.metric("🔄 Squad Turnover Index", f"{vol_profile.squad_turnover_index:.3f}")
    vc3.metric("🌡️ Volatility Dampener", f"{vol_profile.vol_dampener:.3f}" + (" (adjusted)" if vol_profile.adjusted else ""))

    # --- Territory weight calibration + model reliability badge ---
    if "territory_weights_by_league" not in st.session_state:
        # Seed from disk on first touch this session, so a weighting
        # calibrated in a PREVIOUS session/app run is still honoured
        # rather than silently reverting to equal weights.
        st.session_state.territory_weights_by_league = {}
    disk_weights = get_cached_territory_weights(division)
    territory_weights = st.session_state.territory_weights_by_league.get(
        division, disk_weights or E.TERRITORY_WEIGHTS_DEFAULT
    )

    # Everything below (rho, the reliability BSS badge, and the 1X2
    # calibrators) is derived from ONE walk-forward backtest sweep over
    # this league - reusing the SAME cache lookup from the top of this
    # function (n_settled / half-life-mode keyed), so a cache hit there
    # means a cache hit here too, with no second disk read needed.
    if cached_params:
        rho = cached_params["rho"]
        badge_bss = cached_params["badge_bss"]
        calibrators_1x2 = E.deserialize_1x2_calibrators(cached_params["calibrators_1x2"])
        cache_status_note = (
            f"📦 Using cached calibration for this league (computed {cached_params.get('computed_at', 'unknown')}, "
            f"{n_settled} settled matches)."
        )
    else:
        with st.spinner(f"Running the walk-forward backtest for {division} (rho, reliability, and 1X2 calibration)..."):
            badge_backtest_df = E.walk_forward_backtest(settled_div, half_life_days=half_life, weights=territory_weights)
            badge_bss = E.brier_skill_score(badge_backtest_df) if not badge_backtest_df.empty else float("nan")
            rho = E.fit_rho(settled_div)
            # CAVEAT worth knowing: walk_forward_backtest's p_home/p_draw/p_away come from
            # the fast rho=0 Poisson approximation used for the backtest sweep
            # (_quick_lambda_for_backtest in the engine), not the rho-adjusted grid the
            # live prediction below actually uses. Rho's effect on the marginal
            # Home/Draw/Away split is usually modest, but this is a real approximation,
            # not an exact match to what's being corrected.
            calibrators_1x2 = E.fit_1x2_calibrators(badge_backtest_df) if not badge_backtest_df.empty else {}
        set_cached_league_params(
            division, n_settled, half_life_frozen=freeze_decay,
            half_life_days=half_life, rho=rho, badge_bss=badge_bss,
            calibrators_1x2=E.serialize_1x2_calibrators(calibrators_1x2),
        )
        cache_status_note = f"🆕 Freshly computed and cached for next time ({n_settled} settled matches)."

    st.subheader("🩺 Model Reliability for This League")
    badge_emoji, badge_label = E.league_reliability_badge(badge_bss)
    rel_col1, rel_col2 = st.columns([1, 3])
    rel_col1.metric("Status", f"{badge_emoji} {badge_label}")
    bss_display = f"{badge_bss:.3f}" if not math.isnan(badge_bss) else "n/a"
    rel_col2.caption(
        f"Brier Skill Score: **{bss_display}** (0 = no better than guessing this league's own "
        "historical outcome split; positive = genuinely beats it; negative = worse than it). "
        "See the Performance Backtester tab for the full walk-forward detail."
    )
    cache_col1, cache_col2 = st.columns([3, 1])
    cache_col1.caption(cache_status_note)
    if cache_col2.button("🔄 Recalibrate now", key="recalibrate_league_btn", help="Ignore the cache and force a fresh backtest for this league right now."):
        st.session_state["_force_recalibrate_flag"] = True
        st.rerun()

    weights_label = ", ".join(f"{w * 100:.0f}%" for w in territory_weights)
    st.caption(f"Current territory weights (Big Chances / SOT / Box Touches): **{weights_label}**")
    cal_col1, cal_col2 = st.columns([2, 1])
    if cal_col1.button("🎯 Calibrate territory weights for this league (runs a backtest - may take a moment)"):
        with st.spinner(f"Searching territory-weight combinations for {division}..."):
            new_weights, weight_info = E.optimize_territory_weights(settled_div, half_life)
        st.session_state.territory_weights_by_league[division] = new_weights
        set_cached_territory_weights(division, new_weights)
        if "equal_weight_score" in weight_info and "brier_scores_by_weights" in weight_info:
            chosen_score = weight_info["brier_scores_by_weights"][weight_info["chosen"]]
            improvement = weight_info["equal_weight_score"] - chosen_score
            st.success(
                f"✅ Calibrated: Big Chances {new_weights[0]*100:.0f}% / SOT {new_weights[1]*100:.0f}% / "
                f"Box Touches {new_weights[2]*100:.0f}%. Brier score improved from "
                f"{weight_info['equal_weight_score']:.4f} (equal weights) to {chosen_score:.4f} "
                f"({'better' if improvement > 0 else 'no better'} by {abs(improvement):.4f})."
            )
        else:
            st.info(weight_info.get("reason", "Calibration ran, but couldn't compare against equal weights."))
        st.session_state["_force_recalibrate_flag"] = True  # weights changed - rho/BSS/calibrators depend on them
        st.rerun()
    if cal_col2.button("↩️ Reset to equal weights", disabled=division not in st.session_state.territory_weights_by_league):
        st.session_state.territory_weights_by_league.pop(division, None)
        cache = st.session_state.get("_league_params_disk_cache") or load_league_params_cache()
        cache.get("_territory_weights", {}).pop(division, None)
        st.session_state["_league_params_disk_cache"] = cache
        save_league_params_cache(cache)
        st.session_state["_force_recalibrate_flag"] = True
        st.rerun()

    # --- Territory vectors + baseline ---
    baseline = E.compute_league_baseline(settled_div)

    with st.expander(f"📐 This League's Real Baseline ({division})", expanded=True):
        st.caption(
            "These are the actual, real numbers computed directly from this league's own "
            "settled matches - genuinely different from league to league. Team ratings "
            "elsewhere are each team's own stats divided by these numbers, so a rating of "
            "1.00 always means 'exactly average for THIS league' by definition - it's not a "
            "sign the underlying average itself is stuck at a fixed value."
        )
        bl1, bl2 = st.columns(2)
        bl1.metric("⚽ Avg Home Goals", f"{baseline.avg_home_goals:.2f}")
        bl2.metric("⚽ Avg Away Goals", f"{baseline.avg_away_goals:.2f}")
        bl3, bl4, bl5 = st.columns(3)
        bl3.metric("🎯 Avg Big Chances (Home)", f"{baseline.home_big_chances_for:.2f}")
        bl4.metric("🥅 Avg SOT (Home)", f"{baseline.home_sot_for:.2f}")
        bl5.metric("📦 Avg Box Touches (Home)", f"{baseline.home_box_for:.1f}")

    home_profile = E.team_territory_profile(settled_div, home_team, "home", half_life, reference_date, use_match_index=use_match_index)
    away_profile = E.team_territory_profile(settled_div, away_team, "away", half_life, reference_date, use_match_index=use_match_index)

    home_attack = E.attack_strength(home_profile, baseline, "home", territory_weights)
    home_defense = E.defense_strength(home_profile, baseline, "home", territory_weights)
    away_attack = E.attack_strength(away_profile, baseline, "away", territory_weights)
    away_defense = E.defense_strength(away_profile, baseline, "away", territory_weights)
    home_attack_raw, away_attack_raw = home_attack, away_attack

    # --- Momentum banner ---
    home_mom_mult, home_mom_desc = E.team_streak_multiplier(settled_div, home_team)
    away_mom_mult, away_mom_desc = E.team_streak_multiplier(settled_div, away_team)
    st.subheader("🔥 Momentum & Streak Banner")
    mb1, mb2 = st.columns(2)
    mb1.info(f"🏠 **{home_team}**: {home_mom_desc}")
    mb2.info(f"✈️ **{away_team}**: {away_mom_desc}")
    home_attack *= home_mom_mult
    away_attack *= away_mom_mult

    # --- Section 6: tactical multipliers ---
    st.subheader("🎚️ Tactical & Environmental Multipliers")
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown(f"**🏠 {home_team} (Host)**")
        home_newly_relegated = st.checkbox("🔽 Newly relegated", key="home_relegated")
        home_relegation_threat = st.checkbox("📉 Live relegation threat", key="home_threat")
        home_striker_injury = st.checkbox("🏥⚽ Key striker/attacker out", key="home_striker_inj")
        home_defender_injury = st.checkbox("🏥🛡️ Key defender out", key="home_defender_inj")
        home_bogey = st.checkbox("🔮 Historical bogey hex (home venue)", key="home_bogey")
        home_new_manager = st.checkbox("🧠 New manager bounce", key="home_manager")
        home_boardroom_crisis = st.checkbox("⚠️ Boardroom crisis", key="home_crisis")
        home_dead_rubber = st.checkbox("🥱 Dead rubber / beach mode", key="home_dead")
        home_cup_distraction = st.checkbox("🏆 Look-ahead cup penalty", key="home_cup")
        host_travel_units = st.selectbox("🚌 Host's own mid-week travel fatigue", [0, 1, 2, 3], key="host_travel")
        home_tactical_setup = st.selectbox(
            "📐 Host tactical setup",
            ["Standard Open Play", "Deep Ultra-Defensive Low-Block", "High-Intensity Counter-Pressing Style"],
            key="home_tactic_setup",
        )
        home_transfer_impact = st.slider(
            "🔄 Key player signing / departure impact", -20.0, 20.0, 0.0, step=1.0,
            key="home_transfer_impact",
            help="Positive = a quality signing just arrived, negative = a key player just left. "
                 "Set this yourself - there's no universal research figure for a specific "
                 "transfer's impact the way there is for e.g. a manager change (see the "
                 "engine's realism notes).",
        )
    with tc2:
        st.markdown(f"**✈️ {away_team} (Visitor)**")
        away_newly_relegated = st.checkbox("🔽 Newly relegated", key="away_relegated")
        away_relegation_threat = st.checkbox("📉 Live relegation threat", key="away_threat")
        away_striker_injury = st.checkbox("🏥⚽ Key striker/attacker out", key="away_striker_inj")
        away_defender_injury = st.checkbox("🏥🛡️ Key defender out", key="away_defender_inj")
        away_bogey = st.checkbox("🔮 Historical bogey hex (away venue)", key="away_bogey")
        away_new_manager = st.checkbox("🧠 New manager bounce", key="away_manager")
        away_boardroom_crisis = st.checkbox("⚠️ Boardroom crisis", key="away_crisis")
        away_dead_rubber = st.checkbox("🥱 Dead rubber / beach mode", key="away_dead")
        away_cup_distraction = st.checkbox("🏆 Look-ahead cup penalty", key="away_cup")
        away_travel_units = st.selectbox("🚌 Visitor's mid-week travel fatigue (arriving here)", [0, 1, 2, 3], key="away_travel")
        away_tactical_setup = st.selectbox(
            "📐 Visitor tactical setup",
            ["Standard Open Play", "Deep Ultra-Defensive Low-Block", "High-Intensity Counter-Pressing Style"],
            key="away_tactic_setup",
        )
        away_transfer_impact = st.slider(
            "🔄 Key player signing / departure impact", -20.0, 20.0, 0.0, step=1.0,
            key="away_transfer_impact",
            help="Positive = a quality signing just arrived, negative = a key player just left. "
                 "Set this yourself - there's no universal research figure for a specific "
                 "transfer's impact the way there is for e.g. a manager change (see the "
                 "engine's realism notes).",
        )

    st.markdown("**🌍 Universal Match Conditions**")
    uc1, uc2, uc3, uc4 = st.columns(4)
    coastal_shock = uc1.checkbox("🌦️ High-humidity coastal shock (visitor)", key="coastal")
    pre_season = uc2.checkbox("🌱 Pre-season fixture", key="pre_season")
    pitch_surface = uc3.selectbox(
        "🌱 Pitch surface", ["Standard Optimized Turf", "Waterlogged Mud", "Dry Uneven Grass, short and narrow"],
        key="pitch_surface",
    )
    weather = uc4.selectbox(
        "☁️ Weather outlook", ["Clear Sky / Ideal Climate", "Torrential Rain Storm", "Gale-Force Wind Interference"],
        key="weather",
    )
    referee_strictness = st.radio(
        "🟨 Referee Strictness Profile",
        ["Lenient (Flow Enforcer)", "Standard Average", "Hyper-Strict (Card Trigger)"],
        horizontal=True, key="referee_strictness",
    )

    tactics = E.TacticalInputs(
        home_newly_relegated=home_newly_relegated, away_newly_relegated=away_newly_relegated,
        home_relegation_threat=home_relegation_threat, away_relegation_threat=away_relegation_threat,
        home_striker_injury=home_striker_injury, away_striker_injury=away_striker_injury,
        home_defender_injury=home_defender_injury, away_defender_injury=away_defender_injury,
        home_bogey=home_bogey, away_bogey=away_bogey,
        home_new_manager=home_new_manager, away_new_manager=away_new_manager,
        home_boardroom_crisis=home_boardroom_crisis, away_boardroom_crisis=away_boardroom_crisis,
        home_dead_rubber=home_dead_rubber, away_dead_rubber=away_dead_rubber,
        home_travel_fatigue_units=away_travel_units, host_travel_fatigue_units=host_travel_units,
        coastal_shock=coastal_shock,
        home_cup_distraction=home_cup_distraction, away_cup_distraction=away_cup_distraction,
        home_tactical_setup=home_tactical_setup, away_tactical_setup=away_tactical_setup,
        pitch_surface=pitch_surface, weather=weather, referee_strictness=referee_strictness,
        pre_season_fixture=pre_season,
        home_transfer_impact_pct=home_transfer_impact, away_transfer_impact_pct=away_transfer_impact,
    )
    home_adj, away_adj, vol_adjusted, tactic_log = E.apply_tactical_multipliers(
        home_attack, home_defense, away_attack, away_defense, vol_profile.vol_dampener, tactics,
    )
    if tactic_log:
        with st.expander("📜 Applied multiplier log"):
            for line in tactic_log:
                st.text(line)

    # --- Run both engines ---
    lam_home, lam_away = E.expected_goals(home_adj.attack, away_adj.defense, away_adj.attack, home_adj.defense, baseline)
    st.caption(f"⚽ Model expected goals — {home_team}: **{lam_home:.2f}**, {away_team}: **{lam_away:.2f}**")

    st.caption(f"📐 Dixon-Coles ρ (fitted from this division's own low-score history, cached alongside the reliability backtest above): **{rho:.3f}**")
    matrix = E.build_score_matrix(lam_home, lam_away, rho)
    dc_probs = E.market_probs_from_matrix(matrix)

    hg_sim, ag_sim = E.monte_carlo_simulate(lam_home, lam_away, volatility_dampener=vol_adjusted, iterations=E.MC_ITERATIONS)
    mc_probs = E.market_probs_from_simulation(hg_sim, ag_sim)

    # --- 1X2 probability calibration (Platt/Isotonic, auto-selected per league) ---
    # Only touches Home Win / Draw / Away Win (and the Double Chance combos, which are
    # re-derived from the corrected values below to stay internally consistent) - the
    # other 19 markets on the valuation sheet are untouched by this.
    apply_calibration = st.checkbox(
        "🎯 Apply 1X2 probability calibration (Platt/Isotonic, auto-selected per outcome)",
        value=True, key="apply_1x2_calibration",
        help="Corrects Home/Draw/Away probabilities using this league's own backtest "
             "history, via whichever of Platt scaling or isotonic regression scored better "
             "on held-out cross-validation (or no correction, if neither beat the raw "
             "baseline). Uncheck to see the model's raw, uncorrected output.",
    )
    if apply_calibration and calibrators_1x2:
        raw_dc_1x2 = (dc_probs["Home Win"], dc_probs["Draw"], dc_probs["Away Win"])
        raw_mc_1x2 = (mc_probs["Home Win"], mc_probs["Draw"], mc_probs["Away Win"])
        dc_probs["Home Win"], dc_probs["Draw"], dc_probs["Away Win"] = E.apply_1x2_calibration(*raw_dc_1x2, calibrators_1x2)
        mc_probs["Home Win"], mc_probs["Draw"], mc_probs["Away Win"] = E.apply_1x2_calibration(*raw_mc_1x2, calibrators_1x2)
        # Double Chance markets are sums of Home/Draw/Away, so re-derive them from the
        # now-calibrated values rather than leaving them stale from the raw computation.
        for probs in (dc_probs, mc_probs):
            probs["Double Chance 1X"] = probs["Home Win"] + probs["Draw"]
            probs["Double Chance 12"] = probs["Home Win"] + probs["Away Win"]
            probs["Double Chance X2"] = probs["Draw"] + probs["Away Win"]

        with st.expander("🎯 1X2 Calibration Detail (raw vs. corrected)"):
            st.caption(
                "Each outcome class (Home/Draw/Away) independently chose Platt scaling, "
                "isotonic regression, or no correction - whichever scored best on held-out "
                "cross-validated log-loss against this league's own backtest history."
            )
            st.dataframe(E.calibration_summary_table(calibrators_1x2), use_container_width=True, hide_index=True)
            compare_df = pd.DataFrame({
                "Raw Dixon-Coles %": [raw_dc_1x2[0] * 100, raw_dc_1x2[1] * 100, raw_dc_1x2[2] * 100],
                "Calibrated Dixon-Coles %": [dc_probs["Home Win"] * 100, dc_probs["Draw"] * 100, dc_probs["Away Win"] * 100],
                "Raw Monte Carlo %": [raw_mc_1x2[0] * 100, raw_mc_1x2[1] * 100, raw_mc_1x2[2] * 100],
                "Calibrated Monte Carlo %": [mc_probs["Home Win"] * 100, mc_probs["Draw"] * 100, mc_probs["Away Win"] * 100],
            }, index=["Home Win", "Draw", "Away Win"])
            st.dataframe(compare_df.style.format("{:.1f}"), use_container_width=True)
    elif apply_calibration and not calibrators_1x2:
        st.caption(
            "ℹ️ Not enough backtest history yet for this league to fit a calibration "
            "correction (needs 40+ walk-forward predictions per outcome class) - showing "
            "raw, uncorrected probabilities."
        )

    # --- Dynamic prediction explanation ---
    st.subheader("🧾 Why This Prediction Was Made")
    explanation = E.generate_prediction_explanation(
        home_team, away_team, half_life, freeze_decay,
        home_attack_raw, away_attack_raw,
        home_mom_mult, home_mom_desc, away_mom_mult, away_mom_desc,
        tactic_log, rho, lam_home, lam_away, dc_probs, mc_probs, vol_adjusted,
    )
    st.markdown(explanation)

    # --- Charts: exact scoreline comparison + goal totals for both engines ---
    st.subheader("📈 Engine Comparison Charts")
    ch1, ch2 = st.columns(2)
    with ch1:
        st.caption("Dixon-Coles: most likely exact scorelines")
        size = matrix.shape[0]
        flat = [(f"{h}-{a}", matrix[h, a]) for h in range(min(size, 5)) for a in range(min(size, 5))]
        flat_sorted = sorted(flat, key=lambda x: -x[1])[:8]
        score_df = pd.DataFrame(flat_sorted, columns=["Scoreline", "Probability"]).set_index("Scoreline")
        st.bar_chart(score_df)
    with ch2:
        st.caption("Monte Carlo: simulated total-goals distribution")
        total_goals_sim = hg_sim + ag_sim
        totals_counts = pd.Series(total_goals_sim).value_counts().sort_index()
        totals_counts.index = totals_counts.index.astype(str)
        st.bar_chart(totals_counts)

    engine_compare_df = pd.DataFrame({
        "Dixon-Coles %": [dc_probs["Home Win"] * 100, dc_probs["Draw"] * 100, dc_probs["Away Win"] * 100],
        "Monte Carlo %": [mc_probs["Home Win"] * 100, mc_probs["Draw"] * 100, mc_probs["Away Win"] * 100],
    }, index=["Home Win", "Draw", "Away Win"])
    st.caption("Home / Draw / Away probability - both engines side by side")
    st.bar_chart(engine_compare_df)

    # --- Section 8: valuation sheet ---
    st.subheader("📋 22-Market Options Valuation Sheet")
    st.caption("Edit the Bookmaker Odds for any market you want to check - everything else recalculates live.")

    edited_odds = {}
    odds_cols = st.columns(2)
    for i, market in enumerate(E.MARKET_LIST):
        col = odds_cols[i % 2]
        edited_odds[market] = col.number_input(
            f"{market} odds", min_value=1.01, max_value=100.0,
            value=float(st.session_state.bookmaker_odds.get(market, 2.00)),
            step=0.01, key=f"odds_{market}",
        )
    st.session_state.bookmaker_odds = edited_odds

    st.markdown("#### 🎚️ Market Filters")
    st.caption(
        "A market's own confidence is the higher of its Dixon-Coles and Monte Carlo "
        "probability for THAT specific bet - not the match's overall outright-winner "
        "confidence. A market only stays active if it clears BOTH filters below."
    )
    gate_col1, gate_col2 = st.columns(2)
    confidence_floor = gate_col1.slider(
        "🎚️ Confidence Floor (%)", 0, 100, 0, step=5, key="market_confidence_floor",
        help="Markets the model isn't at least this confident about get grayed out and excluded from the parlay builder.",
    )
    min_ev_gate = gate_col2.slider(
        "📈 Minimum EV% Gate", -10.0, 20.0, 0.0, step=0.5, key="market_min_ev_gate",
        help="Only markets with an EV edge at or above this stay active. Set to 0 to just filter out negative-EV bets.",
    )

    sheet_rows = E.build_valuation_sheet(dc_probs, mc_probs, edited_odds, vol_adjusted)
    sheet_data = []
    passing_rows = []
    for r in sheet_rows:
        confidence_pct = max(r.dc_prob, r.mc_prob) * 100
        ev_pct = r.ev * 100
        passes = E.passes_market_gate(confidence_pct, ev_pct, confidence_floor, min_ev_gate)
        if passes:
            passing_rows.append(r)
        sheet_data.append({
            "Market": r.market,
            "Bookmaker Odds": r.bookmaker_odds,
            "Dixon-Coles %": round(r.dc_prob * 100, 2),
            "Monte Carlo %": round(r.mc_prob * 100, 2),
            "Convergence %": round(r.convergence * 100, 1),
            "Fair Odds": round(r.fair_odds, 2) if math.isfinite(r.fair_odds) else None,
            "EV Edge %": round(ev_pct, 2),
            "Volatility Tier": r.volatility_tier,
            "Verdict": r.verdict,
            "Recommended Action": r.recommended_action,
            "Gate Status": E.market_gate_status(confidence_pct, ev_pct, confidence_floor, min_ev_gate),
            "_passes": passes,
        })
    sheet_df = pd.DataFrame(sheet_data)
    display_df = sheet_df.drop(columns=["_passes"])

    def _gate_row_style(row):
        return [''] * len(row) if sheet_df.loc[row.name, "_passes"] else ['background-color: #2b2b2b; color: #808080'] * len(row)

    st.dataframe(display_df.style.apply(_gate_row_style, axis=1), use_container_width=True, hide_index=True, height=560)
    st.caption(f"✅ {len(passing_rows)} of {len(sheet_df)} markets currently allowed under your filters above.")

    st.markdown("#### 🌍 Add to Multi-League Bet Slip")
    st.caption(
        "Unlike the single-match parlay below, this slip persists as you move between "
        "different leagues and fixtures - build a combined ticket across several matches "
        "at once from the '🌍 Multi-League Bet Slip' tab."
    )
    if not passing_rows:
        st.caption("No markets currently pass your filters above to add.")
    else:
        add_market_choice = st.multiselect(
            "Pick market(s) from this fixture to add",
            [r.market for r in passing_rows], key="multi_slip_add_choice",
        )
        if st.button("➕ Add selected to Multi-League Slip", disabled=not add_market_choice):
            existing_keys = {(leg["league"], leg["fixture"], leg["market"]) for leg in st.session_state.multi_bet_slip}
            added = 0
            for r in passing_rows:
                if r.market not in add_market_choice:
                    continue
                key = (division, f"{home_team} vs {away_team}", r.market)
                if key in existing_keys:
                    continue
                st.session_state.multi_bet_slip.append({
                    "league": division, "fixture": f"{home_team} vs {away_team}",
                    "market": r.market, "bookmaker_odds": r.bookmaker_odds,
                    "dc_prob": r.dc_prob, "mc_prob": r.mc_prob, "ev": r.ev,
                })
                added += 1
            if added:
                st.success(f"✅ Added {added} leg(s) to the Multi-League Slip ({len(st.session_state.multi_bet_slip)} total).")
            else:
                st.info("Those legs are already in the slip.")

    # --- Section 9: parlay & Kelly builder ---
    render_parlay_builder(passing_rows, home_team, away_team, explanation)


def render_parlay_builder(sheet_rows, home_team, away_team, explanation_text):
    st.subheader("🎟️ Sisonke Multi-Leg Parlay & Kelly Slip Builder")
    st.caption("Only markets currently passing your Confidence Floor and EV Gate above are offered here.")
    row_by_market = {r.market: r for r in sheet_rows}
    if len(row_by_market) < 2:
        st.info(
            "🚫 Fewer than 2 markets currently pass your filters, so there's nothing "
            "safe to build a parlay from. Lower the Confidence Floor or EV Gate above "
            "if you want more options."
        )
        return
    chosen_markets = st.multiselect(
        "Pick 2 or more qualifying value lines", list(row_by_market.keys()), key="parlay_legs"
    )
    if len(chosen_markets) < 2:
        st.caption("Select at least 2 legs to build a parlay slip.")
        return

    legs = [row_by_market[m] for m in chosen_markets]
    combined_odds, combined_prob = E.combine_parlay_legs(legs)
    combined_ev = E.expected_value(combined_prob, combined_odds)

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("💰 Combined Odds", f"{combined_odds:.2f}")
    pc2.metric("🎯 Joint Model Probability", f"{combined_prob * 100:.2f}%")
    pc3.metric("📈 Combined EV", f"{combined_ev * 100:.2f}%")

    kelly_mult = st.slider("🎚️ Fractional Kelly", 0.05, 1.00, 0.25, step=0.05, key="kelly_slider")
    bankroll = st.number_input("💵 Matchday bankroll (R)", min_value=0.0, value=1000.0, step=50.0, key="bankroll")

    if combined_ev <= 0:
        st.error("🚫 Combined expected value is negative - staking locked out for safety.")
        stake = 0.0
    else:
        kelly_frac = E.kelly_stake_fraction(combined_prob, combined_odds, kelly_mult)
        raw_stake = kelly_frac * bankroll
        stake = E.round_to_nearest(raw_stake, 10)
        st.success(f"✅ Suggested stake: **R{stake:.0f}** (Kelly fraction: {kelly_frac * 100:.2f}%, rounded to nearest R10)")

    ticket_lines = [
        "SISONKE MULTI-LEG PARLAY SLIP", "=" * 40,
        f"Fixture context: {home_team} vs {away_team}", "", "LEGS:",
    ]
    for r in legs:
        ticket_lines.append(f"  - {r.market} @ {r.bookmaker_odds:.2f} (model {max(r.dc_prob, r.mc_prob) * 100:.1f}%, EV {r.ev * 100:.1f}%)")
    ticket_lines += [
        "", f"Combined Odds: {combined_odds:.2f}", f"Joint Model Probability: {combined_prob * 100:.2f}%",
        f"Combined EV: {combined_ev * 100:.2f}%", f"Kelly Fraction Used: {kelly_mult:.2f}", f"Suggested Stake: R{stake:.0f}",
    ]
    ticket_text = "\n".join(ticket_lines)
    dl1, dl2 = st.columns(2)
    dl1.download_button("⬇️ Download Ticket (.txt)", data=ticket_text, file_name="sisonke_bet_slip.txt", mime="text/plain")

    if dl2.button("📱 Send to Telegram", key="send_telegram_btn"):
        token = st.session_state.get("tg_token", "")
        chat_id = st.session_state.get("tg_chat_id", "")
        message = f"{explanation_text}\n\n{ticket_text}"
        ok, msg = E.send_telegram_message(token, chat_id, message)
        if ok:
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ {msg}")


# ---------------------------------------------------------------------------
# Multi-League Bet Slip tab - like the single-fixture parlay builder above,
# but legs persist as plain dicts across different fixtures/leagues, so you
# can combine a pick from the EPL with one from the Bundesliga in the same
# ticket instead of being limited to one match at a time.
# ---------------------------------------------------------------------------
def render_multi_league_slip():
    st.subheader("🌍 Multi-League Bet Slip")
    st.caption(
        "Add legs here from the Active Projections Matrix tab ('➕ Add to Multi-League "
        "Slip') across as many different leagues and fixtures as you like, then price "
        "the combined ticket below."
    )
    slip = st.session_state.multi_bet_slip
    if not slip:
        st.info("Your slip is empty. Go to a fixture on the Active Projections Matrix tab and add some legs.")
        return

    slip_df = pd.DataFrame([
        {
            "League": leg["league"], "Fixture": leg["fixture"], "Market": leg["market"],
            "Odds": leg["bookmaker_odds"], "Model %": round(max(leg["dc_prob"], leg["mc_prob"]) * 100, 1),
            "EV %": round(leg["ev"] * 100, 1),
        }
        for leg in slip
    ])
    st.dataframe(slip_df, use_container_width=True, hide_index=True)

    leg_labels = [f"{leg['league']} | {leg['fixture']} | {leg['market']}" for leg in slip]
    to_remove = st.multiselect("Remove specific leg(s)", leg_labels, key="multi_slip_remove_choice")
    rc1, rc2 = st.columns(2)
    if rc1.button("🗑️ Remove selected", disabled=not to_remove):
        st.session_state.multi_bet_slip = [
            leg for leg, label in zip(slip, leg_labels) if label not in to_remove
        ]
        st.rerun()
    if rc2.button("🧹 Clear entire slip"):
        st.session_state.multi_bet_slip = []
        st.rerun()

    if len(slip) < 2:
        st.caption("Add at least 2 legs (from any league) to price a combined ticket.")
        return

    combined_odds, combined_prob = E.combine_parlay_legs(slip)
    combined_ev = E.expected_value(combined_prob, combined_odds)

    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("💰 Combined Odds", f"{combined_odds:.2f}")
    pc2.metric("🎯 Joint Model Probability", f"{combined_prob * 100:.2f}%")
    pc3.metric("📈 Combined EV", f"{combined_ev * 100:.2f}%")

    kelly_mult = st.slider("🎚️ Fractional Kelly", 0.05, 1.00, 0.25, step=0.05, key="multi_slip_kelly_slider")
    bankroll = st.number_input("💵 Matchday bankroll (R)", min_value=0.0, value=1000.0, step=50.0, key="multi_slip_bankroll")

    if combined_ev <= 0:
        st.error("🚫 Combined expected value is negative - staking locked out for safety.")
        stake = 0.0
    else:
        kelly_frac = E.kelly_stake_fraction(combined_prob, combined_odds, kelly_mult)
        raw_stake = kelly_frac * bankroll
        stake = E.round_to_nearest(raw_stake, 10)
        st.success(f"✅ Suggested stake: **R{stake:.0f}** (Kelly fraction: {kelly_frac * 100:.2f}%, rounded to nearest R10)")

    ticket_lines = ["SISONKE MULTI-LEAGUE BET SLIP", "=" * 40, "LEGS:"]
    for leg in slip:
        ticket_lines.append(
            f"  - [{leg['league']}] {leg['fixture']}: {leg['market']} @ {leg['bookmaker_odds']:.2f} "
            f"(model {max(leg['dc_prob'], leg['mc_prob']) * 100:.1f}%, EV {leg['ev'] * 100:.1f}%)"
        )
    ticket_lines += [
        "", f"Combined Odds: {combined_odds:.2f}", f"Joint Model Probability: {combined_prob * 100:.2f}%",
        f"Combined EV: {combined_ev * 100:.2f}%", f"Kelly Fraction Used: {kelly_mult:.2f}", f"Suggested Stake: R{stake:.0f}",
    ]
    ticket_text = "\n".join(ticket_lines)
    dl1, dl2 = st.columns(2)
    dl1.download_button("⬇️ Download Ticket (.txt)", data=ticket_text, file_name="sisonke_multi_league_slip.txt", mime="text/plain", key="multi_slip_download")

    if dl2.button("📱 Send to Telegram", key="multi_slip_send_telegram_btn"):
        token = st.session_state.get("tg_token", "")
        chat_id = st.session_state.get("tg_chat_id", "")
        ok, msg = E.send_telegram_message(token, chat_id, ticket_text)
        if ok:
            st.success(f"✅ {msg}")
        else:
            st.error(f"❌ {msg}")


# ---------------------------------------------------------------------------
# Live Standings Ledger tab
# ---------------------------------------------------------------------------
def render_standings_ledger():
    division = st.selectbox("🏆 League workspace", divisions, key="standings_division")
    division_df = working_df[working_df[division_col] == division]
    settled_div, upcoming_div = E.split_played_unplayed(division_df)

    st.subheader("📊 Deserved Points Table (xPts)")
    st.caption("Sorted by real (actual) points, highest first.")
    if settled_div.empty:
        st.caption("No settled matches yet for this division.")
    else:
        xpts_table = E.compute_xpts_table(settled_div)
        display_cols = ["team", "played", "wins", "draws", "losses", "goal_difference",
                         "actual_points", "expected_points", "points_difference"]
        st.dataframe(xpts_table[display_cols], use_container_width=True, hide_index=True)

    st.subheader("🔮 10,000-Run Season Forecast")
    if upcoming_div.empty:
        st.caption("No remaining fixtures to simulate - season looks complete in this dataset.")
        return

    all_teams_here = sorted(pd.unique(division_df[["home_team", "away_team"]].values.ravel("K")))
    all_teams_here = [t for t in all_teams_here if isinstance(t, str)]
    with st.expander("💰 Title odds input (optional - adds an Edge column)"):
        title_odds_input = {}
        odd_cols = st.columns(3)
        for i, t in enumerate(all_teams_here):
            val = odd_cols[i % 3].number_input(f"{t} title odds", min_value=0.0, value=0.0, step=1.0, key=f"title_odds_{t}")
            if val > 0:
                title_odds_input[t] = val
        st.session_state.title_odds = title_odds_input

    if st.button("▶️ Run 10,000-iteration season simulation", key="run_season_sim"):
        with st.spinner("Simulating 10,000 seasons..."):
            forecast = E.simulate_season(
                settled_div, upcoming_div, iterations=E.MC_ITERATIONS, title_odds=st.session_state.title_odds
            )
        st.session_state["season_forecast"] = forecast

    if "season_forecast" in st.session_state:
        forecast = st.session_state["season_forecast"]
        core_cols = ["team", "current_points", "title_win_pct", "relegation_risk_pct", "relegation_flag"]
        if "title_odds" in forecast.columns:
            core_cols += ["title_odds", "title_edge_pct"]
        st.caption("Sorted by current real points, highest first.")
        st.dataframe(forecast[core_cols], use_container_width=True, hide_index=True)

        with st.expander("📍 Detailed finishing-position distribution (% chance of each exact position)"):
            pos_cols = [c for c in forecast.columns if c.startswith("finish_pos_")]
            pos_display = forecast[["team"] + pos_cols].copy()
            pos_display.columns = ["team"] + [f"P{c.split('_')[-1]}" for c in pos_cols]
            st.dataframe(pos_display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Performance Backtester tab
# ---------------------------------------------------------------------------
def render_backtester():
    division = st.selectbox("🏆 League workspace", divisions, key="backtest_division")
    division_df = working_df[working_df[division_col] == division]
    settled_div, _ = E.split_played_unplayed(division_df)
    if settled_div.empty:
        st.caption("No settled matches for this division.")
        return

    st.subheader("🎛️ Backtest Controls")
    enable_manual_override = st.checkbox(
        "🔓 Enable manual override (special cases only - leave OFF for normal use)",
        value=False, key="enable_manual_override",
        help="This is a what-if sensitivity check, never the model's main calibrator. Auto-calibrated "
             "half-life + real fitted probabilities are what actually drive predictions - this slider only "
             "nudges the BACKTEST comparison and does nothing unless explicitly turned on here.",
    )
    bc1, bc2 = st.columns(2)
    manual_override_pct = bc1.slider(
        "🎚️ Manual Override (shift Home Win probability, percentage points)",
        -20.0, 20.0, 0.0, step=1.0, key="manual_override_slider",
        disabled=not enable_manual_override,
        help="Nudges every backtest prediction's Home Win probability by this many points (rebalancing Draw/Away proportionally) - a sensitivity check, not a permanent model change.",
    )
    accuracy_floor = bc2.slider(
        "📏 Accuracy Floor (%) - only count high-confidence picks", 0, 100, 0, step=5, key="accuracy_floor_slider",
        help="Filters the accuracy metric to only predictions where the model's top pick exceeded this probability - shows how good the model is when it's genuinely confident.",
    )

    half_life_choice = st.radio(
        "Half-life used for this backtest", ["Auto-optimised", "Frozen 45-day", "Effectively no decay (raw/unweighted)"],
        horizontal=True, key="backtest_hl_choice",
    )
    if half_life_choice == "Frozen 45-day":
        hl_for_backtest = E.FROZEN_HALF_LIFE_DAYS
    elif half_life_choice == "Effectively no decay (raw/unweighted)":
        hl_for_backtest = 100_000.0
    else:
        hl_for_backtest, _ = E.optimize_half_life(settled_div)

    with st.spinner(f"Running a walk-forward backtest across all {len(settled_div)} settled matches..."):
        territory_weights_bt = st.session_state.get("territory_weights_by_league", {}).get(division, E.TERRITORY_WEIGHTS_DEFAULT)
        backtest_df = E.walk_forward_backtest(settled_div, half_life_days=hl_for_backtest, weights=territory_weights_bt)
    if territory_weights_bt != E.TERRITORY_WEIGHTS_DEFAULT:
        st.caption(
            f"Using this league's calibrated territory weights "
            f"(Big Chances {territory_weights_bt[0]*100:.0f}% / SOT {territory_weights_bt[1]*100:.0f}% / "
            f"Box Touches {territory_weights_bt[2]*100:.0f}%) - calibrate or reset from the Active "
            "Projections Matrix tab."
        )

    if backtest_df.empty:
        st.warning("Not enough history yet to backtest (need several settled matches before the first prediction can be made).")
        return

    if enable_manual_override and manual_override_pct != 0:
        adjusted = backtest_df.copy()
        for i, row in adjusted.iterrows():
            ph, pd_, pa = E.apply_manual_override(row["p_home"], row["p_draw"], row["p_away"], manual_override_pct)
            adjusted.at[i, "p_home"], adjusted.at[i, "p_draw"], adjusted.at[i, "p_away"] = ph, pd_, pa
            adjusted.at[i, "predicted_pick"] = max([("H", ph), ("D", pd_), ("A", pa)], key=lambda kv: kv[1])[0]
            adjusted.at[i, "correct"] = adjusted.at[i, "predicted_pick"] == row["actual"]
        backtest_df = adjusted

    bss = E.brier_skill_score(backtest_df)
    accuracy = E.backtest_accuracy_pct(backtest_df)
    confident_mask = backtest_df[["p_home", "p_draw", "p_away"]].max(axis=1) * 100 >= accuracy_floor
    filtered_df = backtest_df[confident_mask]
    filtered_accuracy = E.backtest_accuracy_pct(filtered_df) if not filtered_df.empty else float("nan")

    st.subheader("📐 Model Skill Metrics (whole dataset)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Brier Skill Score", f"{bss:.3f}" if not math.isnan(bss) else "n/a", help=">0 beats naive guessing, <0 is worse than it")
    m2.metric("✅ Accuracy (all picks)", f"{accuracy:.1f}%" if not math.isnan(accuracy) else "n/a")
    m3.metric(f"🔎 Accuracy (≥{accuracy_floor}% confidence)", f"{filtered_accuracy:.1f}%" if not math.isnan(filtered_accuracy) else "n/a", help=f"{len(filtered_df)}/{len(backtest_df)} predictions met the floor")
    m4.metric("📊 Matches backtested", len(backtest_df))

    st.subheader("📈 Weighted vs Raw Comparison")
    raw_backtest = E.walk_forward_backtest(settled_div, half_life_days=100_000.0)
    weighted_backtest = E.walk_forward_backtest(settled_div, half_life_days=E.FROZEN_HALF_LIFE_DAYS)
    compare_df = pd.DataFrame({
        "Accuracy %": [E.backtest_accuracy_pct(raw_backtest), E.backtest_accuracy_pct(weighted_backtest)],
        "Brier Skill Score": [E.brier_skill_score(raw_backtest), E.brier_skill_score(weighted_backtest)],
    }, index=["Raw (no decay)", "Weighted (45-day half-life)"])
    st.bar_chart(compare_df[["Accuracy %"]])
    st.dataframe(compare_df, use_container_width=True)

    st.subheader("📅 Full Backtest Log (whole dataset)")
    st.dataframe(
        backtest_df[["date", "home_team", "away_team", "home_goals", "away_goals", "goal_difference",
                     "p_home", "p_draw", "p_away", "actual", "predicted_pick", "correct"]],
        use_container_width=True, hide_index=True, height=400,
    )

    st.subheader("🎯 Accuracy % for All 22 Markets")
    st.caption(
        "The accuracy shown above (and in the 'Full Backtest Log' actual/predicted_pick "
        "columns) only measures the outright 1X2 pick - whichever of Home/Draw/Away had the "
        "single highest probability. This section is a different, complementary question: "
        "for EACH of the 22 markets as its own standalone bet, how often was the model's own "
        "≥50%-confidence side actually correct? These two numbers can legitimately differ - "
        "e.g. a match where Home Win was the highest of the three outcomes at 45% (so "
        "predicted_pick='H') but didn't itself clear 50% would count as 'wrong' for the "
        "standalone Home Win market here, even though it was still the outright top pick above."
    )
    if st.button("🎯 Calculate accuracy for all 22 markets (re-runs the backtest - may take longer)"):
        with st.spinner("Re-running the walk-forward backtest with all 22 markets..."):
            full_backtest_df = E.walk_forward_backtest(
                settled_div, half_life_days=hl_for_backtest, weights=territory_weights_bt,
                include_all_markets=True,
            )
            st.session_state["_per_market_accuracy_cache"] = E.per_market_backtest_accuracy(full_backtest_df)
    per_market_accuracy = st.session_state.get("_per_market_accuracy_cache")
    if per_market_accuracy:
        market_acc_df = pd.DataFrame(
            [{"Market": m, "Accuracy %": acc} for m, acc in per_market_accuracy.items()]
        ).sort_values("Accuracy %", ascending=False)
        st.dataframe(market_acc_df, use_container_width=True, hide_index=True)
        st.bar_chart(market_acc_df.set_index("Market")["Accuracy %"])
    else:
        st.caption("Click the button above to compute this (not run automatically, since it's more expensive).")

    st.subheader("🎰 Odds Calibration: what does a price actually mean here?")
    st.caption(
        "This uses ONLY the model's own historical predictions - no real bookmaker odds "
        "needed. Every walk-forward pick has an implied price (1 / probability): odds "
        "around 2.00 imply a 50% chance, 3.00 implies 33%, and so on. This groups every "
        "historical pick by that implied price band and checks how often the picked side "
        "ACTUALLY won within each band. A well-calibrated model's Actual Win Rate should "
        "land close to its Model-Implied Win Rate in every row - a consistently positive "
        "or negative Calibration Gap means the model is over- or under-confident at that "
        "price range specifically."
    )
    calibration_table = E.odds_calibration_table(backtest_df)
    if calibration_table.empty:
        st.caption("Not enough backtest predictions yet to build calibration bands.")
    else:
        st.dataframe(calibration_table, use_container_width=True, hide_index=True)

    st.subheader("🎯 1X2 Calibration: Platt vs. Isotonic vs. No Correction")
    st.caption(
        "For each outcome class (Home/Draw/Away), fits BOTH Platt scaling and isotonic "
        "regression on this same backtest sweep, scores both via 5-fold cross-validation "
        "(never on the rows used to fit), and keeps whichever wins on held-out log-loss - "
        "'none' wins if the raw probabilities were already fine, so a calibration correction "
        "is never force-fit to noise. This is what actually gets applied on the Active "
        "Projections Matrix tab when the calibration checkbox is on."
    )
    backtest_calibrators = E.fit_1x2_calibrators(backtest_df)
    st.dataframe(E.calibration_summary_table(backtest_calibrators), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Full Database View tab
# ---------------------------------------------------------------------------
def render_full_database():
    st.subheader("📑 Full Database View")
    st.caption(f"{len(st.session_state.raw_db)} row(s), as originally uploaded (team-name casing already normalised).")

    display_df = working_df.copy()
    if "home_goals" in display_df.columns and "away_goals" in display_df.columns:
        display_df["goal_difference"] = display_df["home_goals"] - display_df["away_goals"]

    if all(c in display_df.columns for c in ["home_big_chances", "away_big_chances", "home_shots_on_target", "away_shots_on_target", "home_box_touches", "away_box_touches"]):
        display_df["home_implied_xg"] = (
            0.55 * display_df["home_big_chances"].fillna(0) * 0.36
            + 0.35 * display_df["home_shots_on_target"].fillna(0) * 0.11
            + 0.10 * display_df["home_box_touches"].fillna(0) * 0.015
        ).round(2)
        display_df["away_implied_xg"] = (
            0.55 * display_df["away_big_chances"].fillna(0) * 0.36
            + 0.35 * display_df["away_shots_on_target"].fillna(0) * 0.11
            + 0.10 * display_df["away_box_touches"].fillna(0) * 0.015
        ).round(2)
        st.caption("ℹ️ `*_implied_xg` is a lightweight per-match estimate from THAT match's own registered stats (not the full team-strength model) - for a quick eyeball check, not a substitute for the Projections Matrix.")

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)


# ---------------------------------------------------------------------------
# Router (Section 1)
# ---------------------------------------------------------------------------
if active_tab == "📁 Research & Sentiment Tracker":
    render_sentiment_tracker()
else:
    st.title("⚽ SISONKE FOOTBALL HUB")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Active Projections Matrix", "📊 Live Standings Ledger",
        "📅 Performance Backtester", "📑 Full Database View",
        "🌍 Multi-League Bet Slip",
    ])
    with tab1:
        render_projections_matrix()
    with tab2:
        render_standings_ledger()
    with tab3:
        render_backtester()
    with tab4:
        render_full_database()
    with tab5:
        render_multi_league_slip()
