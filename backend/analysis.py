from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols, rlm
from statsmodels.stats.anova import anova_lm
from xgboost import XGBRegressor


ALIASES = {
    "driver": ("driver", "driver_number", "abbreviation"),
    "compound": ("compound", "tyre_compound"),
    "lap_number": ("lap_number", "lapnumber", "lap"),
    "lap_time_seconds": ("lap_time_seconds", "laptime", "lap_time"),
    "tyre_age": ("tyre_age", "tyrelife", "tyre_life"),
    "grid_position": ("grid_position", "gridposition", "grid"),
    "finish_position": ("finish_position", "position", "classified_position"),
    "season": ("season", "year"),
    "round": ("round", "round_number"),
}


def normalize_laps(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {str(column).strip().lower().replace(" ", "_"): column for column in frame.columns}
    renamed: dict[Any, str] = {}
    for canonical, candidates in ALIASES.items():
        match = next(
            (columns[candidate.lower().replace(" ", "_")] for candidate in candidates if candidate.lower().replace(" ", "_") in columns),
            None,
        )
        if match is not None:
            renamed[match] = canonical

    laps = frame.rename(columns=renamed).copy()
    required = ("driver", "compound", "lap_number", "lap_time_seconds", "tyre_age")
    missing = [column for column in required if column not in laps.columns]
    if missing:
        raise ValueError(
            "Missing required lap-data columns: "
            + ", ".join(missing)
            + ". Required columns: Driver, Compound, LapNumber, LapTime, TyreLife."
        )

    lap_time = laps["lap_time_seconds"]
    if pd.api.types.is_timedelta64_dtype(lap_time):
        laps["lap_time_seconds"] = lap_time.dt.total_seconds()
    elif lap_time.astype(str).str.contains(":", regex=False).any():
        laps["lap_time_seconds"] = pd.to_timedelta(lap_time, errors="coerce").dt.total_seconds()
    else:
        laps["lap_time_seconds"] = pd.to_numeric(lap_time, errors="coerce")

    for column in ("lap_number", "tyre_age", "grid_position", "finish_position", "season", "round"):
        if column in laps:
            laps[column] = pd.to_numeric(laps[column], errors="coerce")

    laps = laps.replace([np.inf, -np.inf], np.nan)
    laps = laps.dropna(subset=["driver", "compound", "lap_number", "lap_time_seconds", "tyre_age"])
    laps["driver"] = laps["driver"].astype(str).str.strip()
    laps["compound"] = laps["compound"].astype(str).str.title().str.strip()
    laps = laps[(laps["lap_time_seconds"] > 0) & (laps["tyre_age"] >= 0)]
    if len(laps) < 12:
        raise ValueError("At least 12 valid lap records are required for analysis.")
    laps = laps[~laps["compound"].str.upper().isin({"UNKNOWN", "NAN", "NONE", ""})]
    if laps.empty:
        raise ValueError("No laps with a known tyre compound were found.")
    return laps.reset_index(drop=True)


def _json_number(value: Any) -> float | None:
    if value is None or pd.isna(value) or not np.isfinite(value):
        return None
    return float(value)


def _pace_curve(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    race_columns = [column for column in ("season", "round") if column in frame.columns]
    driver_columns = [*race_columns, "driver"]
    pace = frame.copy()
    pace["driver_baseline_seconds"] = pace.groupby(
        driver_columns, dropna=False
    )["lap_time_seconds"].transform("median")
    pace["driver_delta_seconds"] = pace["lap_time_seconds"] - pace["driver_baseline_seconds"]

    race_progress_adjusted = False
    progress_design = ols(
        "driver_delta_seconds ~ race_progress_proxy + C(compound)",
        data=pace,
    )
    if (
        pace["race_progress_proxy"].nunique() > 1
        and np.linalg.matrix_rank(progress_design.exog) == progress_design.exog.shape[1]
    ):
        progress_model = rlm(
            "driver_delta_seconds ~ race_progress_proxy + C(compound)",
            data=pace,
        ).fit()
        pace["pace_delta_seconds"] = (
            pace["driver_delta_seconds"]
            - progress_model.params["race_progress_proxy"] * pace["race_progress_proxy"]
        )
        race_progress_adjusted = True
    else:
        pace["pace_delta_seconds"] = pace["driver_delta_seconds"]

    pace["driver_pace_center"] = pace.groupby(
        driver_columns, dropna=False
    )["pace_delta_seconds"].transform("median")
    pace["driver_pace_deviation"] = (
        pace["pace_delta_seconds"] - pace["driver_pace_center"]
    ).abs()
    pace["driver_pace_mad"] = pace.groupby(
        driver_columns, dropna=False
    )["driver_pace_deviation"].transform("median")
    outlier_limit = (pace["driver_pace_mad"] * 1.4826 * 4.5).clip(lower=4.0)
    outliers_removed = int((pace["driver_pace_deviation"] > outlier_limit).sum())
    pace = pace[pace["driver_pace_deviation"] <= outlier_limit]

    driver_ages = (
        pace.groupby(["compound", "tyre_age", *driver_columns], as_index=False, dropna=False)
        .agg(
            pace_delta_seconds=("pace_delta_seconds", "median"),
            observations=("pace_delta_seconds", "size"),
        )
    )
    summaries = (
        driver_ages.groupby(["compound", "tyre_age"], as_index=False)
        .agg(
            median_delta_seconds=("pace_delta_seconds", "median"),
            q1_delta_seconds=("pace_delta_seconds", lambda values: values.quantile(0.25)),
            q3_delta_seconds=("pace_delta_seconds", lambda values: values.quantile(0.75)),
            drivers=("pace_delta_seconds", "size"),
            observations=("observations", "sum"),
        )
        .sort_values(["compound", "tyre_age"])
    )
    summaries = summaries[summaries["drivers"] >= 3]
    adjustment = {
        "driver_normalized": True,
        "race_progress_adjusted": race_progress_adjusted,
        "minimum_drivers": 3,
        "outliers_removed": outliers_removed,
    }
    return (
        [
            {
                "compound": str(row.compound),
                "tyre_age": float(row.tyre_age),
                "median_delta_seconds": round(float(row.median_delta_seconds), 3),
                "q1_delta_seconds": round(float(row.q1_delta_seconds), 3),
                "q3_delta_seconds": round(float(row.q3_delta_seconds), 3),
                "drivers": int(row.drivers),
                "observations": int(row.observations),
            }
            for row in summaries.itertuples(index=False)
        ],
        adjustment,
    )


def analyze_laps(laps: pd.DataFrame, include_prediction: bool = True, allow_unidentifiable: bool = False) -> dict[str, Any]:
    frame = laps.copy()
    race_columns = [column for column in ("season", "round") if column in frame.columns]
    if race_columns:
        race_lap_max = frame.groupby(race_columns, dropna=False)["lap_number"].transform("max")
    else:
        race_lap_max = frame["lap_number"].max()
    frame["race_progress_proxy"] = race_lap_max - frame["lap_number"]
    compounds = sorted(frame["compound"].unique())
    compound_term = "tyre_age * C(compound)" if len(compounds) > 1 else "tyre_age"
    formula = f"lap_time_seconds ~ {compound_term} + race_progress_proxy + C(driver)"
    design = ols(formula, data=frame)
    degradation: list[dict[str, Any]] = []
    fitted = None
    warning = None
    if np.linalg.matrix_rank(design.exog) < design.exog.shape[1]:
        warning = (
            "Tyre age and race progress are confounded. Include varied stint lengths and compound strategies."
        )
        if not allow_unidentifiable:
            raise ValueError(warning)
    else:
        fitted = design.fit()
        reference = fitted.params.get("tyre_age", 0.0)
        for compound in compounds:
            interaction = fitted.params.get(f"tyre_age:C(compound)[T.{compound}]", 0.0)
            reverse_interaction = fitted.params.get(f"C(compound)[T.{compound}]:tyre_age", 0.0)
            slope = reference + interaction + reverse_interaction
            degradation.append(
                {
                    "compound": compound,
                    "slope_sec_per_lap": round(float(slope), 5),
                    "observations": int((frame["compound"] == compound).sum()),
                }
            )

    compound_p = None
    if fitted is not None and len(compounds) > 1:
        anova_table = anova_lm(fitted, typ=2)
        compound_rows = [index for index in anova_table.index if str(index) == "C(compound)"]
        compound_p = float(anova_table.loc[compound_rows, "PR(>F)"].min()) if compound_rows else None
    grid_finish = _grid_finish_metrics(frame)
    prediction = _position_prediction(frame) if include_prediction else None

    pace_by_age, pace_adjustment = _pace_curve(frame)

    return {
        "rows_analyzed": int(len(frame)),
        "degradation": degradation,
        "pace_by_age": pace_by_age,
        "pace_adjustment": pace_adjustment,
        "race_progress_model": {
            "available": fitted is not None,
            "r_squared": round(float(fitted.rsquared), 5) if fitted is not None else None,
            "adjusted_r_squared": round(float(fitted.rsquared_adj), 5) if fitted is not None else None,
            "formula": formula,
            "message": warning,
        },
        "compound_anova": {
            "p_value": _json_number(compound_p),
            "significant": bool(compound_p is not None and compound_p < 0.05),
            "available": compound_p is not None,
        },
        "grid_finish": grid_finish,
        "prediction": prediction,
        "warnings": [warning] if warning else [],
    }


def _grid_finish_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    needed = {"grid_position", "finish_position"}
    if not needed.issubset(frame.columns):
        return {"spearman_rho": None, "p_value": None, "drivers": 0}
    results = frame.dropna(subset=["grid_position", "finish_position"])
    if "round" in results.columns:
        group_columns = [column for column in ("season", "round", "driver") if column in results]
        results = results.sort_values("lap_number").drop_duplicates(group_columns, keep="last")
    else:
        results = results.drop_duplicates("driver")
    if len(results) < 3 or results["grid_position"].nunique() < 2 or results["finish_position"].nunique() < 2:
        return {"spearman_rho": None, "p_value": None, "drivers": int(len(results))}
    rho, p_value = spearmanr(results["grid_position"], results["finish_position"])
    return {
        "spearman_rho": _json_number(rho),
        "p_value": _json_number(p_value),
        "drivers": int(len(results)),
    }


def _position_prediction(frame: pd.DataFrame) -> dict[str, Any]:
    required = {"grid_position", "finish_position", "driver"}
    if not required.issubset(frame.columns):
        return {"mae": None, "grid_baseline_mae": None, "training_rows": 0, "message": "GridPosition and Position columns are needed."}

    group_columns = [column for column in ("season", "round", "driver") if column in frame.columns]
    races = frame.sort_values("lap_number").drop_duplicates(group_columns, keep="last")
    races = races.dropna(subset=["grid_position", "finish_position"])
    if "season" not in races:
        races["season"] = 0
    if "round" not in races:
        races["round"] = 0
    races["driver_code"] = races["driver"].astype("category").cat.codes

    if len(races) < 8:
        return {
            "mae": None,
            "grid_baseline_mae": None,
            "training_rows": int(len(races)),
            "message": "At least 8 classified driver-race results are needed.",
        }

    features = races[["grid_position", "season", "round", "driver_code"]].astype(float)
    target = races["finish_position"].astype(float)
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.25, random_state=42
    )
    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )
    model.fit(train_features, train_target)
    predicted = np.clip(np.rint(model.predict(test_features)), 1, 20)
    baseline = np.clip(np.rint(test_features["grid_position"]), 1, 20)
    return {
        "mae": round(float(mean_absolute_error(test_target, predicted)), 4),
        "grid_baseline_mae": round(float(mean_absolute_error(test_target, baseline)), 4),
        "training_rows": int(len(train_features)),
        "test_rows": int(len(test_features)),
    }
