from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import wraps
from io import StringIO
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from typing import Any

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from analysis import analyze_laps, normalize_laps

OPENF1_URL = "https://api.openf1.org/v1"
JOLPICA_URL = "https://api.jolpi.ca/ergast/f1"
MIN_SEASON = 2023
MAX_UPLOAD_BYTES = 20_000_000
_openf1_rate_lock = Lock()
_openf1_next_request = 0.0

app = FastAPI(title="Apex Lab Race Analysis API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AnalysisRequest(BaseModel):
    filename: str = Field(default="race-data.csv", max_length=180)
    csv_text: str = Field(min_length=1, max_length=MAX_UPLOAD_BYTES)


def ttl_cache(seconds: int, maxsize: int):
    def decorate(function):
        cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
        lock = Lock()

        @wraps(function)
        def wrapped(*args):
            now = monotonic()
            with lock:
                cached_value = cache.get(args)
                if cached_value and cached_value[0] > now:
                    return cached_value[1]
                cache.pop(args, None)

            value = function(*args)
            with lock:
                if len(cache) >= maxsize:
                    oldest = min(cache, key=lambda key: cache[key][0])
                    cache.pop(oldest, None)
                cache[args] = (monotonic() + seconds, value)
            return value

        return wrapped

    return decorate


def _request_json(url: str, params: dict[str, Any] | None = None) -> Any:
    global _openf1_next_request
    for attempt in range(4):
        if url.startswith(OPENF1_URL):
            with _openf1_rate_lock:
                delay = _openf1_next_request - monotonic()
                if delay > 0:
                    sleep(delay)
                _openf1_next_request = monotonic() + 0.5
        try:
            response = requests.get(url, params=params, timeout=(10, 45))
            if response.status_code == 404:
                return []
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 3:
                    response.raise_for_status()
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = min(15.0, max(0.5, float(retry_after))) if retry_after else 2**attempt
                except ValueError:
                    delay = 2**attempt
                sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
        except requests.Timeout as error:
            if attempt < 2:
                sleep(2**attempt)
                continue
            raise RuntimeError(f"Data provider timed out: {urlparse_host(url)}") from error
        except requests.ConnectionError as error:
            if attempt < 2:
                sleep(2**attempt)
                continue
            raise RuntimeError(f"Data provider connection failed: {urlparse_host(url)}") from error
        except requests.RequestException as error:
            raise RuntimeError(f"Data provider request failed: {error}") from error
        except ValueError as error:
            raise RuntimeError(f"Data provider returned invalid JSON: {urlparse_host(url)}") from error
    raise RuntimeError(f"Data provider could not serve the request: {urlparse_host(url)}")


def urlparse_host(url: str) -> str:
    return url.split("/", 3)[2]


@ttl_cache(seconds=30, maxsize=256)
def _openf1(endpoint: str, session_key: int) -> list[dict[str, Any]]:
    data = _request_json(f"{OPENF1_URL}/{endpoint}", {"session_key": session_key})
    if not isinstance(data, list):
        raise RuntimeError(f"OpenF1 returned an invalid {endpoint} response.")
    return data


@ttl_cache(seconds=3600, maxsize=16)
def _openf1_schedule(year: int) -> list[dict[str, Any]]:
    data = _request_json(
        f"{OPENF1_URL}/sessions",
        {"year": year, "session_name": "Race"},
    )
    if not isinstance(data, list):
        raise RuntimeError("OpenF1 returned an invalid race schedule.")
    return sorted(
        (session for session in data if session.get("session_key") and not session.get("is_cancelled")),
        key=lambda session: session.get("date_start") or "",
    )


@ttl_cache(seconds=21600, maxsize=16)
def _jolpica_schedule(year: int) -> list[dict[str, Any]]:
    data = _request_json(f"{JOLPICA_URL}/{year}.json", {"limit": 100})
    try:
        return data["MRData"]["RaceTable"]["Races"]
    except (TypeError, KeyError) as error:
        raise RuntimeError(f"Jolpica returned an invalid {year} schedule.") from error


@ttl_cache(seconds=3600, maxsize=128)
def _jolpica_round_results(year: int, round_number: int) -> list[dict[str, Any]]:
    data = _request_json(
        f"{JOLPICA_URL}/{year}/{round_number}/results.json",
        {"limit": 100},
    )
    try:
        races = data["MRData"]["RaceTable"]["Races"]
    except (TypeError, KeyError) as error:
        raise RuntimeError(f"Jolpica returned invalid results for {year} round {round_number}.") from error
    return races[0].get("Results", []) if races else []


@ttl_cache(seconds=3600, maxsize=16)
def _jolpica_season_results(year: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    offset = 0
    while True:
        data = _request_json(
            f"{JOLPICA_URL}/{year}/results.json",
            {"limit": 100, "offset": offset},
        )
        try:
            metadata = data["MRData"]
            races = metadata["RaceTable"]["Races"]
        except (TypeError, KeyError) as error:
            raise RuntimeError(f"Jolpica returned invalid {year} season results.") from error

        for race in races:
            for result in race.get("Results", []):
                driver = result.get("Driver", {})
                constructor = result.get("Constructor", {})
                try:
                    grid = int(result.get("grid", "0"))
                    position = int(result.get("position", "0"))
                except (TypeError, ValueError):
                    continue
                if 1 <= grid <= 20 and 1 <= position <= 20:
                    results.append(
                        {
                            "season": year,
                            "round": int(race["round"]),
                            "driver_id": driver.get("driverId", ""),
                            "constructor_id": constructor.get("constructorId", ""),
                            "grid_position": grid,
                            "finish_position": position,
                        }
                    )

        offset += int(metadata.get("limit", 100))
        if offset >= int(metadata.get("total", 0)) or not races:
            break
    return results


def _race_metadata(year: int, round_number: int) -> dict[str, Any]:
    try:
        race = next(
            item for item in _jolpica_schedule(year)
            if int(item["round"]) == round_number
        )
    except StopIteration as error:
        raise HTTPException(status_code=404, detail=f"No round {round_number} exists in the {year} season.") from error

    race_date = race.get("date")
    sessions = _openf1_schedule(year)
    candidates = []
    if race_date:
        target_date = pd.Timestamp(race_date)
        for session in sessions:
            start = pd.Timestamp(session["date_start"])
            distance = abs((start.tz_localize(None).normalize() - target_date).days)
            if distance <= 1:
                candidates.append((distance, session))
    session = min(candidates, key=lambda item: item[0])[1] if candidates else None
    circuit = race.get("Circuit", {})
    location = circuit.get("Location", {})

    now = pd.Timestamp.now(tz="UTC")
    session_start = pd.Timestamp(session["date_start"]) if session else None
    session_end = pd.Timestamp(session["date_end"]) if session else None
    completed = bool(session_end and session_end < now)
    in_progress = bool(session_start and session_end and session_start <= now < session_end)
    return {
        "season": year,
        "round": round_number,
        "name": race.get("raceName", f"Round {round_number}"),
        "date": race_date,
        "time": race.get("time"),
        "location": location.get("locality", ""),
        "country": location.get("country", ""),
        "circuit": circuit.get("circuitName", ""),
        "session_key": session.get("session_key") if session else None,
        "session_start": session.get("date_start") if session else None,
        "session_end": session.get("date_end") if session else None,
        "completed": completed,
        "in_progress": in_progress,
        "status": "LIVE" if in_progress else "COMPLETED" if completed else "UPCOMING",
        "source": "OpenF1 + Jolpica",
    }


def _result_maps(results: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_code: dict[str, dict[str, Any]] = {}
    classified = []
    for result in results:
        driver = result.get("Driver", {})
        constructor = result.get("Constructor", {})
        try:
            grid = int(result.get("grid", "0"))
            finish = int(result.get("position", "0"))
        except (TypeError, ValueError):
            continue
        item = {
            "driver_code": driver.get("code") or driver.get("driverId", "").upper()[:3],
            "driver_name": " ".join(
                part for part in (driver.get("givenName"), driver.get("familyName")) if part
            ),
            "driver_id": driver.get("driverId", ""),
            "constructor_id": constructor.get("constructorId", ""),
            "team": constructor.get("name", ""),
            "team_color": "",
            "grid_position": grid if grid > 0 else None,
            "finish_position": finish if finish > 0 else None,
            "status": result.get("status", ""),
            "points": float(result.get("points", 0) or 0),
        }
        by_code[item["driver_code"]] = item
        if item["finish_position"] is not None:
            classified.append(item)
    return by_code, classified


def _history_before(year: int, round_number: int) -> list[dict[str, Any]]:
    previous_years = range(max(2022, year - 2), year)
    with ThreadPoolExecutor(max_workers=3) as executor:
        previous_seasons = [
            executor.submit(_jolpica_season_results, prior_year)
            for prior_year in previous_years
        ]
        current_season = executor.submit(_jolpica_season_results, year)
        history = []
        for season in previous_seasons:
            history.extend(season.result())
        history.extend(
            result for result in current_season.result()
            if result["round"] < round_number
        )
    return history


def _walk_forward_prediction(
    year: int,
    round_number: int,
    target_drivers: list[dict[str, Any]],
) -> dict[str, Any]:
    history = _history_before(year, round_number)
    history_frame = pd.DataFrame(history)
    target_frame = pd.DataFrame(target_drivers)
    if not target_frame.empty:
        target_frame["season"] = year
        target_frame["round"] = round_number
        target_frame = target_frame.dropna(
            subset=["driver_id", "constructor_id", "grid_position", "finish_position"]
        )
    if len(history_frame) < 40 or len(target_frame) < 8:
        return {
            "mae": None,
            "grid_baseline_mae": None,
            "training_rows": int(len(history_frame)),
            "test_rows": int(len(target_frame)),
            "driver_predictions": [],
            "message": "More prior classified race results are needed for a walk-forward model.",
        }

    categorical = ["driver_id", "constructor_id"]
    numeric = ["grid_position", "season", "round"]
    preprocessor = ColumnTransformer(
        [
            ("identity", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("race", "passthrough", numeric),
        ]
    )
    model = make_pipeline(
        preprocessor,
        XGBRegressor(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
    )
    train_features = history_frame[categorical + numeric]
    target_features = target_frame[categorical + numeric]
    model.fit(train_features, history_frame["finish_position"])
    predicted_scores = model.predict(target_features)
    ranked_indices = np.argsort(predicted_scores, kind="stable")
    predicted_positions = np.empty(len(target_frame), dtype=int)
    predicted_positions[ranked_indices] = np.arange(1, len(target_frame) + 1)

    actual = target_frame["finish_position"].to_numpy(dtype=float)
    grid_baseline = target_frame["grid_position"].to_numpy(dtype=float)
    predictions = []
    for index, row in target_frame.reset_index(drop=True).iterrows():
        predictions.append(
            {
                "driver_code": row["driver_code"],
                "predicted_position": int(predicted_positions[index]),
                "actual_finish_position": int(row["finish_position"]),
                "grid_position": int(row["grid_position"]),
                "driver_name": row["driver_name"],
                "team": row["team"],
                "team_color": row["team_color"],
                "status": row["status"],
            }
        )
    return {
        "mae": round(float(mean_absolute_error(actual, predicted_positions)), 3),
        "grid_baseline_mae": round(float(mean_absolute_error(actual, grid_baseline)), 3),
        "training_rows": int(len(history_frame)),
        "test_rows": int(len(target_frame)),
        "training_window": f"{max(2022, year - 2)}-{year} before round {round_number}",
        "driver_predictions": sorted(predictions, key=lambda item: item["predicted_position"]),
        "message": "Walk-forward holdout: trained on earlier race results only.",
    }


def _make_lap_frame(
    lap_rows: list[dict[str, Any]],
    stint_rows: list[dict[str, Any]],
    pit_rows: list[dict[str, Any]],
    drivers_by_number: dict[int, dict[str, Any]],
    results_by_code: dict[str, dict[str, Any]],
    year: int,
    round_number: int,
) -> pd.DataFrame:
    stints_by_driver: dict[int, list[dict[str, Any]]] = {}
    for stint in stint_rows:
        stints_by_driver.setdefault(int(stint["driver_number"]), []).append(stint)
    pit_laps_by_driver: dict[int, set[int]] = {}
    for pit in pit_rows:
        try:
            pit_laps_by_driver.setdefault(int(pit["driver_number"]), set()).add(int(pit["lap_number"]))
        except (KeyError, TypeError, ValueError):
            continue

    normalized_rows = []
    for lap in lap_rows:
        if lap.get("lap_duration") is None or lap.get("is_pit_out_lap"):
            continue
        driver_number = int(lap["driver_number"])
        lap_number = int(lap["lap_number"])
        if lap_number in pit_laps_by_driver.get(driver_number, set()):
            continue
        stint = next(
            (
                item for item in stints_by_driver.get(driver_number, [])
                if int(item["lap_start"]) <= lap_number
                and (item.get("lap_end") is None or lap_number <= int(item["lap_end"]))
            ),
            None,
        )
        if not stint or not stint.get("compound"):
            continue
        driver = drivers_by_number.get(driver_number, {})
        code = driver.get("name_acronym") or str(driver_number)
        result = results_by_code.get(code, {})
        tyre_age_start = stint.get("tyre_age_at_start")
        try:
            tyre_age_start = float(tyre_age_start or 0)
        except (TypeError, ValueError):
            tyre_age_start = 0.0
        normalized_rows.append(
            {
                "driver": code,
                "compound": stint["compound"],
                "lap_number": lap_number,
                "lap_time_seconds": lap["lap_duration"],
                "tyre_age": tyre_age_start + lap_number - int(stint["lap_start"]) + 1,
                "grid_position": result.get("grid_position"),
                "finish_position": result.get("finish_position"),
                "season": year,
                "round": round_number,
            }
        )
    return normalize_laps(pd.DataFrame(normalized_rows))


def _weather_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False}
    weather = pd.DataFrame(rows)
    output: dict[str, Any] = {"available": True, "samples": len(weather)}
    for source, target in (
        ("track_temperature", "track_temperature_c"),
        ("air_temperature", "air_temperature_c"),
        ("humidity", "humidity_percent"),
        ("wind_speed", "wind_speed_mps"),
        ("rainfall", "rainfall"),
    ):
        if source in weather:
            values = pd.to_numeric(weather[source], errors="coerce").dropna()
            if not values.empty:
                output[target] = round(float(values.mean()), 1)
    return output


def _strategy_rows(
    stints: list[dict[str, Any]],
    drivers_by_number: dict[int, dict[str, Any]],
    results_by_code: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for stint in stints:
        grouped.setdefault(int(stint["driver_number"]), []).append(stint)

    output = []
    for number, driver_stints in grouped.items():
        driver = drivers_by_number.get(number, {})
        code = driver.get("name_acronym") or str(number)
        result = results_by_code.get(code, {})
        ordered_stints = sorted(driver_stints, key=lambda item: int(item["stint_number"]))
        output.append(
            {
                "driver_number": number,
                "driver_code": code,
                "driver_name": driver.get("full_name") or result.get("driver_name") or code,
                "team": driver.get("team_name") or result.get("team", ""),
                "team_color": driver.get("team_colour") or "",
                "grid_position": result.get("grid_position"),
                "finish_position": result.get("finish_position"),
                "stints": [
                    {
                        "number": int(stint["stint_number"]),
                        "compound": str(stint.get("compound") or "UNKNOWN").title(),
                        "start_lap": int(stint["lap_start"]),
                        "end_lap": int(stint["lap_end"]) if stint.get("lap_end") is not None else None,
                        "tyre_age_at_start": stint.get("tyre_age_at_start"),
                    }
                    for stint in ordered_stints
                ],
            }
        )
    return sorted(
        output,
        key=lambda item: (item["finish_position"] is None, item["finish_position"] or 99),
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "apex-lab", "data_source": "OpenF1 + Jolpica"}


@app.get("/api/seasons")
def available_seasons() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    years = []
    for year in range(MIN_SEASON, now.year + 1):
        try:
            if _openf1_schedule(year):
                years.append(year)
        except RuntimeError as error:
            if year == now.year:
                continue
            raise HTTPException(status_code=502, detail=str(error)) from error
    if not years:
        raise HTTPException(status_code=502, detail="No race schedules are currently available from OpenF1.")
    return {"seasons": years, "latest_season": max(years)}


@app.get("/api/seasons/{year}/races")
def season_races(year: int) -> dict[str, Any]:
    if year < MIN_SEASON or year > datetime.now(timezone.utc).year:
        raise HTTPException(status_code=422, detail=f"Choose a season between {MIN_SEASON} and the current year.")
    try:
        races = []
        for race in _jolpica_schedule(year):
            metadata = _race_metadata(year, int(race["round"]))
            metadata["available"] = metadata["session_key"] is not None and (
                metadata["completed"] or metadata["in_progress"]
            )
            races.append(metadata)
        return {"season": year, "races": races}
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@app.post("/api/analyze")
def analyze_csv(request: AnalysisRequest) -> dict[str, Any]:
    if len(request.csv_text.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="CSV uploads must be 20 MB or smaller.")
    try:
        raw = pd.read_csv(StringIO(request.csv_text))
        laps = normalize_laps(raw)
        result = analyze_laps(laps)
        result["filename"] = Path(request.filename).name
        result["source"] = "CSV"
        return result
    except pd.errors.EmptyDataError as error:
        raise HTTPException(status_code=422, detail="The uploaded CSV is empty.") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not analyze CSV: {error}") from error


@app.get("/api/races/{year}/{round_number}")
def load_race(year: int, round_number: int) -> dict[str, Any]:
    current_year = datetime.now(timezone.utc).year
    if year < MIN_SEASON or year > current_year:
        raise HTTPException(status_code=422, detail=f"Choose a season between {MIN_SEASON} and {current_year}.")
    try:
        event = _race_metadata(year, round_number)
        if not event["session_key"]:
            raise HTTPException(
                status_code=409,
                detail="OpenF1 does not have timing data for this race yet.",
            )
        if not event["completed"] and not event["in_progress"]:
            raise HTTPException(
                status_code=409,
                detail="Race timing data is not available until the race session has finished.",
            )
        session_key = int(event["session_key"])
        with ThreadPoolExecutor(max_workers=4) as executor:
            laps_future = executor.submit(_openf1, "laps", session_key)
            stints_future = executor.submit(_openf1, "stints", session_key)
            drivers_future = executor.submit(_openf1, "drivers", session_key)
            weather_future = executor.submit(_openf1, "weather", session_key)
            pits_future = executor.submit(_openf1, "pit", session_key)
            positions_future = (
                executor.submit(_openf1, "position", session_key)
                if event["in_progress"]
                else None
            )
            lap_rows = laps_future.result()
            stint_rows = stints_future.result()
            driver_rows = drivers_future.result()
            weather_rows = weather_future.result()
            pit_rows = pits_future.result()
            position_rows = positions_future.result() if positions_future else []
        results = _jolpica_round_results(year, round_number)
        results_by_code, classified = _result_maps(results)
        drivers_by_number = {
            int(driver["driver_number"]): driver
            for driver in driver_rows
            if driver.get("driver_number") is not None
        }
        for driver in driver_rows:
            code = driver.get("name_acronym")
            if code in results_by_code:
                results_by_code[code]["team_color"] = driver.get("team_colour") or ""
                results_by_code[code]["team"] = driver.get("team_name") or results_by_code[code]["team"]

        if event["in_progress"]:
            positions_by_number: dict[int, list[dict[str, Any]]] = {}
            for position in position_rows:
                try:
                    number = int(position["driver_number"])
                except (KeyError, TypeError, ValueError):
                    continue
                positions_by_number.setdefault(number, []).append(position)
            classified = []
            for number, driver_positions in positions_by_number.items():
                driver = drivers_by_number.get(number, {})
                code = driver.get("name_acronym") or str(number)
                ordered_positions = sorted(driver_positions, key=lambda row: row.get("date", ""))
                grid = ordered_positions[0].get("position")
                current = ordered_positions[-1].get("position")
                if not grid or not current:
                    continue
                result = results_by_code.setdefault(
                    code,
                    {
                        "driver_code": code,
                        "driver_name": driver.get("full_name") or code,
                        "driver_id": code.lower(),
                        "constructor_id": (driver.get("team_name") or "").lower().replace(" ", "_"),
                        "team": driver.get("team_name") or "",
                        "team_color": driver.get("team_colour") or "",
                        "grid_position": int(grid),
                        "finish_position": int(current),
                        "status": "In progress",
                        "points": 0.0,
                    },
                )
                result["grid_position"] = int(grid)
                result["finish_position"] = int(current)
                result["status"] = "In progress"
                classified.append(result)

        laps = _make_lap_frame(
            lap_rows,
            stint_rows,
            pit_rows,
            drivers_by_number,
            results_by_code,
            year,
            round_number,
        )
        analysis = analyze_laps(laps, include_prediction=False, allow_unidentifiable=True)
        event["lap_count"] = int(laps["lap_number"].max())
        event["session_key"] = session_key
        event["event_name"] = event["name"]
        event["name"] = event["name"].replace(" Grand Prix", "")
        driver_results = []
        for result in classified:
            code = result["driver_code"]
            driver = next(
                (
                    item for item in driver_rows
                    if item.get("name_acronym") == code
                ),
                {},
            )
            driver_results.append(
                {
                    **result,
                    "team_color": driver.get("team_colour") or "",
                    "finish_position": result["finish_position"],
                }
            )

        analysis.update(
            {
                "source": "OpenF1 + Jolpica",
                "event": event,
                "weather": _weather_summary(weather_rows),
                "drivers": sorted(driver_results, key=lambda item: item["finish_position"]),
                "strategies": _strategy_rows(stint_rows, drivers_by_number, results_by_code),
                "prediction": _walk_forward_prediction(year, round_number, driver_results)
                if event["completed"]
                else {
                    "mae": None,
                    "grid_baseline_mae": None,
                    "training_rows": 0,
                    "test_rows": 0,
                    "driver_predictions": [],
                    "message": "A finishing-position benchmark is available after the race is classified.",
                },
            }
        )
        return analysis
    except HTTPException:
        raise
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Race analysis failed: {error}") from error
