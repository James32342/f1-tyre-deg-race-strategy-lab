import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import analyze_laps, normalize_laps
from main import _make_lap_frame, app, load_race


def build_laps(single_compound=False):
    rows = []
    for season in (2023, 2024):
        for round_number in (1, 2):
            for driver in range(1, 11):
                soft_end = 6 + driver % 4
                finish = ((driver + round_number + season - 2023 - 1) % 10) + 1
                for lap in range(1, 17):
                    if single_compound:
                        compound = "Hard"
                        tyre_age = lap if driver % 2 else max(1, lap - (driver % 4))
                    else:
                        compound = "Soft" if lap <= soft_end else "Medium"
                        tyre_age = lap if lap <= soft_end else lap - soft_end
                    rows.append(
                        {
                            "Driver": f"D{driver:02}",
                            "Compound": compound,
                            "LapNumber": lap,
                            "LapTime": 80 + driver * 0.025 + lap * 0.018
                            + tyre_age * (0.13 if compound == "Soft" else 0.065),
                            "TyreLife": tyre_age,
                            "GridPosition": driver,
                            "Position": finish,
                            "season": season,
                            "round": round_number,
                        }
                    )
    return pd.DataFrame(rows)


class AnalysisTests(unittest.TestCase):
    def test_normalizes_duration_columns_and_estimates_compound_degradation(self):
        source = build_laps()
        source["LapTime"] = pd.to_timedelta(source["LapTime"], unit="s")
        laps = normalize_laps(source)
        result = analyze_laps(laps)

        slopes = {item["compound"]: item["slope_sec_per_lap"] for item in result["degradation"]}
        self.assertAlmostEqual(slopes["Soft"] - slopes["Medium"], 0.065, places=2)
        self.assertGreater(len(result["pace_by_age"]), 0)
        self.assertIsNotNone(result["compound_anova"]["p_value"])

    def test_single_compound_race_remains_analyzable(self):
        result = analyze_laps(normalize_laps(build_laps(single_compound=True)))
        self.assertEqual(len(result["degradation"]), 1)
        self.assertFalse(result["compound_anova"]["available"])

    def test_pace_curve_uses_driver_adjusted_robust_medians(self):
        rows = []
        for driver in range(1, 6):
            for lap in range(1, 9):
                rows.append(
                    {
                        "Driver": f"D{driver}",
                        "Compound": "Medium",
                        "LapNumber": lap,
                        "LapTime": 80 + driver * 2 + lap * 0.1 + lap**2 * 0.002,
                        "TyreLife": lap,
                        "season": 2025,
                        "round": 1,
                    }
                )
        clean = analyze_laps(
            normalize_laps(pd.DataFrame(rows)),
            include_prediction=False,
            allow_unidentifiable=True,
        )
        rows[3]["LapTime"] = 600
        with_outlier = analyze_laps(
            normalize_laps(pd.DataFrame(rows)),
            include_prediction=False,
            allow_unidentifiable=True,
        )

        self.assertTrue(clean["pace_adjustment"]["driver_normalized"])
        self.assertTrue(clean["pace_adjustment"]["race_progress_adjusted"])
        self.assertEqual(len(clean["pace_by_age"]), 8)
        self.assertEqual(clean["pace_by_age"][3]["drivers"], 5)
        for expected, actual in zip(clean["pace_by_age"], with_outlier["pace_by_age"]):
            self.assertAlmostEqual(
                expected["median_delta_seconds"],
                actual["median_delta_seconds"],
                places=2,
            )

    def test_openf1_first_lap_tyre_age_starts_at_one(self):
        laps = _make_lap_frame(
            [
                {"driver_number": 1, "lap_number": lap, "lap_duration": 90 + lap, "is_pit_out_lap": False}
                for lap in range(1, 13)
            ],
            [
                {
                    "driver_number": 1,
                    "stint_number": 1,
                    "lap_start": 1,
                    "lap_end": 12,
                    "compound": "MEDIUM",
                    "tyre_age_at_start": 0,
                }
            ],
            [],
            {1: {"name_acronym": "TST"}},
            {},
            2025,
            1,
        )
        self.assertEqual(laps.loc[laps["lap_number"] == 1, "tyre_age"].item(), 1)

    def test_live_race_can_return_lap_data_without_unidentifiable_regression(self):
        source = build_laps(single_compound=True)
        source["TyreLife"] = source["LapNumber"]
        result = analyze_laps(
            normalize_laps(source),
            include_prediction=False,
            allow_unidentifiable=True,
        )
        self.assertFalse(result["race_progress_model"]["available"])
        self.assertEqual(result["degradation"], [])
        self.assertTrue(result["warnings"])
        self.assertIsNone(result["prediction"])

    def test_csv_endpoint_returns_analysis_and_validates_required_columns(self):
        client = TestClient(app)
        valid = client.post(
            "/api/analyze",
            json={"filename": "race.csv", "csv_text": build_laps().to_csv(index=False)},
        )
        self.assertEqual(valid.status_code, 200, valid.text)
        self.assertEqual(valid.json()["source"], "CSV")

        invalid = client.post(
            "/api/analyze",
            json={"filename": "invalid.csv", "csv_text": "Driver,LapTime\nVER,91"},
        )
        self.assertEqual(invalid.status_code, 422)

    def test_live_race_uses_current_positions_without_claiming_a_finish(self):
        event = {
            "season": 2025,
            "round": 1,
            "name": "Test Grand Prix",
            "session_key": 123,
            "session_start": "2025-03-01T10:00:00+00:00",
            "session_end": "2025-03-01T12:00:00+00:00",
            "completed": False,
            "in_progress": True,
            "status": "LIVE",
        }
        driver_rows = [
            {
                "driver_number": number,
                "name_acronym": f"D{number}",
                "full_name": f"Driver {number}",
                "team_name": "Test Team",
                "team_colour": "123456",
            }
            for number in range(1, 5)
        ]
        lap_rows = [
            {
                "driver_number": number,
                "lap_number": lap,
                "lap_duration": 90 + number + lap * 0.1,
                "is_pit_out_lap": False,
            }
            for number in range(1, 5)
            for lap in range(1, 5)
        ]
        stint_rows = [
            {
                "driver_number": number,
                "stint_number": 1,
                "lap_start": 1,
                "lap_end": 4,
                "compound": "HARD",
                "tyre_age_at_start": 0,
            }
            for number in range(1, 5)
        ]
        position_rows = [
            {"driver_number": number, "position": position, "date": "2025-03-01T11:00:00+00:00"}
            for number, position in zip(range(1, 5), (2, 1, 3, 4))
        ]
        endpoint_data = {
            "laps": lap_rows,
            "stints": stint_rows,
            "drivers": driver_rows,
            "weather": [],
            "pit": [],
            "position": position_rows,
        }

        with patch("main._race_metadata", return_value=event), patch(
            "main._openf1", side_effect=lambda endpoint, _key: endpoint_data[endpoint]
        ), patch("main._jolpica_round_results", return_value=[]):
            result = load_race(2025, 1)

        self.assertEqual(result["event"]["status"], "LIVE")
        self.assertEqual(result["drivers"][0]["driver_code"], "D2")
        self.assertEqual(result["drivers"][0]["finish_position"], 1)
        self.assertIsNone(result["prediction"]["mae"])


if __name__ == "__main__":
    unittest.main()
