# Apex Lab — F1 Race Strategy Analytics

A React dashboard and Python API for real Formula 1 race timing, tyre stints, weather, classified results, tyre degradation, and walk-forward finishing-position model evaluation.

## Run locally

### Start the API

From the repository root, create an isolated Python environment and install the backend dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m uvicorn main:app --app-dir backend --reload --port 8000
```

The API health check is at `http://localhost:8000/api/health`.

### Start the dashboard

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (usually `http://localhost:5173`). The Vite server proxies `/api` to the Python API. On start, the dashboard loads a live session if one is in progress, otherwise the latest completed race with available timing data. Changing the season or race loads its data automatically. During a live race the dashboard polls session timing every 30 seconds; session API responses are cached for 30 seconds to reduce provider load.

## Real data and analysis

- **Race schedule and lap timing:** [OpenF1](https://openf1.org/) provides the race sessions, driver laps, tyre stints, live positions, and weather. Historical timing is available from 2023 onward.
- **Starting grid and classification:** [Jolpica-F1](https://github.com/jolpica/jolpica-f1) supplies the official grid and finish positions and supplements OpenF1 driver/team details.
- **Tyre pace chart:** lap times are centered on each driver's median, adjusted for the robust race-progress trend when identifiable, and extreme residuals are omitted using a 4.5-scaled-MAD threshold (at least four seconds). The chart then shows the cross-driver median and interquartile range per compound and tyre age. Ages represented by fewer than three drivers are omitted, and gaps are not connected. Pace is a relative comparison, not raw lap time or a direct measure of tyre wear.
- **Tyre degradation:** pit-in and pit-out laps are excluded. Statsmodels OLS estimates lap-time change per tyre-age lap while controlling for lap progression as a fuel-load proxy, compound, and driver. Tyre age counts the completed lap, so the first lap on a new set is age one. If tyre age cannot be distinguished from race progression, the API reports the regression as unavailable rather than returning misleading slopes.
- **Grid relationship:** Spearman correlation compares actual starting grid and classified finish positions.
- **XGBoost:** The selected race is held out; the model trains only on earlier classified results from the preceding two seasons and earlier rounds of the selected season. It is compared with a grid-position ranking baseline. This is a retrospective benchmark, not a forecast for an unrun race.
- **CSV:** Upload a lap-level CSV with FastF1-style `Driver`, `Compound`, `LapNumber`, `LapTime`, `TyreLife` columns. `GridPosition`, `Position`, `season`, and `round` enable the additional position analysis. Lap times can be seconds or durations such as `0 days 00:01:32.500`.

Completed race data is fetched when requested; during a scheduled race session, current positions and lap data refresh as providers publish them. OpenF1 session data is cached for 30 seconds, Jolpica results for one hour, and calendars for up to six hours. This is a polling dashboard, not a millisecond real-time telemetry stream.

## Project layout

```text
frontend/
  src/App.jsx           React dashboard and race data views
  src/styles.css        Responsive dashboard styles
  vite.config.js        Local API proxy
backend/
  main.py               FastAPI endpoints and provider integrations
  analysis.py           Pandas, Statsmodels, SciPy, scikit-learn, XGBoost
  requirements.txt      Python dependencies
```
