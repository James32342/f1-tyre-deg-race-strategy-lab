import { useEffect, useRef, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
const COMPOUND_COLORS = {
  soft: '#fa6262',
  medium: '#f1c84b',
  hard: '#e8edf5',
  intermediate: '#56c68a',
  wet: '#67a3ff'
};

function compoundColor(compound) {
  return COMPOUND_COLORS[compound?.toLowerCase()] || '#b58af6';
}

function Icon({ name, size = 18 }) {
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
    tyre: <><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="4" /><path d="M12 3v5m0 8v5M3 12h5m8 0h5" /></>,
    chart: <><path d="M4 19V5m0 14h17" /><path d="m7 15 4-4 3 2 6-7" /></>,
    upload: <><path d="M12 16V4m-5 5 5-5 5 5" /><path d="M4 16v4h16v-4" /></>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6" /></>,
    chevron: <path d="m9 18 6-6-6-6" />,
    flag: <><path d="M5 21V4" /><path d="M5 4c6-4 8 4 14 0v11c-6 4-8-4-14 0" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    check: <path d="m5 12 4 4L19 6" />
  };
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>;
}

function PaceChart({ points, raceProgressAdjusted }) {
  const compounds = [...new Set(points.map((point) => point.compound))];
  if (!points.length) {
    return <div className="empty-chart">Not enough tyre-age samples from three or more drivers to draw a reliable curve.</div>;
  }

  const width = 680;
  const height = 220;
  const ages = points.map((point) => point.tyre_age);
  const paceBounds = points.flatMap((point) => [point.q1_delta_seconds, point.q3_delta_seconds]);
  const minAge = Math.min(...ages);
  const maxAge = Math.max(...ages);
  const paceRange = Math.max(0.5, Math.max(...paceBounds) - Math.min(...paceBounds));
  const minPace = Math.min(0, ...paceBounds) - paceRange * 0.08;
  const maxPace = Math.max(0, ...paceBounds) + paceRange * 0.08;
  const paceSpan = maxPace - minPace;
  const ageRange = Math.max(1, maxAge - minAge);
  const y = (value) => 12 + (maxPace - value) * (height - 24) / paceSpan;
  const x = (value) => 12 + (value - minAge) * (width - 24) / ageRange;
  const ticks = Array.from({ length: 4 }, (_, index) => maxPace - index * paceSpan / 3);
  const formatDelta = (value) => `${value > 0 ? '+' : ''}${value.toFixed(1)}s`;

  return (
    <div className="chart-wrap">
      <div className="chart-legend">
        {compounds.map((compound) => <span className="legend-item" key={compound}><i className="legend-dot" style={{ background: compoundColor(compound) }} />{compound}</span>)}
        <span className="chart-unit">Pace delta · seconds</span>
      </div>
      <div className="chart-canvas">
        <div className="y-labels">{ticks.map((tick) => <span key={tick}>{formatDelta(tick)}</span>)}</div>
        <svg className="pace-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Driver- and race-progress-adjusted median lap pace by tyre age and compound">
          {ticks.map((tick) => <line key={tick} x1="0" x2={width} y1={y(tick)} y2={y(tick)} className="chart-grid" />)}
          {minPace <= 0 && maxPace >= 0 && <line x1="0" x2={width} y1={y(0)} y2={y(0)} className="chart-zero" />}
          {compounds.map((compound) => {
            const series = points.filter((point) => point.compound === compound).sort((a, b) => a.tyre_age - b.tyre_age);
            const segments = [];
            series.forEach((point) => {
              const segment = segments[segments.length - 1];
              if (!segment || point.tyre_age - segment[segment.length - 1].tyre_age > 1.01) segments.push([point]);
              else segment.push(point);
            });
            return <g key={compound}>
              {segments.filter((segment) => segment.length > 1).map((segment, index) => {
                const path = segment.map((point, pointIndex) => `${pointIndex ? 'L' : 'M'}${x(point.tyre_age).toFixed(1)},${y(point.median_delta_seconds).toFixed(1)}`).join(' ');
                return <path key={index} d={path} fill="none" stroke={compoundColor(compound)} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />;
              })}
              {series.map((point) => <g key={point.tyre_age}>
                <line x1={x(point.tyre_age)} x2={x(point.tyre_age)} y1={y(point.q1_delta_seconds)} y2={y(point.q3_delta_seconds)} stroke={compoundColor(compound)} strokeOpacity=".58" strokeWidth="3" />
                <circle cx={x(point.tyre_age)} cy={y(point.median_delta_seconds)} r="3.5" fill={compoundColor(compound)}>
                  <title>{`${compound} · tyre age ${point.tyre_age} · median ${formatDelta(point.median_delta_seconds)} · middle 50% ${formatDelta(point.q1_delta_seconds)} to ${formatDelta(point.q3_delta_seconds)} · ${point.drivers} drivers · ${point.observations} laps`}</title>
                </circle>
              </g>)}
            </g>;
          })}
        </svg>
        <div className="x-labels"><span>{minAge}</span><span>{((minAge + maxAge) / 2).toFixed(0)}</span><span>{maxAge} laps</span></div>
      </div>
      <div className="chart-axis-caption">TYRE AGE · LAPS <span>{raceProgressAdjusted ? 'DRIVER & RACE-PROGRESS ADJUSTED' : 'DRIVER-ADJUSTED · RACE-PROGRESS UNAVAILABLE'}</span></div>
    </div>
  );
}

function MetricCard({ label, value, unit = '', trend, icon, tone }) {
  return <article className="metric-card"><div className="metric-top"><span>{label}</span><i className={`metric-icon ${tone}`}><Icon name={icon} size={16} /></i></div><div className="metric-value">{value}<small>{unit}</small></div><div className="metric-trend"><span className="trend-marker">·</span>{trend}</div></article>;
}

function App() {
  const [seasons, setSeasons] = useState([]);
  const [season, setSeason] = useState('');
  const [races, setRaces] = useState([]);
  const [roundNumber, setRoundNumber] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [driverCode, setDriverCode] = useState('');
  const [activeTab, setActiveTab] = useState('Overview');
  const [loadingSchedule, setLoadingSchedule] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState('');
  const [notice, setNotice] = useState('');
  const [refreshKey, setRefreshKey] = useState(0);
  const fileInput = useRef(null);

  const selectedRace = races.find((race) => race.round === Number(roundNumber));
  const event = analysis?.event || selectedRace;
  const selectedDriver = analysis?.drivers?.find((driver) => driver.driver_code === driverCode) || analysis?.drivers?.[0];
  const selectedPrediction = analysis?.prediction?.driver_predictions?.find((item) => item.driver_code === (selectedDriver?.driver_code || driverCode));
  const weather = analysis?.weather;
  const topDegradation = analysis?.degradation?.slice().sort((a, b) => b.slope_sec_per_lap - a.slope_sec_per_lap)[0];
  const trackCondition = weather?.available
    ? weather.rainfall > 0 ? 'WET' : 'DRY'
    : '—';

  useEffect(() => {
    const controller = new AbortController();
    async function loadSeasons() {
      try {
        const response = await fetch(`${API_BASE}/api/seasons`, { signal: controller.signal });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Could not load race seasons.');
        setSeasons(result.seasons);
        setSeason(String(result.latest_season));
        setAnalysisError('');
      } catch (error) {
        if (error.name !== 'AbortError') setAnalysisError(error.message || 'Could not connect to the race data API.');
      } finally {
        if (!controller.signal.aborted) setLoadingSchedule(false);
      }
    }
    loadSeasons();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!season) return undefined;
    const controller = new AbortController();
    async function loadSchedule() {
      setLoadingSchedule(true);
      setRaces([]);
      setRoundNumber('');
      setAnalysis(null);
      try {
        const response = await fetch(`${API_BASE}/api/seasons/${season}/races`, { signal: controller.signal });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || `Could not load the ${season} race schedule.`);
        setRaces(result.races);
        const activeRace = result.races.find((race) => race.in_progress && race.available);
        const lastCompleted = result.races.slice().reverse().find((race) => race.completed && race.available);
        const initialRace = activeRace || lastCompleted;
        if (initialRace) {
          setRoundNumber(String(initialRace.round));
          setAnalysisError('');
        } else {
          setAnalysisError(`No completed ${season} race data is available yet.`);
        }
      } catch (error) {
        if (error.name !== 'AbortError') setAnalysisError(error.message || 'Could not load the race schedule.');
      } finally {
        if (!controller.signal.aborted) setLoadingSchedule(false);
      }
    }
    loadSchedule();
    return () => controller.abort();
  }, [season]);

  useEffect(() => {
    if (!season) return undefined;
    const timer = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/seasons/${season}/races`);
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Could not refresh the race calendar.');
        setRaces(result.races);
        const liveRace = result.races.find((race) => race.in_progress && race.available);
        if (liveRace && roundNumber !== String(liveRace.round)) {
          setRoundNumber(String(liveRace.round));
        }
      } catch (error) {
        setAnalysisError(error.message || 'Could not refresh the race calendar.');
      }
    }, 60_000);
    return () => window.clearInterval(timer);
  }, [season, roundNumber]);

  useEffect(() => {
    if (!selectedRace?.in_progress) return undefined;
    const timer = window.setInterval(() => {
      setRefreshKey((value) => value + 1);
    }, 30_000);
    return () => window.clearInterval(timer);
  }, [selectedRace?.in_progress]);

  useEffect(() => {
    if (!season || !roundNumber) return undefined;
    const controller = new AbortController();
    async function loadRace() {
      setIsAnalyzing(true);
      setAnalysis(null);
      setAnalysisError('');
      try {
        const response = await fetch(`${API_BASE}/api/races/${season}/${roundNumber}`, { signal: controller.signal });
        const result = await response.json();
        if (!response.ok) throw new Error(result.detail || 'Could not load race data.');
        setAnalysis(result);
        setDriverCode(result.drivers?.[0]?.driver_code || '');
      } catch (error) {
        if (error.name !== 'AbortError') setAnalysisError(error.message || 'Could not connect to the race analysis API.');
      } finally {
        if (!controller.signal.aborted) setIsAnalyzing(false);
      }
    }
    loadRace();
    return () => controller.abort();
  }, [season, roundNumber, refreshKey]);

  function chooseNav(label, target) {
    setActiveTab(label);
    document.getElementById(target)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function analyzeCsv(file) {
    if (!file) return;
    const controller = new AbortController();
    setIsAnalyzing(true);
    setAnalysisError('');
    setNotice('');
    file.text().then(async (csvText) => {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name, csv_text: csvText }),
        signal: controller.signal
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'CSV analysis failed.');
      setAnalysis(result);
      setNotice(`CSV analysis complete · ${result.rows_analyzed.toLocaleString()} lap records`);
    }).catch((error) => {
      if (error.name !== 'AbortError') setAnalysisError(error.message || 'Could not analyze the CSV.');
    }).finally(() => {
      setIsAnalyzing(false);
      if (fileInput.current) fileInput.current.value = '';
    });
  }

  const seasonsLabel = seasons.length ? `${seasons[0]} — ${seasons[seasons.length - 1]}` : 'Loading seasons';
  const chartPoints = analysis?.pace_by_age || [];
  const strategies = analysis?.strategies || [];
  const predictions = analysis?.prediction;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#" aria-label="Race Strategy Lab home">
          <span className="brand-mark"><span /></span>
          <span className="brand-copy"><strong>APEX<span> / </span>LAB</strong><small>RACE INTELLIGENCE</small></span>
        </a>
        <div className="sidebar-caption">WORKSPACE</div>
        <nav className="side-nav" aria-label="Dashboard sections">
          <button className={activeTab === 'Overview' ? 'nav-item active' : 'nav-item'} onClick={() => chooseNav('Overview', 'overview')}><Icon name="grid" />Overview</button>
          <button className={activeTab === 'Tyre analysis' ? 'nav-item active' : 'nav-item'} onClick={() => chooseNav('Tyre analysis', 'tyres')}><Icon name="tyre" />Tyre analysis</button>
          <button className={activeTab === 'Race results' ? 'nav-item active' : 'nav-item'} onClick={() => chooseNav('Race results', 'predictor')}><Icon name="chart" />Race results</button>
        </nav>
        <div className="sidebar-bottom">
          <div className="season-card">
            <span className="season-kicker"><span className="live-dot" /> DATA WINDOW</span>
            <strong>{seasonsLabel}</strong>
            <small>OpenF1 timing · Jolpica results</small>
          </div>
          <div className="profile-chip"><span className="avatar">AL</span><span><strong>Analyst workspace</strong><small>Live data session</small></span><span className="profile-more">···</span></div>
        </div>
      </aside>

      <main className="main-content" id="overview">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><Icon name="chevron" size={14} /><strong>Race overview</strong></div>
          <div className="top-actions">
            <span className="data-badge"><span className="live-dot" />{event?.in_progress ? 'LIVE SESSION' : analysis ? analysis.source : 'RACE DATA'}</span>
            <button className="icon-button" aria-label="Upload race CSV" title="Upload race CSV" onClick={() => fileInput.current?.click()}><Icon name="upload" /></button>
            <input ref={fileInput} className="visually-hidden" type="file" accept=".csv,text/csv" onChange={(event) => analyzeCsv(event.target.files?.[0])} />
            <span className="avatar small-avatar">AL</span>
          </div>
        </header>

        <div className="page-wrap">
          <section className="page-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> RACE STRATEGY INTELLIGENCE</div>
              <h1>Race overview<span className="heading-period">.</span></h1>
              <p>Real race timing, tyre stints, weather, and classified results.</p>
            </div>
            <div className="selectors">
              <label className="select-field"><span>SEASON</span><select aria-label="SEASON" value={season} onChange={(item) => setSeason(item.target.value)} disabled={!seasons.length}>{seasons.map((year) => <option key={year} value={year}>{year}</option>)}</select></label>
              <label className="select-field race-select"><span>GRAND PRIX</span><select aria-label="GRAND PRIX" value={roundNumber} onChange={(item) => setRoundNumber(item.target.value)} disabled={!races.length}>{races.map((item) => <option key={item.round} value={item.round} disabled={!item.available}>{item.name}</option>)}</select></label>
              <button className="load-race-button" type="button" onClick={() => setRefreshKey((value) => value + 1)} disabled={!roundNumber || isAnalyzing}><Icon name="clock" size={14} />Refresh data</button>
            </div>
          </section>

          {(notice || analysisError) && <div className={analysisError ? 'notice error-notice' : 'notice'} role={analysisError ? 'alert' : 'status'}>{analysisError || notice}<button onClick={() => { setNotice(''); setAnalysisError(''); }} aria-label="Dismiss message">×</button></div>}

          <section className="race-banner">
            <div className="race-watermark">{String(event?.round || roundNumber || '').padStart(2, '0')}</div>
            <div className="race-banner-content">
              <div className="round-label"><span>ROUND {String(event?.round || roundNumber || '—').padStart(2, '0')}</span><span className="round-separator">/</span><span>{event?.season || season || '—'} SEASON</span></div>
              <h2>{event?.event_name || event?.name || (loadingSchedule ? 'Loading race calendar…' : 'Select a race')}</h2>
              <div className="race-meta"><span><span className="meta-dot" />{event?.location ? `${event.location}, ${event.country}` : 'Race location unavailable'}</span><span><Icon name="flag" size={15} /> {event?.lap_count ? `${event.lap_count} LAPS` : '— LAPS'}</span><span><Icon name="clock" size={15} /> {event?.date || 'RACE DATE'}</span></div>
            </div>
            <div className="banner-art" aria-hidden="true"><div className="track-line track-one" /><div className="track-line track-two" /><div className="banner-number">F1</div></div>
            <div className="banner-tag"><span className="live-dot" />{analysis?.source || 'WAITING FOR DATA'}</div>
          </section>

          <section className="metric-grid" aria-label="Race metrics">
            <MetricCard label="TYRE-AGE EFFECT" value={topDegradation ? `${topDegradation.slope_sec_per_lap.toFixed(3)}s` : '—'} unit={topDegradation ? '/ lap' : ''} trend={topDegradation ? `${topDegradation.compound} · race-progress adjusted` : 'Requires lap timing and tyre data'} icon="tyre" tone="orange" />
            <MetricCard label={event?.in_progress ? 'GRID → CURRENT' : 'GRID → FINISH'} value={analysis?.grid_finish?.spearman_rho != null ? analysis.grid_finish.spearman_rho.toFixed(2) : '—'} trend={analysis?.grid_finish?.spearman_rho != null ? `Spearman · ${analysis.grid_finish.drivers} drivers` : 'Race classification unavailable'} icon="chart" tone="blue" />
            <MetricCard label="XGBOOST MAE" value={predictions?.mae != null ? predictions.mae.toFixed(2) : '—'} unit={predictions?.mae != null ? 'pos' : ''} trend={predictions?.mae != null ? `Prior-race holdout · grid ${predictions.grid_baseline_mae.toFixed(2)}` : 'Insufficient prior race data'} icon="grid" tone="violet" />
            <MetricCard label="TRACK TEMPERATURE" value={weather?.track_temperature_c != null ? `${weather.track_temperature_c}°` : '—'} unit={weather?.track_temperature_c != null ? 'C' : ''} trend={weather?.air_temperature_c != null ? `Air ${weather.air_temperature_c}°C · ${trackCondition}` : 'Weather data unavailable'} icon="clock" tone="green" />
          </section>

          <div className="content-grid">
            <section className="panel pace-panel" id="tyres">
              <div className="panel-header">
                <div><div className="section-kicker">TYRE PERFORMANCE</div><h3>Lap pace by tyre age</h3></div>
                <span className="data-source-note">{analysis ? `${analysis.rows_analyzed.toLocaleString()} valid laps` : 'Waiting for race data'}</span>
              </div>
              <div className="chart-summary"><span><i className="summary-dot orange" />MEDIAN PACE DELTA · INTERQUARTILE RANGE</span><span className="illustrative-tag">{analysis ? 'RACE DATA' : 'NO DATA'}</span></div>
              <PaceChart points={chartPoints} raceProgressAdjusted={analysis?.pace_adjustment?.race_progress_adjusted} />
              <div className="chart-footer"><span>Per-driver medians · sparse ages and {analysis?.pace_adjustment?.outliers_removed ?? 0} extreme laps omitted</span><span>FASTER <b>←</b> 0 <b>→</b> SLOWER</span></div>
            </section>

            <section className="panel predictor-panel" id="predictor">
              <div className="panel-header">
                <div><div className="section-kicker">CLASSIFIED RESULT</div><h3>Driver result</h3></div>
                <span className="model-pill"><span />WALK-FORWARD</span>
              </div>
              <div className="driver-picker">
                <label htmlFor="driver-select">SELECT DRIVER</label>
                <select id="driver-select" value={selectedDriver?.driver_code || ''} onChange={(item) => setDriverCode(item.target.value)} disabled={!analysis?.drivers?.length}>
                  {(analysis?.drivers || []).map((item) => <option key={item.driver_code} value={item.driver_code}>{item.driver_name}</option>)}
                </select>
                <span className="driver-team"><span style={{ background: `#${selectedDriver?.team_color || '777'}` }} />{selectedDriver?.team || 'No result loaded'}</span>
              </div>
              <div className="prediction-display">
                <div><span className="prediction-label">{event?.in_progress ? 'CURRENT POSITION' : 'ACTUAL FINISH'}</span><div className="position-number">{selectedDriver?.finish_position ? String(selectedDriver.finish_position).padStart(2, '0') : '—'}<sup>{selectedDriver?.status || 'POS'}</sup></div></div>
                <div className="prediction-change"><span>{selectedPrediction?.predicted_position ? `MODEL P${selectedPrediction.predicted_position}` : `GRID P${selectedDriver?.grid_position ?? '—'}`}</span><small>{selectedPrediction ? 'prior-race estimate' : event?.in_progress ? 'live timing position' : 'actual starting grid'}</small></div>
              </div>
              <div className="prediction-summary"><span>GRID</span><strong>{selectedDriver?.grid_position ?? '—'}</strong><span>MODEL</span><strong>{selectedPrediction?.predicted_position ?? '—'}</strong><span>FINISH</span><strong>{selectedDriver?.finish_position ?? '—'}</strong></div>
              <div className="model-foot"><span>MODEL</span><strong>Earlier race results only</strong><span className="model-foot-note">{predictions?.training_rows ? `${predictions.training_rows} training rows` : 'Walk-forward unavailable'}</span></div>
            </section>
          </div>

          <div className="lower-grid" id="tyre-details">
            <section className="panel strategy-panel">
              <div className="panel-header">
                <div><div className="section-kicker">STINT COMPARISON</div><h3>Actual race strategies</h3></div>
                <span className="result-count">{strategies.length || 0} drivers</span>
              </div>
              <div className="strategy-legend"><span>DRIVER</span><span className="stint-label">TYRE STINT · LAPS 1—{event?.lap_count || '—'}</span><span>FINISH</span></div>
              <div className="strategy-list">
                {strategies.length ? strategies.map((item) => {
                  const raceLaps = event?.lap_count || Math.max(...item.stints.map((stint) => stint.end_lap || stint.start_lap), 1);
                  return <div className="strategy-row" key={item.driver_code}>
                    <div className="strategy-driver"><span className="position-index">{String(item.finish_position || '—').padStart(2, '0')}</span><span><strong>{item.driver_code}</strong><small>{item.team || item.driver_name}</small></span></div>
                    <div className="stint-timeline"><div className="timeline-base" />{item.stints.map((stint) => {
                      const end = stint.end_lap || raceLaps;
                      return <span key={stint.number} className="stint" style={{ left: `${(stint.start_lap - 1) / raceLaps * 100}%`, width: `${Math.max(1, end - stint.start_lap + 1) / raceLaps * 100}%`, background: compoundColor(stint.compound) }} title={`${stint.compound} · laps ${stint.start_lap}–${end}`} />;
                    })}</div>
                    <span className="pace-delta">{item.finish_position ? `P${item.finish_position}` : '—'}</span>
                  </div>;
                }) : <div className="empty-state">Race tyre stint data will appear here when loaded.</div>}
              </div>
              <div className="compound-key"><span><i className="soft-key" />Soft</span><span><i className="medium-key" />Medium</span><span><i className="hard-key" />Hard</span><span><i className="intermediate-key" />Inter</span><span><i className="wet-key" />Wet</span></div>
            </section>

            <section className="panel insight-panel">
              <div className="panel-header">
                <div><div className="section-kicker">RACE CONTEXT</div><h3>Observed conditions</h3></div>
                <span className="insight-icon"><Icon name="chart" size={17} /></span>
              </div>
              <div className="context-list">
                <div><span>TRACK TEMPERATURE</span><strong>{weather?.track_temperature_c != null ? `${weather.track_temperature_c} °C` : 'Unavailable'}</strong></div>
                <div><span>AIR TEMPERATURE</span><strong>{weather?.air_temperature_c != null ? `${weather.air_temperature_c} °C` : 'Unavailable'}</strong></div>
                <div><span>RAINFALL</span><strong>{weather?.rainfall != null ? `${weather.rainfall.toFixed(1)} mm average` : 'Unavailable'}</strong></div>
                <div><span>COMPOUND EFFECT</span><strong>{analysis?.compound_anova?.available ? `p = ${analysis.compound_anova.p_value.toPrecision(3)}` : 'Not testable'}</strong></div>
                <div><span>MODEL FIT · R²</span><strong>{analysis?.race_progress_model?.r_squared != null ? `${(analysis.race_progress_model.r_squared * 100).toFixed(1)}%` : 'Unavailable'}</strong></div>
              </div>
              <div className="insight-disclaimer"><Icon name="check" size={14} />Observed values · sourced from race session APIs</div>
            </section>
          </div>

          {analysis && (
            <section className="panel results-panel">
              <div className="panel-header">
                <div><div className="section-kicker">RACE ANALYSIS</div><h3>{analysis.event?.event_name || analysis.filename || 'Uploaded dataset'}</h3></div>
                <span className="result-count">{analysis.rows_analyzed.toLocaleString()} valid laps</span>
              </div>
              <div className="results-grid">
                <div><span>RACE-PROGRESS MODEL</span><strong>{analysis.race_progress_model?.r_squared != null ? `${(analysis.race_progress_model.r_squared * 100).toFixed(1)}%` : '—'}</strong><small>R² · tyre age + lap progression proxy + driver</small></div>
                <div><span>COMPOUND ANOVA</span><strong>{analysis.compound_anova?.p_value != null ? `p = ${analysis.compound_anova.p_value.toPrecision(3)}` : '—'}</strong><small>{analysis.compound_anova?.significant ? 'Significant compound effect' : analysis.compound_anova?.available ? 'No significant compound effect' : 'Needs at least two compounds'}</small></div>
                <div><span>XGBOOST VS GRID BASELINE</span><strong>{predictions?.mae != null ? `${predictions.mae.toFixed(2)} pos` : '—'}</strong><small>{predictions?.grid_baseline_mae != null ? `Grid baseline MAE ${predictions.grid_baseline_mae.toFixed(2)} pos` : predictions?.message || 'Walk-forward model unavailable'}</small></div>
              </div>
            </section>
          )}

          <footer className="page-footer"><span>APEX / LAB <i /> F1 STRATEGY ANALYTICS</span><span>LIVE SESSION POLLING EVERY 30S · HISTORICAL DATA OTHERWISE</span></footer>
        </div>
      </main>
      {(isAnalyzing || loadingSchedule) && <div className="loading-overlay"><span className="loading-spinner" /><span>{loadingSchedule ? 'Loading race calendar…' : 'Loading race data and fitting models…'}</span></div>}
    </div>
  );
}

export default App;
