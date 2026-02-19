let scoreboardCache = [];
let lastPrediction = null;
let lastNowPrice = null;
let lastNowPriceInr = null;
let lastQuoteCurrency = 'USD';
let lastFxRate = null;
let lastFxUpdatedAt = null;
let lastFxSource = null;
let lastForcedRefreshAt = 0;
let runState = { running: false, last_started_at: null, progress: null };
let lastSummaryRunAt = null;

const APP_ROOT = document.getElementById('app-root');
const MARKET_LABEL = APP_ROOT?.dataset.marketLabel || 'Jeena Sikho';
const API_PREFIX = APP_ROOT?.dataset.apiPrefix || '/api/jeena-sikho';

const PRICE_POLL_MS = 30000;
const PREDICTION_POLL_MS = 30000;
const SUMMARY_POLL_MS = 30000;
const SCOREBOARD_POLL_MS = 60000;
const RUN_STATUS_POLL_MS = 15000;
const COUNTDOWN_TICK_MS = 1000;

const RUN_TIMER_START_KEY = 'js_run_start';
const RUN_TIMER_LAST_KEY = 'js_run_last';

async function getJSON(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

function fmt(num, digits = 2) {
  if (num === null || num === undefined || Number.isNaN(num)) return '--';
  return Number(num).toLocaleString('en-US', { maximumFractionDigits: digits });
}

function fmtFixed(num, digits = 2) {
  if (num === null || num === undefined || Number.isNaN(num)) return '--';
  return Number(num).toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtUsd(num) {
  if (num === null || num === undefined || Number.isNaN(num)) return '--';
  return `$${fmt(num, 2)}`;
}

function fmtInr(num) {
  if (num === null || num === undefined || Number.isNaN(num)) return '--';
  return `INR ${fmt(num, 2)}`;
}

function fmtQuoted(num, quoteCurrency) {
  if (num === null || num === undefined || Number.isNaN(num)) return '--';
  if (quoteCurrency === 'INR') return fmtInr(num);
  if (quoteCurrency === 'USD') return fmtUsd(num);
  return `${quoteCurrency} ${fmt(num, 2)}`;
}

function formatDualPrice(price, fxRate, quoteCurrency = lastQuoteCurrency) {
  if (price === null || price === undefined || Number.isNaN(price)) return '--';
  if (quoteCurrency === 'USD') {
    if (fxRate) return `${fmtInr(price * fxRate)} (${fmtUsd(price)})`;
    return fmtUsd(price);
  }
  return fmtQuoted(price, quoteCurrency);
}

function formatNowPrice(price, inr, fxRate, quoteCurrency = lastQuoteCurrency) {
  if (quoteCurrency === 'USD') {
    if (inr !== null && inr !== undefined && !Number.isNaN(inr)) {
      return `${fmtInr(inr)}${price ? ` (${fmtUsd(price)})` : ''}`;
    }
    return formatDualPrice(price, fxRate, quoteCurrency);
  }
  return fmtQuoted(price, quoteCurrency);
}

function fmtDateTime(iso) {
  if (!iso) return '--';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
}

function fmtDateTimeLower(iso) {
  const text = fmtDateTime(iso);
  if (text === '--') return text;
  return text.replace(' AM', ' am').replace(' PM', ' pm');
}

function fmtTimeOnly(iso) {
  if (!iso) return '--';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  });
}

function formatElapsed(ms) {
  if (!ms || ms < 0) return '--';
  const total = Math.floor(ms / 1000);
  const hh = Math.floor(total / 3600);
  const mm = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
  const ss = String(total % 60).padStart(2, '0');
  if (hh > 0) return `${hh}:${mm}:${ss}`;
  return `${mm}:${ss}`;
}

function renderRunTimer() {
  const el = document.getElementById('run-timer');
  if (!el) return;

  if (runState.running) {
    const startIso = runState.last_started_at || localStorage.getItem(RUN_TIMER_START_KEY);
    if (!startIso) {
      el.textContent = 'Timer: --';
      return;
    }
    const start = new Date(startIso).getTime();
    if (Number.isNaN(start)) {
      el.textContent = 'Timer: --';
      return;
    }
    el.textContent = `Timer: ${formatElapsed(Date.now() - start)}`;
    return;
  }

  const lastMs = localStorage.getItem(RUN_TIMER_LAST_KEY);
  if (lastMs) {
    el.textContent = `Last time: ${formatElapsed(Number(lastMs))}`;
  } else {
    el.textContent = 'Last time: --';
  }
}

function renderProgress(state) {
  const el = document.getElementById('run-progress');
  if (!el) return;
  const progress = state?.progress;
  const total = progress?.total;
  const done = progress?.done;
  const task = progress?.task;
  if (!total || total <= 0) {
    el.style.display = 'none';
    return;
  }
  el.style.display = '';
  const safeDone = Math.max(0, Math.min(Number(done) || 0, total));
  const pct = Math.round((safeDone / total) * 100);
  if (state?.running) {
    const taskLabel = task ? ` | ${task}` : '';
    el.textContent = `Models trained: ${safeDone}/${total} (${pct}%)${taskLabel}`;
    return;
  }
  el.textContent = `Last trained: ${safeDone}/${total} (${pct}%)`;
}

function expectedTimeLabel(candidateCount) {
  if (!candidateCount || candidateCount <= 0) return 'Expected: --';
  if (candidateCount >= 200) return 'Expected: ~15-30 min';
  if (candidateCount >= 120) return 'Expected: ~10-20 min';
  if (candidateCount >= 80) return 'Expected: ~8-15 min';
  if (candidateCount >= 40) return 'Expected: ~6-12 min';
  return 'Expected: ~3-8 min';
}

function formatEtaSeconds(seconds) {
  if (!seconds || seconds <= 0) return 'Expected: --';
  return `Expected: ~${formatElapsed(seconds * 1000)}`;
}

function updateRunState(state) {
  if (!state) return;
  runState.running = !!state.running;
  runState.last_started_at = state.last_started_at || runState.last_started_at;
  runState.progress = state.progress || runState.progress;

  if (runState.running) {
    if (runState.last_started_at) {
      localStorage.setItem(RUN_TIMER_START_KEY, runState.last_started_at);
    }
  } else {
    const startIso = localStorage.getItem(RUN_TIMER_START_KEY);
    if (startIso) {
      const start = new Date(startIso).getTime();
      if (!Number.isNaN(start)) {
        const elapsed = Date.now() - start;
        if (elapsed > 0) {
          localStorage.setItem(RUN_TIMER_LAST_KEY, String(elapsed));
        }
      }
      localStorage.removeItem(RUN_TIMER_START_KEY);
    } else if (state.duration_seconds) {
      const durationMs = Math.max(0, Number(state.duration_seconds) * 1000);
      if (Number.isFinite(durationMs) && durationMs > 0) {
        localStorage.setItem(RUN_TIMER_LAST_KEY, String(durationMs));
      }
    } else if (state.last_started_at && state.last_finished_at) {
      const started = new Date(state.last_started_at).getTime();
      const finished = new Date(state.last_finished_at).getTime();
      if (!Number.isNaN(started) && !Number.isNaN(finished) && finished > started) {
        localStorage.setItem(RUN_TIMER_LAST_KEY, String(finished - started));
      }
    }
  }
  renderRunTimer();
  renderProgress(runState);
}

function normalizePredictions(data) {
  if (!data || !Array.isArray(data.predictions)) return [];
  return data.predictions;
}

function predictionMinutes(pred) {
  if (!pred) return 0;
  return pred.prediction_horizon_min || pred.timeframe_minutes || 0;
}

function labelForPrediction(pred) {
  if (!pred) return '--';
  if (pred.timeframe) return pred.timeframe;
  const mins = predictionMinutes(pred);
  if (!mins) return '--';
  return mins < 60 ? `${mins}m` : `${mins / 60}h`;
}

function sortPredictions(preds) {
  return [...preds].sort((a, b) => predictionMinutes(a) - predictionMinutes(b));
}

function selectPrimaryPrediction(preds) {
  if (!preds.length) return null;
  const byMinutes = sortPredictions(preds);
  const ten = byMinutes.find(p => predictionMinutes(p) === 10) || byMinutes[0];
  return ten || byMinutes[0];
}

function formatCountdown(predictedAt, horizonMin, targetIso = null) {
  let target = null;
  if (targetIso) {
    const parsed = new Date(targetIso).getTime();
    if (!Number.isNaN(parsed)) target = parsed;
  }
  if (target === null) {
    if (!predictedAt) return '--:--';
    const start = new Date(predictedAt).getTime();
    if (Number.isNaN(start)) return '--:--';
    target = start + horizonMin * 60 * 1000;
  }
  const diff = target - Date.now();
  if (diff <= 0) return '00:00';
  const totalSec = Math.floor(diff / 1000);
  const mm = String(Math.floor(totalSec / 60)).padStart(2, '0');
  const ss = String(totalSec % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}

function formatNextMatchLabel(predictedAt, horizonMin, targetIso = null) {
  if (!predictedAt || !horizonMin) return 'Next match in --:--';
  let targetMs = null;
  if (targetIso) {
    const parsed = new Date(targetIso).getTime();
    if (!Number.isNaN(parsed)) targetMs = parsed;
  }
  if (targetMs === null) {
    const start = new Date(predictedAt).getTime();
    if (Number.isNaN(start)) return `Next match in ${formatCountdown(predictedAt, horizonMin, targetIso)}`;
    targetMs = start + horizonMin * 60 * 1000;
  }
  const target = new Date(targetMs);
  const timeLabel = target.toLocaleTimeString('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
  return `Next match in ${formatCountdown(predictedAt, horizonMin, targetIso)} (${timeLabel})`;
}

function statusText(pred) {
  if (!pred) return '--';
  if (pred.match_percent_precise !== null && pred.match_percent_precise !== undefined) {
    return formatMatchPercent(pred);
  }
  if (pred.match_percent !== null && pred.match_percent !== undefined) {
    return formatMatchPercent(pred);
  }
  if (pred.status === 'pending') {
    const horizonMin = predictionMinutes(pred);
    return formatNextMatchLabel(pred.predicted_at, horizonMin, pred.target_iso || null);
  }
  if (pred.status && pred.status !== 'ready') {
    return pred.status.replace('_', ' ');
  }
  return '--';
}

function formatMatchPercent(pred) {
  if (!pred) return '--';
  const hasPrecise = pred.match_percent_precise !== null && pred.match_percent_precise !== undefined;
  const matchVal = hasPrecise ? pred.match_percent_precise : pred.match_percent;
  if (matchVal === null || matchVal === undefined || Number.isNaN(matchVal)) return '--';
  const digits = hasPrecise ? 4 : 1;
  return `${fmtFixed(matchVal, digits)}%`;
}

function formatConfidence(pred) {
  if (!pred) return '--';
  const c = pred.confidence_pct;
  if (c === null || c === undefined || Number.isNaN(c)) return '--';
  return `${fmt(c, 1)}%`;
}

function formatDiffHtml(pred, lineBreak = false) {
  if (!pred) return '';
  const predicted = pred.predicted_price;
  const actual = pred.actual_price;
  if (predicted === null || predicted === undefined || Number.isNaN(predicted)) return '';
  if (actual === null || actual === undefined || Number.isNaN(actual)) return '';
  const diff = Number(predicted) - Number(actual);
  const sign = diff > 0 ? '+' : diff < 0 ? '-' : '';
  const signClass = diff > 0 ? 'diff-plus' : diff < 0 ? 'diff-minus' : '';
  const absVal = Math.abs(diff);
  const diffText = fmtQuoted(absVal, lastQuoteCurrency);
  const prefix = lineBreak ? '<br>' : ' | ';
  return `${prefix}<span class="diff-label">Difference:</span> <span class="diff-value"><span class="${signClass}">${sign}</span>${diffText}</span>`;
}

function renderPriceRow(primary, nowPriceUsd, nowPriceInr) {
  const nowDisplay = formatNowPrice(nowPriceUsd, nowPriceInr, lastFxRate);
  const predList = document.getElementById('pred-list');
  const lastLine = document.getElementById('pred-last-line');
  const actualLine = document.getElementById('pred-actual-line');
  const priceRow = document.getElementById('price-row');

  if (!primary) {
    document.getElementById('price-now').textContent = `${MARKET_LABEL} Now: ${nowDisplay}`;
    if (priceRow) {
      priceRow.style.display = '';
      priceRow.innerHTML = '<span class="price-left">Predicted: -- | Match: --</span><span class="price-right"></span>';
    }
    if (lastLine) {
      lastLine.style.display = '';
      lastLine.textContent = 'Last matched prediction: --';
    }
    if (actualLine) {
      actualLine.style.display = '';
      actualLine.textContent = 'Actual price at match time: --';
    }
    return;
  }

  const horizonMin = predictionMinutes(primary) || 10;
  const label = labelForPrediction(primary);
  const predDisplay = primary.predicted_price
    ? formatDualPrice(primary.predicted_price, lastFxRate)
    : '--';
  const band = (primary.predicted_price_low && primary.predicted_price_high)
    ? ` [${formatDualPrice(primary.predicted_price_low, lastFxRate)} - ${formatDualPrice(primary.predicted_price_high, lastFxRate)}]`
    : '';
  const conf = formatConfidence(primary);
  const match = statusText(primary);
  const isSingle = Array.isArray(lastPrediction?.predictions) && lastPrediction.predictions.length <= 1;

  const lastReady = primary.last_ready;

  if (isSingle) {
    document.getElementById('price-now').textContent = `${MARKET_LABEL} Now: ${nowDisplay}`;
    if (priceRow) {
      priceRow.style.display = '';
      priceRow.innerHTML = `
        <span class="price-left">Predicted (${label || `${horizonMin}m`}): ${predDisplay}${band} | Conf: ${conf}${primary.low_confidence ? ' low' : ''} | Match: ${match}</span>
        <span class="price-right"></span>
      `;
    }
    if (lastLine) lastLine.style.display = '';
    if (actualLine) actualLine.style.display = '';
    if (lastReady) {
      const actualTime = lastReady.actual_at ? `at ${fmtDateTimeLower(lastReady.actual_at)}` : '';
      if (lastLine) {
        lastLine.innerHTML = `Last matched prediction: ${formatDualPrice(lastReady.predicted_price, lastFxRate)} (${formatMatchPercent(lastReady)})${formatDiffHtml(lastReady)}`;
      }
      if (actualLine) {
        const actualText = lastReady.actual_price !== null && lastReady.actual_price !== undefined
          ? `Last match actual price: ${formatDualPrice(lastReady.actual_price, lastFxRate)}`
          : 'Last match actual price: --';
        actualLine.textContent = `${actualText}${actualTime ? ' ' + actualTime : ''}`;
      }
    } else {
      if (lastLine) lastLine.textContent = 'Last matched prediction: --';
      if (actualLine) actualLine.textContent = 'Actual price at match time: --';
    }
    if (predList) predList.style.display = 'none';
  } else {
    document.getElementById('price-now').textContent = `${MARKET_LABEL} Now: ${nowDisplay}`;
    if (priceRow) {
      priceRow.style.display = 'none';
      priceRow.innerHTML = '';
    }
    if (lastLine) {
      lastLine.style.display = 'none';
      lastLine.textContent = '';
    }
    if (actualLine) {
      actualLine.style.display = 'none';
      actualLine.textContent = '';
    }
    if (predList) predList.style.display = '';
  }
}

function renderPredList(predictions) {
  const list = document.getElementById('pred-list');
  if (!list) return;
  if (predictions.length <= 1) {
    list.innerHTML = '';
    list.style.display = 'none';
    return;
  }
  if (!predictions.length) {
    list.innerHTML = '';
    list.style.display = 'none';
    return;
  }
  list.style.display = '';
  const ordered = sortPredictions(predictions);
  list.innerHTML = '';
  ordered.forEach((pred, idx) => {
    const label = labelForPrediction(pred);
    const horizonMin = predictionMinutes(pred) || 0;
    const predPrice = pred.predicted_price
      ? formatDualPrice(pred.predicted_price, lastFxRate)
      : '--';
    const band = (pred.predicted_price_low && pred.predicted_price_high)
      ? ` [${formatDualPrice(pred.predicted_price_low, lastFxRate)} - ${formatDualPrice(pred.predicted_price_high, lastFxRate)}]`
      : '';
    const conf = formatConfidence(pred);
    const match = statusText(pred);

    let lastMatchedLine = 'Last matched on last predicted price: --';
    let diffActualLine = 'Difference: -- | Actual: --';
    if (pred.last_ready) {
      const lr = pred.last_ready;
      lastMatchedLine = `Last matched on last predicted price: ${formatDualPrice(lr.predicted_price, lastFxRate)} (${formatMatchPercent(lr)})`;
      const actualLine = lr.actual_price !== null && lr.actual_price !== undefined
        ? `Actual: ${formatDualPrice(lr.actual_price, lastFxRate)}`
        : 'Actual: --';
      const actualTime = lr.actual_at ? `@ ${fmtDateTimeLower(lr.actual_at)}` : '';
      const diffHtml = formatDiffHtml(lr);
      const diffOnly = diffHtml
        ? (diffHtml.startsWith(' | ') ? diffHtml.slice(3) : diffHtml)
        : 'Difference: --';
      diffActualLine = `${diffOnly} | ${actualLine}${actualTime ? ' ' + actualTime : ''}`;
    }

    const line = `Predicted (${label}): ${predPrice}${band} | Conf: ${conf}${pred.low_confidence ? ' low' : ''} | Match: ${match}<br>${lastMatchedLine}<br>${diffActualLine}`;

    const item = document.createElement('div');
    item.className = 'pred-item';
    item.innerHTML = `<div class="pred-idx">${idx + 1}</div><div class="pred-text">${line}</div>`;
    list.appendChild(item);
  });
}

function renderHorizonMetrics(data) {
  const el = document.getElementById('horizon-metrics');
  if (!el) return;
  const rows = Array.isArray(data?.metrics_by_horizon) ? data.metrics_by_horizon : [];
  if (!rows.length) {
    el.innerHTML = '';
    return;
  }

  const ordered = [...rows].sort((a, b) => (a.horizon_minutes || 0) - (b.horizon_minutes || 0));
  const lines = [];
  ordered.forEach((row) => {
    const tf = row.timeframe || '--';
    const target = row.target || '--';
    const metrics = row.metrics;
    if (!metrics || !metrics.samples) {
      lines.push(`<div class="metric-row">${tf} (${target}): pending matches</div>`);
      return;
    }
    const mae = fmt(metrics.mae, 4);
    const mape = metrics.mape === null || metrics.mape === undefined ? '--' : `${fmt(metrics.mape, 2)}%`;
    const hit = metrics.hit_rate === null || metrics.hit_rate === undefined ? '--' : `${fmt(metrics.hit_rate, 2)}%`;
    const util = metrics.directional_utility === null || metrics.directional_utility === undefined ? '--' : fmt(metrics.directional_utility, 5);
    const calib = metrics.calibration_rmse === null || metrics.calibration_rmse === undefined ? '--' : fmt(metrics.calibration_rmse, 5);
    lines.push(
      `<div class="metric-row">${tf} (${target}) | n=${metrics.samples} | MAE=${mae} | MAPE=${mape} | Hit=${hit} | Util=${util} | CalRMSE=${calib}</div>`,
    );
  });
  const report = Array.isArray(data?.backtest_report) ? data.backtest_report : [];
  report.forEach((row) => {
    const ready = row.production_ready ? 'READY' : 'not-ready';
    lines.push(`<div class="metric-row">${row.timeframe} pack: ${ready}</div>`);
  });
  el.innerHTML = lines.join('');
}

function updateTimeframePill(predictions) {
  const pill = document.getElementById('timeframe-pill');
  if (!pill) return;
  if (!predictions || predictions.length === 0) {
    pill.textContent = 'timeframe: --';
    return;
  }
  if (predictions.length === 1) {
    pill.textContent = `${labelForPrediction(predictions[0])} candles`;
  } else {
    pill.textContent = 'multi-timeframe';
  }
}

function renderPredictionUI() {
  const predictions = normalizePredictions(lastPrediction);
  updateTimeframePill(predictions);
  const primary = selectPrimaryPrediction(predictions.filter(p => p.predicted_price));
  renderPriceRow(primary, lastNowPrice, lastNowPriceInr);
  renderPredList(predictions);
  renderHorizonMetrics(lastPrediction);
}

async function queryPriceAt() {
  const input = document.getElementById('price-at-input');
  const result = document.getElementById('price-at-result');
  if (!input || !result) return;
  const value = input.value.trim();
  if (!value) {
    result.textContent = 'Price at timestamp: --';
    return;
  }
  result.textContent = 'Price at timestamp: loading...';
  try {
    const data = await getJSON(`${API_PREFIX}/price_at?ts=${encodeURIComponent(value)}`);
    const quoteCurrency = data.quote_currency || lastQuoteCurrency;
    const display = formatNowPrice(data.price, data.price_inr, data.fx_rate || lastFxRate, quoteCurrency);
    const ts = data.aligned_at || data.timestamp_utc || data.requested_at;
    const timeLabel = ts ? fmtDateTimeLower(ts) : '--';
    const alignedNote = data.aligned && data.aligned_at ? ' (aligned)' : '';
    const tfLabel = data.timeframe ? ` [${data.timeframe}]` : '';
    result.textContent = `Price at ${timeLabel}${alignedNote}${tfLabel}: ${display}`;
  } catch (err) {
    result.textContent = 'Price at timestamp: not found';
  }
}

async function loadPrice() {
  try {
    const data = await getJSON(`${API_PREFIX}/price`);
    const updated = new Date(data.updated_at).toLocaleTimeString();
    let fxText = '';
    let marketText = '';
    const quoteCurrency = data.quote_currency || lastQuoteCurrency;
    if (quoteCurrency === 'USD' && data.fx_rate) {
      fxText = ` | FX: 1 USD = ${fmt(data.fx_rate, 2)} INR`;
      if (data.fx_source) fxText += ` (${data.fx_source})`;
      if (data.fx_stale) fxText += ' stale';
    }
    if (data.market_status) {
      const status = data.market_status === 'open' ? 'Open' : 'Closed';
      const mode = data.price_mode === 'last_traded' ? 'Last traded' : 'Live';
      marketText = ` | Market: ${status} (${mode})`;
    }
    document.getElementById('price-updated').textContent = `Updated: ${updated}${fxText}${marketText}`;
    lastNowPrice = data.price;
    lastNowPriceInr = data.price_inr;
    lastQuoteCurrency = quoteCurrency;
    if (data.fx_rate) lastFxRate = data.fx_rate;
    if (data.fx_updated_at) lastFxUpdatedAt = data.fx_updated_at;
    if (data.fx_source) lastFxSource = data.fx_source;
    renderPredictionUI();
    return data.price;
  } catch (err) {
    return null;
  }
}

async function loadSummary() {
  try {
    const data = await getJSON(`${API_PREFIX}/tournament/summary`);
    const summaryRunAt = data.last_run_at || null;
    if (summaryRunAt && summaryRunAt !== lastSummaryRunAt) {
      lastSummaryRunAt = summaryRunAt;
      loadScoreboard();
    }
    const candidateCount = data.candidate_count || 0;
    document.getElementById('candidate-count').textContent = `${candidateCount} models`;
    const lastStarted = data.last_run_started_at || data.last_run_at;
    const lastFinished = data.last_run_finished_at || data.last_run_at;
    document.getElementById('last-run').textContent = `Last run: ${fmtDateTime(lastStarted)}`;
    document.getElementById('last-completed').textContent = `Last tournament completed: ${fmtDateTimeLower(lastFinished)}`;
    document.getElementById('run-mode').textContent = `mode: ${data.run_mode || '--'}`;
    const next = document.getElementById('run-next');
    if (next) {
      next.textContent = `Next: ${fmtTimeOnly(data.next_run_at)}`;
    }

    const champs = data.champions || {};
    document.getElementById('champ-direction').textContent = `Direction champion: ${champs.direction?.model_id || '--'}`;
    document.getElementById('champ-return').textContent = `Return champion: ${champs.return?.model_id || '--'}`;
    document.getElementById('champ-range').textContent = `Range champion: ${champs.range?.model_id || '--'}`;
    const drift = data.drift_status || {};
    const driftEl = document.getElementById('drift-status');
    if (driftEl) {
      driftEl.textContent = `Drift: ${drift.alert ? 'ALERT (retrain recommended)' : 'stable'}`;
    }
    const byH = data.champions_by_horizon || {};
    const horizonEl = document.getElementById('champions-horizon');
    if (horizonEl) {
      const lines = Object.entries(byH).map(([tf, obj]) => {
        const ret = obj?.return || {};
        const model = ret.model_id || '--';
        const conf = ret.confidence_pct === null || ret.confidence_pct === undefined ? '--' : `${fmt(ret.confidence_pct, 1)}%`;
        const trend = ret.trend_delta === null || ret.trend_delta === undefined ? '--' : fmt(ret.trend_delta, 4);
        return `${tf}: ${model} | conf ${conf} | trend ${trend}`;
      });
      horizonEl.textContent = `Horizon champions: ${lines.join(' || ') || '--'}`;
    }
    const compRows = Array.isArray(data.completeness_by_horizon) ? data.completeness_by_horizon : [];
    const compEl = document.getElementById('completeness-horizon');
    if (compEl) {
      if (!compRows.length) {
        compEl.textContent = 'Completeness: --';
      } else {
        const parts = compRows.map((r) => `${r.timeframe}:${fmt(r.completeness_pct, 2)}% (${r.actual}/${r.expected})`);
        compEl.textContent = `Completeness (${compRows[0].lookback_days}d): ${parts.join(' || ')}`;
      }
    }
  } catch (err) {
    // ignore
  }
}

function renderScoreboard(rows) {
  const tbody = document.getElementById('scoreboard-body');
  tbody.innerHTML = '';
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    if (row.is_champion) tr.classList.add('winner');
    const badge = row.is_champion ? ' <span class="badge">Champion</span>' : '';
    tr.innerHTML = `
      <td>${row.rank}</td>
      <td>${row.target}</td>
      <td>${row.feature_set}</td>
      <td>${row.model_name}${badge}</td>
      <td>${row.family}</td>
      <td>${fmt(row.final_score, 4)}</td>
      <td>${row.primary_metric?.name || ''}: ${fmt(row.primary_metric?.value, 4)}</td>
      <td>${fmt(row.trading_score, 4)}</td>
      <td>${fmt(row.stability_penalty, 4)}</td>
      <td>${row.run_at || '--'}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderChart(rows) {
  const chart = document.getElementById('score-chart');
  chart.innerHTML = '';
  const top = rows.slice(0, 10);
  const maxScore = Math.max(...top.map(r => r.final_score || 0), 1e-6);
  top.forEach((r) => {
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.height = `${Math.max(5, (r.final_score / maxScore) * 100)}%`;
    bar.innerHTML = `<span>${fmt(r.final_score, 3)}</span>`;
    chart.appendChild(bar);
  });
}

function applyFilters() {
  let rows = [...scoreboardCache];
  const target = document.getElementById('filter-target').value;
  const feature = document.getElementById('filter-feature').value;
  const text = document.getElementById('filter-text').value.toLowerCase();
  const sortBy = document.getElementById('sort-by').value;

  if (target !== 'all') rows = rows.filter(r => r.target === target);
  if (feature !== 'all') rows = rows.filter(r => r.feature_set === feature);
  if (text) rows = rows.filter(r => (r.model_name || '').toLowerCase().includes(text));

  rows.sort((a, b) => {
    if (sortBy === 'trading_score') return (b.trading_score || 0) - (a.trading_score || 0);
    if (sortBy === 'primary') return (b.primary_metric?.value || 0) - (a.primary_metric?.value || 0);
    return (b.final_score || 0) - (a.final_score || 0);
  });

  renderScoreboard(rows);
  renderChart(rows);
}

async function loadScoreboard() {
  try {
    const rows = await getJSON(`${API_PREFIX}/tournament/scoreboard?limit=500`);
    scoreboardCache = rows;
    applyFilters();
  } catch (err) {
    // ignore
  }
}

async function refreshPrediction() {
  try {
    await getJSON(`${API_PREFIX}/prediction/refresh`, { method: 'POST' });
  } catch (err) {
    // ignore
  }
}

async function loadPrediction() {
  try {
    const data = await getJSON(`${API_PREFIX}/prediction/latest`);
    lastPrediction = data;
    renderPredictionUI();
  } catch (err) {
    // ignore
  }
}

async function refreshPredictionAndLoad() {
  await refreshPrediction();
  await loadPrediction();
}

async function runNow() {
  const button = document.getElementById('run-now');
  const state = document.getElementById('run-state');
  if (!button || !state) return;
  button.disabled = true;
  state.textContent = 'running...';
  const localStart = new Date().toISOString();
  updateRunState({ running: true, last_started_at: localStart });
  try {
    const res = await getJSON(`${API_PREFIX}/tournament/run`, { method: 'POST', body: '{}' });
    state.textContent = res.status || 'started';
    if (!res.running) {
      button.disabled = false;
    }
    updateRunState(res);
  } catch (err) {
    state.textContent = 'error';
  }
}

async function pollRunStatus() {
  try {
    const state = await getJSON(`${API_PREFIX}/tournament/run/status`);
    const badge = document.getElementById('run-state');
    const runBtn = document.getElementById('run-now');
    if (state.running) {
      badge.textContent = 'running';
      if (runBtn) runBtn.disabled = true;
    } else {
      badge.textContent = 'idle';
      if (runBtn) runBtn.disabled = false;
    }
    updateRunState(state);
  } catch (err) {
    // ignore
  }
}

function updateCountdownOnly() {
  const predictions = normalizePredictions(lastPrediction);
  if (predictions.length) {
    const now = Date.now();
    const due = predictions.some((pred) => {
      if (!pred || pred.status !== 'pending' || !pred.predicted_at) return false;
      const horizonMin = predictionMinutes(pred) || 0;
      const start = new Date(pred.predicted_at).getTime();
      if (Number.isNaN(start)) return false;
      return now >= start + horizonMin * 60 * 1000;
    });
    if (due && now - lastForcedRefreshAt > 5000) {
      lastForcedRefreshAt = now;
      loadPrediction();
      return;
    }
  }
  renderPredictionUI();
}

async function init() {
  await loadSummary();
  await loadScoreboard();
  await loadPrice();
  await refreshPredictionAndLoad();
  await pollRunStatus();

  const runNowBtn = document.getElementById('run-now');
  if (runNowBtn) runNowBtn.addEventListener('click', runNow);
  document.getElementById('filter-target').addEventListener('change', applyFilters);
  document.getElementById('filter-feature').addEventListener('change', applyFilters);
  document.getElementById('filter-text').addEventListener('input', applyFilters);
  document.getElementById('sort-by').addEventListener('change', applyFilters);
  const priceAtBtn = document.getElementById('price-at-btn');
  const priceAtInput = document.getElementById('price-at-input');
  if (priceAtBtn) priceAtBtn.addEventListener('click', queryPriceAt);
  if (priceAtInput) {
    priceAtInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') queryPriceAt();
    });
  }

  setInterval(loadPrice, PRICE_POLL_MS);
  setInterval(refreshPredictionAndLoad, PREDICTION_POLL_MS);
  setInterval(loadSummary, SUMMARY_POLL_MS);
  setInterval(loadScoreboard, SCOREBOARD_POLL_MS);
  setInterval(pollRunStatus, RUN_STATUS_POLL_MS);
  setInterval(updateCountdownOnly, COUNTDOWN_TICK_MS);
  setInterval(renderRunTimer, COUNTDOWN_TICK_MS);
}

init();

