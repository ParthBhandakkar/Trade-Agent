/* Strategy 09 Backtest Dashboard — App Logic (v2) */

let currentResults = null;
let currentSymbolIdx = 0;
let allTrades = [];
let equityChart = null;
let logLines = 0;
let logSource = null;
let logsCollapsed = false;

// ── Init ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSymbols();
  connectLogStream();
});

async function loadSymbols() {
  try {
    const res = await fetch('/api/symbols');
    const data = await res.json();
    const list = document.getElementById('symbolList');
    list.innerHTML = '';

    // Add "ALL" option
    const allOpt = document.createElement('option');
    allOpt.value = 'ALL';
    list.appendChild(allOpt);

    if (data.symbols && data.symbols.length > 0) {
      data.symbols.forEach(sym => {
        const opt = document.createElement('option');
        opt.value = sym;
        list.appendChild(opt);
      });
      document.getElementById('symbolInput').value = data.symbols[0];
    }
  } catch (err) {
    console.error('Failed to load symbols:', err);
  }
}

// ── SSE Log Streaming ────────────────────────────────────────────────
function connectLogStream() {
  if (logSource) logSource.close();

  logSource = new EventSource('/api/logs');
  logSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      appendLog(data.level, data.message);
    } catch {}
  };
  logSource.onerror = () => {
    setTimeout(connectLogStream, 3000);
  };
}

function appendLog(level, message) {
  const content = document.getElementById('logContent');
  const body = document.getElementById('logBody');

  // Determine extra styling
  let extraClass = level;
  if (message.includes('TRADE:')) extraClass += ' trade';
  else if (message.includes('✅') || message.includes('complete')) extraClass += ' highlight';

  const line = document.createElement('div');
  line.className = `log-line ${extraClass}`;
  line.textContent = message;
  content.appendChild(line);

  logLines++;
  document.getElementById('logCount').textContent = `${logLines} lines`;

  // Auto-scroll to bottom
  body.scrollTop = body.scrollHeight;

  // Keep max 500 lines
  while (content.children.length > 500) {
    content.removeChild(content.firstChild);
  }
}

function clearLogs() {
  document.getElementById('logContent').innerHTML = '';
  logLines = 0;
  document.getElementById('logCount').textContent = '0 lines';
}

function toggleLogs() {
  const body = document.getElementById('logBody');
  const btn = document.getElementById('logToggle');
  logsCollapsed = !logsCollapsed;
  body.classList.toggle('collapsed', logsCollapsed);
  btn.textContent = logsCollapsed ? '▶ Expand' : '▼ Collapse';
}

// ── Run Backtest ─────────────────────────────────────────────────────
async function runBacktest() {
  const btn = document.getElementById('btnRun');
  const symbol = document.getElementById('symbolInput').value.trim().toUpperCase();
  const startDate = document.getElementById('startDate').value;
  const endDate = document.getElementById('endDate').value;
  const dot = document.getElementById('statusDot');

  if (!symbol || !startDate || !endDate) {
    showError('Please enter a symbol and select dates');
    return;
  }

  // Set loading state
  btn.classList.add('loading');
  btn.disabled = true;
  dot.className = 'status-dot running';
  document.getElementById('headerStatus').textContent = 'Running...';
  hideError();
  clearLogs();

  const payload = {
    symbol: symbol,
    start_date: startDate,
    end_date: endDate,
    enable_ema: document.getElementById('filterEma').checked,
    enable_killzone: document.getElementById('filterKillzone').checked,
    enable_london_block: document.getElementById('filterLondon').checked,
    enable_blacklist: document.getElementById('filterBlacklist').checked,
    min_quality: parseInt(document.getElementById('minQuality').value),
    auto_fetch: document.getElementById('autoFetch').checked,
    verbose_logs: document.getElementById('verboseLogs')?.checked ?? false,
  };

  try {
    // Short-lived POST: avoids "Failed to fetch" when the job runs 10+ minutes (browser / OS idle timeouts).
    const startRes = await fetch('/api/backtest/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!startRes.ok) {
      let detail = startRes.statusText;
      try {
        const errBody = await startRes.json();
        detail = typeof errBody.detail === 'string' ? errBody.detail : JSON.stringify(errBody.detail || errBody);
      } catch (_) {}
      throw new Error(detail || `HTTP ${startRes.status}`);
    }

    const pollMs = 400;
    let waitedMs = 0;
    let data = null;

    while (true) {
      const stRes = await fetch('/api/backtest/status');
      if (!stRes.ok) throw new Error(`Status poll failed: HTTP ${stRes.status}`);
      const st = await stRes.json();

      if (!st.running && st.result) {
        data = st.result;
        break;
      }
      if (!st.running && !st.result) {
        throw new Error('Backtest stopped without a result (server may have restarted).');
      }

      waitedMs += pollMs;
      document.getElementById('headerStatus').textContent =
        `Running… (${Math.round(waitedMs / 1000)}s)`;
      await new Promise((r) => setTimeout(r, pollMs));
    }

    if (!data.success) {
      showError(data.error || 'Backtest failed');
      document.getElementById('resultsContainer').style.display = 'none';
      document.getElementById('emptyState').style.display = 'block';
      dot.className = 'status-dot error';
      document.getElementById('headerStatus').textContent = 'Error';
    } else {
      currentResults = data;
      currentSymbolIdx = 0;
      renderResults(data);
      dot.className = 'status-dot';
      if (data.error) showError('Partial: ' + data.error);
    }

    const elapsed = data.elapsed_sec ? ` (${data.elapsed_sec}s)` : '';
    document.getElementById('headerStatus').textContent = `Done${elapsed}`;
    loadSymbols();
  } catch (err) {
    showError('Connection error: ' + err.message);
    dot.className = 'status-dot error';
    document.getElementById('headerStatus').textContent = 'Error';
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
  }
}

// ── Render Results ───────────────────────────────────────────────────
function renderResults(data) {
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('resultsContainer').style.display = 'block';

  if (data.aggregate && data.results.length > 1) {
    renderAggregate(data.aggregate);
    renderSymbolTabs(data.results);
  } else {
    document.getElementById('aggregateBanner').style.display = 'none';
    document.getElementById('symbolTabs').style.display = 'none';
  }

  if (data.results.length > 0) {
    renderSymbolResult(data.results[currentSymbolIdx]);
  }
}

function renderAggregate(agg) {
  const banner = document.getElementById('aggregateBanner');
  banner.style.display = 'flex';
  const pnlClass = agg.combined_pnl_pct >= 0 ? 'green' : 'red';
  const pnlSign = agg.combined_pnl_pct >= 0 ? '+' : '';
  const inrVal = agg.combined_pnl_inr || 0;
  const inrSign = inrVal >= 0 ? '+' : '';
  const inrColor = inrVal >= 0 ? 'green' : 'red';

  banner.innerHTML = `
    <div class="aggregate-item"><div class="agg-label">Symbols Tested</div><div class="agg-value">${agg.symbols_tested}</div></div>
    <div class="aggregate-item"><div class="agg-label">Total Trades</div><div class="agg-value">${agg.total_trades}</div></div>
    <div class="aggregate-item"><div class="agg-label">Win Rate</div><div class="agg-value" style="color:${agg.win_rate>=50?'var(--green)':'var(--red)'}">${agg.win_rate}%</div></div>
    <div class="aggregate-item"><div class="agg-label">Combined PnL</div><div class="agg-value ${pnlClass}">${pnlSign}${agg.combined_pnl_pct.toFixed(3)}%</div></div>
    <div class="aggregate-item"><div class="agg-label">Combined PnL (₹)</div><div class="agg-value ${inrColor}">${inrSign}₹${inrVal.toFixed(0)}</div></div>
    <div class="aggregate-item"><div class="agg-label">Avg Per Trade</div><div class="agg-value ${pnlClass}">${pnlSign}${agg.avg_pnl_per_trade.toFixed(4)}%</div></div>`;
}

function renderSymbolTabs(results) {
  const container = document.getElementById('symbolTabs');
  container.style.display = 'flex';
  container.innerHTML = '';

  const allTab = document.createElement('button');
  allTab.className = 'symbol-tab';
  allTab.textContent = 'All Combined';
  allTab.onclick = () => {
    currentSymbolIdx = -1;
    document.querySelectorAll('.symbol-tab').forEach(t => t.classList.remove('active'));
    allTab.classList.add('active');
    renderCombinedView(results);
  };
  container.appendChild(allTab);

  results.forEach((r, idx) => {
    const tab = document.createElement('button');
    tab.className = `symbol-tab${idx === currentSymbolIdx ? ' active' : ''}`;
    const pnlSign = r.total_pnl_pct >= 0 ? '+' : '';
    const pnlClass = r.total_pnl_pct >= 0 ? 'positive' : 'negative';
    tab.innerHTML = `${r.symbol} <span class="tab-pnl ${pnlClass}">${pnlSign}${r.total_pnl_pct.toFixed(3)}%</span>`;
    tab.onclick = () => {
      currentSymbolIdx = idx;
      document.querySelectorAll('.symbol-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      renderSymbolResult(r);
    };
    container.appendChild(tab);
  });
}

function renderCombinedView(results) {
  allTrades = [];
  results.forEach(r => { allTrades = allTrades.concat(r.trades); });
  const executed = allTrades.filter(t => !t.rejected);
  const wins = executed.filter(t => (t.pnl_inr || 0) > 0);
  const losses = executed.filter(t => (t.pnl_inr || 0) < 0);
  const totalPnl = executed.reduce((s, t) => s + t.pnl_pct, 0);
  const totalInr = executed.reduce((s, t) => s + (t.pnl_inr || 0), 0);

  renderStatsGrid({
    signals_found: results.reduce((s, r) => s + r.signals_found, 0),
    signals_rejected: results.reduce((s, r) => s + r.signals_rejected, 0),
    trades_executed: executed.length, wins: wins.length, losses: losses.length,
    win_rate: executed.length ? (wins.length / executed.length * 100) : 0,
    total_pnl_pct: totalPnl, total_pnl_inr: totalInr, max_drawdown: 0,
  });

  const curve = [0]; let total = 0;
  const curveInr = [0]; let totalI = 0;
  executed.forEach(t => { total += t.pnl_pct; curve.push(total); totalI += (t.pnl_inr || 0); curveInr.push(totalI); });
  renderEquityChart(curve, curveInr);

  const rejections = {};
  results.forEach(r => { Object.entries(r.rejection_breakdown).forEach(([k,v]) => { rejections[k] = (rejections[k]||0) + v; }); });
  renderRejections(rejections);
  renderTradesTable(allTrades);
}

function renderSymbolResult(result) {
  allTrades = result.trades;
  renderStatsGrid(result);
  renderEquityChart(result.equity_curve, result.equity_curve_inr);
  renderRejections(result.rejection_breakdown);
  renderTradesTable(result.trades);
}

// ── Stats Grid ───────────────────────────────────────────────────────
function renderStatsGrid(r) {
  const grid = document.getElementById('statsGrid');
  const pnlSign = r.total_pnl_pct >= 0 ? '+' : '';
  const pnlColor = r.total_pnl_pct >= 0 ? 'green' : 'red';
  const pnlClass = r.total_pnl_pct >= 0 ? 'positive' : 'negative';
  const wrColor = r.win_rate >= 50 ? 'green' : 'red';
  const avgPnl = r.trades_executed > 0 ? r.total_pnl_pct / r.trades_executed : 0;

  const totalInr = r.total_pnl_inr || 0;
  const inrSign = totalInr >= 0 ? '+' : '';
  const inrColor = totalInr >= 0 ? 'green' : 'red';
  const avgInr = r.trades_executed > 0 ? totalInr / r.trades_executed : 0;

  grid.innerHTML = `
    <div class="stat-card"><div class="stat-label">Signals Found</div><div class="stat-value">${r.signals_found}</div><div class="stat-sub">${r.signals_rejected} rejected</div></div>
    <div class="stat-card"><div class="stat-label">Trades Executed</div><div class="stat-value">${r.trades_executed}</div><div class="stat-sub">${r.wins}W / ${r.losses}L</div></div>
    <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value ${wrColor}">${r.win_rate.toFixed(1)}%</div><div class="stat-sub">${r.wins} of ${r.trades_executed}</div></div>
    <div class="stat-card ${pnlClass}"><div class="stat-label">Total PnL</div><div class="stat-value ${pnlColor}">${pnlSign}${r.total_pnl_pct.toFixed(3)}%</div><div class="stat-sub">${inrSign}₹${totalInr.toFixed(0)}</div></div>
    <div class="stat-card"><div class="stat-label">Avg PnL / Trade</div><div class="stat-value ${pnlColor}">${avgPnl >= 0 ? '+' : ''}${avgPnl.toFixed(4)}%</div><div class="stat-sub">${avgInr >= 0 ? '+' : ''}₹${avgInr.toFixed(0)}/trade</div></div>
    <div class="stat-card"><div class="stat-label">Total PnL (₹)</div><div class="stat-value ${inrColor}">${inrSign}₹${totalInr.toFixed(0)}</div><div class="stat-sub">₹1000 × 2000x leverage</div></div>
    <div class="stat-card"><div class="stat-label">Max Drawdown</div><div class="stat-value red">${r.max_drawdown.toFixed(3)}%</div></div>`;
}

// ── Equity Chart ─────────────────────────────────────────────────────
function renderEquityChart(curve, curveInr) {
  const ctx = document.getElementById('equityChart').getContext('2d');
  if (equityChart) equityChart.destroy();

  const labels = curve.map((_, i) => i === 0 ? 'Start' : `T${i}`);
  const pos = curve[curve.length - 1] >= 0;
  const lineColor = pos ? '#10b981' : '#ef4444';
  const fillColor = pos ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)';

  const datasets = [{
    label: 'Cumulative PnL %', data: curve, borderColor: lineColor,
    backgroundColor: fillColor, fill: true, tension: 0.3, borderWidth: 2,
    pointRadius: curve.length > 30 ? 0 : 3, pointHoverRadius: 5,
    pointBackgroundColor: lineColor, yAxisID: 'y',
  }];

  const hasInr = curveInr && curveInr.length > 0;
  if (hasInr) {
    const posInr = curveInr[curveInr.length - 1] >= 0;
    datasets.push({
      label: 'PnL ₹', data: curveInr, borderColor: posInr ? '#6ee7b7' : '#fca5a5',
      borderWidth: 1.5, borderDash: [4, 3], tension: 0.3, fill: false,
      pointRadius: 0, pointHoverRadius: 4, yAxisID: 'y1',
    });
  }

  equityChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: hasInr, labels: { color: '#94a3b8', font: { size: 11 } } },
        tooltip: {
          backgroundColor: 'rgba(17,24,39,0.95)', titleColor: '#f1f5f9',
          bodyColor: '#94a3b8', borderColor: 'rgba(75,85,99,0.3)', borderWidth: 1,
          cornerRadius: 8, displayColors: true,
          callbacks: {
            label: item => {
              if (item.datasetIndex === 0) return `PnL: ${item.parsed.y >= 0 ? '+' : ''}${item.parsed.y.toFixed(4)}%`;
              return `PnL: ${item.parsed.y >= 0 ? '+' : ''}₹${item.parsed.y.toFixed(0)}`;
            }
          }
        }
      },
      scales: {
        x: { display: false },
        y: {
          position: 'left',
          grid: { color: 'rgba(75,85,99,0.15)' },
          ticks: { color: '#64748b', font: { family: "'JetBrains Mono'" }, callback: v => v.toFixed(3) + '%' },
        },
        ...(hasInr ? { y1: {
          position: 'right',
          grid: { drawOnChartArea: false },
          ticks: { color: '#94a3b8', font: { family: "'JetBrains Mono'", size: 10 }, callback: v => '₹' + v.toFixed(0) },
        }} : {}),
      },
      interaction: { intersect: false, mode: 'index' },
    }
  });
}

// ── Rejections ───────────────────────────────────────────────────────
function renderRejections(breakdown) {
  const panel = document.getElementById('rejectionsPanel');
  const grid = document.getElementById('rejectionsGrid');
  const entries = Object.entries(breakdown);
  if (!entries.length) { panel.style.display = 'none'; return; }

  panel.style.display = 'block';
  const maxCount = Math.max(...entries.map(([,v]) => v));

  grid.innerHTML = entries.map(([reason, count]) => {
    const pct = (count / maxCount * 100).toFixed(0);
    return `<div class="rejection-bar"><span class="bar-label">${reason}</span><div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div><span class="bar-count">${count}</span></div>`;
  }).join('');
}

// ── Trades Table ─────────────────────────────────────────────────────
function renderTradesTable(trades) {
  const body = document.getElementById('tradesBody');
  const title = document.getElementById('tradesTitle');
  const executed = trades.filter(t => !t.rejected);
  title.textContent = `📋 Trades (${executed.length} executed, ${trades.length - executed.length} rejected)`;

  body.innerHTML = trades.map((t, i) => {
    const dirBadge = t.direction === 'bullish' ? '<span class="badge badge-bullish">LONG</span>' : '<span class="badge badge-bearish">SHORT</span>';
    let outcomeBadge = '';
    if (t.rejected) outcomeBadge = '<span class="badge badge-rejected">REJECTED</span>';
    else if (t.outcome === 'TP') outcomeBadge = '<span class="badge badge-tp">TP ✓</span>';
    else if (t.outcome === 'SL') outcomeBadge = '<span class="badge badge-sl">SL ✗</span>';
    else if (t.outcome === 'TSL') outcomeBadge = '<span class="badge badge-tp">TSL ✓</span>';
    else if (t.outcome === 'TIMEOUT') outcomeBadge = '<span class="badge badge-timeout">TIMEOUT</span>';
    else outcomeBadge = t.outcome || '—';

    const pnlStr = t.rejected ? '—' : `<span style="color:${t.pnl_pct >= 0 ? 'var(--green)' : 'var(--red)'}">${t.pnl_pct >= 0 ? '+' : ''}${t.pnl_pct.toFixed(4)}%</span>`;
    const inrVal = t.pnl_inr || 0;
    const pnlInrStr = t.rejected ? '—' : `<span style="color:${inrVal >= 0 ? 'var(--green)' : 'var(--red)'}">${inrVal >= 0 ? '+' : ''}₹${inrVal.toFixed(0)}</span>`;
    const pfmt = t.market === 'FOREX' ? 5 : 6;
    const detail = t.rejected ? truncate(t.reject_reason, 40) : t.sl_reason ? truncate(t.sl_reason, 40) : '';

    return `<tr data-rejected="${t.rejected}" data-outcome="${t.outcome || ''}">
      <td>${i+1}</td><td style="font-weight:600;color:var(--text-primary)">${t.symbol}</td><td>${dirBadge}</td>
      <td style="color:var(--text-secondary)">${t.entry_time ? formatTime(t.entry_time) : '—'}</td>
      <td>${t.entry_price.toFixed(pfmt)}</td><td>${t.rejected ? '—' : t.sl_price.toFixed(pfmt)}</td><td>${t.rejected ? '—' : t.tp_price.toFixed(pfmt)}</td>
      <td style="text-align:center">${t.quality}</td><td>${outcomeBadge}</td><td>${pnlStr}</td><td>${pnlInrStr}</td>
      <td style="color:var(--text-muted)">${t.rejected ? '—' : (t.exit_time ? formatTime(t.exit_time) : '—')}</td>
      <td style="color:var(--text-muted);font-family:'Inter',sans-serif;font-size:11px" title="${detail}">${detail}</td></tr>`;
  }).join('');
}

function filterTrades(filter, btnEl) {
  document.querySelectorAll('.table-filters button').forEach(b => b.classList.remove('active'));
  if (btnEl) btnEl.classList.add('active');

  document.querySelectorAll('#tradesBody tr').forEach(row => {
    const rejected = row.dataset.rejected === 'true';
    const outcome = row.dataset.outcome;
    let show = true;
    switch (filter) {
      case 'executed': show = !rejected; break;
      case 'tp': show = outcome === 'TP' || outcome === 'TSL'; break;
      case 'sl': show = outcome === 'SL'; break;
      case 'rejected': show = rejected; break;
    }
    row.style.display = show ? '' : 'none';
  });
}

// ── Helpers ──────────────────────────────────────────────────────────
function formatTime(isoStr) {
  if (!isoStr) return '—';
  try {
    const d = new Date(isoStr);
    const ist = new Date(d.getTime() + 5.5 * 60 * 60 * 1000);
    const y = ist.getUTCFullYear();
    const mo = String(ist.getUTCMonth() + 1).padStart(2, '0');
    const day = String(ist.getUTCDate()).padStart(2, '0');
    const h = String(ist.getUTCHours()).padStart(2, '0');
    const m = String(ist.getUTCMinutes()).padStart(2, '0');
    return `${y}-${mo}-${day} ${h}:${m}`;
  } catch { return isoStr; }
}

function truncate(str, max) { return !str ? '' : str.length > max ? str.substring(0, max) + '…' : str; }
function showError(msg) { document.getElementById('errorBanner').style.display = 'flex'; document.getElementById('errorText').textContent = msg; }
function hideError() { document.getElementById('errorBanner').style.display = 'none'; }
