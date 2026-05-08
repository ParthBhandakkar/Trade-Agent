const form = document.querySelector("#runForm");
const symbolsEl = document.querySelector("#symbols");
const statusEl = document.querySelector("#status");
const runButton = document.querySelector("#runButton");
const tradesBody = document.querySelector("#tradesBody");
const dialog = document.querySelector("#tradeDialog");
const closeDialog = document.querySelector("#closeDialog");
const dialogTitle = document.querySelector("#dialogTitle");
const dialogMeta = document.querySelector("#dialogMeta");
const timeline = document.querySelector("#timeline");

let trades = [];

function today(offsetDays = 0) {
  const d = new Date();
  d.setDate(d.getDate() + offsetDays);
  return d.toISOString().slice(0, 10);
}

function selectedSymbols() {
  return Array.from(symbolsEl.selectedOptions).map((option) => option.value);
}

function money(value) {
  return Number(value || 0).toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function setSummary(summary) {
  document.querySelector("#sumTrades").textContent = summary?.trades ?? 0;
  document.querySelector("#sumWinRate").textContent = `${summary?.win_rate ?? 0}%`;
  document.querySelector("#sumR").textContent = Number(summary?.total_r ?? 0).toFixed(2);
  document.querySelector("#sumInr").textContent = money(summary?.total_pnl_inr ?? 0);
}

function renderTrades(rows) {
  tradesBody.innerHTML = "";
  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="7">No trades matched the strategy rules for this range.</td>`;
    tradesBody.appendChild(tr);
    return;
  }
  rows.forEach((trade, index) => {
    const tr = document.createElement("tr");
    tr.className = "tradeRow";
    tr.dataset.index = index;
    const pnlClass = Number(trade.pnl_inr) >= 0 ? "good" : "bad";
    tr.innerHTML = `
      <td>${trade.entry_time_ist}</td>
      <td>${trade.symbol}</td>
      <td>${trade.direction}</td>
      <td>${trade.quality}</td>
      <td>${trade.entry}</td>
      <td>${trade.exit_reason}<br><small>${trade.exit_time_ist}</small></td>
      <td class="${pnlClass}">${trade.pnl_r}R<br><small>INR ${money(trade.pnl_inr)}</small></td>
    `;
    tr.addEventListener("click", () => openTrade(index));
    tradesBody.appendChild(tr);
  });
}

function openTrade(index) {
  const trade = trades[index];
  dialogTitle.textContent = `${trade.symbol} ${trade.direction}`;
  dialogMeta.textContent = `Entry ${trade.entry} | SL ${trade.sl} | TP1 ${trade.tp1} | TP2 ${trade.tp2} | ${trade.session} | spread ${trade.spread_pips} pips`;
  timeline.innerHTML = "";
  trade.events.forEach((event) => {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${event.phase}</strong><span>${event.time_ist}</span><div>${event.detail}${event.price === null ? "" : ` @ ${event.price}`}</div>`;
    timeline.appendChild(li);
  });
  dialog.showModal();
}

async function loadDefaults() {
  const res = await fetch("/api/defaults");
  const data = await res.json();
  symbolsEl.innerHTML = "";
  data.symbols.forEach((symbol) => {
    const option = document.createElement("option");
    option.value = symbol;
    option.textContent = symbol;
    option.selected = data.default_symbols.includes(symbol);
    symbolsEl.appendChild(option);
  });
  document.querySelector("#risk").value = data.risk_inr;
  document.querySelector("#quality").value = data.min_quality;
  document.querySelector("#start").value = today(-14);
  document.querySelector("#end").value = today(0);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  runButton.disabled = true;
  statusEl.textContent = "Fetching/cache-checking MT5 data and replaying closed 5M candles...";
  trades = [];
  renderTrades(trades);
  try {
    const payload = {
      symbols: selectedSymbols(),
      start: document.querySelector("#start").value,
      end: document.querySelector("#end").value,
      risk_inr: Number(document.querySelector("#risk").value),
      min_quality: Number(document.querySelector("#quality").value),
      force_refresh: document.querySelector("#refresh").checked,
    };
    const res = await fetch("/api/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Backtest failed");
    }
    trades = data.trades || [];
    setSummary(data.summary);
    renderTrades(trades);
    const cacheSources = new Set((data.cache || []).map((item) => item.source));
    const sourceText = cacheSources.size ? Array.from(cacheSources).join(", ") : "none";
    statusEl.textContent = `Completed. Data source: ${sourceText}. ${data.errors?.length ? `Errors: ${data.errors.join(" | ")}` : ""}`;
  } catch (error) {
    statusEl.textContent = error.message;
    setSummary(null);
  } finally {
    runButton.disabled = false;
  }
});

closeDialog.addEventListener("click", () => dialog.close());
loadDefaults().catch((error) => {
  statusEl.textContent = error.message;
});

