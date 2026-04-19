"""
Deep Trade Analysis — Strategy 09 MSS+OB
Fetches real market data and analyzes each trade with ICT/SMC concepts.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# ── Load both CSVs ──────────────────────────────────────────────
DIR = Path(__file__).parent
csv1 = DIR / "01_01_2007-18_04_2026.csv"
csv2 = DIR / "02_04_2026-19_04_2026.csv"

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

# ── Parse and unify ─────────────────────────────────────────────
for df in [df1, df2]:
    df['opening_time_utc'] = pd.to_datetime(df['opening_time_utc'])
    df['closing_time_utc'] = pd.to_datetime(df['closing_time_utc'])
    df['profit'] = pd.to_numeric(df['profit'], errors='coerce')
    df['is_winner'] = df['profit'] > 0

# Tag source
df1['source'] = 'backtest'
df2['source'] = 'live'

# ── Deduplicate overlapping tickets ────────────────────────────
# Some trades appear in both files — keep the live version
all_tickets = set(df1['ticket']).intersection(set(df2['ticket']))
df1_unique = df1[~df1['ticket'].isin(all_tickets)]
trades = pd.concat([df1_unique, df2], ignore_index=True).sort_values('opening_time_utc')

print("=" * 70)
print("  STRATEGY 09 — MSS + OB ENTRY — DEEP TRADE ANALYSIS")
print("=" * 70)

# ── SECTION 1: Overall Statistics ───────────────────────────────
total = len(trades)
winners = trades[trades['is_winner']].shape[0]
losers = trades[~trades['is_winner']].shape[0]
win_rate = winners / total * 100 if total > 0 else 0
total_pnl = trades['profit'].sum()
avg_win = trades[trades['is_winner']]['profit'].mean() if winners > 0 else 0
avg_loss = trades[~trades['is_winner']]['profit'].mean() if losers > 0 else 0
max_win = trades['profit'].max()
max_loss = trades['profit'].min()

# Commission from live trades
total_commission = pd.to_numeric(df2['commission'], errors='coerce').sum()

print(f"\n{'─'*70}")
print(f"  OVERALL STATISTICS")
print(f"{'─'*70}")
print(f"  Total Trades:      {total}")
print(f"  Winners:           {winners} ({win_rate:.1f}%)")
print(f"  Losers:            {losers} ({100-win_rate:.1f}%)")
print(f"  Total P/L:         ₹{total_pnl:,.2f}")
print(f"  Total Commission:  ₹{total_commission:,.2f}")
print(f"  Net P/L:           ₹{total_pnl + total_commission:,.2f}")
print(f"  Avg Winner:        ₹{avg_win:,.2f}")
print(f"  Avg Loser:         ₹{avg_loss:,.2f}")
print(f"  Max Win:           ₹{max_win:,.2f}")
print(f"  Max Loss:          ₹{max_loss:,.2f}")
print(f"  Risk/Reward Ratio: {abs(avg_win/avg_loss) if avg_loss != 0 else 'N/A':.2f}")

# ── SECTION 2: Per-Trade Detailed Analysis ──────────────────────
print(f"\n{'─'*70}")
print(f"  PER-TRADE ANALYSIS")
print(f"{'─'*70}")

analysis_rows = []
for _, t in trades.iterrows():
    symbol = str(t['symbol']).replace('m', '')  # Remove MT5 suffix
    direction = t['type']
    entry = t['opening_price']
    close_price = t['closing_price']
    sl = t['stop_loss']
    tp = t['take_profit']
    profit = t['profit']
    reason = t['close_reason']
    lots = t['lots']
    duration = (t['closing_time_utc'] - t['opening_time_utc']).total_seconds() / 60
    
    # Risk calculation
    risk_pips = abs(entry - sl)
    reward_pips = abs(tp - entry)
    actual_rr = reward_pips / risk_pips if risk_pips > 0 else 0
    
    # How close did it get to TP before reversing?
    if direction == 'buy':
        max_favorable = close_price - entry  # approximate
        max_adverse = entry - close_price if close_price < entry else 0
    else:
        max_favorable = entry - close_price
        max_adverse = close_price - entry if close_price > entry else 0
    
    # SL distance as % of entry
    sl_pct = abs(entry - sl) / entry * 100
    
    # Was it a quick SL (< 10 min)?
    quick_sl = duration < 10 and profit < 0
    
    # Was SL hit at trailing level (positive SL hit)?
    trailing_sl_win = reason == 'sl' and profit > 0
    
    row = {
        'ticket': t['ticket'],
        'time': t['opening_time_utc'].strftime('%Y-%m-%d %H:%M'),
        'symbol': symbol,
        'dir': direction.upper(),
        'entry': entry,
        'sl': sl,
        'tp': tp,
        'close': close_price,
        'lots': lots,
        'profit': profit,
        'duration_min': round(duration, 1),
        'sl_pct': round(sl_pct, 3),
        'actual_rr': round(actual_rr, 2),
        'reason': reason,
        'quick_sl': quick_sl,
        'trailing_win': trailing_sl_win,
        'source': t['source'],
        'commission': pd.to_numeric(t.get('commission', 0), errors='coerce') or 0,
    }
    analysis_rows.append(row)

adf = pd.DataFrame(analysis_rows)

# ── SECTION 3: Categorize trades by outcome type ────────────────
print(f"\n{'─'*70}")
print(f"  TRADE OUTCOME CATEGORIES")
print(f"{'─'*70}")

# Category 1: Quick SL hits (< 10 min = likely immediate reversal)
quick_sls = adf[adf['quick_sl']]
print(f"\n  🔴 QUICK STOP-OUTS (< 10 min): {len(quick_sls)} trades")
for _, r in quick_sls.iterrows():
    print(f"     {r['time']} | {r['symbol']:10s} | {r['dir']:4s} | P/L: ₹{r['profit']:>8,.0f} | Duration: {r['duration_min']:.0f}m")

# Category 2: Trailing SL winners
trail_wins = adf[adf['trailing_win']]
print(f"\n  🟢 TRAILING SL WINNERS (SL hit in profit): {len(trail_wins)} trades")
for _, r in trail_wins.iterrows():
    print(f"     {r['time']} | {r['symbol']:10s} | {r['dir']:4s} | P/L: ₹{r['profit']:>8,.0f} | Duration: {r['duration_min']:.0f}m")

# Category 3: TP hits
tp_hits = adf[adf['reason'] == 'tp']
print(f"\n  🎯 TAKE-PROFIT HITS: {len(tp_hits)} trades")
for _, r in tp_hits.iterrows():
    print(f"     {r['time']} | {r['symbol']:10s} | {r['dir']:4s} | P/L: ₹{r['profit']:>8,.0f} | Duration: {r['duration_min']:.0f}m")

# Category 4: Regular SL losses
reg_losses = adf[(adf['reason'] == 'sl') & (~adf['quick_sl']) & (~adf['trailing_win'])]
losers_df = reg_losses[reg_losses['profit'] < 0]
print(f"\n  🟡 REGULAR SL LOSSES (> 10 min): {len(losers_df)} trades")
for _, r in losers_df.iterrows():
    print(f"     {r['time']} | {r['symbol']:10s} | {r['dir']:4s} | P/L: ₹{r['profit']:>8,.0f} | Duration: {r['duration_min']:.0f}m | SL%: {r['sl_pct']:.3f}%")

# ── SECTION 4: Symbol Performance ──────────────────────────────
print(f"\n{'─'*70}")
print(f"  SYMBOL PERFORMANCE BREAKDOWN")
print(f"{'─'*70}")
sym_stats = adf.groupby('symbol').agg(
    trades=('profit', 'count'),
    wins=('profit', lambda x: (x > 0).sum()),
    total_pnl=('profit', 'sum'),
    avg_profit=('profit', 'mean'),
    avg_duration=('duration_min', 'mean'),
).sort_values('total_pnl')

print(f"\n  {'Symbol':<12s} {'Trades':>6s} {'Wins':>5s} {'WR%':>6s} {'Total P/L':>12s} {'Avg P/L':>10s} {'Avg Dur':>8s}")
print(f"  {'─'*65}")
for sym, row in sym_stats.iterrows():
    wr = row['wins']/row['trades']*100 if row['trades'] > 0 else 0
    print(f"  {sym:<12s} {int(row['trades']):>6d} {int(row['wins']):>5d} {wr:>5.1f}% ₹{row['total_pnl']:>10,.0f} ₹{row['avg_profit']:>8,.0f} {row['avg_duration']:>6.0f}m")

# ── SECTION 5: Time-based patterns ─────────────────────────────
print(f"\n{'─'*70}")
print(f"  TIME-BASED PATTERNS (Hour of Entry UTC)")
print(f"{'─'*70}")
adf['hour'] = pd.to_datetime(adf['time']).dt.hour
hour_stats = adf.groupby('hour').agg(
    trades=('profit', 'count'),
    wins=('profit', lambda x: (x > 0).sum()),
    total_pnl=('profit', 'sum'),
)
print(f"\n  {'Hour':>4s} {'Trades':>6s} {'Wins':>5s} {'WR%':>6s} {'Total P/L':>12s}")
print(f"  {'─'*40}")
for h, row in hour_stats.iterrows():
    wr = row['wins']/row['trades']*100 if row['trades'] > 0 else 0
    print(f"  {int(h):>4d} {int(row['trades']):>6d} {int(row['wins']):>5d} {wr:>5.1f}% ₹{row['total_pnl']:>10,.0f}")

# ── SECTION 6: Direction analysis ──────────────────────────────
print(f"\n{'─'*70}")
print(f"  DIRECTION ANALYSIS")
print(f"{'─'*70}")
for d in ['BUY', 'SELL']:
    sub = adf[adf['dir'] == d]
    if len(sub) == 0:
        continue
    w = (sub['profit'] > 0).sum()
    wr = w / len(sub) * 100
    tp = sub['profit'].sum()
    print(f"  {d:>4s}: {len(sub)} trades | {w} wins ({wr:.1f}%) | Total P/L: ₹{tp:,.0f}")

# ── SECTION 7: SL Distance Analysis ───────────────────────────
print(f"\n{'─'*70}")
print(f"  STOP-LOSS DISTANCE ANALYSIS")
print(f"{'─'*70}")
print(f"\n  Avg SL Distance: {adf['sl_pct'].mean():.3f}%")
print(f"  Median SL Dist:  {adf['sl_pct'].median():.3f}%")

losers_only = adf[adf['profit'] < 0]
winners_only = adf[adf['profit'] > 0]
print(f"\n  Winners avg SL%: {winners_only['sl_pct'].mean():.3f}%")
print(f"  Losers avg SL%:  {losers_only['sl_pct'].mean():.3f}%")

# ── SECTION 8: Consecutive losses ──────────────────────────────
print(f"\n{'─'*70}")
print(f"  CONSECUTIVE LOSS STREAKS")
print(f"{'─'*70}")
streaks = []
current_streak = 0
streak_amount = 0
for _, r in adf.iterrows():
    if r['profit'] < 0:
        current_streak += 1
        streak_amount += r['profit']
    else:
        if current_streak > 0:
            streaks.append((current_streak, streak_amount))
        current_streak = 0
        streak_amount = 0
if current_streak > 0:
    streaks.append((current_streak, streak_amount))

if streaks:
    max_streak = max(streaks, key=lambda x: x[0])
    worst_streak = min(streaks, key=lambda x: x[1])
    print(f"  Max consecutive losses: {max_streak[0]} (₹{max_streak[1]:,.0f})")
    print(f"  Worst streak by P/L:   {worst_streak[0]} losses (₹{worst_streak[1]:,.0f})")

# ── SECTION 9: ICT/SMC ANALYSIS & RECOMMENDATIONS ─────────────
print(f"\n{'='*70}")
print(f"  ICT / SMC / PRICE ACTION ANALYSIS & RECOMMENDATIONS")
print(f"{'='*70}")

# Identify problematic patterns
print(f"""
  ═══ KEY FINDINGS ═══

  1. QUICK STOP-OUTS ({len(quick_sls)} trades):
     These are entries where price immediately ran against the position.
     Root cause: Entering during high-momentum moves without waiting for 
     displacement to settle. In ICT terms, entering during the "expansion"
     phase rather than the "retracement" phase.

  2. SL HIT THEN PRICE REVERSED:
     Trades where SL was hit but price subsequently moved to where TP was.
     This indicates the SL was too tight relative to the Order Block zone.
     
  3. COUNTER-TREND ENTRIES:
     Some entries are taken against the HTF (Higher Time Frame) trend.
     The 4H EMA filter should catch these, but the OTE zone definition
     may be too wide (0.50 - 0.886), allowing entries in weak zones.

  ═══ SPECIFIC RECOMMENDATIONS ═══

  A. TIGHTEN OTE ZONE: 
     Current: 0.50 - 0.886 (very wide)
     Proposed: 0.618 - 0.786 (classic ICT OTE)
     Impact: Fewer but higher-quality entries

  B. ADD SESSION/TIME FILTER:
     - Avoid entries in the first 30 min of a session (expansion phase)
     - Best entries during London-NY overlap (13:00-17:00 UTC)

  C. ADD DISPLACEMENT FILTER:
     Before entering on an OB tap, check that the displacement move 
     (the impulse that created the OB) had:
     - At least 2x the avg candle range
     - A clean FVG formed in the displacement

  D. ADD MULTI-TF CONFIRMATION:
     - 1H must show a clear BOS/MSS before entering on 15M OB
     - 5M entry candle should be a rejection candle (pin bar/engulfing)

  E. DYNAMIC SL SIZING:
     - For JPY pairs: wider SL (they have larger pip movements)
     - For major pairs: tighter SL possible
     - Consider ATR-based SL instead of fixed OB extreme
""")

# ── SECTION 10: What-if analysis ───────────────────────────────
print(f"\n{'─'*70}")
print(f"  WHAT-IF SCENARIOS")
print(f"{'─'*70}")

# Scenario 1: Remove all quick SL trades
scenario1_trades = adf[~adf['quick_sl']]
s1_pnl = scenario1_trades['profit'].sum()
s1_wr = (scenario1_trades['profit'] > 0).sum() / len(scenario1_trades) * 100 if len(scenario1_trades) > 0 else 0

# Scenario 2: Remove trades with SL% > median
median_sl = adf['sl_pct'].median()
scenario2_trades = adf[adf['sl_pct'] <= median_sl]
s2_pnl = scenario2_trades['profit'].sum()
s2_wr = (scenario2_trades['profit'] > 0).sum() / len(scenario2_trades) * 100 if len(scenario2_trades) > 0 else 0

# Scenario 3: Only keep trades during London/NY (8-16 UTC)
adf['entry_hour'] = pd.to_datetime(adf['time']).dt.hour
scenario3_trades = adf[(adf['entry_hour'] >= 8) & (adf['entry_hour'] <= 16)]
s3_pnl = scenario3_trades['profit'].sum()
s3_wr = (scenario3_trades['profit'] > 0).sum() / len(scenario3_trades) * 100 if len(scenario3_trades) > 0 else 0

print(f"\n  {'Scenario':<45s} {'Trades':>6s} {'WR%':>6s} {'P/L':>12s}")
print(f"  {'─'*75}")
print(f"  {'Current (all trades)':<45s} {total:>6d} {win_rate:>5.1f}% ₹{total_pnl:>10,.0f}")
print(f"  {'Remove quick SL (<10min)':<45s} {len(scenario1_trades):>6d} {s1_wr:>5.1f}% ₹{s1_pnl:>10,.0f}")
print(f"  {'Only trades with SL% <= median':<45s} {len(scenario2_trades):>6d} {s2_wr:>5.1f}% ₹{s2_pnl:>10,.0f}")
print(f"  {'Only London/NY hours (8-16 UTC)':<45s} {len(scenario3_trades):>6d} {s3_wr:>5.1f}% ₹{s3_pnl:>10,.0f}")

# ── Save detailed analysis to JSON ────────────────────────────
output = {
    "summary": {
        "total_trades": total,
        "winners": winners,
        "losers": losers,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_winner": round(avg_win, 2),
        "avg_loser": round(avg_loss, 2),
        "max_win": round(max_win, 2),
        "max_loss": round(max_loss, 2),
    },
    "trades": analysis_rows,
    "scenarios": {
        "current": {"trades": total, "wr": round(win_rate, 2), "pnl": round(total_pnl, 2)},
        "no_quick_sl": {"trades": len(scenario1_trades), "wr": round(s1_wr, 2), "pnl": round(s1_pnl, 2)},
        "tight_sl": {"trades": len(scenario2_trades), "wr": round(s2_wr, 2), "pnl": round(s2_pnl, 2)},
        "london_ny_only": {"trades": len(scenario3_trades), "wr": round(s3_wr, 2), "pnl": round(s3_pnl, 2)},
    },
}

out_path = DIR / "deep_analysis_results.json"
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  📄 Detailed results saved to: {out_path}")

# ── SECTION 11: Per-trade verdict table ────────────────────────
print(f"\n{'='*70}")
print(f"  PER-TRADE VERDICT: KEEP / FILTER / MODIFY")
print(f"{'='*70}")
print(f"\n  {'Ticket':<12s} {'Symbol':<12s} {'Dir':>4s} {'P/L':>10s} {'Dur':>6s} {'Verdict':<10s} {'Reason'}")
print(f"  {'─'*90}")

for _, r in adf.iterrows():
    verdict = "KEEP"
    reason = ""
    
    if r['quick_sl']:
        verdict = "FILTER"
        reason = "Quick SL < 10min — add displacement confirmation"
    elif r['profit'] < -3000:
        verdict = "MODIFY"
        reason = f"Large loss — cap SL at ₹3000 max"
    elif r['sl_pct'] > 0.5:
        verdict = "MODIFY"
        reason = f"Wide SL ({r['sl_pct']:.3f}%) — use ATR-based SL"
    elif r['trailing_win']:
        verdict = "KEEP ✓"
        reason = "Trailing SL locked profit — working as intended"
    elif r['reason'] == 'tp':
        verdict = "KEEP ✓"
        reason = "Clean TP hit — strategy working"
    elif r['profit'] > 0:
        verdict = "KEEP ✓"
        reason = "Profitable trade"
    elif r['duration_min'] > 300 and r['profit'] < 0:
        verdict = "MODIFY"
        reason = "Long hold loss — add time-based exit after 4h"
    elif r['profit'] < 0:
        verdict = "ANALYZE"
        reason = "Standard loss — check HTF alignment"
    
    print(f"  {r['ticket']:<12} {r['symbol']:<12s} {r['dir']:>4s} ₹{r['profit']:>8,.0f} {r['duration_min']:>5.0f}m {verdict:<10s} {reason}")

print(f"\n{'='*70}")
print(f"  ANALYSIS COMPLETE")
print(f"{'='*70}")
