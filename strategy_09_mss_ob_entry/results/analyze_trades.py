import pandas as pd
import numpy as np

# CSV 1 - MT5 trades
df1 = pd.read_csv('01_01_2007-18_04_2026.csv')
print('=== CSV 1: MT5 Trades (01_01_2007-18_04_2026) ===')
print(f'Total trades: {len(df1)}')
print()

for _, row in df1.iterrows():
    profit = row['profit']
    direction = row['type']
    sym = row['symbol']
    close_reason = row['close_reason']
    entry = row['opening_price']
    exit_p = row['closing_price']
    sl = row['stop_loss']
    tp = row['take_profit']
    open_t = row['opening_time_utc']
    close_t = row['closing_time_utc']
    risk = abs(entry - sl) if pd.notna(sl) else 0
    reward = abs(tp - entry) if pd.notna(tp) else 0
    rr = reward / risk if risk > 0 else 0
    status = 'WIN' if profit >= 0 else 'LOSS'
    print(f'{sym:12s} | {direction:4s} | {open_t} -> {close_t} | Entry: {entry:.5f} SL: {sl:.5f} TP: {tp:.5f} | P/L: {profit:>+10.2f} | RR: {rr:.2f} | {close_reason} | {status}')

print()
wins1 = df1[df1['profit'] >= 0]
losses1 = df1[df1['profit'] < 0]
total_profit = df1['profit'].sum()
print(f'Wins: {len(wins1)}, Losses: {len(losses1)}, Win Rate: {len(wins1)/len(df1)*100:.1f}%')
print(f'Total P/L: {total_profit:+.2f}')
print(f'Avg Win: {wins1["profit"].mean():+.2f}')
print(f'Avg Loss: {losses1["profit"].mean():+.2f}')
if losses1["profit"].sum() != 0:
    print(f'Profit Factor: {wins1["profit"].sum() / abs(losses1["profit"].sum()):.2f}')
print()

# CSV 2 - Live trades with commissions
df2 = pd.read_csv('02_04_2026-19_04_2026.csv')
print('=== CSV 2: Live Trades (02_04_2026-19_04_2026) ===')
print(f'Total trades: {len(df2)}')
print()

for _, row in df2.iterrows():
    profit = row['profit']
    commission = row['commission'] if pd.notna(row['commission']) else 0
    net = profit + commission
    direction = row['type']
    sym = row['symbol']
    close_reason = row['close_reason']
    entry = row['opening_price']
    exit_p = row['closing_price']
    sl = row['stop_loss']
    tp = row['take_profit']
    open_t = row['opening_time_utc']
    close_t = row['closing_time_utc']
    risk = abs(entry - sl) if pd.notna(sl) else 0
    reward = abs(tp - entry) if pd.notna(tp) else 0
    rr = reward / risk if risk > 0 else 0
    status = 'WIN' if net >= 0 else 'LOSS'
    print(f'{sym:12s} | {direction:4s} | {open_t} -> {close_t} | Gross: {profit:>+10.2f} Comm: {commission:>+8.2f} Net: {net:>+10.2f} | RR: {rr:.2f} | {close_reason} | {status}')

print()
df2['commission'] = df2['commission'].fillna(0)
df2['net'] = df2['profit'] + df2['commission']
wins2 = df2[df2['net'] >= 0]
losses2 = df2[df2['net'] < 0]
total_net = df2['net'].sum()
print(f'Wins: {len(wins2)}, Losses: {len(losses2)}, Win Rate: {len(wins2)/len(df2)*100:.1f}%')
print(f'Total Net P/L: {total_net:+.2f}')
if len(wins2) > 0:
    print(f'Avg Win: {wins2["net"].mean():+.2f}')
if len(losses2) > 0:
    print(f'Avg Loss: {losses2["net"].mean():+.2f}')
if losses2['net'].sum() != 0:
    print(f'Profit Factor: {wins2["net"].sum() / abs(losses2["net"].sum()):.2f}')

# Hold times
df1['open_dt'] = pd.to_datetime(df1['opening_time_utc'])
df1['close_dt'] = pd.to_datetime(df1['closing_time_utc'])
df1['hold_min'] = (df1['close_dt'] - df1['open_dt']).dt.total_seconds() / 60

print(f'\n=== Hold Time Analysis (CSV1) ===')
for _, row in df1.iterrows():
    print(f'{row["symbol"]:12s} | Hold: {row["hold_min"]:>6.0f} min ({row["hold_min"]/60:.1f}h) | P/L: {row["profit"]:>+10.2f} | {row["close_reason"]}')

df2['open_dt'] = pd.to_datetime(df2['opening_time_utc'])
df2['close_dt'] = pd.to_datetime(df2['closing_time_utc'])
df2['hold_min'] = (df2['close_dt'] - df2['open_dt']).dt.total_seconds() / 60

print(f'\n=== Hold Time Analysis (CSV2) ===')
for _, row in df2.iterrows():
    print(f'{row["symbol"]:12s} | Hold: {row["hold_min"]:>6.0f} min ({row["hold_min"]/60:.1f}h) | Net: {row["net"]:>+10.2f} | {row["close_reason"]}')

# Quick killing trades
print(f'\n=== QUICK STOP OUTS (< 15 min hold) ===')
combined = pd.concat([
    df1.assign(source='CSV1', net_pl=df1['profit']),
    df2.assign(source='CSV2', net_pl=df2['net'])
])
quick_stops = combined[combined['hold_min'] < 15]
for _, row in quick_stops.iterrows():
    print(f'{row["source"]} | {row["symbol"]:12s} | Hold: {row["hold_min"]:.0f} min | Net: {row["net_pl"]:>+10.2f} | {row["close_reason"]}')

# Trades killed at same time (clustered)
print(f'\n=== CLUSTERED STOP-OUTS (same close time) ===')
df2['close_rounded'] = df2['close_dt'].dt.floor('5min')
clusters = df2.groupby('close_rounded').filter(lambda x: len(x) > 1)
if len(clusters) > 0:
    for ts, group in clusters.groupby('close_rounded'):
        print(f'\nCluster at {ts}:')
        for _, row in group.iterrows():
            print(f'  {row["symbol"]:12s} | {row["type"]:4s} | Net: {row["net"]:>+10.2f} | {row["close_reason"]}')
        print(f'  Cluster total: {group["net"].sum():+.2f}')

# SL distance analysis 
print(f'\n=== SL DISTANCE ANALYSIS ===')
for _, row in df2.iterrows():
    entry = row['opening_price']
    sl = row['stop_loss']
    sl_dist_pips = abs(entry - sl)
    if 'JPY' in str(row['symbol']):
        sl_dist_pips = sl_dist_pips * 100  # JPY pairs
    else:
        sl_dist_pips = sl_dist_pips * 10000  # non-JPY
    sl_dist_pct = abs(entry - sl) / entry * 100
    commission = row['commission'] if pd.notna(row['commission']) else 0
    net = row['profit'] + commission
    status = 'WIN' if net >= 0 else 'LOSS'
    print(f'{row["symbol"]:12s} | SL dist: {sl_dist_pips:>7.1f} pips ({sl_dist_pct:.3f}%) | Net: {net:>+10.2f} | {status}')
