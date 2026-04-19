"""Optimize all filter parameters against real trade data + real 1H EMA values."""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from scripts.utils.env_loader import load_root_env
load_root_env(REPO)
os.environ["TV_SOURCE_TZ"] = os.getenv("TV_SOURCE_TZ", "Asia/Kolkata").strip() or "Asia/Kolkata"

import pandas as pd, pytz, requests, json
UTC = pytz.UTC
TV_TZ = pytz.timezone(os.getenv("TV_SOURCE_TZ", "Asia/Kolkata"))

from tvDatafeed import TvDatafeed, Interval

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def atr(df, p=14):
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift(1)),
                     abs(df['low']-df['close'].shift(1))], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def resolve_token(sid):
    import re
    try:
        s = requests.Session()
        s.cookies.set('sessionid', sid, domain='.tradingview.com')
        r = s.get('https://www.tradingview.com/chart/',
                  headers={'User-Agent':'Mozilla/5.0'}, timeout=30)
        m = re.search(r'"auth_token":"([^"]+)"', r.text)
        return m.group(1) if m else None
    except: return None

def fetch(tv, sym, n=500):
    for a in range(3):
        try:
            df = tv.get_hist(symbol=sym, exchange="OANDA",
                             interval=Interval.in_1_hour, n_bars=n)
            if df is not None and not df.empty:
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                if df.index.tz is None: df.index = df.index.tz_localize(TV_TZ)
                df.index = df.index.tz_convert(UTC)
                return df[['open','high','low','close','volume']]
        except: time.sleep(2)
    return None

# All 42 trades
ALL = [
 {"t":1042640024,"dt":"2026-04-02T09:08","s":"NZDCHF","d":"SELL","p":501.63},
 {"t":1042876863,"dt":"2026-04-02T09:48","s":"GBPNZD","d":"BUY","p":3071.70},
 {"t":1044562087,"dt":"2026-04-02T13:37","s":"GBPAUD","d":"BUY","p":497.02},
 {"t":1044801344,"dt":"2026-04-02T13:57","s":"AUDJPY","d":"SELL","p":500.78},
 {"t":1044918126,"dt":"2026-04-02T14:07","s":"GBPNZD","d":"BUY","p":1496.84},
 {"t":1045286518,"dt":"2026-04-02T14:42","s":"AUDCAD","d":"SELL","p":-882.97},
 {"t":1045287991,"dt":"2026-04-02T14:42","s":"NZDCAD","d":"SELL","p":494.38},
 {"t":1046307259,"dt":"2026-04-02T16:47","s":"EURAUD","d":"BUY","p":505.04},
 {"t":1047895385,"dt":"2026-04-03T16:07","s":"EURCAD","d":"BUY","p":4001.15},
 {"t":1052341725,"dt":"2026-04-06T07:57","s":"USDCAD","d":"BUY","p":-1571.68},
 {"t":1055090649,"dt":"2026-04-06T14:43","s":"AUDCHF","d":"BUY","p":1485.55},
 {"t":1055091016,"dt":"2026-04-06T14:43","s":"AUDCHF","d":"BUY","p":1985.07},
 {"t":1059472654,"dt":"2026-04-07T08:08","s":"EURUSD","d":"SELL","p":-2453.48},
 {"t":1059472893,"dt":"2026-04-07T08:08","s":"GBPUSD","d":"SELL","p":-2540.68},
 {"t":1059481827,"dt":"2026-04-07T08:11","s":"EURUSD","d":"SELL","p":-1752.73},
 {"t":1059483729,"dt":"2026-04-07T08:11","s":"GBPUSD","d":"SELL","p":-1743.63},
 {"t":1059860509,"dt":"2026-04-07T09:01","s":"USDJPY","d":"BUY","p":1500.70},
 {"t":1059869379,"dt":"2026-04-07T09:03","s":"USDJPY","d":"BUY","p":1500.70},
 {"t":1074951684,"dt":"2026-04-09T08:27","s":"GBPAUD","d":"SELL","p":1002.34},
 {"t":1075704758,"dt":"2026-04-09T11:03","s":"GBPCAD","d":"BUY","p":1000.60},
 {"t":1076946507,"dt":"2026-04-09T13:57","s":"EURCHF","d":"BUY","p":1002.36},
 {"t":1083550917,"dt":"2026-04-10T14:22","s":"EURCHF","d":"BUY","p":4503.13},
 {"t":1083671633,"dt":"2026-04-10T14:36","s":"EURGBP","d":"BUY","p":2004.21},
 {"t":1091405657,"dt":"2026-04-13T14:03","s":"EURCHF","d":"BUY","p":-4828.16},
 {"t":1102178641,"dt":"2026-04-15T08:21","s":"CADJPY","d":"BUY","p":3000.00},
 {"t":1102370871,"dt":"2026-04-15T09:07","s":"CADJPY","d":"BUY","p":1494.00},
 {"t":1102902409,"dt":"2026-04-15T10:39","s":"ETHUSD","d":"BUY","p":1414.51},
 {"t":1107625702,"dt":"2026-04-16T07:37","s":"CHFJPY","d":"BUY","p":1499.20},
 {"t":1107638691,"dt":"2026-04-16T07:40","s":"USDCHF","d":"SELL","p":-2982.99},
 {"t":1107825244,"dt":"2026-04-16T08:23","s":"USDJPY","d":"SELL","p":-2817.25},
 {"t":1107825857,"dt":"2026-04-16T08:24","s":"USDJPY","d":"SELL","p":-2370.37},
 {"t":1107842879,"dt":"2026-04-16T08:28","s":"CHFJPY","d":"BUY","p":-3211.42},
 {"t":1108084641,"dt":"2026-04-16T09:33","s":"EURGBP","d":"SELL","p":1993.69},
 {"t":1108097633,"dt":"2026-04-16T09:36","s":"EURGBP","d":"SELL","p":998.95},
 {"t":1109518654,"dt":"2026-04-16T13:58","s":"BTCUSD","d":"BUY","p":-3030.89},
 {"t":1109668204,"dt":"2026-04-16T14:09","s":"EURCHF","d":"BUY","p":2491.00},
 {"t":1109847843,"dt":"2026-04-16T14:30","s":"EURGBP","d":"SELL","p":-409.62},
 {"t":1109893129,"dt":"2026-04-16T14:38","s":"GBPCHF","d":"BUY","p":1999.00},
 {"t":1109972897,"dt":"2026-04-16T14:44","s":"GBPCHF","d":"BUY","p":2999.40},
 {"t":1113538369,"dt":"2026-04-17T08:17","s":"NZDJPY","d":"BUY","p":2006.00},
 {"t":1114142016,"dt":"2026-04-17T10:47","s":"NZDUSD","d":"BUY","p":7602.06},
 {"t":1114144800,"dt":"2026-04-17T10:47","s":"GBPNZD","d":"SELL","p":5171.79},
]

if __name__ == "__main__":
    sid = os.environ.get("TV_SESSION_TOKEN","").strip()
    tv = TvDatafeed()
    if sid:
        tok = resolve_token(sid)
        if tok: tv.token = tok; print("Auth OK")

    # Fetch 1H for all SELL symbols
    sell_syms = sorted(set(t["s"] for t in ALL if t["d"]=="SELL"))
    sdata = {}
    print(f"\nFetching 1H for {len(sell_syms)} SELL symbols...")
    for sym in sell_syms:
        print(f"  {sym}...", end=" ", flush=True)
        df = fetch(tv, sym)
        if df is not None:
            sdata[sym] = df
            print(f"OK ({len(df)} bars)")
        else: print("FAIL")
        time.sleep(1)

    # Compute gap% for each SELL trade
    for t in ALL:
        if t["d"] != "SELL":
            t["gap_pct"] = None
            continue
        sym = t["s"]
        if sym not in sdata:
            t["gap_pct"] = None; continue
        df = sdata[sym]
        tt = pd.Timestamp(t["dt"]).tz_localize(UTC)
        mask = df.index <= tt
        if mask.sum() < 25:
            t["gap_pct"] = None; continue
        sub = df[mask]
        c = float(sub["close"].iloc[-1])
        e = float(ema(sub["close"], 21).iloc[-1])
        t["gap_pct"] = (c - e) / e * 100  # positive = above EMA
        t["1h_close"] = c
        t["ema21"] = e

    # Print gap analysis
    print(f"\n{'='*80}")
    print("  SELL TRADE GAP ANALYSIS (Close vs EMA21)")
    print(f"{'='*80}")
    sells = [t for t in ALL if t["d"]=="SELL" and t.get("gap_pct") is not None]
    sells_sorted = sorted(sells, key=lambda x: x["gap_pct"])
    for t in sells_sorted:
        win = "WIN " if t["p"]>0 else "LOSS"
        print(f"  {win} Rs{t['p']:>8,.0f} | {t['s']:<8} | gap: {t['gap_pct']:>+.4f}% | {t['dt'][5:]}")

    # ── OPTIMIZATION: Test all filter combos ─────────────────
    london_windows = [
        ("OFF", None, None),
        ("07:00-08:00", 7*60, 8*60),
        ("07:00-08:30", 7*60, 8*60+30),
        ("07:00-09:00", 7*60, 9*60),
        ("07:30-08:30", 7*60+30, 8*60+30),
    ]

    sell_thresholds = [
        ("OFF", None),
        ("0.05%", 0.05),
        ("0.08%", 0.08),
        ("0.10%", 0.10),
        ("0.15%", 0.15),
        ("0.20%", 0.20),
        ("0.25%", 0.25),
        ("0.30%", 0.30),
        ("ALL_SELLS_BLOCKED", -999),
    ]

    corr_options = [("OFF", False), ("ON", True)]

    print(f"\n{'='*80}")
    print("  FILTER OPTIMIZATION — TESTING ALL COMBINATIONS")
    print(f"{'='*80}")

    results = []
    for l_name, l_start, l_end in london_windows:
        for s_name, s_thresh in sell_thresholds:
            for c_name, c_on in corr_options:
                kept = []
                blocked_wins = []
                blocked_losses = []
                exposed_currencies = set()

                for t in ALL:
                    trade_blocked = False
                    block_reason = ""
                    h = int(t["dt"][11:13])
                    m = int(t["dt"][14:16])
                    t_min = h * 60 + m

                    # London block
                    if l_start is not None and l_start <= t_min <= l_end:
                        trade_blocked = True
                        block_reason = "london"

                    # Correlation guard
                    if not trade_blocked and c_on:
                        sym = t["s"]
                        base = sym[:3]
                        quote = sym[3:6] if len(sym) >= 6 else ""
                        if base in exposed_currencies or (quote and quote in exposed_currencies):
                            trade_blocked = True
                            block_reason = "correl"

                    # SELL filter
                    if not trade_blocked and t["d"] == "SELL" and s_thresh is not None:
                        gp = t.get("gap_pct")
                        if s_thresh == -999:
                            trade_blocked = True
                            block_reason = "sell_all"
                        elif gp is not None and gp > s_thresh:
                            trade_blocked = True
                            block_reason = "sell_gap"

                    if trade_blocked:
                        if t["p"] > 0:
                            blocked_wins.append(t)
                        else:
                            blocked_losses.append(t)
                    else:
                        kept.append(t)
                        # Track exposed currencies
                        if c_on:
                            sym = t["s"]
                            exposed_currencies.add(sym[:3])
                            if len(sym) >= 6:
                                exposed_currencies.add(sym[3:6])

                total_kept = len(kept)
                wins_kept = sum(1 for x in kept if x["p"] > 0)
                pnl_kept = sum(x["p"] for x in kept)
                wr = wins_kept/total_kept*100 if total_kept > 0 else 0
                loss_saved = sum(abs(x["p"]) for x in blocked_losses)
                win_lost = sum(x["p"] for x in blocked_wins)

                results.append({
                    "london": l_name, "sell": s_name, "corr": c_name,
                    "trades": total_kept, "wins": wins_kept, "wr": wr,
                    "pnl": pnl_kept, "loss_saved": loss_saved,
                    "win_lost": win_lost, "net_impact": loss_saved - win_lost,
                    "blocked_w": len(blocked_wins), "blocked_l": len(blocked_losses),
                })

    # Sort by net P/L (what you actually keep)
    results.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"\n  TOP 20 CONFIGURATIONS (sorted by NET P/L kept)")
    print(f"  {'London':<14} {'SELL':<18} {'Corr':>4} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'Net P/L':>10} {'Saved':>8} {'Lost':>8} {'Impact':>8}")
    print(f"  {'-'*100}")
    for r in results[:20]:
        print(f"  {r['london']:<14} {r['sell']:<18} {r['corr']:>4} {r['trades']:>6} {r['wins']:>5} {r['wr']:>5.1f}% Rs{r['pnl']:>8,.0f} Rs{r['loss_saved']:>6,.0f} Rs{r['win_lost']:>6,.0f} Rs{r['net_impact']:>+6,.0f}")

    # Find the best WR with decent P/L
    print(f"\n  TOP 10 BY WIN RATE (min 20 trades)")
    print(f"  {'London':<14} {'SELL':<18} {'Corr':>4} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'Net P/L':>10}")
    print(f"  {'-'*70}")
    wr_sorted = sorted([r for r in results if r["trades"] >= 20], key=lambda x: x["wr"], reverse=True)
    for r in wr_sorted[:10]:
        print(f"  {r['london']:<14} {r['sell']:<18} {r['corr']:>4} {r['trades']:>6} {r['wins']:>5} {r['wr']:>5.1f}% Rs{r['pnl']:>8,.0f}")

    # Find the BEST balance (highest P/L with WR > 75%)
    print(f"\n{'='*80}")
    print("  RECOMMENDED OPTIMAL SETUP")
    print(f"{'='*80}")
    balanced = sorted([r for r in results if r["wr"] >= 75 and r["trades"] >= 20],
                       key=lambda x: x["pnl"], reverse=True)
    if balanced:
        b = balanced[0]
        print(f"\n  London Block:     {b['london']}")
        print(f"  SELL Filter:      {b['sell']}")
        print(f"  Correl Guard:     {b['corr']}")
        print(f"  ---")
        print(f"  Trades Taken:     {b['trades']}")
        print(f"  Winners:          {b['wins']} ({b['wr']:.1f}%)")
        print(f"  Net P/L:          Rs {b['pnl']:,.0f}")
        print(f"  Losses Saved:     Rs {b['loss_saved']:,.0f}")
        print(f"  Winners Given Up: Rs {b['win_lost']:,.0f}")
        print(f"  Net Improvement:  Rs {b['net_impact']:+,.0f}")

    # Also find best if WR > 80%
    strict = sorted([r for r in results if r["wr"] >= 80 and r["trades"] >= 15],
                     key=lambda x: x["pnl"], reverse=True)
    if strict:
        b = strict[0]
        print(f"\n  AGGRESSIVE (WR > 80%):")
        print(f"  London: {b['london']} | SELL: {b['sell']} | Corr: {b['corr']}")
        print(f"  {b['trades']} trades, {b['wins']} wins ({b['wr']:.1f}%), Rs {b['pnl']:,.0f}")

    # Save all results
    out = Path(__file__).parent / "optimization_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to: {out}")
    print(f"{'='*80}")
