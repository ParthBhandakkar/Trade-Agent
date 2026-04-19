"""
Find GENERIC, UNBIASED filters that work for BOTH buy and sell.
Fetches real 1H data for ALL 42 trades and tests universal ICT/SMC filters.
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from scripts.utils.env_loader import load_root_env
load_root_env(REPO)
os.environ["TV_SOURCE_TZ"] = os.getenv("TV_SOURCE_TZ", "Asia/Kolkata").strip() or "Asia/Kolkata"

import pandas as pd, numpy as np, pytz, requests, json
UTC = pytz.UTC
TV_TZ = pytz.timezone(os.getenv("TV_SOURCE_TZ", "Asia/Kolkata"))
from tvDatafeed import TvDatafeed, Interval

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def atr_calc(df, p=14):
    tr = pd.concat([df['high']-df['low'], abs(df['high']-df['close'].shift(1)),
                     abs(df['low']-df['close'].shift(1))], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()
def rsi_calc(s, p=14):
    d = s.diff(); g = d.where(d>0,0); l = -d.where(d<0,0)
    ag = g.ewm(alpha=1/p, adjust=False).mean()
    al = l.ewm(alpha=1/p, adjust=False).mean()
    return 100 - 100/(1 + ag/(al+1e-10))

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

def fetch_tf(tv, sym, interval, n=500):
    for a in range(3):
        try:
            df = tv.get_hist(symbol=sym, exchange="OANDA",
                             interval=interval, n_bars=n)
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

    # Fetch 1H for ALL unique symbols
    all_syms = sorted(set(t["s"] for t in ALL if t["s"] != "ETHUSD" and t["s"] != "BTCUSD"))
    sdata = {}
    print(f"\nFetching 1H for {len(all_syms)} forex symbols...")
    for sym in all_syms:
        print(f"  {sym}...", end=" ", flush=True)
        df = fetch_tf(tv, sym, Interval.in_1_hour, 500)
        if df is not None:
            sdata[sym] = df; print(f"OK ({len(df)})")
        else: print("FAIL")
        time.sleep(1)

    # ── Compute GENERIC metrics for each trade ────────────────
    print(f"\n{'='*100}")
    print("  COMPUTING UNIVERSAL METRICS FOR ALL 42 TRADES")
    print(f"{'='*100}\n")

    for t in ALL:
        sym = t["s"]
        tt = pd.Timestamp(t["dt"]).tz_localize(UTC)
        t["hour"] = tt.hour
        t["win"] = t["p"] > 0

        if sym not in sdata:
            t["ema_aligned"] = None
            t["ema_gap_pct"] = None
            t["atr_ratio"] = None
            t["rsi"] = None
            continue

        df = sdata[sym]
        mask = df.index <= tt
        if mask.sum() < 30:
            t["ema_aligned"] = None; t["ema_gap_pct"] = None
            t["atr_ratio"] = None; t["rsi"] = None
            continue

        sub = df[mask]
        c = float(sub["close"].iloc[-1])
        e21 = float(ema(sub["close"], 21).iloc[-1])
        e50 = float(ema(sub["close"], 50).iloc[-1])
        at = float(atr_calc(sub).iloc[-1])
        rs = float(rsi_calc(sub["close"]).iloc[-1])

        # GENERIC: Is 1H aligned with trade direction?
        # BUY needs close > EMA21, SELL needs close < EMA21
        if t["d"] == "BUY":
            aligned = c > e21
        else:
            aligned = c < e21

        # GENERIC: How far is price from EMA21 (in ATR units)?
        # Positive = correct side, Negative = wrong side
        if t["d"] == "BUY":
            dist = (c - e21) / at  # positive = price above EMA (good for buy)
        else:
            dist = (e21 - c) / at  # positive = price below EMA (good for sell)

        # GENERIC: EMA21 vs EMA50 alignment
        if t["d"] == "BUY":
            ema_trend = e21 > e50
        else:
            ema_trend = e21 < e50

        t["ema_aligned"] = aligned
        t["ema_gap_atr"] = round(dist, 3)  # distance in ATR units
        t["ema_trend_1h"] = ema_trend
        t["rsi"] = round(rs, 1)
        t["atr"] = round(at, 6)
        t["close_1h"] = c
        t["ema21_1h"] = e21
        t["ema50_1h"] = e50

    # ── PRINT ALL TRADES WITH METRICS ─────────────────────────
    print(f"  {'WL':>3} {'P/L':>8} {'Sym':<8} {'Dir':>4} {'Hr':>3} {'1H Aligned':>10} {'Dist(ATR)':>10} {'EMA Trend':>10} {'RSI':>5}")
    print(f"  {'-'*75}")

    for t in sorted(ALL, key=lambda x: x.get("ema_gap_atr", 0) or 0):
        if t.get("ema_aligned") is None:
            al_str = "N/A"
            dist_str = "N/A"
            trend_str = "N/A"
            rsi_str = "N/A"
        else:
            al_str = "YES" if t["ema_aligned"] else "NO"
            dist_str = f"{t['ema_gap_atr']:+.3f}"
            trend_str = "YES" if t.get("ema_trend_1h") else "NO"
            rsi_str = f"{t['rsi']:.0f}"

        wl = "W" if t["win"] else "L"
        print(f"  {wl:>3} Rs{t['p']:>7,.0f} {t['s']:<8} {t['d']:>4} {t['hour']:>3} {al_str:>10} {dist_str:>10} {trend_str:>10} {rsi_str:>5}")

    # ── ANALYSIS: What separates winners from losers? ──────────
    print(f"\n{'='*100}")
    print("  PATTERN ANALYSIS: WHAT SEPARATES WINNERS FROM LOSERS?")
    print(f"{'='*100}")

    # Filter 1: 1H EMA21 Alignment (direction-neutral)
    w_aligned = [t for t in ALL if t["win"] and t.get("ema_aligned") is not None]
    l_aligned = [t for t in ALL if not t["win"] and t.get("ema_aligned") is not None]

    w_yes = sum(1 for t in w_aligned if t["ema_aligned"])
    w_no = sum(1 for t in w_aligned if not t["ema_aligned"])
    l_yes = sum(1 for t in l_aligned if t["ema_aligned"])
    l_no = sum(1 for t in l_aligned if not t["ema_aligned"])

    print(f"\n  FILTER: 1H Close on correct side of EMA21 (generic for both BUY & SELL)")
    print(f"  Winners: {w_yes} aligned, {w_no} misaligned")
    print(f"  Losers:  {l_yes} aligned, {l_no} misaligned")
    if w_no > 0:
        mis_wins = [t for t in w_aligned if not t["ema_aligned"]]
        print(f"  Misaligned winners (would be blocked):")
        for t in mis_wins:
            print(f"    {t['s']} {t['d']} Rs{t['p']:,.0f} gap_atr={t['ema_gap_atr']}")

    # Filter 2: 1H EMA21 > EMA50 trend (generic)
    print(f"\n  FILTER: 1H EMA21 vs EMA50 trend alignment (generic)")
    w_trend = sum(1 for t in w_aligned if t.get("ema_trend_1h"))
    w_notrend = sum(1 for t in w_aligned if not t.get("ema_trend_1h"))
    l_trend = sum(1 for t in l_aligned if t.get("ema_trend_1h"))
    l_notrend = sum(1 for t in l_aligned if not t.get("ema_trend_1h"))
    print(f"  Winners: {w_trend} trend-aligned, {w_notrend} counter-trend")
    print(f"  Losers:  {l_trend} trend-aligned, {l_notrend} counter-trend")

    # Filter 3: Distance from EMA21 in ATR units
    print(f"\n  FILTER: Distance from EMA21 (in ATR units, negative = wrong side)")
    w_dists = [t["ema_gap_atr"] for t in w_aligned if t.get("ema_gap_atr") is not None]
    l_dists = [t["ema_gap_atr"] for t in l_aligned if t.get("ema_gap_atr") is not None]
    if w_dists: print(f"  Winners: min={min(w_dists):.3f}, median={sorted(w_dists)[len(w_dists)//2]:.3f}, max={max(w_dists):.3f}")
    if l_dists: print(f"  Losers:  min={min(l_dists):.3f}, median={sorted(l_dists)[len(l_dists)//2]:.3f}, max={max(l_dists):.3f}")

    # Filter 4: London open hour (generic for both directions)
    print(f"\n  FILTER: London Open block (generic, applies to BUY and SELL equally)")
    for h_start, h_end, label in [(7,8,"07:00-08:00"),(7,8.5,"07:00-08:30")]:
        w_in = sum(1 for t in ALL if t["win"] and h_start*60 <= t["hour"]*60+int(t["dt"][14:16]) <= h_end*60)
        l_in = sum(1 for t in ALL if not t["win"] and h_start*60 <= t["hour"]*60+int(t["dt"][14:16]) <= h_end*60)
        print(f"  {label}: {w_in} winners blocked, {l_in} losers blocked")

    # ── COMBINED GENERIC FILTER TEST ──────────────────────────
    print(f"\n{'='*100}")
    print("  TESTING GENERIC FILTER COMBINATIONS (direction-neutral)")
    print(f"{'='*100}")

    # Test: 1H EMA alignment + London block combos
    combos = []
    for london in [("OFF",None,None), ("07-08",7*60,8*60)]:
        for ema_filter in [("OFF",False), ("1H_align",True)]:
            for ema_trend in [("OFF",False), ("1H_trend",True)]:
                ln, ls, le = london
                en, eon = ema_filter
                tn, ton = ema_trend

                kept = []
                for t in ALL:
                    blocked = False
                    h = t["hour"]; m = int(t["dt"][14:16])
                    tmin = h*60+m

                    # London block (GENERIC - blocks both directions)
                    if ls is not None and ls <= tmin <= le:
                        blocked = True

                    # 1H EMA alignment (GENERIC - same rule for buy & sell)
                    if not blocked and eon and t.get("ema_aligned") is not None:
                        if not t["ema_aligned"]:
                            blocked = True

                    # 1H EMA trend (GENERIC)
                    if not blocked and ton and t.get("ema_trend_1h") is not None:
                        if not t["ema_trend_1h"]:
                            blocked = True

                    if not blocked:
                        kept.append(t)

                n = len(kept)
                w = sum(1 for x in kept if x["p"]>0)
                wr = w/n*100 if n>0 else 0
                pnl = sum(x["p"] for x in kept)
                blocked_trades = [x for x in ALL if x not in kept]
                loss_saved = sum(abs(x["p"]) for x in blocked_trades if x["p"]<0)
                win_lost = sum(x["p"] for x in blocked_trades if x["p"]>0)

                combos.append({
                    "london":ln, "ema":en, "trend":tn,
                    "n":n, "w":w, "wr":wr, "pnl":pnl,
                    "saved":loss_saved, "lost":win_lost
                })

    combos.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"\n  {'London':<8} {'1H EMA':>8} {'1H Trend':>9} {'Trades':>6} {'Wins':>5} {'WR%':>6} {'P/L':>10} {'Saved':>8} {'Lost':>8}")
    print(f"  {'-'*80}")
    for c in combos:
        print(f"  {c['london']:<8} {c['ema']:>8} {c['trend']:>9} {c['n']:>6} {c['w']:>5} {c['wr']:>5.1f}% Rs{c['pnl']:>8,.0f} Rs{c['saved']:>6,.0f} Rs{c['lost']:>6,.0f}")

    # ── SHOW INDIVIDUAL TRADE IMPACT FOR BEST COMBO ───────────
    print(f"\n{'='*100}")
    print("  BEST GENERIC SETUP: PER-TRADE BREAKDOWN")
    print(f"{'='*100}")

    best = combos[0]
    print(f"\n  Config: London={best['london']} | 1H EMA={best['ema']} | 1H Trend={best['trend']}")
    print(f"  Result: {best['n']} trades, {best['w']} wins ({best['wr']:.1f}%), Rs {best['pnl']:,.0f}\n")

    for t in ALL:
        blocked = False
        reason = ""
        h = t["hour"]; m = int(t["dt"][14:16])
        tmin = h*60+m

        if best["london"] != "OFF":
            ls = 7*60; le = 8*60
            if ls <= tmin <= le:
                blocked = True; reason = "LONDON BLOCK"

        if not blocked and best["ema"] != "OFF" and t.get("ema_aligned") is not None:
            if not t["ema_aligned"]:
                blocked = True; reason = f"1H EMA misaligned (gap={t['ema_gap_atr']:+.3f} ATR)"

        if not blocked and best["trend"] != "OFF" and t.get("ema_trend_1h") is not None:
            if not t["ema_trend_1h"]:
                blocked = True; reason = f"1H EMA21 vs EMA50 counter-trend"

        wl = "WIN " if t["win"] else "LOSS"
        status = "BLOCKED" if blocked else "KEPT"
        icon = "X" if blocked else ">"

        if blocked:
            if t["win"]:
                impact = f"(lost Rs{t['p']:,.0f})"
            else:
                impact = f"(saved Rs{abs(t['p']):,.0f})"
        else:
            impact = ""

        print(f"  {icon} {wl} Rs{t['p']:>8,.0f} {t['s']:<8} {t['d']:>4} {t['dt'][5:16]} {status:<8} {reason} {impact}")

    print(f"\n{'='*100}")
