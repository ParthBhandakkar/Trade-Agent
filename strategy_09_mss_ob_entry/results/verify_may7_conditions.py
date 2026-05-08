
import os, sys, json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pytz

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'strategy_09_mss_ob_entry'))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
except Exception:
    pass

import MetaTrader5 as mt5
from strategy_09_mss_ob_entry.strategy import MSSOrderBlockStrategy, BiasType
from strategy_09_mss_ob_entry.auto_trader import _drop_incomplete, _check_strict_ema, compute_smart_sl, compute_tp
from scripts.utils.indicators import Direction, is_bullish_candle, is_bearish_candle

IST=pytz.timezone('Asia/Kolkata')
UTC=pytz.UTC
TF={'4h': mt5.TIMEFRAME_H4, '1h': mt5.TIMEFRAME_H1, '15m': mt5.TIMEFRAME_M15, '5m': mt5.TIMEFRAME_M5}
MINUTES={'4h':240,'1h':60,'15m':15,'5m':5}

TRADES=[
    dict(symbol='EURAUD', direction='bearish', detected='2026-05-07 18:46:03', signal='2026-05-07 18:40:00', entry=1.62403, sl=1.62697, tp=1.61962, quality=50),
    dict(symbol='GBPNZD', direction='bearish', detected='2026-05-07 20:51:00', signal='2026-05-07 20:45:00', entry=2.28150, sl=2.28450, tp=2.27700, quality=54),
    dict(symbol='AUDUSD', direction='bullish', detected='2026-05-07 21:51:00', signal='2026-05-07 21:45:00', entry=0.72347, sl=0.72138, tp=0.726605, quality=69),
    dict(symbol='NZDUSD', direction='bullish', detected='2026-05-07 21:51:04', signal='2026-05-07 21:45:00', entry=0.59572, sl=0.59350, tp=0.59905, quality=54),
]

def ist_dt(s):
    return IST.localize(datetime.strptime(s, '%Y-%m-%d %H:%M:%S'))

def resolve(sym):
    for c in [sym, sym+'m', sym+'.m']:
        if mt5.symbol_select(c, True): return c
    matches=mt5.symbols_get(f'*{sym}*') or []
    for m in matches:
        if mt5.symbol_select(m.name, True): return m.name
    return None

def fetch(sym, interval, end_utc, days=25):
    start=end_utc - timedelta(days=days)
    rates=mt5.copy_rates_range(sym, TF[interval], start, end_utc + timedelta(minutes=MINUTES[interval]))
    if rates is None or len(rates)==0: return None
    df=pd.DataFrame(rates)
    df['datetime']=pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('datetime', inplace=True)
    vol='tick_volume' if 'tick_volume' in df.columns else 'real_volume'
    df['volume']=df[vol] if vol in df.columns else 0
    return df[['open','high','low','close','volume']].sort_index()

def to_ist(ts):
    if ts is None: return None
    if hasattr(ts, 'to_pydatetime'): ts=ts.to_pydatetime()
    if ts.tzinfo is None: ts=UTC.localize(ts)
    return ts.astimezone(IST).strftime('%Y-%m-%d %H:%M')

def near(a,b,tol): return abs(float(a)-float(b)) <= tol

def enumerate_ob_taps(strategy, df15, df5, bias, mss_conf, expected_tap, expected_entry):
    out=[]
    mss_idx=df15.index.searchsorted(mss_conf.timestamp)
    if mss_idx >= len(df15): mss_idx=len(df15)-1
    lookback_start=max(0,mss_idx-60)
    seg=df15.iloc[lookback_start:mss_idx+1]
    if len(seg)<3: return out
    if bias.direction == BiasType.BULLISH:
        swing_low=seg['low'].min(); swing_high=mss_conf.mss.break_price
        if swing_high <= swing_low: return out
        fib_range=swing_high-swing_low
        zone_top=swing_high-strategy.ote_fib_low*fib_range
        zone_bottom=swing_high-strategy.ote_fib_high*fib_range
        direction=Direction.BULLISH
    else:
        swing_high=seg['high'].max(); swing_low=mss_conf.mss.break_price
        if swing_high <= swing_low: return out
        fib_range=swing_high-swing_low
        zone_bottom=swing_low+strategy.ote_fib_low*fib_range
        zone_top=swing_low+strategy.ote_fib_high*fib_range
        direction=Direction.BEARISH

    search_start=max(0,mss_idx-strategy.max_ob_age_candles)
    for i in range(mss_idx-1, search_start-1, -1):
        c=df15.iloc[i]
        if direction == Direction.BULLISH:
            if not is_bearish_candle(c): continue
            body_top=c['open']; body_bottom=c['close']
            ok = body_top >= zone_bottom and body_bottom <= zone_top
            top=c['high']; bottom=c['low']
        else:
            if not is_bullish_candle(c): continue
            body_top=c['close']; body_bottom=c['open']
            ok = body_top >= zone_bottom and body_bottom <= zone_top
            top=c['high']; bottom=c['low']
        if not ok: continue
        ob_mid=(top+bottom)/2
        fib_level=(swing_high-ob_mid)/fib_range if direction==Direction.BULLISH else (ob_mid-swing_low)/fib_range
        start_idx=df5.index.searchsorted(mss_conf.timestamp)
        for j in range(start_idx, len(df5)):
            k=df5.iloc[j]
            if direction == Direction.BULLISH:
                tapped=k['low'] <= body_top
                entry=max(body_top, k['close']) if tapped else None
            else:
                tapped=k['high'] >= body_bottom
                entry=min(body_bottom, k['close']) if tapped else None
            if tapped:
                tap_ts=df5.index[j]
                # Keep all taps near expected, plus first tap for context
                if len(out)<2 or abs((tap_ts-expected_tap).total_seconds()) <= 3600 or near(entry, expected_entry, 0.0008 if expected_entry<1 else 0.008):
                    out.append(dict(
                        ob_time=to_ist(df15.index[i]), ob_top=float(top), ob_bottom=float(bottom),
                        body_top=float(body_top), body_bottom=float(body_bottom),
                        fib=round(float(fib_level),3), in_ote= strategy.ote_fib_low <= fib_level <= strategy.ote_fib_high,
                        tap_time=to_ist(tap_ts), entry=float(entry), first=(j==start_idx),
                        zone_bottom=float(zone_bottom), zone_top=float(zone_top),
                        swing_low=float(swing_low), swing_high=float(swing_high),
                    ))
                break
    return out

if not mt5.initialize():
    print(json.dumps({'error':'mt5 init failed','last_error':mt5.last_error()})); raise SystemExit
login=os.getenv('MT5_LOGIN') or os.getenv('XM_MT5_LOGIN')
pw=os.getenv('MT5_PASSWORD') or os.getenv('XM_MT5_PASSWORD')
server=os.getenv('MT5_SERVER') or os.getenv('XM_MT5_SERVER')
if login and pw: mt5.login(int(login), password=pw, server=server or None)
acct=mt5.account_info()
print('ACCOUNT', acct.login if acct else None, acct.server if acct else None, acct.currency if acct else None, acct.leverage if acct else None)
strategy=MSSOrderBlockStrategy()

for tr in TRADES:
    print('\n'+'='*100)
    print(tr['symbol'], tr['direction'], 'detected', tr['detected'], 'signal', tr['signal'])
    sym=resolve(tr['symbol'])
    print('MT5_SYMBOL', sym)
    det_utc=ist_dt(tr['detected']).astimezone(UTC)
    sig_utc=ist_dt(tr['signal']).astimezone(UTC)
    dfs={iv: fetch(sym,iv,det_utc) for iv in TF}
    for iv,df in dfs.items():
        if df is not None:
            dfs[iv]=_drop_incomplete(df, MINUTES[iv], det_utc)
            print('BARS',iv,len(dfs[iv]),'last',to_ist(dfs[iv].index[-1]))
        else:
            print('BARS',iv,'NONE')
    if any(v is None or v.empty for v in dfs.values()): continue

    biases=strategy.determine_bias(dfs['4h'],72)
    exp_bias=BiasType.BULLISH if tr['direction']=='bullish' else BiasType.BEARISH
    print('4H_BIASES', len(biases))
    for b in biases:
        if b.direction==exp_bias:
            print('  BIAS_MATCH', b.direction.value, b.reason, 'sweep', to_ist(b.sweep_timestamp), 'level_time', to_ist(b.source_timestamp), 'conf', b.confidence)

    matches=[]
    for b in biases:
        if b.direction != exp_bias: continue
        m=strategy.confirm_mss(dfs['1h'], b)
        if not m: continue
        obs=enumerate_ob_taps(strategy, dfs['15m'], dfs['5m'], b, m, sig_utc, tr['entry'])
        q_entries=[]
        for o in obs:
            is_near_time=(o['tap_time']==ist_dt(tr['signal']).strftime('%Y-%m-%d %H:%M'))
            tol=0.0008 if tr['entry']<1 else 0.008
            is_near_entry=near(o['entry'], tr['entry'], tol)
            if is_near_time or is_near_entry:
                # build actual ob_entry from strategy on filtered 5m after previous tap is hard; use details and score approximation if exact first returned
                q_entries.append((o,is_near_time,is_near_entry))
        matches.append((b,m,obs,q_entries))
    print('PHASE_CHAINS_WITH_MSS', len(matches))
    for b,m,obs,q_entries in matches:
        print('  MSS', m.details, 'mss_time', to_ist(m.timestamp), 'after_sweep', m.timestamp>b.sweep_timestamp)
        print('  OB/TAP candidates shown', len(obs))
        for o in obs[:8]:
            mark='*' if o['tap_time']==ist_dt(tr['signal']).strftime('%Y-%m-%d %H:%M') else ' '
            print(f"   {mark}OB {o['ob_time']} body {o['body_bottom']:.5f}-{o['body_top']:.5f} top/bot {o['ob_top']:.5f}/{o['ob_bottom']:.5f} fib {o['fib']} OTE {o['in_ote']} tap {o['tap_time']} entry {o['entry']:.5f} zone {o['zone_bottom']:.5f}-{o['zone_top']:.5f}")
        if q_entries:
            print('  MATCHED_EXPECTED_TAP_OR_ENTRY yes')
    ema_ok, ema_detail=_check_strict_ema(dfs['4h'], tr['direction'])
    print('4H_EMA_FILTER', ema_ok, ema_detail)
    # find raw 5m candle at signal time
    if sig_utc in dfs['5m'].index:
        c=dfs['5m'].loc[sig_utc]
        print('5M_SIGNAL_CANDLE', 'open',float(c.open),'high',float(c.high),'low',float(c.low),'close',float(c.close))
    else:
        loc=dfs['5m'].index.searchsorted(sig_utc)
        print('5M_SIGNAL_CANDLE_NOT_FOUND nearest', to_ist(dfs['5m'].index[loc-1]) if loc else None, to_ist(dfs['5m'].index[loc]) if loc<len(dfs['5m']) else None)
mt5.shutdown()
