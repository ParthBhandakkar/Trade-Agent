
import os, sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd, pytz
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT), str(ROOT/'scripts'), str(ROOT/'strategy_09_mss_ob_entry')]
try:
 from dotenv import load_dotenv; load_dotenv(ROOT/'.env')
except Exception: pass
import MetaTrader5 as mt5
from strategy_09_mss_ob_entry.strategy import MSSOrderBlockStrategy, BiasType
from strategy_09_mss_ob_entry.auto_trader import _drop_incomplete, _check_strict_ema
from scripts.utils.indicators import Direction, is_bullish_candle, is_bearish_candle
IST=pytz.timezone('Asia/Kolkata'); UTC=pytz.UTC
TF={'4h':mt5.TIMEFRAME_H4,'1h':mt5.TIMEFRAME_H1,'15m':mt5.TIMEFRAME_M15,'5m':mt5.TIMEFRAME_M5}; MIN={'4h':240,'1h':60,'15m':15,'5m':5}
TRADES=[('EURAUD','bearish','2026-05-07 18:46:03','2026-05-07 18:40:00',1.62403),('GBPNZD','bearish','2026-05-07 20:51:00','2026-05-07 20:45:00',2.28150),('AUDUSD','bullish','2026-05-07 21:51:00','2026-05-07 21:45:00',0.72347),('NZDUSD','bullish','2026-05-07 21:51:04','2026-05-07 21:45:00',0.59572)]
def idt(s): return IST.localize(datetime.strptime(s,'%Y-%m-%d %H:%M:%S'))
def ti(ts):
 if hasattr(ts,'to_pydatetime'): ts=ts.to_pydatetime()
 if ts.tzinfo is None: ts=UTC.localize(ts)
 return ts.astimezone(IST).strftime('%Y-%m-%d %H:%M')
def resolve(sym):
 for c in [sym,sym+'m',sym+'.m']:
  if mt5.symbol_select(c,True): return c
 for m in (mt5.symbols_get(f'*{sym}*') or []):
  if mt5.symbol_select(m.name,True): return m.name
 return None
def fetch(sym,iv,end):
 r=mt5.copy_rates_range(sym,TF[iv],end-timedelta(days=25),end+timedelta(minutes=MIN[iv]))
 if r is None or len(r)==0: return None
 df=pd.DataFrame(r); df['datetime']=pd.to_datetime(df['time'],unit='s',utc=True); df.set_index('datetime',inplace=True)
 vol='tick_volume' if 'tick_volume' in df.columns else 'real_volume'; df['volume']=df[vol] if vol in df.columns else 0
 return df[['open','high','low','close','volume']].sort_index()
mt5.initialize(); login=os.getenv('MT5_LOGIN') or os.getenv('XM_MT5_LOGIN'); pw=os.getenv('MT5_PASSWORD') or os.getenv('XM_MT5_PASSWORD'); server=os.getenv('MT5_SERVER') or os.getenv('XM_MT5_SERVER')
if login and pw: mt5.login(int(login),password=pw,server=server or None)
st=MSSOrderBlockStrategy()
for sym0,dirn,det,sigt,entry_expected in TRADES:
 print('\n==',sym0,dirn,sigt)
 sym=resolve(sym0); detu=idt(det).astimezone(UTC); sigu=idt(sigt).astimezone(UTC)
 dfs={iv:_drop_incomplete(fetch(sym,iv,detu),MIN[iv],detu) for iv in TF}
 exp=BiasType.BULLISH if dirn=='bullish' else BiasType.BEARISH; sig_c=dfs['5m'].loc[sigu]
 print('signal candle OHLC',float(sig_c.open),float(sig_c.high),float(sig_c.low),float(sig_c.close))
 for b in st.determine_bias(dfs['4h'],72):
  if b.direction!=exp: continue
  m=st.confirm_mss(dfs['1h'],b)
  if not m: continue
  mss_idx=dfs['15m'].index.searchsorted(m.timestamp); seg=dfs['15m'].iloc[max(0,mss_idx-60):mss_idx+1]
  if exp==BiasType.BULLISH:
   swing_low=seg.low.min(); swing_high=m.mss.break_price; rng=swing_high-swing_low; ztop=swing_high-st.ote_fib_low*rng; zbot=swing_high-st.ote_fib_high*rng; direction=Direction.BULLISH
  else:
   swing_high=seg.high.max(); swing_low=m.mss.break_price; rng=swing_high-swing_low; zbot=swing_low+st.ote_fib_low*rng; ztop=swing_low+st.ote_fib_high*rng; direction=Direction.BEARISH
  print('bias',b.reason,'sweep',ti(b.sweep_timestamp),'mss',m.details,'mss_time',ti(m.timestamp),'OTE',round(zbot,5),round(ztop,5))
  hits=[]
  for i in range(mss_idx-1,max(0,mss_idx-st.max_ob_age_candles)-1,-1):
   c=dfs['15m'].iloc[i]
   if direction==Direction.BULLISH:
    if not is_bearish_candle(c): continue
    bt,bb=float(c.open),float(c.close); top,bottom=float(c.high),float(c.low)
    overlaps=bt>=zbot and bb<=ztop
    tapped=float(sig_c.low)<=bt
    ent=max(bt,float(sig_c.close)) if tapped else None
    fib=(swing_high-(top+bottom)/2)/rng
   else:
    if not is_bullish_candle(c): continue
    bt,bb=float(c.close),float(c.open); top,bottom=float(c.high),float(c.low)
    overlaps=bt>=zbot and bb<=ztop
    tapped=float(sig_c.high)>=bb
    ent=min(bb,float(sig_c.close)) if tapped else None
    fib=((top+bottom)/2-swing_low)/rng
   if overlaps and tapped:
    hits.append((abs(ent-entry_expected),ti(dfs['15m'].index[i]),bb,bt,top,bottom,round(fib,3),st.ote_fib_low<=fib<=st.ote_fib_high,ent))
  for h in sorted(hits)[:5]:
   _,ot,bb,bt,top,bottom,fib,inote,ent=h
   print('  exact 5M tap OB',ot,'body',round(bb,5),round(bt,5),'top/bot',round(top,5),round(bottom,5),'fib',fib,'inOTE',inote,'entry_from_candle',round(ent,5),'delta_entry',round(ent-entry_expected,5))
mt5.shutdown()
