"""MT5 Client for connecting to MetaTrader 5"""

import os
from typing import Optional

class MT5Credentials:
    """MT5 connection credentials"""
    
    def __init__(self, login: str = "", password: str = "", server: str = ""):
        self.login = login
        self.password = password
        self.server = server
    
    @classmethod
    def from_env(cls):
        """Load credentials from environment variables"""
        login = (os.getenv("MT5_LOGIN", "") or "").strip()
        password = (os.getenv("MT5_PASSWORD", "") or "").strip()
        server = (os.getenv("MT5_SERVER", "") or "").strip()
        return cls(login, password, server)

class MT5Client:
    """MT5 Client wrapper"""
    
    def __init__(self, credentials: MT5Credentials):
        self.credentials = credentials
        self._mt5 = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize MT5 connection"""
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
            
            # Attempt to connect
            if not mt5.initialize():
                print(f"MT5 initialization failed")
                return False
            
            # Attempt login only when credentials are provided.
            # This allows using the already-open MT5 terminal session.
            if self.credentials.login:
                if not mt5.login(
                    login=int(self.credentials.login),
                    password=self.credentials.password,
                    server=self.credentials.server or None,
                ):
                    print(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
            
            self._initialized = True
            return True
        except ImportError:
            print("MetaTrader5 package not installed. Install with: pip install MetaTrader5")
            return False
        except Exception as e:
            print(f"MT5 initialization error: {e}")
            return False
    
    def ensure_symbol(self, symbol: str) -> Optional[str]:
        """Ensure symbol is available on MT5"""
        if not self._mt5 or not self._initialized:
            return None
        
        try:
            if self._mt5.symbol_select(symbol, True):
                return symbol
        except Exception:
            pass
        
        return None
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        if self._mt5 and self._initialized:
            try:
                self._mt5.shutdown()
                self._initialized = False
            except Exception:
                pass
