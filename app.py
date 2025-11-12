import streamlit as st
import time
import threading
from collections import deque
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="⚡ Bot HFT Futuros MEXC - Dual BTC & ETH",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PersistentStateManager:
    def __init__(self, state_file='bot_persistent_state.json'):
        self.state_file = state_file
        self.lock = threading.Lock()
    
    def save_state(self, state_data):
        with self.lock:
            try:
                temp_file = f"{self.state_file}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(state_data, f, default=str, indent=2)
                os.replace(temp_file, self.state_file)
            except Exception as e:
                print(f"Error guardando estado: {e}")
    
    def load_state(self):
        with self.lock:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            return {'cash_balance': 255.0, 'positions': {}, 'positions_history': [], 'tick_data': {}, 'total_profit': 0}

class MexcFuturesTradingBot:
    def __init__(self, api_key="", secret_key="", symbols=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.persistence = PersistentStateManager()
        self._state = self.persistence.load_state()
        for s in self.symbols:
            self._state.setdefault('positions', {}).setdefault(s, {'position': 0, 'side': '', 'entry_price': 0})
            self._state.setdefault('tick_data', {}).setdefault(s, [])
        self._tick_data = {s: deque(self._state['tick_data'][s], maxlen=100) for s in self.symbols}
        self.base_position_size = 0.12
        self.position_size = self.base_position_size
        self.max_position_size = 0.35
        self.leverage = 3
        self.is_running = False
        self.total_profit = self._state.get('total_profit', 0)
        self.initial_capital = 255.0
        self.max_positions = 2

    def get_futures_price(self, symbol):
        try:
            url = "https://api.mexc.com/api/v3/ticker/price"
            r = requests.get(url, params={"symbol": symbol}, timeout=5)
            data = r.json()
            price = float(data['price'])
            spread = price * 0.00005
            return {'bid': price - spread, 'ask': price + spread, 'timestamp': datetime.now(), 'symbol': symbol}
        except:
            base = 3500 if symbol == "ETHUSDT" else 103000
            price = base * (1 + np.random.uniform(-0.01, 0.01))
            return {'bid': price, 'ask': price, 'timestamp': datetime.now(), 'symbol': symbol}

    def calculate_indicators(self, ticks):
        if len(ticks) < 5:
            return None
        prices = [t['bid'] for t in ticks]
        rsi = 100 - (100 / (1 + (pd.Series(prices).diff().clip(lower=0).mean() /
                                 abs(pd.Series(prices).diff().clip(upper=0)).mean())))
        return {'rsi': rsi, 'momentum': (prices[-1] - prices[-2]) / prices[-2]}

    def strategy(self, indicators):
        if not indicators: return 'hold'
        if indicators['rsi'] < 40 and indicators['momentum'] < 0:
            return 'buy'
        if indicators['rsi'] > 60 and indicators['momentum'] > 0:
            return 'sell'
        return 'hold'

    def execute_trade(self, symbol, action, price):
        pos = self._state['positions'][symbol]
        qty = (self._state['cash_balance'] * self.position_size * self.leverage) / price
        if action == 'buy' and pos['position'] == 0:
            pos.update({'position': qty, 'side': 'long', 'entry_price': price})
            self._state['cash_balance'] -= (qty * price / self.leverage)
        elif action == 'sell' and pos['position'] == 0:
            pos.update({'position': qty, 'side': 'short', 'entry_price': price})
            self._state['cash_balance'] -= (qty * price / self.leverage)
        elif action == 'close' and pos['position'] > 0:
            pnl = (price - pos['entry_price']) * qty * self.leverage if pos['side'] == 'long' else (pos['entry_price'] - price) * qty * self.leverage
            self._state['cash_balance'] += (qty * price / self.leverage) + pnl
            pos.update({'position': 0, 'side': '', 'entry_price': 0})
            self.total_profit += pnl
        s
