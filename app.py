import streamlit as st
import time
import threading
from collections import deque
from datetime import datetime
import pandas as pd
import numpy as np
import hmac
import hashlib
import requests
import json
import plotly.graph_objects as go
import os

# Configurar la página de Streamlit
st.set_page_config(
    page_title="⚡ Bot HFT MEXC",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CLASE ORIGINAL ----------------
class MexcHighFrequencyTradingBot:
    def __init__(self, api_key: str, secret_key: str, symbol: str = 'BTCUSDT'):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.base_url = 'https://api.mexc.com'
        self.load_state()

        # Configuración de alta frecuencia ajustada
        self.position_size = 0.15
        self.max_positions = 1
        self.momentum_threshold = 0.001
        self.mean_reversion_threshold = 0.0015
        self.volatility_multiplier = 1.8
        self.min_profit_target = 0.003
        self.max_loss_stop = 0.002

        self.trading_thread = None

    # ------------------- ESTADO -------------------
    def save_state(self):
        try:
            state = {
                'cash_balance': self.cash_balance,
                'position': self.position,
                'entry_price': self.entry_price,
                'positions_history': [],
                'open_positions': self.open_positions,
                'log_messages': self.log_messages,
                'tick_data': [],
                'is_running': self.is_running,
                'total_profit': self.total_profit
            }
            for pos in self.positions_history:
                pos_copy = pos.copy()
                pos_copy['timestamp'] = pos['timestamp'].isoformat() if isinstance(pos['timestamp'], datetime) else str(pos['timestamp'])
                state['positions_history'].append(pos_copy)
            with open(f'bot_state_{self.symbol}.json', 'w') as f:
                json.dump(state, f, default=str, indent=2)
        except Exception as e:
            print(f"Error guardando estado: {e}")

    def load_state(self):
        try:
            if os.path.exists(f'bot_state_{self.symbol}.json'):
                with open(f'bot_state_{self.symbol}.json', 'r') as f:
                    state = json.load(f)
                tick_data = deque(maxlen=50)
                self.bot_data = {
                    'cash_balance': state.get('cash_balance', 255.0),
                    'position': state.get('position', 0),
                    'entry_price': state.get('entry_price', 0),
                    'positions_history': state.get('positions_history', []),
                    'open_positions': state.get('open_positions', 0),
                    'log_messages': state.get('log_messages', []),
                    'tick_data': tick_data,
                    'is_running': state.get('is_running', False),
                    'total_profit': state.get('total_profit', 0)
                }
            else:
                self.bot_data = {
                    'cash_balance': 255.0,
                    'position': 0,
                    'entry_price': 0,
                    'positions_history': [],
                    'open_positions': 0,
                    'log_messages': [],
                    'tick_data': deque(maxlen=50),
                    'is_running': False,
                    'total_profit': 0
                }
        except:
            self.bot_data = {
                'cash_balance': 255.0,
                'position': 0,
                'entry_price': 0,
                'positions_history': [],
                'open_positions': 0,
                'log_messages': [],
                'tick_data': deque(maxlen=50),
                'is_running': False,
                'total_profit': 0
            }

    # Propiedades
    @property
    def cash_balance(self): return self.bot_data['cash_balance']
    @cash_balance.setter
    def cash_balance(self, v): self.bot_data['cash_balance'] = v; self.save_state()
    @property
    def position(self): return self.bot_data['position']
    @position.setter
    def position(self, v): self.bot_data['position'] = v; self.save_state()
    @property
    def entry_price(self): return self.bot_data['entry_price']
    @entry_price.setter
    def entry_price(self, v): self.bot_data['entry_price'] = v; self.save_state()
    @property
    def positions_history(self): return self.bot_data['positions_history']
    @property
    def open_positions(self): return self.bot_data['open_positions']
    @open_positions.setter
    def open_positions(self, v): self.bot_data['open_positions'] = v; self.save_state()
    @property
    def log_messages(self): return self.bot_data['log_messages']
    @property
    def tick_data(self): return self.bot_data['tick_data']
    @property
    def is_running(self): return self.bot_data['is_running']
    @is_running.setter
    def is_running(self, v): self.bot_data['is_running'] = v; self.save_state()
    @property
    def total_profit(self): return self.bot_data['total_profit']
    @total_profit.setter
    def total_profit(self, v): self.bot_data['total_profit'] = v; self.save_state()

    def log_message(self, msg, lvl="INFO"):
        t = datetime.now().strftime("%H:%M:%S")
        entry = f"[{t}] {lvl}: {msg}"
        self.log_messages.append(entry)
        if len(self.log_messages) > 50: self.log_messages.pop(0)
        self.save_state()

    # ------------------- DATOS Y ESTRATEGIA -------------------
    def get_real_price_from_api(self):
        try:
            r = requests.get("https://api.mexc.com/api/v3/ticker/price", params={'symbol': self.symbol}, timeout=5)
            data = r.json()
            if 'price' in data:
                p = float(data['price'])
                s = p * 0.0001
                return {'timestamp': datetime.now(), 'bid': p - s, 'ask': p + s, 'symbol': self.symbol}
        except:
            return {'timestamp': datetime.now(), 'bid': 0, 'ask': 0, 'symbol': self.symbol}

    def calculate_indicators(self):
        if len(self.tick_data) < 10: return {}
        prices = [t['bid'] for t in self.tick_data]
        df = pd.DataFrame(prices, columns=['price'])
        df['returns'] = df['price'].pct_change()
        df['momentum'] = df['returns'].rolling(3).mean()
        df['sma_5'] = df['price'].rolling(5).mean()
        df['sma_10'] = df['price'].rolling(10).mean()
        df['price_deviation'] = (df['price'] - df['sma_5']) / df['sma_5']
        df['volatility'] = df['returns'].rolling(8).std() * self.volatility_multiplier
        latest = df.iloc[-1]
        return {'momentum': latest['momentum'], 'price_deviation': latest['price_deviation'], 'current_price': latest['price']}

    def trading_strategy(self, ind):
        if not ind: return 'hold'
        if ind['momentum'] > self.momentum_threshold: return 'buy'
        elif ind['momentum'] < -self.momentum_threshold: return 'sell'
        return 'hold'

    def execute_trade(self, action, price):
        if action == 'buy' and self.open_positions < self.max_positions:
            # interés compuesto dinámico
            self.position_size = min(0.15 + (self.total_profit / 1000), 0.25)
            inv = self.cash_balance * self.position_size
            qty = inv / price
            if inv > self.cash_balance: return
            self.cash_balance -= inv
            self.position += qty
            self.entry_price = price
            self.open_positions += 1
            self.log_message(f"BUY {qty:.6f} {self.symbol} @ ${price:.2f}", "TRADE")
        elif action == 'sell' and self.position > 0:
            sale = self.position * price
            profit = sale - (self.position * self.entry_price)
            self.cash_balance += sale
            self.position = 0
            self.open_positions = 0
            self.total_profit += profit
            self.log_message(f"SELL {self.symbol} P/L: {profit:.2f}", "TRADE")
        self.save_state()

    def trading_cycle(self):
        self.log_message(f"🚀 Iniciando HFT para {self.symbol}")
        while self.is_running:
            try:
                tick = self.get_real_price_from_api()
                if tick: self.tick_data.append(tick)
                ind = self.calculate_indicators()
                sig = self.trading_strategy(ind)
                if sig != 'hold':
                    price = tick['bid'] if sig == 'buy' else tick['ask']
                    self.execute_trade(sig, price)
                time.sleep(0.5)  # Alta frecuencia (cada medio segundo)
            except Exception as e:
                self.log_message(f"Error ciclo: {e}", "ERROR")
                time.sleep(1)

    def start_trading(self):
        if not self.is_running:
            self.is_running = True
            self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
            self.trading_thread.start()

    def stop_trading(self):
        self.is_running = False
        self.log_message("Bot detenido", "INFO")


# ---------------- INTERFAZ ----------------
def main():
    st.title("⚡ Bot HFT MEXC - BTC + ETH simultáneo")
    st.markdown("---")

    if 'btc_bot' not in st.session_state:
        st.session_state.btc_bot = MexcHighFrequencyTradingBot("", "", "BTCUSDT")
    if 'eth_bot' not in st.session_state:
        st.session_state.eth_bot = MexcHighFrequencyTradingBot("", "", "ETHUSDT")

    btc_bot = st.session_state.btc_bot
    eth_bot = st.session_state.eth_bot

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("BTCUSDT")
        if st.button("▶️ Iniciar BTC", use_container_width=True):
            btc_bot.start_trading()
        if st.button("⏹️ Detener BTC", use_container_width=True):
            btc_bot.stop_trading()
        st.info(f"Saldo: ${btc_bot.cash_balance:.2f} | Profit: ${btc_bot.total_profit:.2f}")

    with col2:
        st.subheader("ETHUSDT")
        if st.button("▶️ Iniciar ETH", use_container_width=True):
            eth_bot.start_trading()
        if st.button("⏹️ Detener ETH", use_container_width=True):
            eth_bot.stop_trading()
        st.info(f"Saldo: ${eth_bot.cash_balance:.2f} | Profit: ${eth_bot.total_profit:.2f}")

    st.markdown("---")
    st.success("Ambos bots pueden correr en simultáneo con interés compuesto y frecuencia 0.5s.")


if __name__ == "__main__":
    main()
