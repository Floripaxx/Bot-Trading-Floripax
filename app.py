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

# ✅ Page config must be first in Streamlit
st.set_page_config(
    page_title="⚡ Bot HFT MEXC Dual",
    page_icon="⚡",
    layout="wide"
)

# =======================
#  Persistencia del estado
# =======================
class PersistentStateManager:
    def __init__(self, file='bot_state.json'):
        self.file = file
        self.lock = threading.Lock()
    
    def save(self, data):
        with self.lock:
            try:
                tmp = f"{self.file}.tmp"
                with open(tmp, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                os.replace(tmp, self.file)
            except Exception as e:
                print("❌ Error guardando estado:", e)
    
    def load(self):
        if os.path.exists(self.file):
            with open(self.file, 'r') as f:
                return json.load(f)
        return {'cash_balance': 255.0, 'positions': {}, 'total_profit': 0, 'tick_data': {}}


# =======================
#  Bot principal
# =======================
class MexcFuturesTradingBot:
    def __init__(self, symbols=None):
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self.persistence = PersistentStateManager()
        self._state = self.persistence.load()

        # Asegurar estructura por símbolo
        for s in self.symbols:
            self._state.setdefault('positions', {}).setdefault(s, {'position': 0, 'side': '', 'entry_price': 0})
            self._state.setdefault('tick_data', {}).setdefault(s, [])
        self._tick_data = {s: deque(self._state['tick_data'][s], maxlen=100) for s in self.symbols}

        # Parámetros
        self.base_position_size = 0.12
        self.position_size = self.base_position_size
        self.leverage = 3
        self.is_running = False
        self.total_profit = self._state.get('total_profit', 0)

    def get_price(self, symbol):
        try:
            url = "https://api.mexc.com/api/v3/ticker/price"
            r = requests.get(url, params={"symbol": symbol}, timeout=5)
            data = r.json()
            p = float(data['price'])
            spread = p * 0.00005
            return {'bid': p - spread, 'ask': p + spread, 'timestamp': datetime.now(), 'symbol': symbol}
        except:
            # Simulación si no responde la API
            base = 103000 if "BTC" in symbol else 3500
            p = base * (1 + np.random.uniform(-0.01, 0.01))
            return {'bid': p, 'ask': p, 'timestamp': datetime.now(), 'symbol': symbol}

    def calc_indicators(self, ticks):
        if len(ticks) < 5:
            return None
        prices = [t['bid'] for t in ticks]
        rsi = 100 - (100 / (1 + (pd.Series(prices).diff().clip(lower=0).mean() /
                                 abs(pd.Series(prices).diff().clip(upper=0)).mean())))
        momentum = (prices[-1] - prices[-2]) / prices[-2]
        return {'rsi': rsi, 'momentum': momentum}

    def decide(self, ind):
        if not ind:
            return 'hold'
        if ind['rsi'] < 40 and ind['momentum'] < 0:
            return 'buy'
        if ind['rsi'] > 60 and ind['momentum'] > 0:
            return 'sell'
        return 'hold'

    def trade(self, symbol, action, price):
        pos = self._state['positions'][symbol]
        size = (self._state['cash_balance'] * self.position_size * self.leverage) / price
        if action == 'buy' and pos['position'] == 0:
            pos.update({'position': size, 'side': 'long', 'entry_price': price})
            self._state['cash_balance'] -= (size * price / self.leverage)
        elif action == 'sell' and pos['position'] == 0:
            pos.update({'position': size, 'side': 'short', 'entry_price': price})
            self._state['cash_balance'] -= (size * price / self.leverage)
        elif action == 'close' and pos['position'] > 0:
            pnl = (price - pos['entry_price']) * size * self.leverage if pos['side'] == 'long' else (pos['entry_price'] - price) * size * self.leverage
            self._state['cash_balance'] += (size * price / self.leverage) + pnl
            pos.update({'position': 0, 'side': '', 'entry_price': 0})
            self.total_profit += pnl
            # interés compuesto automático controlado
            self.position_size = min(self.base_position_size * (1 + self.total_profit / 500), 0.35)
        self.persistence.save(self._state)

    def cycle(self):
        while self.is_running:
            for s in self.symbols:
                tick = self.get_price(s)
                self._tick_data[s].append(tick)
                ind = self.calc_indicators(self._tick_data[s])
                signal = self.decide(ind)
                if signal in ['buy', 'sell']:
                    self.trade(s, signal, tick['ask'] if signal == 'buy' else tick['bid'])
            time.sleep(2)

    def start(self):
        if not self.is_running:
            self.is_running = True
            threading.Thread(target=self.cycle, daemon=True).start()

    def stop(self):
        self.is_running = False
        self.persistence.save(self._state)


# =======================
#  Interfaz Streamlit
# =======================
def main():
    st.title("⚡ Bot HFT Dual - BTCUSDT + ETHUSDT")
    st.caption("Opera ambos pares en simultáneo con interés compuesto automático.")

    if 'bot' not in st.session_state:
        st.session_state.bot = MexcFuturesTradingBot()
    bot = st.session_state.bot

    with st.sidebar:
        st.header("🎛 Control del Bot")
        if st.button("▶ Iniciar"):
            bot.start()
        if st.button("⏹ Detener"):
            bot.stop()
        st.metric("💰 Balance", f"${bot._state['cash_balance']:.2f}")
        st.metric("📈 Profit Total", f"${bot.total_profit:.2f}")
        st.info("Operando BTCUSDT y ETHUSDT en paralelo")

    col1, col2 = st.columns(2)
    for i, s in enumerate(bot.symbols):
        data = list(bot._tick_data[s])
        if len(data) > 3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[t['timestamp'] for t in data],
                y=[t['bid'] for t in data],
                mode='lines',
                name=s
            ))
            fig.update_layout(title=f"{s} Precio", template="plotly_dark", height=300)
            (col1 if i == 0 else col2).plotly_chart(fig, use_container_width=True)
        else:
            (col1 if i == 0 else col2).info(f"Esperando datos de {s}...")

    st.markdown("---")
    st.subheader("📊 Posiciones Actuales")
    df = pd.DataFrame([
        {'Símbolo': s, 'Lado': p['side'], 'Cantidad': p['position'], 'Entrada': p['entry_price']}
        for s, p in bot._state['positions'].items()
    ])
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
