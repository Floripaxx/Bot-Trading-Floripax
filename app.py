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
import pickle

# Configurar la página de Streamlit
st.set_page_config(
    page_title="🤖 Bot HFT MEXC",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MexcHighFrequencyTradingBot:
    def __init__(self, api_key: str, secret_key: str, symbol: str = 'BTCUSDT'):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.base_url = 'https://api.mexc.com'
        
        # Cargar estado desde archivo o inicializar
        self.load_state()
        
        # Configuración HFT OPTIMIZADA PARA MAYORES GANANCIAS
        self.position_size = 0.15  # AUMENTADO a 15% para mayores ganancias
        self.max_positions = 2     # REDUCIDO para mejor gestión
        self.momentum_threshold = 0.0025  # AUMENTADO para señales más fuertes
        self.mean_reversion_threshold = 0.002
        self.volatility_multiplier = 1.5
        self.min_profit_target = 0.005  # 0.5% de ganancia mínima (AUMENTADO)
        self.max_loss_stop = 0.003     # 0.3% de stop loss
        
        self.trading_thread = None
        
    def save_state(self):
        """Guardar estado en archivo JSON"""
        state = {
            'cash_balance': self.cash_balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'positions_history': self.positions_history,
            'open_positions': self.open_positions,
            'log_messages': self.log_messages,
            'tick_data': list(self.tick_data),
            'is_running': self.is_running,
            'total_profit': self.total_profit
        }
        try:
            with open('bot_state.json', 'w') as f:
                json.dump(state, f, default=str, indent=2)
        except Exception as e:
            self.log_message(f"Error guardando estado: {e}", "ERROR")
    
    def load_state(self):
        """Cargar estado desde archivo JSON"""
        try:
            if os.path.exists('bot_state.json'):
                with open('bot_state.json', 'r') as f:
                    state = json.load(f)
                
                # Convertir deque de tick_data
                tick_data = deque(maxlen=50)
                for tick in state.get('tick_data', []):
                    tick['timestamp'] = datetime.fromisoformat(tick['timestamp'])
                    tick_data.append(tick)
                
                # Convertir timestamps en positions_history
                positions_history = []
                for pos in state.get('positions_history', []):
                    pos['timestamp'] = datetime.fromisoformat(pos['timestamp'])
                    positions_history.append(pos)
                
                self.bot_data = {
                    'cash_balance': state.get('cash_balance', 250.0),
                    'position': state.get('position', 0),
                    'entry_price': state.get('entry_price', 0),
                    'positions_history': positions_history,
                    'open_positions': state.get('open_positions', 0),
                    'log_messages': state.get('log_messages', []),
                    'tick_data': tick_data,
                    'is_running': state.get('is_running', False),
                    'total_profit': state.get('total_profit', 0)
                }
            else:
                # Estado inicial
                self.bot_data = {
                    'cash_balance': 250.0,
                    'position': 0,
                    'entry_price': 0,
                    'positions_history': [],
                    'open_positions': 0,
                    'log_messages': [],
                    'tick_data': deque(maxlen=50),
                    'is_running': False,
                    'total_profit': 0
                }
        except Exception as e:
            self.log_message(f"Error cargando estado: {e}", "ERROR")
            # Estado inicial por defecto
            self.bot_data = {
                'cash_balance': 250.0,
                'position': 0,
                'entry_price': 0,
                'positions_history': [],
                'open_positions': 0,
                'log_messages': [],
                'tick_data': deque(maxlen=50),
                'is_running': False,
                'total_profit': 0
            }
    
    @property
    def cash_balance(self):
        return self.bot_data['cash_balance']
    
    @cash_balance.setter
    def cash_balance(self, value):
        self.bot_data['cash_balance'] = value
        self.save_state()
        
    @property
    def position(self):
        return self.bot_data['position']
    
    @position.setter
    def position(self, value):
        self.bot_data['position'] = value
        self.save_state()
        
    @property
    def entry_price(self):
        return self.bot_data['entry_price']
    
    @entry_price.setter
    def entry_price(self, value):
        self.bot_data['entry_price'] = value
        self.save_state()
        
    @property
    def positions_history(self):
        return self.bot_data['positions_history']
    
    @property
    def open_positions(self):
        return self.bot_data['open_positions']
    
    @open_positions.setter
    def open_positions(self, value):
        self.bot_data['open_positions'] = value
        self.save_state()
        
    @property
    def log_messages(self):
        return self.bot_data['log_messages']
    
    @property
    def tick_data(self):
        return self.bot_data['tick_data']
    
    @property
    def is_running(self):
        return self.bot_data['is_running']
    
    @is_running.setter
    def is_running(self, value):
        self.bot_data['is_running'] = value
        self.save_state()
    
    @property
    def total_profit(self):
        return self.bot_data['total_profit']
    
    @total_profit.setter
    def total_profit(self, value):
        self.bot_data['total_profit'] = value
        self.save_state()

    def log_message(self, message: str, level: str = "INFO"):
        """Agregar mensaje al log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_messages.append(log_entry)
        if len(self.log_messages) > 50:
            self.log_messages.pop(0)
        self.save_state()

    # ... (los métodos get_real_price_from_api, get_binance_price, get_realistic_price se mantienen igual)

    def calculate_indicators(self) -> dict:
        """Calcular indicadores técnicos OPTIMIZADOS"""
        if len(self.tick_data) < 10:
            return {}
        
        prices = [tick['bid'] for tick in self.tick_data]
        df = pd.DataFrame(prices, columns=['price'])
        
        # Indicadores más agresivos
        df['returns'] = df['price'].pct_change()
        df['momentum'] = df['returns'].rolling(3).mean()  # Ventana más corta
        df['sma_5'] = df['price'].rolling(5).mean()
        df['sma_10'] = df['price'].rolling(10).mean()
        df['price_deviation'] = (df['price'] - df['sma_5']) / df['sma_5']
        df['volatility'] = df['returns'].rolling(8).std() * self.volatility_multiplier
        
        # RSI más rápido
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD más agresivo
        exp12 = df['price'].ewm(span=5, adjust=False).mean()
        exp26 = df['price'].ewm(span=8, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=3, adjust=False).mean()
        
        # Bollinger Bands para mejores entradas
        df['bb_middle'] = df['price'].rolling(10).mean()
        df['bb_std'] = df['price'].rolling(10).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 1.5)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 1.5)
        
        latest = df.iloc[-1]
        
        return {
            'momentum': latest['momentum'],
            'price_deviation': latest['price_deviation'],
            'current_price': latest['price'],
            'sma_5': latest['sma_5'],
            'sma_10': latest['sma_10'],
            'rsi': latest['rsi'],
            'volatility': latest['volatility'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal'],
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'bb_middle': latest['bb_middle']
        }

    def trading_strategy(self, indicators: dict) -> str:
        """Estrategia OPTIMIZADA para mayores ganancias"""
        if not indicators:
            return 'hold'
        
        momentum = indicators['momentum']
        deviation = indicators['price_deviation']
        rsi = indicators['rsi']
        volatility = indicators['volatility']
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        current_price = indicators['current_price']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        
        # ESTRATEGIA MÁS AGRESIVA Y SELECTIVA
        buy_conditions = [
            momentum > self.momentum_threshold,      # Momentum fuerte
            rsi < 45,                               # No sobrecomprado
            macd > macd_signal,                     # Tendencia alcista
            current_price < bb_lower,               # Precio en zona de soporte
            volatility < 0.02                       # Mercado estable
        ]
        
        sell_conditions = [
            momentum < -self.momentum_threshold,     # Momentum bajista
            rsi > 70,                               # Sobrecomprado
            macd < macd_signal,                     # Tendencia bajista  
            current_price > bb_upper,               # Precio en zona de resistencia
            self.position > 0                       # Solo vender si tenemos posición
        ]
        
        # TOMA DE GANANCIAS MÁS AGRESIVA
        if self.position > 0:
            current_profit_pct = (current_price - self.entry_price) / self.entry_price
            current_loss_pct = (self.entry_price - current_price) / self.entry_price
            
            # Tomar ganancias rápido
            if current_profit_pct >= self.min_profit_target:
                self.log_message(f"🎯 TOMANDO GANANCIAS: {current_profit_pct:.3%} (+)", "PROFIT")
                return 'sell'
            
            # Stop loss protector
            if current_loss_pct >= self.max_loss_stop:
                self.log_message(f"🛑 STOP LOSS: {current_loss_pct:.3%} (-)", "LOSS")
                return 'sell'
        
        # SEÑALES PRINCIPALES
        if sum(buy_conditions) >= 4:  # Necesita 4 de 5 condiciones
            self.log_message(f"🔥 SEÑAL COMPRA FUERTE: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'buy'
        elif sum(sell_conditions) >= 3:  # Necesita 3 de 5 condiciones
            self.log_message(f"🔥 SEÑAL VENTA: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'sell'
        
        return 'hold'

    def execute_trade(self, action: str, price: float):
        """Ejecutar operación - OPTIMIZADA PARA MAYORES GANANCIAS"""
        try:
            if action == 'buy':
                if self.open_positions < self.max_positions:
                    # Inversión más grande (15% del balance)
                    investment_amount = self.cash_balance * self.position_size
                    quantity = investment_amount / price
                    
                    if investment_amount > self.cash_balance:
                        self.log_message("❌ Fondos insuficientes", "ERROR")
                        return
                    
                    # Actualizar balances
                    self.cash_balance -= investment_amount
                    self.position += quantity
                    self.entry_price = price
                    self.open_positions += 1
                    
                    trade_info = f"🟢 COMPRA: {quantity:.6f} {self.symbol} @ ${price:.2f} | Inversión: ${investment_amount:.2f} | Cash: ${self.cash_balance:.2f}"
                    self.log_message(trade_info, "TRADE")
                    
            elif action == 'sell' and self.position > 0:
                # Vender toda la posición
                quantity_to_sell = self.position
                sale_amount = quantity_to_sell * price
                profit_loss = sale_amount - (self.position * self.entry_price)
                
                # Actualizar balances
                self.cash_balance += sale_amount
                self.position = 0
                self.open_positions = 0
                self.total_profit += profit_loss
                
                profit_color = "🟢" if profit_loss > 0 else "🔴"
                trade_info = f"{profit_color} VENTA: {quantity_to_sell:.6f} {self.symbol} @ ${price:.2f} | Monto: ${sale_amount:.2f} | P/L: ${profit_loss:.4f} | Profit Total: ${self.total_profit:.2f}"
                self.log_message(trade_info, "TRADE")
            
            # Registrar posición
            current_equity = self.cash_balance + (self.position * price)
            self.positions_history.append({
                'timestamp': datetime.now(),
                'action': action,
                'price': price,
                'quantity': quantity_to_sell if action == 'sell' else quantity,
                'investment': investment_amount if action == 'buy' else sale_amount,
                'cash_balance': self.cash_balance,
                'position_value': self.position * price,
                'total_equity': current_equity,
                'open_positions': self.open_positions,
                'profit_loss': profit_loss if action == 'sell' else 0
            })
            
            self.save_state()
            
        except Exception as e:
            self.log_message(f"Error ejecutando trade: {e}", "ERROR")

    # ... (los demás métodos se mantienen similares)

def main():
    st.title("🤖 Bot HFT MEXC - VERSIÓN OPTIMIZADA 🚀")
    st.markdown("---")
    
    # Inicializar el bot
    if 'bot' not in st.session_state:
        st.session_state.bot = MexcHighFrequencyTradingBot("", "", "BTCUSDT")
    
    bot = st.session_state.bot
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        api_key = st.text_input("API Key MEXC", type="password")
        secret_key = st.text_input("Secret Key MEXC", type="password")
        symbol = st.selectbox("Símbolo", ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"])
        
        bot.api_key = api_key
        bot.secret_key = secret_key
        bot.symbol = symbol
        
        st.markdown("---")
        st.header("🎯 Control del Bot")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Iniciar Bot", use_container_width=True):
                bot.start_trading()
                st.rerun()
        
        with col2:
            if st.button("⏹️ Detener Bot", use_container_width=True):
                bot.stop_trading()
                st.rerun()
        
        if st.button("🧹 Cerrar Todas las Posiciones", use_container_width=True):
            bot.close_all_positions()
            st.rerun()
        
        st.markdown("---")
        st.header("💰 Estadísticas Clave")
        st.info(f"**Tamaño posición:** {bot.position_size*100}%")
        st.info(f"**Target ganancia:** {bot.min_profit_target*100}%")
        st.info(f"**Stop loss:** {bot.max_loss_stop*100}%")
        
        if bot.is_running:
            st.success("✅ Bot Ejecutándose - ESTRATEGIA AGRESIVA")
        else:
            st.warning("⏸️ Bot Detenido")

    # ... (el resto del main se mantiene similar)

if __name__ == "__main__":
    main()
