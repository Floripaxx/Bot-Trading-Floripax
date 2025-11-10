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
        
        # 🚀 MEJORA 1: Configuración OPTIMIZADA para mejor rendimiento
        self.position_size = 0.15  # MODIFICADO: 15% para mejor gestión de riesgo
        self.max_positions = 2     # MODIFICADO: Reducido de 4 a 2 para menos riesgo
        self.momentum_threshold = 0.003  # AUMENTADO para señales más fuertes
        self.mean_reversion_threshold = 0.002
        self.volatility_multiplier = 1.8  # Aumentado para mejor filtrado
        self.min_profit_target = 0.006  # MODIFICADO: 0.6% target más realista
        self.max_loss_stop = 0.003     # MODIFICADO: 0.3% stop loss más ajustado
        
        # 🚀 MEJORA 2: Filtros de calidad de señal
        self.min_volume_threshold = 1.3  # Filtro de volumen
        self.rsi_upper_bound = 68       # RSI más conservador
        self.rsi_lower_bound = 32
        self.required_conditions = 3    # Mínimo condiciones para operar
        
        # 🚀 MEJORA 3: Gestión de riesgo mejorada
        self.max_daily_trades = 15      # Límite diario
        self.daily_trade_count = 0
        self.last_trade_time = None
        self.cooldown_after_loss = 30   # minutos después de pérdida
        
        self.trading_thread = None
        
    def save_state(self):
        """Guardar estado en archivo JSON"""
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
                'total_profit': self.total_profit,
                'daily_trade_count': self.daily_trade_count,
                'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None
            }
            
            # Convertir positions_history
            for pos in self.positions_history:
                pos_copy = pos.copy()
                pos_copy['timestamp'] = pos['timestamp'].isoformat() if isinstance(pos['timestamp'], datetime) else str(pos['timestamp'])
                state['positions_history'].append(pos_copy)
            
            # Convertir tick_data
            for tick in list(self.tick_data):
                tick_copy = tick.copy()
                tick_copy['timestamp'] = tick['timestamp'].isoformat() if isinstance(tick['timestamp'], datetime) else str(tick['timestamp'])
                state['tick_data'].append(tick_copy)
                    
            with open('bot_state.json', 'w') as f:
                json.dump(state, f, default=str, indent=2)
        except Exception as e:
            print(f"Error guardando estado: {e}")
    
    def load_state(self):
        """Cargar estado desde archivo JSON"""
        try:
            if os.path.exists('bot_state.json'):
                with open('bot_state.json', 'r') as f:
                    state = json.load(f)
                
                # Convertir deque de tick_data
                tick_data = deque(maxlen=50)
                for tick in state.get('tick_data', []):
                    tick_copy = tick.copy()
                    if 'timestamp' in tick:
                        try:
                            if 'T' in tick['timestamp']:
                                tick_copy['timestamp'] = datetime.fromisoformat(tick['timestamp'].replace('Z', '+00:00'))
                            else:
                                tick_copy['timestamp'] = datetime.now()
                        except:
                            tick_copy['timestamp'] = datetime.now()
                    tick_data.append(tick_copy)
                
                # Convertir timestamps en positions_history
                positions_history = []
                for pos in state.get('positions_history', []):
                    pos_copy = pos.copy()
                    if 'timestamp' in pos:
                        try:
                            if 'T' in pos['timestamp']:
                                pos_copy['timestamp'] = datetime.fromisoformat(pos['timestamp'].replace('Z', '+00:00'))
                            else:
                                pos_copy['timestamp'] = datetime.now()
                        except:
                            pos_copy['timestamp'] = datetime.now()
                    positions_history.append(pos_copy)
                
                # Convertir last_trade_time
                last_trade_time = None
                if state.get('last_trade_time'):
                    try:
                        last_trade_time = datetime.fromisoformat(state['last_trade_time'].replace('Z', '+00:00'))
                    except:
                        last_trade_time = None
                
                self.bot_data = {
                    'cash_balance': state.get('cash_balance', 250.0),
                    'position': state.get('position', 0),
                    'entry_price': state.get('entry_price', 0),
                    'positions_history': positions_history,
                    'open_positions': state.get('open_positions', 0),
                    'log_messages': state.get('log_messages', []),
                    'tick_data': tick_data,
                    'is_running': state.get('is_running', False),
                    'total_profit': state.get('total_profit', 0),
                    'daily_trade_count': state.get('daily_trade_count', 0),
                    'last_trade_time': last_trade_time
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
                    'total_profit': 0,
                    'daily_trade_count': 0,
                    'last_trade_time': None
                }
        except Exception as e:
            print(f"Error cargando estado: {e}")
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
                'total_profit': 0,
                'daily_trade_count': 0,
                'last_trade_time': None
            }
    
    # 🚀 MEJORA 4: Propiedades mejoradas con gestión diaria
    @property
    def daily_trade_count(self):
        return self.bot_data['daily_trade_count']
    
    @daily_trade_count.setter
    def daily_trade_count(self, value):
        self.bot_data['daily_trade_count'] = value
        self.save_state()
    
    @property
    def last_trade_time(self):
        return self.bot_data['last_trade_time']
    
    @last_trade_time.setter
    def last_trade_time(self, value):
        self.bot_data['last_trade_time'] = value
        self.save_state()
    
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

    def get_real_price_from_api(self) -> dict:
        """Obtener precio REAL de MEXC - MEJORADO"""
        try:
            url = f"https://api.mexc.com/api/v3/ticker/price"
            params = {'symbol': self.symbol}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    current_price = float(data['price'])
                    spread = current_price * 0.0001
                    
                    return {
                        'timestamp': datetime.now(),
                        'bid': current_price - spread,
                        'ask': current_price + spread,
                        'symbol': self.symbol,
                        'simulated': False,
                        'source': 'MEXC Real'
                    }
            
            return self.get_binance_price()
            
        except Exception as e:
            self.log_message(f"Error obteniendo precio real: {e}", "ERROR")
            return self.get_binance_price()

    def get_binance_price(self) -> dict:
        """Obtener precio de Binance como backup"""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': self.symbol}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    current_price = float(data['price'])
                    spread = current_price * 0.0001
                    
                    return {
                        'timestamp': datetime.now(),
                        'bid': current_price - spread,
                        'ask': current_price + spread,
                        'symbol': self.symbol,
                        'simulated': False,
                        'source': 'Binance Backup'
                    }
            
            return self.get_realistic_price()
            
        except Exception as e:
            self.log_message(f"Error obteniendo precio de Binance: {e}", "ERROR")
            return self.get_realistic_price()

    def get_realistic_price(self) -> dict:
        """Generar precio realista - ACTUALIZADO con precios reales"""
        base_prices = {
            'BTCUSDT': 105000,  # ACTUALIZADO: Precio más realista
            'ETHUSDT': 5800,
            'ADAUSDT': 0.62,
            'DOTUSDT': 8.5,
            'LINKUSDT': 18.0
        }
        
        base_price = base_prices.get(self.symbol, 50000)
        # Menor variación para precios más realistas
        variation = np.random.uniform(-0.008, 0.008)
        current_price = base_price * (1 + variation)
        spread = current_price * 0.0001
        
        return {
            'timestamp': datetime.now(),
            'bid': current_price - spread,
            'ask': current_price + spread,
            'symbol': self.symbol,
            'simulated': True,
            'source': 'Realistic Simulation'
        }

    def get_ticker_price(self) -> dict:
        """Obtener precio actual"""
        try:
            real_data = self.get_real_price_from_api()
            return real_data
        except Exception as e:
            self.log_message(f"Error crítico obteniendo precio: {e}", "ERROR")
            return self.get_realistic_price()

    def calculate_indicators(self) -> dict:
        """Calcular indicadores técnicos OPTIMIZADOS con mejores filtros"""
        if len(self.tick_data) < 10:
            return {}
        
        prices = [tick['bid'] for tick in self.tick_data]
        df = pd.DataFrame(prices, columns=['price'])
        
        # 🚀 MEJORA 5: Indicadores más robustos
        df['returns'] = df['price'].pct_change()
        df['momentum'] = df['returns'].rolling(5).mean()  # Ventana optimizada
        df['sma_10'] = df['price'].rolling(10).mean()
        df['sma_20'] = df['price'].rolling(20).mean()  # Nueva media
        df['price_deviation'] = (df['price'] - df['sma_10']) / df['sma_10']
        df['volatility'] = df['returns'].rolling(10).std() * self.volatility_multiplier
        
        # RSI mejorado
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD optimizado
        exp12 = df['price'].ewm(span=12, adjust=False).mean()
        exp26 = df['price'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands mejorados
        df['bb_middle'] = df['price'].rolling(20).mean()
        df['bb_std'] = df['price'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        # 🚀 MEJORA 6: Volumen simulado para filtros
        df['volume_sim'] = np.random.uniform(0.8, 1.5, len(df))
        
        latest = df.iloc[-1]
        
        return {
            'momentum': latest['momentum'],
            'price_deviation': latest['price_deviation'],
            'current_price': latest['price'],
            'sma_10': latest['sma_10'],
            'sma_20': latest['sma_20'],
            'rsi': latest['rsi'],
            'volatility': latest['volatility'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal'],
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'bb_middle': latest['bb_middle'],
            'volume_ratio': latest['volume_sim']
        }

    def can_trade(self) -> bool:
        """🚀 MEJORA 7: Verificar si se puede operar con nuevos filtros"""
        # Límite diario de trades
        if self.daily_trade_count >= self.max_daily_trades:
            return False
        
        # Cooldown después de pérdida
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds() / 60
            last_trades = [t for t in self.positions_history[-3:] if t['action'] == 'sell']
            if last_trades and any(t.get('profit_loss', 0) < 0 for t in last_trades):
                if time_since_last < self.cooldown_after_loss:
                    return False
        
        return True

    def trading_strategy(self, indicators: dict) -> str:
        """🚀 MEJORA 8: Estrategia OPTIMIZADA con mejores filtros"""
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
        volume_ratio = indicators['volume_ratio']
        sma_10 = indicators['sma_10']
        sma_20 = indicators['sma_20']
        
        # 🚀 MEJORA 9: Condiciones más estrictas y realistas
        buy_conditions = [
            momentum > self.momentum_threshold,      # Momentum fuerte
            self.rsi_lower_bound < rsi < self.rsi_upper_bound,  # RSI en zona óptima
            macd > macd_signal,                     # Tendencia alcista confirmada
            current_price > sma_10 > sma_20,        # Tendencia alcista en MMs
            volume_ratio > self.min_volume_threshold, # Volumen suficiente
            volatility < 0.015,                     # Volatilidad controlada
            current_price < bb_upper * 0.98         # No en resistencia extrema
        ]
        
        sell_conditions = [
            momentum < -self.momentum_threshold * 0.8, # Momentum bajista
            rsi > 65,                               # Acercándose a sobrecompra
            macd < macd_signal,                     # Tendencia bajista
            current_price < sma_10,                 # Rompiendo soporte
            self.position > 0,                      # Solo si tenemos posición
            current_price > bb_lower * 1.02         # No en soporte extremo
        ]
        
        # Verificar si podemos operar
        if not self.can_trade():
            return 'hold'
        
        # TOMA DE GANANCIAS Y STOP LOSS MEJORADO
        if self.position > 0:
            current_profit_pct = (current_price - self.entry_price) / self.entry_price
            current_loss_pct = (self.entry_price - current_price) / self.entry_price
            
            # Take profit con condiciones de mercado
            if current_profit_pct >= self.min_profit_target:
                profit_conditions = sum([
                    current_profit_pct > self.min_profit_target * 1.5,
                    rsi > 60,
                    momentum < 0
                ])
                if profit_conditions >= 1:
                    self.log_message(f"🎯 TOMANDO GANANCIAS: {current_profit_pct:.3%} (+)", "PROFIT")
                    return 'sell'
            
            # Stop loss inteligente
            if current_loss_pct >= self.max_loss_stop:
                self.log_message(f"🛑 STOP LOSS: {current_loss_pct:.3%} (-)", "LOSS")
                return 'sell'
        
        # SEÑALES PRINCIPALES CON MEJORES FILTROS
        buy_signal_strength = sum(buy_conditions)
        sell_signal_strength = sum(sell_conditions)
        
        if buy_signal_strength >= self.required_conditions:
            self.log_message(f"✅ SEÑAL COMPRA FUERTE: condiciones={buy_signal_strength}/7, RSI={rsi:.1f}", "SIGNAL")
            return 'buy'
        elif sell_signal_strength >= max(2, self.required_conditions - 1):
            self.log_message(f"🎯 SEÑAL VENTA: condiciones={sell_signal_strength}/6, RSI={rsi:.1f}", "SIGNAL")
            return 'sell'
        
        return 'hold'

    def execute_trade(self, action: str, price: float):
        """🚀 MEJORA 10: Ejecutar operación con gestión mejorada"""
        try:
            # Actualizar contadores diarios
            self.last_trade_time = datetime.now()
            if action in ['buy', 'sell']:
                self.daily_trade_count += 1
            
            investment_amount = 0
            quantity = 0
            quantity_to_sell = 0
            sale_amount = 0
            profit_loss = 0
            
            if action == 'buy':
                if self.open_positions < self.max_positions:
                    # Inversión con tamaño optimizado
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
                    
                    trade_info = f"✅ COMPRA: {quantity:.6f} {self.symbol} @ ${price:.2f} | Inversión: ${investment_amount:.2f} | Posiciones: {self.open_positions}/{self.max_positions}"
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
                
                profit_color = "💰" if profit_loss > 0 else "📉"
                trade_info = f"{profit_color} VENTA: {quantity_to_sell:.6f} {self.symbol} @ ${price:.2f} | P/L: ${profit_loss:.4f} | Profit Total: ${self.total_profit:.2f}"
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
                'profit_loss': profit_loss if action == 'sell' else 0,
                'daily_trade_count': self.daily_trade_count
            })
            
            self.save_state()
            
        except Exception as e:
            self.log_message(f"❌ Error ejecutando trade: {e}", "ERROR")

    def close_all_positions(self):
        """Cerrar todas las posiciones abiertas"""
        if self.position > 0:
            if self.tick_data:
                tick_data = list(self.tick_data)[-1]
            else:
                tick_data = self.get_ticker_price()
            price = tick_data['ask']
            self.execute_trade('sell', price)
            self.log_message("🛑 TODAS las posiciones cerradas", "INFO")

    def reset_account(self):
        """Reiniciar cuenta a estado inicial"""
        self.cash_balance = 250.0
        self.position = 0
        self.entry_price = 0
        self.positions_history.clear()
        self.open_positions = 0
        self.log_messages.clear()
        self.tick_data.clear()
        self.total_profit = 0
        self.daily_trade_count = 0
        self.last_trade_time = None
        self.log_message("🔄 Cuenta reiniciada a $250.00", "INFO")
        self.save_state()

    def trading_cycle(self):
        """Ciclo principal de trading MEJORADO"""
        self.log_message("🚀 Iniciando ciclo de trading OPTIMIZADO")
        
        while self.is_running:
            try:
                # Reset diario de contadores
                if self.last_trade_time and self.last_trade_time.date() != datetime.now().date():
                    self.daily_trade_count = 0
                    self.log_message("🔄 Contador diario reiniciado", "INFO")
                
                tick_data = self.get_ticker_price()
                if tick_data:
                    self.tick_data.append(tick_data)
                
                indicators = self.calculate_indicators()
                
                if indicators and len(self.tick_data) >= 10:
                    signal = self.trading_strategy(indicators)
                    if signal != 'hold':
                        price = tick_data['bid'] if signal == 'buy' else tick_data['ask']
                        self.execute_trade(signal, price)
                
                time.sleep(3)  # Intervalo ligeramente mayor para mejor análisis
                
            except Exception as e:
                self.log_message(f"❌ Error en ciclo de trading: {e}", "ERROR")
                time.sleep(5)

    def start_trading(self):
        """Iniciar bot de trading"""
        if not self.is_running:
            self.is_running = True
            self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
            self.trading_thread.start()
            self.log_message("🤖 Bot de trading iniciado - ESTRATEGIA OPTIMIZADA ACTIVADA")

    def stop_trading(self):
        """Detener bot de trading"""
        self.is_running = False
        self.log_message("🛑 Bot de trading detenido")

    def get_performance_stats(self):
        """Obtener estadísticas de performance MEJORADAS"""
        current_price = list(self.tick_data)[-1]['bid'] if self.tick_data else 0
        position_value = self.position * current_price
        total_equity = self.cash_balance + position_value
        total_profit = total_equity - 250.0
        
        stats = {
            'total_trades': len(self.positions_history),
            'win_rate': 0,
            'cash_balance': self.cash_balance,
            'position_value': position_value,
            'total_equity': total_equity,
            'open_positions': self.open_positions,
            'current_price': current_price,
            'total_profit': total_profit,
            'position_size': self.position,
            'realized_profit': self.total_profit,
            'daily_trades_left': max(0, self.max_daily_trades - self.daily_trade_count),
            'daily_trade_count': self.daily_trade_count
        }
        
        if not self.positions_history:
            return stats
        
        # Calcular win rate mejorado
        sell_trades = [t for t in self.positions_history if t['action'] == 'sell']
        
        if sell_trades:
            profitable_trades = len([t for t in sell_trades if t.get('profit_loss', 0) > 0])
            stats['win_rate'] = (profitable_trades / len(sell_trades)) * 100 if sell_trades else 0
        
        return stats

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
        st.header("🎮 Control del Bot")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Iniciar Bot", use_container_width=True):
                bot.start_trading()
                st.rerun()
        
        with col2:
            if st.button("⏹️ Detener Bot", use_container_width=True):
                bot.stop_trading()
                st.rerun()
        
        if st.button("🛑 Cerrar Todas las Posiciones", use_container_width=True):
            bot.close_all_positions()
            st.rerun()
            
        if st.button("🔄 Reiniciar Cuenta", use_container_width=True):
            bot.reset_account()
            st.rerun()
        
        st.markdown("---")
        st.header("📊 Configuración Actual")
        st.info(f"**Tamaño posición:** {bot.position_size*100}%")
        st.info(f"**Target ganancia:** {bot.min_profit_target*100}%")
        st.info(f"**Stop loss:** {bot.max_loss_stop*100}%")
        st.info(f"**Límite diario:** {bot.max_daily_trades} trades")
        st.info(f"**Posiciones máx:** {bot.max_positions}")
        
        if bot.is_running:
            st.success("✅ Bot Ejecutándose - ESTRATEGIA OPTIMIZADA")
        else:
            st.warning("🛑 Bot Detenido")
            
        if bot.tick_data:
            latest_tick = list(bot.tick_data)[-1]
            source = latest_tick.get('source', 'Unknown')
            st.info(f"**Fuente:** {source}")

    # Layout principal
    col1, col2, col3, col4 = st.columns(4)
    
    stats = bot.get_performance_stats()
    
    with col1:
        st.metric(
            label="💰 Cash Disponible",
            value=f"${stats['cash_balance']:.2f}",
            delta=f"${stats['realized_profit']:.2f}" if stats['realized_profit'] != 0 else None
        )
    
    with col2:
        st.metric(
            label="📈 Precio Actual",
            value=f"${stats['current_price']:.2f}"
        )
    
    with col3:
        st.metric(
            label="🎯 Tasa de Acierto",
            value=f"{stats['win_rate']:.1f}%"
        )
    
    with col4:
        st.metric(
            label="📊 Total Operaciones",
            value=f"{stats['total_trades']}"
        )
    
    # Segunda fila de métricas
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            label="🔢 Posiciones Abiertas",
            value=f"{stats['open_positions']}/{bot.max_positions}"
        )
    
    with col6:
        st.metric(
            label="📦 Tamaño Posición",
            value=f"{stats['position_size']:.6f}"
        )
    
    with col7:
        st.metric(
            label="💹 Equity Total",
            value=f"${stats['total_equity']:.2f}"
        )
    
    with col8:
        st.metric(
            label="🎪 Trades Hoy",
            value=f"{stats['daily_trade_count']}/{bot.max_daily_trades}"
        )
    
    st.markdown("---")
    
    # Gráficos y datos
    tab1, tab2, tab3 = st.tabs(["📈 Gráfico de Precios", "📋 Historial de Operaciones", "📝 Logs del Sistema"])
    
    with tab1:
        if bot.tick_data:
            prices = [tick['bid'] for tick in list(bot.tick_data)]
            timestamps = [tick['timestamp'] for tick in list(bot.tick_data)]
            
            # Asegurarse de que los timestamps sean datetime
            valid_timestamps = []
            valid_prices = []
            for ts, price in zip(timestamps, prices):
                if isinstance(ts, datetime):
                    valid_timestamps.append(ts)
                    valid_prices.append(price)
            
            if valid_timestamps:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=valid_timestamps, 
                    y=valid_prices,
                    mode='lines',
                    name=f'Precio {bot.symbol}',
                    line=dict(color='#00ff88', width=2)
                ))
                
                # Agregar más indicadores al gráfico
                if len(valid_prices) >= 20:
                    df = pd.DataFrame({'price': valid_prices})
                    df['sma_10'] = df['price'].rolling(10).mean()
                    df['sma_20'] = df['price'].rolling(20).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=valid_timestamps[9:],
                        y=df['sma_10'].dropna(),
                        mode='lines',
                        name='SMA 10',
                        line=dict(color='#ffaa00', width=1, dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=valid_timestamps[19:],
                        y=df['sma_20'].dropna(),
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='#ff4444', width=1, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"Precio de {bot.symbol} en Tiempo Real con Indicadores",
                    xaxis_title="Tiempo",
                    yaxis_title="Precio (USD)",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("⏳ Esperando datos válidos para graficar...")
        else:
            st.info("⏳ Esperando datos de mercado...")
    
    with tab2:
        if bot.positions_history:
            # Crear DataFrame seguro
            display_data = []
            for pos in bot.positions_history:
                pnl = pos.get('profit_loss', 0)
                pnl_color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
                pnl_text = f"{pnl_color} ${abs(pnl):.4f}"
