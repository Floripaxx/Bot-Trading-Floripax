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
    page_title="🤖 Bot Futures MEXC 3x",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MexcFuturesTradingBot:
    def __init__(self, api_key: str, secret_key: str, symbol: str = 'BTCUSDT'):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.base_url = 'https://api.mexc.com'
        
        # Cargar estado desde archivo o inicializar
        self.load_state()
        
        # CONFIGURACIÓN FUTUROS 3x - MÍNIMOS CAMBIOS
        self.leverage = 3                       # NUEVO: Apalancamiento 3x
        self.risk_per_trade = 0.02              # NUEVO: 2% riesgo por operación
        self.position_size = 0.06               # MODIFICADO: 6% (2% * 3x)
        self.max_positions = 1                  # MANTENIDO: 1 posición máxima
        self.momentum_threshold = 0.001         # MANTENIDO: 0.1% para señales sensibles
        self.mean_reversion_threshold = 0.0015  # MANTENIDO: Más sensible
        self.volatility_multiplier = 1.8        # MANTENIDO: Más tolerante
        self.min_profit_target = 0.003          # MANTENIDO: 0.3% ganancia (0.9% con 3x)
        self.max_loss_stop = 0.002              # MANTENIDO: 0.2% stop (0.6% con 3x)
        
        self.trading_thread = None
        self.position_side = None  # NUEVO: 'long' o 'short'
        
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
                'position_side': self.position_side  # NUEVO: Guardar lado de posición
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
                    'position_side': state.get('position_side', None)  # NUEVO: Cargar lado de posición
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
                    'position_side': None  # NUEVO: Inicializar lado de posición
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
                'position_side': None  # NUEVO: Inicializar lado de posición
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
    
    @property
    def position_side(self):
        return self.bot_data['position_side']
    
    @position_side.setter
    def position_side(self, value):
        self.bot_data['position_side'] = value
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
        """Obtener precio REAL de MEXC Futures"""
        try:
            # CORRECCIÓN: Endpoint correcto para futures de MEXC
            url = "https://contract.mexc.com/api/v1/contract/detail"
            params = {'symbol': self.symbol}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # CORRECCIÓN: Estructura correcta de la respuesta MEXC Futures
                if 'data' in data and data['data']:
                    current_price = float(data['data']['lastPrice'])
                    spread = current_price * 0.0001
                    
                    return {
                        'timestamp': datetime.now(),
                        'bid': current_price - spread,
                        'ask': current_price + spread,
                        'symbol': self.symbol,
                        'simulated': False,
                        'source': 'MEXC Futures Real'
                    }
            
            # Fallback a precio de spot si futures falla
            return self.get_spot_price_fallback()
            
        except Exception as e:
            self.log_message(f"Error obteniendo precio futures real: {e}", "ERROR")
            return self.get_spot_price_fallback()

    def get_spot_price_fallback(self) -> dict:
        """Obtener precio de spot MEXC como fallback"""
        try:
            url = "https://api.mexc.com/api/v3/ticker/price"
            params = {'symbol': self.symbol.replace('USDT', '_USDT')}  # Formato para spot
            
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
                        'source': 'MEXC Spot Fallback'
                    }
            
            return self.get_binance_price()
            
        except Exception as e:
            self.log_message(f"Error obteniendo precio spot: {e}", "ERROR")
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
        """Generar precio realista"""
        base_prices = {
            'BTCUSDT': 105000,
            'ETHUSDT': 3500,
            'ADAUSDT': 0.45,
            'DOTUSDT': 7.5,
            'LINKUSDT': 15.0
        }
        
        base_price = base_prices.get(self.symbol, 50000)
        variation = np.random.uniform(-0.02, 0.02)
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
        """Estrategia FUTURES con posiciones largas y cortas"""
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
        
        if np.isnan(rsi):
            return 'hold'
        
        # SEÑALES LARGAS (COMPRA)
        long_conditions = [
            momentum > self.momentum_threshold,
            rsi < 60,
            macd > macd_signal,
            current_price < bb_lower,
            volatility < 0.02
        ]
        
        # SEÑALES CORTAS (VENTA)
        short_conditions = [
            momentum < -self.momentum_threshold,
            rsi > 70,
            macd < macd_signal,
            current_price > bb_upper,
            volatility < 0.02
        ]
        
        # TOMA DE GANANCIAS Y STOP LOSS CON APALANCAMIENTO
        if self.position > 0:
            if self.position_side == 'long':
                current_profit_pct = (current_price - self.entry_price) / self.entry_price
                current_loss_pct = (self.entry_price - current_price) / self.entry_price
            else:  # short
                current_profit_pct = (self.entry_price - current_price) / self.entry_price
                current_loss_pct = (current_price - self.entry_price) / self.entry_price
            
            # Aplicar apalancamiento 3x a ganancias/pérdidas
            leveraged_profit = current_profit_pct * self.leverage
            leveraged_loss = current_loss_pct * self.leverage
            
            if leveraged_profit >= self.min_profit_target:
                self.log_message(f"🎯 TOMANDO GANANCIAS {self.leverage}x: {leveraged_profit:.3%} (+)", "PROFIT")
                return 'close'
            
            if leveraged_loss >= self.max_loss_stop:
                self.log_message(f"🛑 STOP LOSS {self.leverage}x: {leveraged_loss:.3%} (-)", "LOSS")
                return 'close'
        
        # SEÑALES PRINCIPALES
        if sum(long_conditions) >= 4 and not self.position:
            self.log_message(f"✅ SEÑAL LONG: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            self.position_side = 'long'
            return 'buy'
        elif sum(short_conditions) >= 4 and not self.position:
            self.log_message(f"🔻 SEÑAL SHORT: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            self.position_side = 'short'
            return 'sell'
        
        return 'hold'

    def execute_trade(self, action: str, price: float):
        """Ejecutar operación FUTURES con apalancamiento"""
        try:
            investment_amount = 0
            quantity = 0
            quantity_to_sell = 0
            sale_amount = 0
            profit_loss = 0
            
            if action == 'buy' and not self.position:
                # ENTRADA LONG
                investment_amount = self.cash_balance * self.position_size
                quantity = (investment_amount * self.leverage) / price
                
                if investment_amount > self.cash_balance:
                    self.log_message("❌ Fondos insuficientes", "ERROR")
                    return
                
                self.cash_balance -= investment_amount
                self.position += quantity
                self.entry_price = price
                self.open_positions += 1
                self.position_side = 'long'
                
                trade_info = f"✅ LONG {self.leverage}x: {quantity:.6f} {self.symbol} @ ${price:.2f} | Inversión: ${investment_amount:.2f} | Exposición: ${investment_amount * self.leverage:.2f}"
                self.log_message(trade_info, "TRADE")
                
            elif action == 'sell' and not self.position:
                # ENTRADA SHORT
                investment_amount = self.cash_balance * self.position_size
                quantity = (investment_amount * self.leverage) / price
                
                if investment_amount > self.cash_balance:
                    self.log_message("❌ Fondos insuficientes", "ERROR")
                    return
                
                self.cash_balance -= investment_amount
                self.position += quantity
                self.entry_price = price
                self.open_positions += 1
                self.position_side = 'short'
                
                trade_info = f"🔻 SHORT {self.leverage}x: {quantity:.6f} {self.symbol} @ ${price:.2f} | Inversión: ${investment_amount:.2f} | Exposición: ${investment_amount * self.leverage:.2f}"
                self.log_message(trade_info, "TRADE")
                
            elif action == 'close' and self.position > 0:
                # CERRAR POSICIÓN
                quantity_to_sell = self.position
                
                if self.position_side == 'long':
                    sale_amount = quantity_to_sell * price
                    profit_loss = (sale_amount - (self.position * self.entry_price)) / self.leverage
                else:  # short
                    sale_amount = quantity_to_sell * self.entry_price
                    profit_loss = ((self.position * self.entry_price) - (quantity_to_sell * price)) / self.leverage
                
                # Actualizar balances
                self.cash_balance += (investment_amount + profit_loss) if action == 'close' else sale_amount
                self.position = 0
                self.open_positions = 0
                self.total_profit += profit_loss
                
                profit_color = "💰" if profit_loss > 0 else "📉"
                side_str = "LONG" if self.position_side == 'long' else "SHORT"
                trade_info = f"{profit_color} CLOSE {side_str}: P/L: ${profit_loss:.4f} | Profit Total: ${self.total_profit:.2f}"
                self.log_message(trade_info, "TRADE")
                
                self.position_side = None
            
            # Registrar posición
            current_equity = self.cash_balance + (self.position * price / self.leverage)
            self.positions_history.append({
                'timestamp': datetime.now(),
                'action': action,
                'price': price,
                'quantity': quantity_to_sell if action == 'close' else quantity,
                'investment': investment_amount,
                'cash_balance': self.cash_balance,
                'position_value': self.position * price / self.leverage,
                'total_equity': current_equity,
                'open_positions': self.open_positions,
                'profit_loss': profit_loss if action == 'close' else 0,
                'position_side': self.position_side,
                'leverage': self.leverage
            })
            
            self.save_state()
            
        except Exception as e:
            self.log_message(f"❌ Error ejecutando trade futures: {e}", "ERROR")

    def close_all_positions(self):
        """Cerrar todas las posiciones abiertas"""
        if self.position > 0:
            if self.tick_data:
                tick_data = list(self.tick_data)[-1]
            else:
                tick_data = self.get_ticker_price()
            price = tick_data['bid'] if self.position_side == 'long' else tick_data['ask']
            self.execute_trade('close', price)
            self.log_message("🛑 TODAS las posiciones futures cerradas", "INFO")

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
        self.position_side = None
        self.log_message("🔄 Cuenta futures reiniciada a $250.00", "INFO")
        self.save_state()

    def trading_cycle(self):
        """Ciclo principal de trading FUTURES"""
        self.log_message(f"🚀 Iniciando ciclo de trading FUTURES {self.leverage}x - RIESGO {self.risk_per_trade*100}%")
        
        while self.is_running:
            try:
                tick_data = self.get_ticker_price()
                if tick_data:
                    self.tick_data.append(tick_data)
                
                indicators = self.calculate_indicators()
                
                if indicators and len(self.tick_data) >= 10:
                    signal = self.trading_strategy(indicators)
                    if signal != 'hold':
                        if signal == 'buy':
                            price = tick_data['ask']
                        elif signal == 'sell':
                            price = tick_data['bid']
                        else:  # close
                            price = tick_data['bid'] if self.position_side == 'long' else tick_data['ask']
                        self.execute_trade(signal, price)
                
                time.sleep(2)
                
            except Exception as e:
                self.log_message(f"❌ Error en ciclo de trading futures: {e}", "ERROR")
                time.sleep(5)

    def start_trading(self):
        """Iniciar bot de trading futures"""
        if not self.is_running:
            self.is_running = True
            self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
            self.trading_thread.start()
            self.log_message(f"🤖 Bot Futures {self.leverage}x iniciado - RIESGO CONTROLADO")

    def stop_trading(self):
        """Detener bot de trading"""
        self.is_running = False
        self.log_message("🛑 Bot de trading futures detenido")

    def get_performance_stats(self):
        """Obtener estadísticas de performance"""
        current_price = list(self.tick_data)[-1]['bid'] if self.tick_data else 0
        position_value = self.position * current_price / self.leverage
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
            'leverage': self.leverage,
            'position_side': self.position_side
        }
        
        if not self.positions_history:
            return stats
        
        # Calcular win rate
        close_trades = [t for t in self.positions_history if t['action'] == 'close']
        
        if close_trades:
            profitable_trades = 0
            for trade in self.positions_history:
                if trade['action'] == 'close' and trade.get('profit_loss', 0) > 0:
                    profitable_trades += 1
            
            stats['win_rate'] = (profitable_trades / len(close_trades)) * 100 if close_trades else 0
        
        return stats

def main():
    st.title("🤖 Bot Futures MEXC 3x - APALANCAMIENTO CONTROLADO 🚀")
    st.markdown("---")
    
    # Inicializar el bot
    if 'bot' not in st.session_state:
        st.session_state.bot = MexcFuturesTradingBot("", "", "BTCUSDT")
    
    bot = st.session_state.bot
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración Futures")
        
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
        st.header("📊 Configuración Futures")
        st.info(f"**Apalancamiento:** {bot.leverage}x")
        st.info(f"**Riesgo por operación:** {bot.risk_per_trade*100}%")
        st.info(f"**Tamaño posición:** {bot.position_size*100}%")
        st.info(f"**Target ganancia:** {bot.min_profit_target*100}% ({bot.min_profit_target*bot.leverage*100:.1f}% con {bot.leverage}x)")
        st.info(f"**Stop loss:** {bot.max_loss_stop*100}% ({bot.max_loss_stop*bot.leverage*100:.1f}% con {bot.leverage}x)")
        
        if bot.is_running:
            st.success(f"✅ Bot Futures {bot.leverage}x Ejecutándose")
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
            label="⚡ Apalancamiento",
            value=f"{stats['leverage']}x"
        )
    
    with col6:
        position_status = "LONG" if stats['position_side'] == 'long' else "SHORT" if stats['position_side'] == 'short' else "CASH"
        st.metric(
            label="📊 Posición Actual",
            value=position_status
        )
    
    with col7:
        st.metric(
            label="💹 Equity Total",
            value=f"${stats['total_equity']:.2f}"
        )
    
    with col8:
        status = "🎪 LIVE" if bot.is_running else "🛑 STOP"
        st.metric(
            label="🎪 Estado",
            value=status
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
                
                if len(valid_prices) >= 10:
                    df = pd.DataFrame({'price': valid_prices})
                    df['sma_10'] = df['price'].rolling(10).mean()
                    fig.add_trace(go.Scatter(
                        x=valid_timestamps[9:],
                        y=df['sma_10'].dropna(),
                        mode='lines',
                        name='SMA 10',
                        line=dict(color='#ffaa00', width=1, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"Precio Futures de {bot.symbol} en Tiempo Real",
                    xaxis_title="Tiempo",
                    yaxis_title="Precio (USD)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos válidos para graficar")
        else:
            st.info("Esperando datos de mercado...")
    
    with tab2:
        if bot.positions_history:
            # Crear DataFrame seguro
            display_data = []
            for pos in bot.positions_history:
                row = {
                    'timestamp': pos['timestamp'].strftime('%H:%M:%S') if isinstance(pos['timestamp'], datetime) else str(pos['timestamp']),
                    'action': pos['action'],
                    'side': pos.get('position_side', 'N/A'),
                    'leverage': f"{pos.get('leverage', 1)}x",
                    'price': f"${pos['price']:.2f}",
                    'quantity': f"{pos['quantity']:.6f}",
                    'cash_balance': f"${pos['cash_balance']:.2f}",
                    'total_equity': f"${pos['total_equity']:.2f}",
                    'profit_loss': f"${pos.get('profit_loss', 0):.4f}"
                }
                display_data.append(row)
            
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.info("No hay operaciones futures registradas aún.")
    
    with tab3:
        log_container = st.container(height=400)
        with log_container:
            for log_entry in reversed(bot.log_messages[-20:]):
                if "ERROR" in log_entry:
                    st.error(log_entry)
                elif "TRADE" in log_entry:
                    if "LONG" in log_entry or "SHORT" in log_entry:
                        st.success(log_entry)
                    elif "CLOSE" in log_entry:
                        if "📉" in log_entry:
                            st.error(log_entry)
                        else:
                            st.success(log_entry)
                    else:
                        st.info(log_entry)
                elif "SEÑAL" in log_entry or "PROFIT" in log_entry or "LOSS" in log_entry:
                    st.warning(log_entry)
                else:
                    st.info(log_entry)
    
    # Auto-refresh
    if bot.is_running:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
