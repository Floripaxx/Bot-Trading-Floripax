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
        
        # Configuración HFT ULTRA-RÁPIDO - MÁS OPERACIONES
        self.position_size = 0.15       # MODIFICADO: 15% por operación
        self.max_positions = 1          # MODIFICADO: SOLO 1 posición máxima
        self.momentum_threshold = 0.001 # MODIFICADO: 0.1% para señales más sensibles
        self.mean_reversion_threshold = 0.0015  # MODIFICADO: Más sensible
        self.volatility_multiplier = 1.8        # MODIFICADO: Más tolerante a volatilidad
        self.min_profit_target = 0.003  # MODIFICADO: 0.3% de ganancia mínima
        self.max_loss_stop = 0.002      # MODIFICADO: 0.2% de stop loss
        
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
                'total_profit': self.total_profit
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

    def get_real_price_from_api(self) -> dict:
        """Obtener precio REAL de MEXC"""
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
        """Generar precio realista"""
        base_prices = {
            'BTCUSDT': 100000,
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
        
        # 🚨 MEJORA 1: Validar que RSI no sea NaN
        if np.isnan(rsi):
            return 'hold'
        
        # ESTRATEGIA MÁS AGRESIVA Y SELECTIVA
        buy_conditions = [
            momentum > self.momentum_threshold,      # Momentum fuerte
            rsi < 60,                               # 🚨 MEJORA 2: De 45 a 60 (más flexible, evita RSI alto)
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
            self.log_message(f"✅ SEÑAL COMPRA FUERTE: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'buy'
        elif sum(sell_conditions) >= 3:  # Necesita 3 de 5 condiciones
            self.log_message(f"🎯 SEÑAL VENTA: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'sell'
        
        return 'hold'

    def execute_trade(self, action: str, price: float):
        """Ejecutar operación - OPTIMIZADA PARA MAYORES GANANCIAS"""
        try:
            investment_amount = 0
            quantity = 0
            quantity_to_sell = 0
            sale_amount = 0
            profit_loss = 0
            
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
                    
                    trade_info = f"✅ COMPRA: {quantity:.6f} {self.symbol} @ ${price:.2f} | Inversión: ${investment_amount:.2f} | Cash: ${self.cash_balance:.2f}"
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
        self.log_message("🔄 Cuenta reiniciada a $250.00", "INFO")
        self.save_state()

    def trading_cycle(self):
        """Ciclo principal de trading"""
        self.log_message("🚀 Iniciando ciclo de trading HFT ULTRA-RÁPIDO - MÁS OPERACIONES")
        
        while self.is_running:
            try:
                tick_data = self.get_ticker_price()
                if tick_data:
                    self.tick_data.append(tick_data)
                
                indicators = self.calculate_indicators()
                
                if indicators and len(self.tick_data) >= 10:
                    signal = self.trading_strategy(indicators)
                    if signal != 'hold':
                        price = tick_data['bid'] if signal == 'buy' else tick_data['ask']
                        self.execute_trade(signal, price)
                
                time.sleep(2)
                
            except Exception as e:
                self.log_message(f"❌ Error en ciclo de trading: {e}", "ERROR")
                time.sleep(5)

    def start_trading(self):
        """Iniciar bot de trading"""
        if not self.is_running:
            self.is_running = True
            self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
            self.trading_thread.start()
            self.log_message("🤖 Bot de trading ULTRA-RÁPIDO iniciado - MÁS OPERACIONES")

    def stop_trading(self):
        """Detener bot de trading"""
        self.is_running = False
        self.log_message("🛑 Bot de trading detenido")

    def get_performance_stats(self):
        """Obtener estadísticas de performance"""
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
            'realized_profit': self.total_profit
        }
        
        if not self.positions_history:
            return stats
        
        # Calcular win rate
        sell_trades = [t for t in self.positions_history if t['action'] == 'sell']
        
        if sell_trades:
            profitable_trades = 0
            for trade in self.positions_history:
                if trade['action'] == 'sell' and trade.get('profit_loss', 0) > 0:
                    profitable_trades += 1
            
            stats['win_rate'] = (profitable_trades / len(sell_trades)) * 100 if sell_trades else 0
        
        return stats

def main():
    st.title("🤖 Bot HFT MEXC - ESTRATEGIA ULTRA-RÁPIDA 🚀")
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
        st.header("📊 Estadísticas Clave")
        st.info(f"**Tamaño posición:** {bot.position_size*100}%")
        st.info(f"**Target ganancia:** {bot.min_profit_target*100}%")
        st.info(f"**Stop loss:** {bot.max_loss_stop*100}%")
        st.info(f"**Posiciones máx:** {bot.max_positions}")
        
        if bot.is_running:
            st.success("✅ Bot Ejecutándose - ESTRATEGIA ULTRA-RÁPIDA")
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
            value=f"{stats['open_positions']}"
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
                    title=f"Precio de {bot.symbol} en Tiempo Real",
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
                    'price': f"${pos['price']:.2f}",
                    'quantity': f"{pos['quantity']:.6f}",
                    'cash_balance': f"${pos['cash_balance']:.2f}",
                    'total_equity': f"${pos['total_equity']:.2f}",
                    'open_positions': pos['open_positions']
                }
                display_data.append(row)
            
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True, height=400)
        else:
            st.info("No hay operaciones registradas aún.")
    
    with tab3:
        log_container = st.container(height=400)
        with log_container:
            for log_entry in reversed(bot.log_messages[-20:]):
                if "ERROR" in log_entry:
                    st.error(log_entry)
                elif "TRADE" in log_entry:
                    if "COMPRA" in log_entry:
                        st.success(log_entry)
                    elif "VENTA" in log_entry:
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
