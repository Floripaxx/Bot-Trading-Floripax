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
    page_title="🚀 Bot HFT Futuros MEXC - ESTRATEGIA MOMENTUM",
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
        
        # *** SOLO CAMBIO: NUEVA ESTRATEGIA MOMENTUM HFT ***
        self.leverage = 3
        self.position_size = 0.08  # 8% del capital por operación
        self.max_positions = 1
        
        # PARÁMETROS ESTRATEGIA MOMENTUM
        self.ema_fast_period = 8
        self.ema_slow_period = 21
        self.volume_multiplier = 2.0
        self.stop_loss_pct = 0.08    # 0.08% stop loss
        self.take_profit_pct = 0.12  # 0.12% take profit
        
        # *** FIN CAMBIO ESTRATEGIA ***
        
        self.trading_thread = None
        
    def save_state(self):
        """Guardar estado en archivo JSON"""
        try:
            state = {
                'cash_balance': self.cash_balance,
                'position': self.position,
                'position_side': self.position_side,  # 'long' o 'short'
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
                    
            with open('futures_bot_state.json', 'w') as f:
                json.dump(state, f, default=str, indent=2)
        except Exception as e:
            print(f"Error guardando estado: {e}")
    
    def load_state(self):
        """Cargar estado desde archivo JSON"""
        try:
            if os.path.exists('futures_bot_state.json'):
                with open('futures_bot_state.json', 'r') as f:
                    state = json.load(f)
                
                # Convertir deque de tick_data
                tick_data = deque(maxlen=100)  # Más datos para mejor análisis
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
                    'cash_balance': state.get('cash_balance', 255.0),  # CAPITAL INICIAL $255
                    'position': state.get('position', 0),
                    'position_side': state.get('position_side', ''),  # 'long' o 'short'
                    'entry_price': state.get('entry_price', 0),
                    'positions_history': positions_history,
                    'open_positions': state.get('open_positions', 0),
                    'log_messages': state.get('log_messages', []),
                    'tick_data': tick_data,
                    'is_running': state.get('is_running', False),
                    'total_profit': state.get('total_profit', 0)
                }
            else:
                # Estado inicial FUTUROS
                self.bot_data = {
                    'cash_balance': 255.0,  # CAPITAL INICIAL $255
                    'position': 0,
                    'position_side': '',  # 'long' o 'short'
                    'entry_price': 0,
                    'positions_history': [],
                    'open_positions': 0,
                    'log_messages': [],
                    'tick_data': deque(maxlen=100),  # Más datos
                    'is_running': False,
                    'total_profit': 0
                }
        except Exception as e:
            print(f"Error cargando estado: {e}")
            # Estado inicial por defecto FUTUROS
            self.bot_data = {
                'cash_balance': 255.0,  # CAPITAL INICIAL $255
                'position': 0,
                'position_side': '',  # 'long' o 'short'
                'entry_price': 0,
                'positions_history': [],
                'open_positions': 0,
                'log_messages': [],
                'tick_data': deque(maxlen=100),
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
    def position_side(self):
        return self.bot_data['position_side']
    
    @position_side.setter
    def position_side(self, value):
        self.bot_data['position_side'] = value
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
        if len(self.log_messages) > 100:  # Más logs
            self.log_messages.pop(0)
        self.save_state()

    def get_futures_price(self) -> dict:
        """Obtener precio de FUTUROS MEXC"""
        try:
            url = f"https://api.mexc.com/api/v3/ticker/price"
            params = {'symbol': self.symbol}
            
            response = requests.get(url, params=params, timeout=5)  # Timeout más corto
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    current_price = float(data['price'])
                    spread = current_price * 0.00005  # Spread más ajustado para futuros
                    
                    return {
                        'timestamp': datetime.now(),
                        'bid': current_price - spread,
                        'ask': current_price + spread,
                        'symbol': self.symbol,
                        'simulated': False,
                        'source': 'MEXC Futures'
                    }
            
            return self.get_backup_price()
            
        except Exception as e:
            self.log_message(f"Error obteniendo precio futuros: {e}", "ERROR")
            return self.get_backup_price()

    def get_backup_price(self) -> dict:
        """Precio de backup para futuros"""
        base_prices = {
            'BTCUSDT': 106000,  # Precio actualizado
            'ETHUSDT': 3500,
            'ADAUSDT': 0.45,
            'DOTUSDT': 7.5,
            'LINKUSDT': 15.0
        }
        
        base_price = base_prices.get(self.symbol, 106000)
        variation = np.random.uniform(-0.01, 0.01)  # Menor variación
        current_price = base_price * (1 + variation)
        spread = current_price * 0.00005  # Spread ajustado
        
        return {
            'timestamp': datetime.now(),
            'bid': current_price - spread,
            'ask': current_price + spread,
            'symbol': self.symbol,
            'simulated': True,
            'source': 'Futures Simulation'
        }

    # *** SOLO CAMBIO: NUEVA FUNCIÓN DE INDICADORES MOMENTUM ***
    def calculate_indicators(self) -> dict:
        """Calcular indicadores para estrategia MOMENTUM HFT"""
        if len(self.tick_data) < 30:  # Necesitamos más datos para EMAs
            return {}
        
        prices = [tick['bid'] for tick in self.tick_data]
        volumes = [tick.get('volume', 1000) for tick in self.tick_data]  # Volumen simulado
        
        df = pd.DataFrame({
            'price': prices,
            'volume': volumes
        })
        
        # Calcular EMAs para momentum
        df['ema_fast'] = df['price'].ewm(span=self.ema_fast_period, adjust=False).mean()
        df['ema_slow'] = df['price'].ewm(span=self.ema_slow_period, adjust=False).mean()
        
        # Calcular volumen promedio
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        
        # Calcular VWAP
        typical_price = (df['price'] * 3)  # Simulación de high/low/close
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        return {
            'ema_fast': latest['ema_fast'],
            'ema_slow': latest['ema_slow'],
            'ema_fast_prev': previous['ema_fast'],
            'ema_slow_prev': previous['ema_slow'],
            'volume_ratio': latest['volume_ratio'],
            'vwap': latest['vwap'],
            'current_price': latest['price'],
            'returns': latest['price'] / previous['price'] - 1 if len(df) > 1 else 0
        }

    # *** SOLO CAMBIO: NUEVA ESTRATEGIA MOMENTUM ***
    def trading_strategy(self, indicators: dict) -> str:
        """ESTRATEGIA MOMENTUM HFT - EMA Crossover con confirmación volumétrica"""
        if not indicators:
            return 'hold'
        
        ema_fast = indicators['ema_fast']
        ema_slow = indicators['ema_slow']
        ema_fast_prev = indicators['ema_fast_prev']
        ema_slow_prev = indicators['ema_slow_prev']
        volume_ratio = indicators['volume_ratio']
        vwap = indicators['vwap']
        current_price = indicators['current_price']
        
        # SEÑAL LONG: EMA rápido cruza arriba EMA lento + volumen + sobre VWAP
        long_conditions = [
            ema_fast > ema_slow and ema_fast_prev <= ema_slow_prev,  # EMA crossover
            volume_ratio >= self.volume_multiplier,                   # Volumen confirmación
            current_price > vwap,                                     # Tendencia alcista
            self.position == 0                                        # Sin posición abierta
        ]
        
        # SEÑAL SHORT: EMA rápido cruza abajo EMA lento + volumen + bajo VWAP
        short_conditions = [
            ema_fast < ema_slow and ema_fast_prev >= ema_slow_prev,  # EMA crossover
            volume_ratio >= self.volume_multiplier,                   # Volumen confirmación  
            current_price < vwap,                                     # Tendencia bajista
            self.position == 0                                        # Sin posición abierta
        ]
        
        # GESTIÓN DE POSICIÓN ABIERTA
        if self.position > 0:
            if self.position_side == 'long':
                current_profit_pct = (current_price - self.entry_price) / self.entry_price
                current_loss_pct = (self.entry_price - current_price) / self.entry_price
            else:  # short
                current_profit_pct = (self.entry_price - current_price) / self.entry_price
                current_loss_pct = (current_price - self.entry_price) / self.entry_price
            
            # TAKE PROFIT
            if current_profit_pct >= self.take_profit_pct:
                self.log_message(f"🎯 TOMA DE GANANCIAS: {current_profit_pct:.3%} (+)", "PROFIT")
                return 'close'
            
            # STOP LOSS
            if current_loss_pct >= self.stop_loss_pct:
                self.log_message(f"🛑 STOP LOSS: {current_loss_pct:.3%} (-)", "LOSS")
                return 'close'
        
        # SEÑALES PRINCIPALES MOMENTUM
        if all(long_conditions):
            self.log_message(f"📈 SEÑAL LONG: EMA{self.ema_fast_period}/{self.ema_slow_period} crossover, Vol: {volume_ratio:.1f}x", "SIGNAL")
            return 'buy'
        elif all(short_conditions):
            self.log_message(f"📉 SEÑAL SHORT: EMA{self.ema_fast_period}/{self.ema_slow_period} crossover, Vol: {volume_ratio:.1f}x", "SIGNAL")
            return 'sell'
        
        return 'hold'

    def execute_trade(self, action: str, price: float):
        """Ejecutar operación en FUTUROS - MANTENIDO ORIGINAL"""
        try:
            investment_amount = 0
            quantity = 0
            quantity_to_close = 0
            close_amount = 0
            profit_loss = 0
            
            if action in ['buy', 'sell'] and self.position == 0:
                # ABRIR POSICIÓN
                investment_amount = self.cash_balance * self.position_size * self.leverage
                quantity = investment_amount / price
                
                if investment_amount > self.cash_balance * self.leverage:
                    self.log_message("❌ Margen insuficiente", "ERROR")
                    return
                
                # Actualizar balances
                self.cash_balance -= (investment_amount / self.leverage)  # Margen utilizado
                self.position = quantity
                self.entry_price = price
                self.position_side = 'long' if action == 'buy' else 'short'
                self.open_positions = 1
                
                side_emoji = "🟢" if action == 'buy' else "🔴"
                trade_info = f"{side_emoji} ABRIR {self.position_side.upper()}: {quantity:.6f} {self.symbol} @ ${price:.2f} | Margen: ${investment_amount/self.leverage:.2f} | Leverage: {self.leverage}x"
                self.log_message(trade_info, "TRADE")
                
            elif action == 'close' and self.position > 0:
                # CERRAR POSICIÓN
                quantity_to_close = self.position
                
                if self.position_side == 'long':
                    close_amount = quantity_to_close * price
                    profit_loss = (close_amount - (self.position * self.entry_price)) * self.leverage
                else:  # short
                    close_amount = quantity_to_close * price
                    profit_loss = ((self.position * self.entry_price) - close_amount) * self.leverage
                
                # Actualizar balances
                self.cash_balance += (self.position * self.entry_price / self.leverage) + profit_loss
                self.position = 0
                self.open_positions = 0
                self.position_side = ''
                self.total_profit += profit_loss
                
                profit_color = "🟢" if profit_loss > 0 else "🔴"
                trade_info = f"{profit_color} CERRAR: {quantity_to_close:.6f} {self.symbol} @ ${price:.2f} | P/L: ${profit_loss:.4f} | Profit Total: ${self.total_profit:.2f}"
                self.log_message(trade_info, "TRADE")
            
            # Registrar posición
            current_equity = self.cash_balance + (self.position * self.entry_price / self.leverage if self.position > 0 else 0)
            self.positions_history.append({
                'timestamp': datetime.now(),
                'action': action,
                'side': self.position_side if action != 'close' else '',
                'leverage': self.leverage if action != 'close' else '',
                'price': price,
                'quantity': quantity_to_close if action == 'close' else quantity,
                'cash_balance': self.cash_balance,
                'total_equity': current_equity,
                'profit_loss': profit_loss if action == 'close' else 0
            })
            
            self.save_state()
            
        except Exception as e:
            self.log_message(f"Error ejecutando trade futuros: {e}", "ERROR")

    def close_all_positions(self):
        """Cerrar todas las posiciones abiertas"""
        if self.position > 0:
            tick_data = self.get_futures_price()
            price = tick_data['bid'] if self.position_side == 'long' else tick_data['ask']
            self.execute_trade('close', price)
            self.log_message("🔄 TODAS las posiciones futuros cerradas", "INFO")

    def reset_account(self):
        """Reiniciar cuenta a estado inicial"""
        self.cash_balance = 255.0  # CAPITAL INICIAL $255
        self.position = 0
        self.position_side = ''
        self.entry_price = 0
        self.positions_history.clear()
        self.open_positions = 0
        self.log_messages.clear()
        self.tick_data.clear()
        self.total_profit = 0
        self.log_message("🔄 Cuenta futuros reiniciada a $255.00", "INFO")
        self.save_state()

    def trading_cycle(self):
        """Ciclo principal de trading HFT"""
        self.log_message("🚀 Iniciando HFT Futuros - ESTRATEGIA MOMENTUM ACTIVADA")
        
        while self.is_running:
            try:
                tick_data = self.get_futures_price()
                if tick_data:
                    self.tick_data.append(tick_data)
                
                indicators = self.calculate_indicators()
                
                if indicators and len(self.tick_data) >= 30:  # Más datos necesarios
                    signal = self.trading_strategy(indicators)
                    if signal != 'hold':
                        price = tick_data['ask'] if signal == 'buy' else tick_data['bid']
                        self.execute_trade(signal, price)
                
                time.sleep(0.8)  # HFT: 800ms entre ciclos
                
            except Exception as e:
                self.log_message(f"Error en ciclo HFT: {e}", "ERROR")
                time.sleep(2)

    def start_trading(self):
        """Iniciar bot de trading HFT"""
        if not self.is_running:
            self.is_running = True
            self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
            self.trading_thread.start()
            self.log_message("✅ Bot HFT Futuros iniciado - ESTRATEGIA MOMENTUM ACTIVADA")

    def stop_trading(self):
        """Detener bot de trading"""
        self.is_running = False
        self.log_message("🛑 Bot HFT Futuros detenido")

    def get_performance_stats(self):
        """Obtener estadísticas de performance"""
        current_price = list(self.tick_data)[-1]['bid'] if self.tick_data else 0
        
        # Calcular equity considerando posición abierta
        if self.position > 0:
            if self.position_side == 'long':
                position_value = self.position * current_price
                unrealized_pl = (position_value - (self.position * self.entry_price)) * self.leverage
            else:  # short
                position_value = self.position * current_price
                unrealized_pl = ((self.position * self.entry_price) - position_value) * self.leverage
            
            total_equity = self.cash_balance + unrealized_pl
        else:
            total_equity = self.cash_balance
        
        total_profit = total_equity - 255.0  # Desde capital inicial
        
        stats = {
            'total_trades': len([p for p in self.positions_history if p['action'] in ['buy', 'sell']]),
            'win_rate': 0,
            'cash_balance': self.cash_balance,
            'total_equity': total_equity,
            'open_positions': self.open_positions,
            'current_price': current_price,
            'total_profit': total_profit,
            'position_size': self.position,
            'position_side': self.position_side,
            'realized_profit': self.total_profit,
            'leverage': self.leverage
        }
        
        if not self.positions_history:
            return stats
        
        # Calcular win rate
        close_trades = [t for t in self.positions_history if t['action'] == 'close']
        
        if close_trades:
            profitable_trades = len([t for t in close_trades if t.get('profit_loss', 0) > 0])
            stats['win_rate'] = (profitable_trades / len(close_trades)) * 100 if close_trades else 0
        
        return stats

def main():
    st.title("🚀 Bot HFT Futuros MEXC - ESTRATEGIA MOMENTUM 🚀")
    st.markdown("**CAPITAL INICIAL: $255.00 | APALANCAMIENTO: 3x | ESTRATEGIA: MOMENTUM HFT**")
    st.markdown("---")
    
    # Inicializar el bot
    if 'bot' not in st.session_state:
        st.session_state.bot = MexcFuturesTradingBot("", "", "BTCUSDT")
    
    bot = st.session_state.bot
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuración Futuros")
        
        api_key = st.text_input("API Key MEXC", type="password")
        secret_key = st.text_input("Secret Key MEXC", type="password")
        symbol = st.selectbox("Símbolo Futuros", ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"])
        
        bot.api_key = api_key
        bot.secret_key = secret_key
        bot.symbol = symbol
        
        st.markdown("---")
        st.header("🎮 Control HFT Futuros")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Iniciar HFT", use_container_width=True, type="primary"):
                bot.start_trading()
                st.rerun()
        
        with col2:
            if st.button("🛑 Detener HFT", use_container_width=True):
                bot.stop_trading()
                st.rerun()
        
        if st.button("🔄 Cerrar Posición", use_container_width=True):
            bot.close_all_positions()
            st.rerun()
            
        if st.button("🔄 Reiniciar $255", use_container_width=True):
            bot.reset_account()
            st.rerun()
        
        st.markdown("---")
        st.header("⚙️ Configuración HFT")
        st.info(f"**Tamaño posición:** {bot.position_size*100}%")
        st.info(f"**Take Profit:** {bot.take_profit_pct*100}%")
        st.info(f"**Stop Loss:** {bot.stop_loss_pct*100}%")
        st.info(f"**Apalancamiento:** {bot.leverage}x")
        st.info(f"**EMAs:** {bot.ema_fast_period}/{bot.ema_slow_period}")
        st.info(f"**Volumen:** {bot.volume_multiplier}x")
        
        if bot.is_running:
            st.success("✅ HFT EJECUTÁNDOSE - ESTRATEGIA MOMENTUM")
        else:
            st.warning("🛑 HFT DETENIDO")
            
        if bot.tick_data:
            latest_tick = list(bot.tick_data)[-1]
            source = latest_tick.get('source', 'Unknown')
            st.info(f"**Fuente:** {source}")

    # Layout principal
    col1, col2, col3, col4 = st.columns(4)
    
    stats = bot.get_performance_stats()
    
    with col1:
        st.metric(
            label="💰 Margen Disponible",
            value=f"${stats['cash_balance']:.2f}",
            delta=f"${stats['realized_profit']:.2f}" if stats['realized_profit'] != 0 else None
        )
    
    with col2:
        st.metric(
            label="📈 Precio Futuros",
            value=f"${stats['current_price']:.2f}"
        )
    
    with col3:
        st.metric(
            label="🎯 Tasa Acierto",
            value=f"{stats['win_rate']:.1f}%"
        )
    
    with col4:
        st.metric(
            label="📊 Operaciones",
            value=f"{stats['total_trades']}"
        )
    
    # Segunda fila de métricas
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        position_status = f"{stats['position_side'].upper()}" if stats['position_side'] else "SIN POSICIÓN"
        st.metric(
            label="📊 Posición Actual",
            value=position_status
        )
    
    with col6:
        st.metric(
            label="⚖️ Tamaño Pos",
            value=f"{stats['position_size']:.6f}"
        )
    
    with col7:
        st.metric(
            label="💰 Equity Total",
            value=f"${stats['total_equity']:.2f}"
        )
    
    with col8:
        leverage_info = f"{stats['leverage']}x" if stats['position_side'] else "---"
        st.metric(
            label="🎯 Apalancamiento",
            value=leverage_info
        )
    
    st.markdown("---")
    
    # Gráficos y datos
    tab1, tab2, tab3 = st.tabs(["📈 Precios Futuros", "📋 Operaciones Futuros", "📝 Logs HFT"])
    
    with tab1:
        if bot.tick_data:
            prices = [tick['bid'] for tick in list(bot.tick_data)]
            timestamps = [tick['timestamp'] for tick in list(bot.tick_data)]
            
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
                
                # Mostrar posición actual si existe
                if bot.position > 0 and bot.entry_price > 0:
                    fig.add_hline(
                        y=bot.entry_price, 
                        line_dash="dash", 
                        line_color="yellow",
                        annotation_text=f"Entrada: ${bot.entry_price:.2f}"
                    )
                
                fig.update_layout(
                    title=f"Precio Futuros {bot.symbol} - ESTRATEGIA MOMENTUM HFT",
                    xaxis_title="Tiempo",
                    yaxis_title="Precio (USD)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos válidos para graficar")
        else:
            st.info("Esperando datos de futuros...")
    
    with tab2:
        if bot.positions_history:
            # Crear DataFrame para futuros
            display_data = []
            for pos in bot.positions_history:
                row = {
                    'timestamp': pos['timestamp'].strftime('%H:%M:%S') if isinstance(pos['timestamp'], datetime) else str(pos['timestamp']),
                    'action': pos['action'],
                    'side': pos.get('side', ''),
                    'leverage': pos.get('leverage', ''),
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
            st.info("No hay operaciones de futuros registradas aún.")
    
    with tab3:
        log_container = st.container(height=400)
        with log_container:
            for log_entry in reversed(bot.log_messages[-30:]):  # Más logs
                if "ERROR" in log_entry:
                    st.error(log_entry)
                elif "TRADE" in log_entry:
                    if "ABRIR" in log_entry:
                        st.success(log_entry)
                    elif "CERRAR" in log_entry:
                        if "🔴" in log_entry:
                            st.error(log_entry)
                        else:
                            st.success(log_entry)
                    else:
                        st.info(log_entry)
                elif "SEÑAL" in log_entry or "PROFIT" in log_entry or "LOSS" in log_entry:
                    st.warning(log_entry)
                else:
                    st.info(log_entry)
    
    # Auto-refresh más rápido para HFT
    if bot.is_running:
        time.sleep(3)
        st.rerun()

if __name__ == "__main__":
    main()
