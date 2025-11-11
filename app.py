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
import atexit
import signal
import sys

# =============================================
# SISTEMA DE PERSISTENCIA INDEPENDIENTE
# =============================================

class PersistentStateManager:
    """Gestor de estado INDEPENDIENTE de Streamlit"""
    
    def __init__(self, state_file='bot_persistent_state.json'):
        self.state_file = state_file
        self.lock = threading.Lock()
        self._register_cleanup()
    
    def _register_cleanup(self):
        """Registrar handlers para cierre limpio"""
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Manejar señales de sistema"""
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Limpieza al cerrar"""
        print("🛑 Cerrando bot persistentemente...")
    
    def save_state(self, state_data):
        """Guardar estado de forma atómica y robusta"""
        with self.lock:
            try:
                # Crear copia profunda
                state_copy = self._deep_copy_state(state_data)
                
                # Guardar en temporal primero
                temp_file = f"{self.state_file}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(state_copy, f, default=self._json_serializer, indent=2)
                
                # Mover atómicamente
                os.replace(temp_file, self.state_file)
                
                # Backup cada 30 minutos
                if int(time.time()) % 1800 < 2:  # Cada 30 minutos aprox
                    backup_file = f"backups/{self.state_file}.backup.{int(time.time())}"
                    os.makedirs('backups', exist_ok=True)
                    with open(backup_file, 'w') as f:
                        json.dump(state_copy, f, default=self._json_serializer, indent=2)
                
                return True
            except Exception as e:
                print(f"🚨 ERROR guardando estado persistente: {e}")
                # Intentar guardado de emergencia
                try:
                    emergency_file = f"{self.state_file}.emergency"
                    with open(emergency_file, 'w') as f:
                        json.dump({'timestamp': datetime.now().isoformat(), 'error': str(e)}, f)
                except:
                    pass
                return False
    
    def load_state(self):
        """Cargar estado con recuperación múltiple"""
        with self.lock:
            try:
                # Intentar archivo principal
                if os.path.exists(self.state_file):
                    with open(self.state_file, 'r') as f:
                        state = json.load(f)
                    return self._deserialize_state(state)
                
                # Intentar archivo temporal
                temp_file = f"{self.state_file}.tmp"
                if os.path.exists(temp_file):
                    with open(temp_file, 'r') as f:
                        state = json.load(f)
                    os.replace(temp_file, self.state_file)  # Recuperar
                    return self._deserialize_state(state)
                
                # Intentar backups
                backup_files = [f for f in os.listdir('backups') if f.startswith(self.state_file)] if os.path.exists('backups') else []
                if backup_files:
                    latest_backup = max(backup_files)
                    with open(f"backups/{latest_backup}", 'r') as f:
                        state = json.load(f)
                    return self._deserialize_state(state)
                
                # Estado inicial
                return self._get_initial_state()
                
            except Exception as e:
                print(f"🚨 ERROR cargando estado persistente: {e}")
                return self._get_initial_state()
    
    def _deep_copy_state(self, state):
        """Copia profunda del estado"""
        if isinstance(state, dict):
            return {k: self._deep_copy_state(v) for k, v in state.items()}
        elif isinstance(state, list):
            return [self._deep_copy_state(item) for item in state]
        else:
            return state
    
    def _json_serializer(self, obj):
        """Serializar objetos para JSON"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _deserialize_state(self, state):
        """Deserializar estado desde JSON"""
        def convert_timestamps(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == 'timestamp' and isinstance(v, str):
                        try:
                            obj[k] = datetime.fromisoformat(v.replace('Z', '+00:00'))
                        except:
                            obj[k] = datetime.now()
                    else:
                        convert_timestamps(v)
            elif isinstance(obj, list):
                for item in obj:
                    convert_timestamps(item)
            return obj
        
        return convert_timestamps(state)
    
    def _get_initial_state(self):
        """Estado inicial del bot"""
        return {
            'cash_balance': 255.0,
            'position': 0,
            'position_side': '',
            'entry_price': 0,
            'positions_history': [],
            'open_positions': 0,
            'log_messages': ["🔄 SISTEMA INICIADO - Estado persistente activado"],
            'tick_data': [],
            'is_running': False,
            'total_profit': 0,
            'start_time': datetime.now().isoformat(),
            'session_id': str(int(time.time()))
        }

# =============================================
# BOT CON PERSISTENCIA INDEPENDIENTE
# =============================================

class MexcFuturesTradingBot:
    def __init__(self, api_key: str, secret_key: str, symbol: str = 'BTCUSDT'):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.base_url = 'https://api.mexc.com'
        
        # SISTEMA DE PERSISTENCIA INDEPENDIENTE
        self.persistence = PersistentStateManager()
        self._state = self.persistence.load_state()
        
        # Configuración del bot (igual que antes)
        self.leverage = 3
        self.position_size = 0.12
        self.max_positions = 2
        self.momentum_threshold = 0.0012
        self.mean_reversion_threshold = 0.001
        self.volatility_multiplier = 1.8
        self.min_profit_target = 0.0015
        self.max_loss_stop = 0.0020
        
        self.trading_thread = None
        self._auto_save_thread = None
        self._running = False
        
        # Iniciar auto-guardado
        self._start_auto_save()
    
    def _start_auto_save(self):
        """Hilo independiente para auto-guardado cada 30 segundos"""
        def auto_save_loop():
            while getattr(self, '_running', True):
                try:
                    self.persistence.save_state(self._state)
                    time.sleep(30)  # Guardar cada 30 segundos
                except:
                    time.sleep(10)
        
        self._auto_save_thread = threading.Thread(target=auto_save_loop, daemon=True)
        self._auto_save_thread.start()
    
    # PROPIEDADES CON PERSISTENCIA AUTOMÁTICA
    @property
    def cash_balance(self):
        return self._state['cash_balance']
    
    @cash_balance.setter
    def cash_balance(self, value):
        self._state['cash_balance'] = value
        self._auto_save()
    
    @property
    def position(self):
        return self._state['position']
    
    @position.setter
    def position(self, value):
        self._state['position'] = value
        self._auto_save()
    
    @property
    def position_side(self):
        return self._state['position_side']
    
    @position_side.setter
    def position_side(self, value):
        self._state['position_side'] = value
        self._auto_save()
    
    @property
    def entry_price(self):
        return self._state['entry_price']
    
    @entry_price.setter
    def entry_price(self, value):
        self._state['entry_price'] = value
        self._auto_save()
    
    @property
    def positions_history(self):
        return self._state['positions_history']
    
    @property
    def open_positions(self):
        return self._state['open_positions']
    
    @open_positions.setter
    def open_positions(self, value):
        self._state['open_positions'] = value
        self._auto_save()
    
    @property
    def log_messages(self):
        return self._state['log_messages']
    
    @property
    def tick_data(self):
        # Convertir lista a deque para uso interno
        if 'tick_data' not in self._state:
            self._state['tick_data'] = []
        return deque(self._state['tick_data'], maxlen=100)
    
    @tick_data.setter
    def tick_data(self, value):
        # Convertir deque a lista para persistencia
        self._state['tick_data'] = list(value)
        self._auto_save()
    
    @property
    def is_running(self):
        return self._state['is_running']
    
    @is_running.setter
    def is_running(self, value):
        self._state['is_running'] = value
        self._auto_save()
    
    @property
    def total_profit(self):
        return self._state['total_profit']
    
    @total_profit.setter
    def total_profit(self, value):
        self._state['total_profit'] = value
        self._auto_save()
    
    def _auto_save(self):
        """Guardado automático no bloqueante"""
        try:
            # Actualizar tick_data antes de guardar
            if hasattr(self, '_current_tick_data'):
                self._state['tick_data'] = list(self._current_tick_data)
            
            # Guardar en hilo separado para no bloquear
            threading.Thread(
                target=self.persistence.save_state, 
                args=(self._state,),
                daemon=True
            ).start()
        except:
            pass
    
    def log_message(self, message: str, level: str = "INFO"):
        """Agregar mensaje al log con persistencia"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_messages.append(log_entry)
        if len(self.log_messages) > 100:
            self.log_messages.pop(0)
        self._auto_save()

    # MANTENER TODAS LAS DEMÁS FUNCIONES IGUAL (get_futures_price, calculate_indicators, trading_strategy, etc.)
    # ... [Todas las funciones anteriores se mantienen igual] ...
    
    def get_futures_price(self) -> dict:
        """Obtener precio de FUTUROS MEXC"""
        try:
            url = f"https://api.mexc.com/api/v3/ticker/price"
            params = {'symbol': self.symbol}
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    current_price = float(data['price'])
                    spread = current_price * 0.00005
                    
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
            'BTCUSDT': 103000,
            'ETHUSDT': 3500,
            'ADAUSDT': 0.45,
            'DOTUSDT': 7.5,
            'LINKUSDT': 15.0
        }
        
        base_price = base_prices.get(self.symbol, 103000)
        variation = np.random.uniform(-0.01, 0.01)
        current_price = base_price * (1 + variation)
        spread = current_price * 0.00005
        
        return {
            'timestamp': datetime.now(),
            'bid': current_price - spread,
            'ask': current_price + spread,
            'symbol': self.symbol,
            'simulated': True,
            'source': 'Futures Simulation'
        }

    def calculate_indicators(self) -> dict:
        """Calcular indicadores técnicos OPTIMIZADOS PARA HFT"""
        if len(self.tick_data) < 10:
            return {}
        
        prices = [tick['bid'] for tick in list(self.tick_data)]
        df = pd.DataFrame(prices, columns=['price'])
        
        # Indicadores ULTRA RÁPIDOS para HFT
        df['returns'] = df['price'].pct_change()
        df['momentum'] = df['returns'].rolling(2).mean()
        df['sma_3'] = df['price'].rolling(3).mean()
        df['sma_8'] = df['price'].rolling(8).mean()
        df['price_deviation'] = (df['price'] - df['sma_3']) / df['sma_3']
        df['volatility'] = df['returns'].rolling(5).std() * self.volatility_multiplier
        
        # RSI ultra rápido
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=3).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD ultra rápido
        exp8 = df['price'].ewm(span=3, adjust=False).mean()
        exp16 = df['price'].ewm(span=6, adjust=False).mean()
        df['macd'] = exp8 - exp16
        df['macd_signal'] = df['macd'].ewm(span=2, adjust=False).mean()
        
        # Bollinger Bands ajustadas
        df['bb_middle'] = df['price'].rolling(8).mean()
        df['bb_std'] = df['price'].rolling(8).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 1.2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 1.2)
        
        latest = df.iloc[-1]
        
        return {
            'momentum': latest['momentum'],
            'price_deviation': latest['price_deviation'],
            'current_price': latest['price'],
            'sma_3': latest['sma_3'],
            'sma_8': latest['sma_8'],
            'rsi': latest['rsi'],
            'volatility': latest['volatility'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal'],
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'bb_middle': latest['bb_middle']
        }

    def trading_strategy(self, indicators: dict) -> str:
        """ESTRATEGIA ULTRA AGRESIVA - MÁXIMAS OPERACIONES"""
        if not indicators:
            return 'hold'
        
        momentum = indicators['momentum']
        rsi = indicators['rsi']
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        current_price = indicators['current_price']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        
        # ESTRATEGIA ULTRA AGRESIVA - MÁXIMAS ENTRADAS
        
        # SHORT ULTRA AGRESIVO
        short_conditions = [
            momentum > 0.0003,
            rsi > 25,
            current_price > bb_lower,
            macd > macd_signal * 0.8,
        ]
        
        # LONG ULTRA AGRESIVO  
        long_conditions = [
            momentum < -0.0003,
            rsi < 75,
            current_price < bb_upper,
            macd < macd_signal * 1.2,
        ]
        
        # GESTIÓN DE POSICIÓN ULTRA AGRESIVA
        if self.position > 0:
            if self.position_side == 'long':
                current_profit_pct = (current_price - self.entry_price) / self.entry_price
                current_loss_pct = (self.entry_price - current_price) / self.entry_price
            else:  # short
                current_profit_pct = (self.entry_price - current_price) / self.entry_price
                current_loss_pct = (current_price - self.entry_price) / self.entry_price
            
            # TOMA DE GANANCIAS ULTRA RÁPIDA
            if current_profit_pct >= self.min_profit_target:
                self.log_message(f"🎯 GANANCIA TURBO: {current_profit_pct:.3%} (+)", "PROFIT")
                return 'close'
            
            # STOP LOSS PROTECTOR
            if current_loss_pct >= self.max_loss_stop:
                self.log_message(f"🛑 STOP LOSS TURBO: {current_loss_pct:.3%} (-)", "LOSS")
                return 'close'
        
        # SEÑALES ULTRA AGRESIVAS - MÁXIMAS OPERACIONES
        if sum(short_conditions) >= 2 and self.open_positions < self.max_positions:
            self.log_message(f"⚡ SEÑAL SHORT TURBO: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'sell'
        elif sum(long_conditions) >= 2 and self.open_positions < self.max_positions:
            self.log_message(f"⚡ SEÑAL LONG TURBO: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'buy'
        
        return 'hold'

    def execute_trade(self, action: str, price: float):
        """Ejecutar operación en FUTUROS - MODO TURBO"""
        try:
            investment_amount = 0
            quantity = 0
            quantity_to_close = 0
            close_amount = 0
            profit_loss = 0
            
            if action in ['buy', 'sell'] and self.open_positions < self.max_positions:
                # ABRIR POSICIÓN
                investment_amount = self.cash_balance * self.position_size * self.leverage
                quantity = investment_amount / price
                
                if investment_amount > self.cash_balance * self.leverage:
                    self.log_message("❌ Margen insuficiente", "ERROR")
                    return
                
                # Actualizar balances
                self.cash_balance -= (investment_amount / self.leverage)
                self.position += quantity
                self.entry_price = price if self.position == quantity else ((self.entry_price * (self.position - quantity)) + (price * quantity)) / self.position
                self.position_side = 'long' if action == 'buy' else 'short'
                self.open_positions += 1
                
                side_emoji = "🟢" if action == 'buy' else "🔴"
                trade_info = f"{side_emoji} TURBO {self.position_side.upper()}: {quantity:.6f} {self.symbol} @ ${price:.2f} | Margen: ${investment_amount/self.leverage:.2f} | Leverage: {self.leverage}x"
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
                trade_info = f"{profit_color} CERRAR TURBO: {quantity_to_close:.6f} {self.symbol} @ ${price:.2f} | P/L: ${profit_loss:.4f} | Profit Total: ${self.total_profit:.2f}"
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
            
            self._auto_save()
            
        except Exception as e:
            self.log_message(f"🚨 ERROR ejecutando trade: {e}", "ERROR")

    def close_all_positions(self):
        """Cerrar todas las posiciones abiertas"""
        if self.position > 0:
            tick_data = self.get_futures_price()
            price = tick_data['bid'] if self.position_side == 'long' else tick_data['ask']
            self.execute_trade('close', price)
            self.log_message("🧹 TODAS las posiciones cerradas - MODO SEGURIDAD", "INFO")

    def reset_account(self):
        """Reiniciar cuenta a estado inicial"""
        self.cash_balance = 255.0
        self.position = 0
        self.position_side = ''
        self.entry_price = 0
        self.positions_history.clear()
        self.open_positions = 0
        self.log_messages.clear()
        self._state['tick_data'] = []
        self.total_profit = 0
        self.log_message("🔄 Cuenta reiniciada a $255.00 - MODO TURBO", "INFO")

    def trading_cycle(self):
        """Ciclo principal de trading HFT ULTRA RÁPIDO"""
        self.log_message("🚀 INICIANDO MODO TURBO HFT - PERSISTENCIA ACTIVA")
        
        # Inicializar tick_data interno
        self._current_tick_data = deque(maxlen=100)
        if self._state['tick_data']:
            self._current_tick_data.extend(self._state['tick_data'])
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                tick_data = self.get_futures_price()
                if tick_data:
                    self._current_tick_data.append(tick_data)
                    # Actualizar estado periódicamente
                    if len(self._current_tick_data) % 10 == 0:
                        self._state['tick_data'] = list(self._current_tick_data)
                        self._auto_save()
                
                indicators = self.calculate_indicators()
                
                if indicators and len(self._current_tick_data) >= 10:
                    signal = self.trading_strategy(indicators)
                    if signal != 'hold':
                        price = tick_data['ask'] if signal == 'buy' else tick_data['bid']
                        self.execute_trade(signal, price)
                
                consecutive_errors = 0
                time.sleep(0.3)
                
            except Exception as e:
                consecutive_errors += 1
                self.log_message(f"🚨 ERROR en ciclo #{consecutive_errors}: {e}", "ERROR")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.log_message("🛑 DEMASIADOS ERRORES - Cerrando todas las posiciones", "CRITICAL")
                    self.close_all_positions()
                    self.is_running = False
                    break
                
                time.sleep(5)
                continue

    def start_trading(self):
        """Iniciar bot de trading HFT"""
        if not self.is_running:
            self.is_running = True
            self._running = True
            self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
            self.trading_thread.start()
            self.log_message("✅ MODO TURBO ACTIVADO - PERSISTENCIA GARANTIZADA", "SYSTEM")

    def stop_trading(self):
        """Detener bot de trading"""
        self.is_running = False
        self._running = False
        self.log_message("⏹️ MODO TURBO DETENIDO", "SYSTEM")
        # Guardar estado final
        self._auto_save()

    def get_performance_stats(self):
        """Obtener estadísticas de performance"""
        current_data = list(self._current_tick_data) if hasattr(self, '_current_tick_data') and self._current_tick_data else self.tick_data
        current_price = current_data[-1]['bid'] if current_data else 0
        
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
        
        total_profit = total_equity - 255.0
        
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

# =============================================
# INTERFAZ STREAMLIT (SIN CAMBIOS)
# =============================================

def main():
    st.title("🤖 Bot HFT Futuros MEXC - PERSISTENCIA TOTAL ⚡")
    st.markdown("**CAPITAL INICIAL: $255.00 | APALANCAMIENTO: 3x | PERSISTENCIA INDEPENDIENTE**")
    st.markdown("---")
    
    # Inicializar el bot
    if 'bot' not in st.session_state:
        st.session_state.bot = MexcFuturesTradingBot("", "", "BTCUSDT")
    
    bot = st.session_state.bot
    
    # Sidebar (igual que antes)
    with st.sidebar:
        st.header("⚙️ Configuración Turbo")
        
        api_key = st.text_input("API Key MEXC", type="password")
        secret_key = st.text_input("Secret Key MEXC", type="password")
        symbol = st.selectbox("Símbolo Futuros", ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"])
        
        bot.api_key = api_key
        bot.secret_key = secret_key
        bot.symbol = symbol
        
        st.markdown("---")
        st.header("🎯 Control Turbo")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Activar Turbo", use_container_width=True, type="primary"):
                bot.start_trading()
                st.rerun()
        
        with col2:
            if st.button("⏹️ Desactivar Turbo", use_container_width=True):
                bot.stop_trading()
                st.rerun()
        
        if st.button("🧹 Cerrar Posiciones", use_container_width=True):
            bot.close_all_positions()
            st.rerun()
            
        if st.button("🔄 Reiniciar $255", use_container_width=True):
            bot.reset_account()
            st.rerun()
        
        st.markdown("---")
        st.header("💰 Configuración Turbo")
        st.info(f"**Tamaño posición:** {bot.position_size*100}%")
        st.info(f"**Target ganancia:** {bot.min_profit_target*100}%")
        st.info(f"**Stop loss:** {bot.max_loss_stop*100}%")
        st.info(f"**Apalancamiento:** {bot.leverage}x")
        st.info(f"**Operaciones máx:** {bot.max_positions}")
        
        if bot.is_running:
            st.success("✅ TURBO ACTIVO - PERSISTENCIA ACTIVA")
        else:
            st.warning("⏸️ SISTEMA EN STANDBY")
            
        if hasattr(bot, '_current_tick_data') and bot._current_tick_data:
            latest_tick = list(bot._current_tick_data)[-1]
            source = latest_tick.get('source', 'Unknown')
            st.info(f"**Fuente:** {source}")

    # Layout principal (igual que antes)
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
            label="🔓 Posición Actual",
            value=position_status
        )
    
    with col6:
        st.metric(
            label="📦 Tamaño Pos",
            value=f"{stats['position_size']:.6f}"
        )
    
    with col7:
        st.metric(
            label="💹 Equity Total",
            value=f"${stats['total_equity']:.2f}"
        )
    
    with col8:
        leverage_info = f"{stats['leverage']}x" if stats['position_side'] else "---"
        st.metric(
            label="⚡ Apalancamiento",
            value=leverage_info
        )
    
    st.markdown("---")
    
    # Gráficos y datos
    tab1, tab2, tab3 = st.tabs(["📈 Precios Futuros", "📋 Operaciones Turbo", "📝 Logs del Sistema"])
    
    with tab1:
        current_data = list(bot._current_tick_data) if hasattr(bot, '_current_tick_data') and bot._current_tick_data else list(bot.tick_data)
        if current_data:
            prices = [tick['bid'] for tick in current_data]
            timestamps = [tick['timestamp'] for tick in current_data]
            
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
                    title=f"Precio Futuros {bot.symbol} - MODO TURBO",
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
                    'price': f"${pos['price
