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

# Configurar la página de Streamlit
st.set_page_config(
    page_title="🤖 Bot HFT Futuros MEXC - ESTRATEGIA ULTRA AGRESIVA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== SISTEMA DE SINCRONIZACIÓN ==========
def force_sync_position_state(bot):
    """
    SINCRONIZACIÓN MÍNIMA: Resuelve el problema de posición fantasma
    """
    try:
        if (bot.position > 0 or bot.open_positions > 0) and not bot._running:
            print("🔄 SINCRONIZACIÓN: Reseteando estado de posición fantasma")
            bot.position = 0
            bot.open_positions = 0
            bot.position_side = ''
            bot.entry_price = 0
            return True
        return False
    except Exception as e:
        print(f"❌ Error en sincronización: {e}")
        return False

class PersistentStateManager:
    def __init__(self, state_file='bot_persistent_state.json'):
        self.state_file = state_file
        self.lock = threading.Lock()
    
    def save_state(self, state_data):
        with self.lock:
            try:
                state_copy = json.loads(json.dumps(state_data, default=self._json_serializer))
                temp_file = f"{self.state_file}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(state_copy, f, indent=2)
                os.replace(temp_file, self.state_file)
                return True
            except Exception as e:
                print(f"❌ ERROR guardando estado: {e}")
                return False
    
    def load_state(self):
        with self.lock:
            try:
                if os.path.exists(self.state_file):
                    with open(self.state_file, 'r') as f:
                        state = json.load(f)
                    return self._deserialize_state(state)
                return self._get_initial_state()
            except Exception as e:
                print(f"❌ ERROR cargando estado: {e}")
                return self._get_initial_state()
    
    def _json_serializer(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _deserialize_state(self, state):
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
        return {
            'cash_balance': 255.0,
            'position': 0,
            'position_side': '',
            'entry_price': 0,
            'positions_history': [],
            'open_positions': 0,
            'log_messages': ["🤖 SISTEMA INICIADO - Estado persistente activado"],
            'tick_data': [],
            'is_running': False,
            'total_profit': 0,
            'start_time': datetime.now().isoformat()
        }

class MexcFuturesTradingBot:
    def __init__(self, api_key: str, secret_key: str, symbol: str = 'BTCUSDT'):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.base_url = 'https://api.mexc.com'
        
        # Configuración simplificada - SOLO BTC por ahora
        self.symbols_to_trade = ["BTCUSDT"]  # Solo BTC hasta que funcione
        self.current_symbol_index = 0
        self.initial_capital = 255.0
        
        # Sistema de persistencia
        self.persistence = PersistentStateManager()
        self._state = self.persistence.load_state()
        
        # Configuración CONSERVADORA
        self.leverage = 3
        self.position_size_base = 0.03  # 3% base - MUCHO MÁS CONSERVADOR
        self.max_positions = 1  # Solo 1 posición a la vez
        self.momentum_threshold = 0.0012
        self.mean_reversion_threshold = 0.001
        self.volatility_multiplier = 1.8
        self.min_profit_target = 0.0015
        self.max_loss_stop = 0.0020
        
        self.trading_thread = None
        self._running = False
        
        # Inicializar datos
        self._current_tick_data = deque(maxlen=100)
        if self._state['tick_data']:
            self._current_tick_data.extend(self._state['tick_data'])
        
        force_sync_position_state(self)
    
    def calculate_dynamic_position_size(self):
        """Tamaño de posición CONSERVADOR"""
        current_capital = self.cash_balance + self.total_profit
        if current_capital > self.initial_capital * 1.10:  # +10% para activar
            growth_factor = min(1.3, current_capital / self.initial_capital)
            dynamic_size = self.position_size_base * growth_factor
            return min(dynamic_size, 0.05)  # Máximo 5%
        return self.position_size_base
    
    def get_next_symbol(self):
        self.current_symbol_index = (self.current_symbol_index + 1) % len(self.symbols_to_trade)
        return self.symbols_to_trade[self.current_symbol_index]
    
    def _auto_save(self):
        try:
            self._state['tick_data'] = list(self._current_tick_data)
            threading.Thread(
                target=self.persistence.save_state, 
                args=(self._state,),
                daemon=True
            ).start()
        except:
            pass
    
    # Propiedades (mantener igual)
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
        return deque(self._state['tick_data'], maxlen=100)
    
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

    def log_message(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_messages.append(log_entry)
        if len(self.log_messages) > 100:
            self.log_messages.pop(0)
        self._auto_save()

    def get_futures_price(self) -> dict:
        try:
            url = "https://api.mexc.com/api/v3/ticker/price"
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
            self.log_message(f"Error obteniendo precio: {e}", "ERROR")
            return self.get_backup_price()

    def get_backup_price(self) -> dict:
        base_prices = {'BTCUSDT': 103000, 'ETHUSDT': 3500}
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
            'source': 'Simulation'
        }

    def calculate_indicators(self) -> dict:
        if len(self._current_tick_data) < 10:
            return {}
        
        prices = [tick['bid'] for tick in list(self._current_tick_data)]
        df = pd.DataFrame(prices, columns=['price'])
        
        # Indicadores básicos
        df['returns'] = df['price'].pct_change()
        df['momentum'] = df['returns'].rolling(2).mean()
        df['sma_3'] = df['price'].rolling(3).mean()
        df['sma_8'] = df['price'].rolling(8).mean()
        df['volatility'] = df['returns'].rolling(5).std() * self.volatility_multiplier
        
        # RSI rápido
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(3).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(3).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        latest = df.iloc[-1]
        return {
            'momentum': latest['momentum'],
            'current_price': latest['price'],
            'sma_3': latest['sma_3'],
            'sma_8': latest['sma_8'],
            'rsi': latest['rsi'],
            'volatility': latest['volatility']
        }

    def trading_strategy(self, indicators: dict) -> str:
        if not indicators:
            return 'hold'
        
        momentum = indicators['momentum']
        rsi = indicators['rsi']
        current_price = indicators['current_price']
        
        # Gestión de posición existente
        if self.position > 0:
            if self.position_side == 'long':
                profit_pct = (current_price - self.entry_price) / self.entry_price
                loss_pct = (self.entry_price - current_price) / self.entry_price
            else:
                profit_pct = (self.entry_price - current_price) / self.entry_price
                loss_pct = (current_price - self.entry_price) / self.entry_price
            
            if profit_pct >= self.min_profit_target:
                self.log_message(f"💰 GANANCIA: {profit_pct:.3%}", "PROFIT")
                return 'close'
            if loss_pct >= self.max_loss_stop:
                self.log_message(f"💸 PÉRDIDA: {loss_pct:.3%}", "LOSS")
                return 'close'
        
        # Señales de entrada
        if momentum > 0.0003 and rsi > 25 and self.open_positions < self.max_positions:
            return 'sell'
        elif momentum < -0.0003 and rsi < 75 and self.open_positions < self.max_positions:
            return 'buy'
        
        return 'hold'

    def execute_trade(self, action: str, price: float):
        """VERSIÓN CORREGIDA Y SEGURA"""
        try:
            # Tamaño CONSERVADOR
            position_size = self.calculate_dynamic_position_size()
            
            if action in ['buy', 'sell'] and self.open_positions < self.max_positions:
                # Cálculo SEGURO
                investment = self.cash_balance * position_size * self.leverage
                required_margin = investment / self.leverage
                
                if required_margin > self.cash_balance * 0.9:
                    self.log_message("❌ MARGEN INSUFICIENTE", "ERROR")
                    return
                
                quantity = investment / price
                
                # Actualizar estado
                self.cash_balance -= required_margin
                self.position = quantity
                self.entry_price = price
                self.position_side = 'long' if action == 'buy' else 'short'
                self.open_positions = 1
                
                side_emoji = "🟢" if action == 'buy' else "🔴"
                self.log_message(f"{side_emoji} ABRIR: {quantity:.6f} {self.symbol} @ ${price:.2f} | Size: {position_size*100:.1f}%", "TRADE")
                
            elif action == 'close' and self.position > 0:
                # Cerrar posición
                quantity = self.position
                
                if self.position_side == 'long':
                    close_value = quantity * price
                    profit_loss = (close_value - (quantity * self.entry_price)) * self.leverage
                else:
                    close_value = quantity * price
                    profit_loss = ((quantity * self.entry_price) - close_value) * self.leverage
                
                # Actualizar balances
                self.cash_balance += (quantity * self.entry_price / self.leverage) + profit_loss
                self.position = 0
                self.open_positions = 0
                self.position_side = ''
                self.total_profit += profit_loss
                
                emoji = "💰" if profit_loss > 0 else "💸"
                self.log_message(f"{emoji} CERRAR: P/L ${profit_loss:.4f} | Total: ${self.total_profit:.2f}", "TRADE")
            
            # Registrar en historial
            current_equity = self.cash_balance + (self.position * self.entry_price / self.leverage if self.position > 0 else 0)
            self.positions_history.append({
                'timestamp': datetime.now(),
                'action': action,
                'side': self.position_side if action != 'close' else '',
                'leverage': self.leverage,
                'price': price,
                'quantity': quantity,
                'cash_balance': self.cash_balance,
                'total_equity': current_equity,
                'profit_loss': profit_loss if action == 'close' else 0
            })
            
            self._auto_save()
            
        except Exception as e:
            self.log_message(f"❌ ERROR en trade: {e}", "ERROR")

    def close_all_positions(self):
        if self.position > 0:
            tick_data = self.get_futures_price()
            price = tick_data['bid'] if self.position_side == 'long' else tick_data['ask']
            self.execute_trade('close', price)
            self.log_message("🛑 TODAS las posiciones cerradas", "INFO")
        force_sync_position_state(self)

    def reset_account(self):
        self.cash_balance = 255.0
        self.position = 0
        self.position_side = ''
        self.entry_price = 0
        self.positions_history.clear()
        self.open_positions = 0
        self.log_messages.clear()
        self._current_tick_data.clear()
        self._state['tick_data'] = []
        self.total_profit = 0
        self.log_message("🔄 Cuenta reiniciada a $255.00", "INFO")
        self._auto_save()

    def trading_cycle(self):
        self.log_message("🚀 INICIANDO BOT - MODO CONSERVADOR")
        
        consecutive_errors = 0
        while self.is_running:
            try:
                self.symbol = self.get_next_symbol()
                tick_data = self.get_futures_price()
                if tick_data:
                    self._current_tick_data.append(tick_data)
                
                indicators = self.calculate_indicators()
                if indicators and len(self._current_tick_data) >= 10:
                    signal = self.trading_strategy(indicators)
                    if signal != 'hold':
                        price = tick_data['ask'] if signal == 'buy' else tick_data['bid']
                        self.execute_trade(signal, price)
                
                consecutive_errors = 0
                time.sleep(1)  # Más lento para seguridad
                
            except Exception as e:
                consecutive_errors += 1
                self.log_message(f"❌ ERROR: {e}", "ERROR")
                if consecutive_errors >= 3:
                    self.close_all_positions()
                    self.is_running = False
                    break
                time.sleep(5)

    def start_trading(self):
        if not self.is_running:
            force_sync_position_state(self)
            self.is_running = True
            self._running = True
            self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
            self.trading_thread.start()
            self.log_message("✅ BOT ACTIVADO", "SYSTEM")

    def stop_trading(self):
        self.is_running = False
        self._running = False
        self.log_message("🛑 BOT DETENIDO", "SYSTEM")
        self._auto_save()
        force_sync_position_state(self)

    def get_performance_stats(self):
        current_data = list(self._current_tick_data) if self._current_tick_data else []
        current_price = current_data[-1]['bid'] if current_data else 0
        
        if self.position > 0:
            if self.position_side == 'long':
                unrealized_pl = (self.position * (current_price - self.entry_price)) * self.leverage
            else:
                unrealized_pl = (self.position * (self.entry_price - current_price)) * self.leverage
            total_equity = self.cash_balance + unrealized_pl
        else:
            total_equity = self.cash_balance
        
        stats = {
            'total_trades': len([p for p in self.positions_history if p['action'] in ['buy', 'sell']]),
            'win_rate': 0,
            'cash_balance': self.cash_balance,
            'total_equity': total_equity,
            'open_positions': self.open_positions,
            'current_price': current_price,
            'total_profit': total_equity - 255.0,
            'position_size': self.position,
            'position_side': self.position_side,
            'realized_profit': self.total_profit,
            'leverage': self.leverage,
            'current_symbol': self.symbol,
            'dynamic_position_size': self.calculate_dynamic_position_size() * 100
        }
        
        close_trades = [t for t in self.positions_history if t['action'] == 'close']
        if close_trades:
            profitable = len([t for t in close_trades if t.get('profit_loss', 0) > 0])
            stats['win_rate'] = (profitable / len(close_trades)) * 100 if close_trades else 0
        
        return stats

# INTERFAZ STREAMLIT SIMPLIFICADA
def main():
    st.title("🤖 Bot Trading MEXC - MODO SEGURO")
    st.markdown("**CAPITAL: $255.00 | LEVERAGE: 3x | MODO CONSERVADOR**")
    
    if 'bot' not in st.session_state:
        st.session_state.bot = MexcFuturesTradingBot("", "", "BTCUSDT")
    
    bot = st.session_state.bot
    
    with st.sidebar:
        st.header("🎛️ Control")
        api_key = st.text_input("API Key", type="password")
        secret_key = st.text_input("Secret Key", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Iniciar", type="primary"):
                bot.start_trading()
                st.rerun()
        with col2:
            if st.button("🛑 Detener"):
                bot.stop_trading()
                st.rerun()
        
        if st.button("🔒 Cerrar Posiciones"):
            bot.close_all_positions()
            st.rerun()
        if st.button("🔄 Reiniciar $255"):
            bot.reset_account()
            st.rerun()
        
        st.markdown("---")
        if bot.is_running:
            st.success("🟢 BOT ACTIVO")
        else:
            st.warning("🔴 BOT DETENIDO")

    # Métricas
    stats = bot.get_performance_stats()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Capital", f"${stats['cash_balance']:.2f}")
    with col2:
        st.metric("📈 Precio", f"${stats['current_price']:.2f}")
    with col3:
        st.metric("🎯 Win Rate", f"{stats['win_rate']:.1f}%")
    with col4:
        st.metric("📊 Operaciones", f"{stats['total_trades']}")
    
    # Pestañas
    tab1, tab2 = st.tabs(["📈 Operaciones", "📋 Logs"])
    
    with tab1:
        if bot.positions_history:
            df_data = []
            for pos in bot.positions_history:
                df_data.append({
                    'Hora': pos['timestamp'].strftime('%H:%M:%S') if isinstance(pos['timestamp'], datetime) else str(pos['timestamp']),
                    'Acción': pos['action'],
                    'Lado': pos.get('side', ''),
                    'Precio': f"${pos['price']:.2f}",
                    'Cantidad': f"{pos['quantity']:.6f}",
                    'P/L': f"${pos.get('profit_loss', 0):.4f}"
                })
            st.dataframe(pd.DataFrame(df_data))
        else:
            st.info("No hay operaciones")
    
    with tab2:
        for log in reversed(bot.log_messages[-20:]):
            if "ERROR" in log:
                st.error(log)
            elif "PROFIT" in log:
                st.success(log)
            elif "LOSS" in log:
                st.error(log)
            else:
                st.info(log)
    
    if bot.is_running:
        time.sleep(3)
        st.rerun()

if __name__ == "__main__":
    main()
