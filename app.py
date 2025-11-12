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
    page_title="?? Bot HFT Futuros MEXC - ESTRATEGIA ULTRA AGRESIVA",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PersistentStateManager:
    """Gestor de estado INDEPENDIENTE de Streamlit - VERSIÓN SIMPLIFICADA"""
    
    def __init__(self, state_file='bot_persistent_state.json'):
        self.state_file = state_file
        self.lock = threading.Lock()
    
    def save_state(self, state_data):
        """Guardar estado de forma atómica"""
        with self.lock:
            try:
                # Crear copia del estado
                state_copy = self._deep_copy_state(state_data)
                
                # Guardar en temporal primero
                temp_file = f"{self.state_file}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(state_copy, f, default=self._json_serializer, indent=2)
                
                # Mover atómicamente
                os.replace(temp_file, self.state_file)
                return True
                
            except Exception as e:
                print(f"?? ERROR guardando estado: {e}")
                return False
    
    def load_state(self):
        """Cargar estado con recuperación"""
        with self.lock:
            try:
                # Intentar archivo principal
                if os.path.exists(self.state_file):
                    with open(self.state_file, 'r') as f:
                        state = json.load(f)
                    return self._deserialize_state(state)
                
                # Estado inicial
                return self._get_initial_state()
                
            except Exception as e:
                print(f"?? ERROR cargando estado: {e}")
                return self._get_initial_state()
    
    def _deep_copy_state(self, state):
        """Copia simple del estado"""
        return json.loads(json.dumps(state, default=self._json_serializer))
    
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
            # posiciones por símbolo -> cada entrada: {'position':0,'position_side':'','entry_price':0}
            'positions': {},
            'positions_history': [],
            'open_positions': 0,
            'log_messages': ["?? SISTEMA INICIADO - Estado persistente activado"],
            # tick_data por símbolo (listas)
            'tick_data': {},
            'is_running': False,
            'total_profit': 0,
            'start_time': datetime.now().isoformat(),
            # para interés compuesto controlado
            'base_position_size': 0.12
        }

class MexcFuturesTradingBot:
    def __init__(self, api_key: str, secret_key: str, symbol: str = 'BTCUSDT'):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol  # símbolo principal mostrado en UI
        self.base_url = 'https://api.mexc.com'
        
        # SISTEMA DE PERSISTENCIA SIMPLIFICADO
        self.persistence = PersistentStateManager()
        self._state = self.persistence.load_state()
        
        # Símbolos a operar simultáneamente: siempre símbolo principal + ETHUSDT
        self.symbols = []
        if self.symbol.upper() not in self.symbols:
            self.symbols.append(self.symbol.upper())
        if 'ETHUSDT' not in self.symbols:
            self.symbols.append('ETHUSDT')
        
        # Inicializar estructuras por símbolo
        # posiciones: si no existe en el estado, configurar por símbolo
        for s in self.symbols:
            if s not in self._state.get('positions', {}):
                self._state.setdefault('positions', {})[s] = {'position': 0.0, 'position_side': '', 'entry_price': 0.0}
            if s not in self._state.get('tick_data', {}):
                self._state.setdefault('tick_data', {})[s] = []
        
        # Convertir tick_data del estado en deques por símbolo
        self._current_tick_data = {}
        for s, lst in self._state.get('tick_data', {}).items():
            self._current_tick_data[s] = deque(lst, maxlen=100)
        
        # Configuración del bot (MANTENER IGUAL)
        self.leverage = 3
        # ahora base_position_size persistido en estado y position_size calculada desde base + compound
        self.base_position_size = float(self._state.get('base_position_size', 0.12))
        self.max_position_size = 0.35  # tope para evitar sizing extremo (ajustable)
        # position_size inicial calculada desde base
        self.position_size = self.base_position_size
        
        self.max_positions = 2
        self.momentum_threshold = 0.0012
        self.mean_reversion_threshold = 0.001
        self.volatility_multiplier = 1.8
        self.min_profit_target = 0.0015
        self.max_loss_stop = 0.0020
        
        self.trading_thread = None
        self._running = False
        
        self._state.setdefault('total_profit', 0.0)
        self.initial_capital = 255.0  # referencia para compound
        
    def _auto_save(self):
        """Guardado automático simple"""
        try:
            # Actualizar tick_data antes de guardar
            # convertir deques a listas
            self._state['tick_data'] = {s: list(d) for s, d in self._current_tick_data.items()}
            # guardar base_position_size también
            self._state['base_position_size'] = self.base_position_size
            # Guardar en hilo separado
            threading.Thread(
                target=self.persistence.save_state, 
                args=(self._state,),
                daemon=True
            ).start()
        except:
            pass

    # Accessors para posiciones por símbolo
    def get_symbol_position(self, sym):
        return float(self._state['positions'].get(sym, {}).get('position', 0.0))
    def set_symbol_position(self, sym, value):
        self._state['positions'].setdefault(sym, {})['position'] = float(value)
        self._auto_save()
    def get_symbol_side(self, sym):
        return self._state['positions'].get(sym, {}).get('position_side', '')
    def set_symbol_side(self, sym, value):
        self._state['positions'].setdefault(sym, {})['position_side'] = value
        self._auto_save()
    def get_symbol_entry_price(self, sym):
        return float(self._state['positions'].get(sym, {}).get('entry_price', 0.0))
    def set_symbol_entry_price(self, sym, value):
        self._state['positions'].setdefault(sym, {})['entry_price'] = float(value)
        self._auto_save()

    @property
    def cash_balance(self):
        return self._state['cash_balance']
    
    @cash_balance.setter
    def cash_balance(self, value):
        self._state['cash_balance'] = value
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
    def is_running(self):
        return self._state['is_running']
    
    @is_running.setter
    def is_running(self, value):
        self._state['is_running'] = value
        self._auto_save()
    
    @property
    def total_profit(self):
        return float(self._state.get('total_profit', 0.0))
    
    @total_profit.setter
    def total_profit(self, value):
        self._state['total_profit'] = float(value)
        self._auto_save()

    def log_message(self, message: str, level: str = "INFO"):
        """Agregar mensaje al log con persistencia"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_messages.append(log_entry)
        if len(self.log_messages) > 200:
            self.log_messages.pop(0)
        self._auto_save()

    def get_futures_price(self, symbol: str) -> dict:
        """Obtener precio de FUTUROS MEXC para un símbolo dado"""
        try:
            url = f"https://api.mexc.com/api/v3/ticker/price"
            params = {'symbol': symbol}
            
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
                        'symbol': symbol,
                        'simulated': False,
                        'source': 'MEXC Futures'
                    }
            
            return self.get_backup_price(symbol)
            
        except Exception as e:
            self.log_message(f"Error obteniendo precio futuros {symbol}: {e}", "ERROR")
            return self.get_backup_price(symbol)

    def get_backup_price(self, symbol: str) -> dict:
        """Precio de backup para futuros (por símbolo)"""
        base_prices = {
            'BTCUSDT': 103000,
            'ETHUSDT': 3500,
            'ADAUSDT': 0.45,
            'DOTUSDT': 7.5,
            'LINKUSDT': 15.0
        }
        
        base_price = base_prices.get(symbol, 103000)
        variation = np.random.uniform(-0.01, 0.01)
        current_price = base_price * (1 + variation)
        spread = current_price * 0.00005
        
        return {
            'timestamp': datetime.now(),
            'bid': current_price - spread,
            'ask': current_price + spread,
            'symbol': symbol,
            'simulated': True,
            'source': 'Futures Simulation'
        }

    def calculate_indicators_for_ticks(self, ticks_deque: deque) -> dict:
        """Calcular indicadores técnicos OPTIMIZADOS PARA HFT para un deque de ticks"""
        if len(ticks_deque) < 10:
            return {}
        
        prices = [tick['bid'] for tick in list(ticks_deque)]
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

    def update_position_size_from_profit(self):
        """
        Interés compuesto controlado: ajustar position_size en función de ganancias acumuladas.
        Fórmula simple y gradual:
            position_size = min(max_position_size, base_position_size * (1 + total_profit / initial_capital))
        """
        try:
            gain = max(0.0, float(self.total_profit))
            factor = 1.0 + (gain / max(1.0, self.initial_capital))
            new_size = self.base_position_size * factor
            # evitar cambios bruscos: limitar incremento por ciclo
            # (mínimo cambio: 0, máximo: self.max_position_size)
            self.position_size = min(self.max_position_size, float(new_size))
            # persistir base en el estado (por si deseas cambiar más tarde desde UI)
            self._state['base_position_size'] = self.base_position_size
            self._auto_save()
        except Exception as e:
            self.log_message(f"ERROR actualizando position_size por compound: {e}", "ERROR")

    def trading_strategy(self, indicators: dict, symbol: str) -> str:
        """ESTRATEGIA ULTRA AGRESIVA - MÁXIMAS OPERACIONES (adaptada por símbolo)"""
        if not indicators:
            return 'hold'
        
        momentum = indicators['momentum']
        rsi = indicators['rsi']
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        current_price = indicators['current_price']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        
        # obtener estado de la posición para este símbolo
        pos_qty = self.get_symbol_position(symbol)
        pos_side = self.get_symbol_side(symbol)
        entry_price = self.get_symbol_entry_price(symbol)
        
        # GESTIÓN DE POSICIÓN ULTRA AGRESIVA (por símbolo)
        if pos_qty > 0:
            if pos_side == 'long':
                current_profit_pct = (current_price - entry_price) / entry_price if entry_price else 0
                current_loss_pct = (entry_price - current_price) / entry_price if entry_price else 0
            else:  # short
                current_profit_pct = (entry_price - current_price) / entry_price if entry_price else 0
                current_loss_pct = (current_price - entry_price) / entry_price if entry_price else 0
            
            # TOMA DE GANANCIAS ULTRA RÁPIDA
            if current_profit_pct >= self.min_profit_target:
                self.log_message(f"?? GANANCIA TURBO {symbol}: {current_profit_pct:.3%} (+)", "PROFIT")
                return 'close'
            
            # STOP LOSS PROTECTOR
            if current_loss_pct >= self.max_loss_stop:
                self.log_message(f"?? STOP LOSS TURBO {symbol}: {current_loss_pct:.3%} (-)", "LOSS")
                return 'close'
        
        # SEÑALES ULTRA AGRESIVAS - MÁXIMAS ENTRADAS
        short_conditions = [
            momentum > 0.0003,
            rsi > 25,
            current_price > bb_lower,
            macd > macd_signal * 0.8,
        ]
        
        long_conditions = [
            momentum < -0.0003,
            rsi < 75,
            current_price < bb_upper,
            macd < macd_signal * 1.2,
        ]
        
        if sum(short_conditions) >= 2 and self.open_positions < self.max_positions:
            self.log_message(f"? SEÑAL SHORT TURBO {symbol}: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'sell'
        elif sum(long_conditions) >= 2 and self.open_positions < self.max_positions:
            self.log_message(f"? SEÑAL LONG TURBO {symbol}: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'buy'
        
        return 'hold'

    def execute_trade(self, action: str, price: float, symbol: str):
        """Ejecutar operación en FUTUROS - MODO TURBO (por símbolo)"""
        try:
            investment_amount = 0
            quantity = 0
            quantity_to_close = 0
            close_amount = 0
            profit_loss = 0
            
            pos_qty = self.get_symbol_position(symbol)
            pos_side = self.get_symbol_side(symbol)
            entry_price = self.get_symbol_entry_price(symbol)
            
            if action in ['buy', 'sell'] and self.open_positions < self.max_positions:
                # ABRIR POSICIÓN para este símbolo
                investment_amount = self.cash_balance * self.position_size * self.leverage
                quantity = investment_amount / price
                
                if investment_amount > self.cash_balance * self.leverage:
                    self.log_message(f"? Margen insuficiente para {symbol}", "ERROR")
                    return
                
                # Actualizar balances
                self.cash_balance -= (investment_amount / self.leverage)
                
                # actualizar posición por símbolo (acumulativa)
                new_total_qty = pos_qty + quantity
                new_entry = price if pos_qty == 0 else ((entry_price * pos_qty) + (price * quantity)) / new_total_qty
                self.set_symbol_position(symbol, new_total_qty)
                self.set_symbol_entry_price(symbol, new_entry)
                self.set_symbol_side(symbol, 'long' if action == 'buy' else 'short')
                self.open_positions += 1
                
                side_emoji = "??" if action == 'buy' else "??"
                trade_info = f"{side_emoji} TURBO {symbol} { 'LONG' if action=='buy' else 'SHORT' }: {quantity:.6f} @ ${price:.2f} | Margen: ${investment_amount/self.leverage:.2f} | Lev: {self.leverage}x"
                self.log_message(trade_info, "TRADE")
                
            elif action == 'close' and pos_qty > 0:
                # CERRAR POSICIÓN para este símbolo
                quantity_to_close = pos_qty
                
                if pos_side == 'long':
                    close_amount = quantity_to_close * price
                    profit_loss = (close_amount - (pos_qty * entry_price)) * self.leverage
                else:  # short
                    close_amount = quantity_to_close * price
                    profit_loss = ((pos_qty * entry_price) - close_amount) * self.leverage
                
                # Actualizar balances
                self.cash_balance += (pos_qty * entry_price / self.leverage) + profit_loss
                # reset símbolo
                self.set_symbol_position(symbol, 0)
                self.set_symbol_side(symbol, '')
                self.set_symbol_entry_price(symbol, 0)
                self.open_positions = max(0, self.open_positions - 1)
                # acumular profit
                self.total_profit = self.total_profit + profit_loss
                
                profit_color = "??" if profit_loss > 0 else "??"
                trade_info = f"{profit_color} CERRAR TURBO {symbol}: {quantity_to_close:.6f} @ ${price:.2f} | P/L: ${profit_loss:.4f} | Profit Total: ${self.total_profit:.2f}"
                self.log_message(trade_info, "TRADE")
            
            # Registrar posición (registro por símbolo)
            # calcular equity aproximada (considerando posiciones abiertas)
            total_unrealized = 0
            for s in self.symbols:
                sq = self.get_symbol_position(s)
                s_entry = self.get_symbol_entry_price(s)
                if sq > 0:
                    # usar último precio si existe en tick data
                    last_price = list(self._current_tick_data.get(s, deque()))[-1]['bid'] if self._current_tick_data.get(s) else 0
                    if self.get_symbol_side(s) == 'long':
                        unreal = (last_price - s_entry) * sq * self.leverage
                    else:
                        unreal = (s_entry - last_price) * sq * self.leverage
                    total_unrealized += unreal
            current_equity = self.cash_balance + total_unrealized
            
            self.positions_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'side': self.get_symbol_side(symbol) if action != 'close' else '',
                'leverage': self.leverage if action != 'close' else '',
                'price': price,
                'quantity': quantity_to_close if action == 'close' else quantity,
                'cash_balance': self.cash_balance,
                'total_equity': current_equity,
                'profit_loss': profit_loss if action == 'close' else 0
            })
            
            # actualizar position_size en base a profit (compound controlado)
            self.update_position_size_from_profit()
            self._auto_save()
            
        except Exception as e:
            self.log_message(f"?? ERROR ejecutando trade {symbol}: {e}", "ERROR")

    def close_all_positions(self):
        """Cerrar todas las posiciones abiertas (para todos los símbolos)"""
        for s in list(self.symbols):
            if self.get_symbol_position(s) > 0:
                tick_data = self.get_futures_price(s)
                price = tick_data['bid'] if self.get_symbol_side(s) == 'long' else tick_data['ask']
                self.execute_trade('close', price, s)
        self.log_message("?? TODAS las posiciones cerradas - MODO SEGURIDAD", "INFO")

    def reset_account(self):
        """Reiniciar cuenta a estado inicial"""
        self.cash_balance = 255.0
        # reset posiciones por símbolo
        for s in self.symbols:
            self.set_symbol_position(s, 0)
            self.set_symbol_side(s, '')
            self.set_symbol_entry_price(s, 0)
        self.positions_history.clear()
        self.open_positions = 0
        self.log_messages.clear()
        for s in self.symbols:
            self._current_tick_data[s].clear()
            self._state['tick_data'][s] = []
        self.total_profit = 0
        self.base_position_size = 0.12
        self.position_size = self.base_position_size
        self.log_message("?? Cuenta reiniciada a $255.00 - MODO TURBO", "INFO")
        self._auto_save()

    def trading_cycle(self):
        """Ciclo principal de trading HFT ULTRA RÁPIDO - ahora por símbolo"""
        self.log_message("?? INICIANDO MODO TURBO HFT - PERSISTENCIA ACTIVA")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                # para cada símbolo, obtener tick y procesar indicadores/estrategia
                for s in self.symbols:
                    tick_data = self.get_futures_price(s)
                    if tick_data:
                        # asegurarse deque existe
                        if s not in self._current_tick_data:
                            self._current_tick_data[s] = deque(maxlen=100)
                        self._current_tick_data[s].append(tick_data)
                        # Auto-guardar cada 10 ticks por símbolo
                        if len(self._current_tick_data[s]) % 10 == 0:
                            self._auto_save()
                    
                    indicators = self.calculate_indicators_for_ticks(self._current_tick_data.get(s, deque()))
                    
                    if indicators and len(self._current_tick_data.get(s, deque())) >= 10:
                        signal = self.trading_strategy(indicators, s)
                        if signal != 'hold':
                            price = tick_data['ask'] if signal == 'buy' else tick_data['bid']
                            self.execute_trade(signal, price, s)
                
                consecutive_errors = 0
                # HFT: mantener latencia muy baja; sleep pequeño
                time.sleep(0.3)
                
            except Exception as e:
                consecutive_errors += 1
                self.log_message(f"?? ERROR en ciclo #{consecutive_errors}: {e}", "ERROR")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.log_message("?? DEMASIADOS ERRORES - Cerrando todas las posiciones", "CRITICAL")
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
            self.log_message("? MODO TURBO ACTIVADO - PERSISTENCIA GARANTIZADA", "SYSTEM")

    def stop_trading(self):
        """Detener bot de trading"""
        self.is_running = False
        self._running = False
        self.log_message("?? MODO TURBO DETENIDO", "SYSTEM")
        # Guardar estado final
        self._auto_save()

    def get_performance_stats(self, for_symbol: str = None):
        """Obtener estadísticas de performance (agregadas o por símbolo)"""
        # calcular equity considerando posiciones abiertas
        total_unrealized = 0
        total_position_size = 0
        for s in self.symbols:
            sq = self.get_symbol_position(s)
            s_entry = self.get_symbol_entry_price(s)
            if sq > 0:
                last_price = list(self._current_tick_data.get(s, deque()))[-1]['bid'] if self._current_tick_data.get(s) else 0
                if self.get_symbol_side(s) == 'long':
                    unreal = (last_price - s_entry) * sq * self.leverage
                else:
                    unreal = (s_entry - last_price) * sq * self.leverage
                total_unrealized += unreal
            total_position_size += sq
        
        total_equity = self.cash_balance + total_unrealized
        total_profit = total_equity - self.initial_capital
        
        stats = {
            'total_trades': len([p for p in self.positions_history if p['action'] in ['buy', 'sell']]),
            'win_rate': 0,
            'cash_balance': self.cash_balance,
            'total_equity': total_equity,
            'open_positions': self.open_positions,
            'current_price': (list(self._current_tick_data.get(for_symbol or self.symbol, deque()))[-1]['bid'] if self._current_tick_data.get(for_symbol or self.symbol) else 0),
            'total_profit': total_profit,
            'position_size': total_position_size,
            'position_side': self.get_symbol_side(for_symbol or self.symbol),
            'realized_profit': self.total_profit,
            'leverage': self.leverage,
            'computed_position_size': self.position_size,
            'base_position_size': self.base_position_size
        }
        
        if not self.positions_history:
            return stats
        
        # Calcular win rate
        close_trades = [t for t in self.positions_history if t['action'] == 'close']
        
        if close_trades:
            profitable_trades = len([t for t in close_trades if t.get('profit_loss', 0) > 0])
            stats['win_rate'] = (profitable_trades / len(close_trades)) * 100 if close_trades else 0
        
        return stats

# INTERFAZ STREAMLIT (MANTENER IGUAL)
def main():
    st.title("?? Bot HFT Futuros MEXC - PERSISTENCIA TOTAL ?")
    st.markdown("**CAPITAL INICIAL: $255.00 | APALANCAMIENTO: 3x | PERSISTENCIA ACTIVA**")
    st.markdown("---")
    
    # Inicializar el bot
    if 'bot' not in st.session_state:
        st.session_state.bot = MexcFuturesTradingBot("", "", "BTCUSDT")
    
    bot = st.session_state.bot
    
    # Sidebar (igual que antes) - el bot siempre operará BTCUSDT + ETHUSDT simultáneamente
    with st.sidebar:
        st.header("?? Configuración Turbo")
        
        api_key = st.text_input("API Key MEXC", type="password")
        secret_key = st.text_input("Secret Key MEXC", type="password")
        # permita seleccionar cuál símbolo mostrar en el panel principal (pero ETH se operará siempre)
        symbol = st.selectbox("Símbolo para visualizar", ["BTCUSDT", "ETHUSDT"])
        
        bot.api_key = api_key
        bot.secret_key = secret_key
        bot.symbol = symbol
        # asegurar que la lista de símbolos incluye el seleccionado y ETHUSDT
        bot.symbols = []
        if bot.symbol.upper() not in bot.symbols:
            bot.symbols.append(bot.symbol.upper())
        if 'ETHUSDT' not in bot.symbols:
            bot.symbols.append('ETHUSDT')
        # asegurarse que existen las estructuras internas
        for s in bot.symbols:
            if s not in bot._current_tick_data:
                bot._current_tick_data[s] = deque(maxlen=100)
            if s not in bot._state['positions']:
                bot._state['positions'][s] = {'position':0,'position_side':'','entry_price':0}
            if s not in bot._state['tick_data']:
                bot._state['tick_data'][s] = []
        
        st.markdown("---")
        st.header("?? Control Turbo")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("?? Activar Turbo", use_container_width=True, type="primary"):
                bot.start_trading()
                st.rerun()
        
        with col2:
            if st.button("?? Desactivar Turbo", use_container_width=True):
                bot.stop_trading()
                st.rerun()
        
        if st.button("?? Cerrar Posiciones", use_container_width=True):
            bot.close_all_positions()
            st.rerun()
            
        if st.button("?? Reiniciar $255", use_container_width=True):
            bot.reset_account()
            st.rerun()
        
        st.markdown("---")
        st.header("?? Configuración Turbo")
        st.info(f"**Tamaño posición base:** {bot.base_position_size*100:.2f}%")
        st.info(f"**Tamaño posición actual (compound):** {bot.position_size*100:.2f}%")
        st.info(f"**Target ganancia:** {bot.min_profit_target*100}%")
        st.info(f"**Stop loss:** {bot.max_loss_stop*100}%")
        st.info(f"**Apalancamiento:** {bot.leverage}x")
        st.info(f"**Operaciones máx:** {bot.max_positions}")
        
        if bot.is_running:
            st.success("? TURBO ACTIVO - PERSISTENCIA ACTIVA")
        else:
            st.warning("?? SISTEMA EN STANDBY")
            
        # mostrar fuente del último tick del símbolo visible
        if bot._current_tick_data.get(bot.symbol):
            latest_tick = list(bot._current_tick_data[bot.symbol])[-1]
            source = latest_tick.get('source', 'Unknown')
            st.info(f"**Fuente (último):** {source}")

    # Layout principal (igual que antes)
    col1, col2, col3, col4 = st.columns(4)
    
    stats = bot.get_performance_stats(for_symbol=bot.symbol)
    
    with col1:
        st.metric(
            label="?? Margen Disponible",
            value=f"${stats['cash_balance']:.2f}",
            delta=f"${stats['realized_profit']:.2f}" if stats['realized_profit'] != 0 else None
        )
    
    with col2:
        st.metric(
            label="?? Precio Futuros",
            value=f"${stats['current_price']:.2f}"
        )
    
    with col3:
        st.metric(
            label="?? Tasa Acierto",
            value=f"{stats['win_rate']:.1f}%"
        )
    
    with col4:
        st.metric(
            label="?? Operaciones",
            value=f"{stats['total_trades']}"
        )
    
    # Segunda fila de métricas
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        position_status = f"{stats['position_side'].upper()}" if stats['position_side'] else "SIN POSICIÓN"
        st.metric(
            label="?? Posición Actual (símbolo visible)",
            value=position_status
        )
    
    with col6:
        st.metric(
            label="?? Tamaño Pos total",
            value=f"{stats['position_size']:.6f}"
        )
    
    with col7:
        st.metric(
            label="?? Equity Total",
            value=f"${stats['total_equity']:.2f}"
        )
    
    with col8:
        leverage_info = f"{stats['leverage']}x" if stats['position_side'] else "---"
        st.metric(
            label="? Apalancamiento",
            value=leverage_info
        )
    
    st.markdown("---")
    
    # Gráficos y datos
    tab1, tab2, tab3 = st.tabs(["?? Precios Futuros", "?? Operaciones Turbo", "?? Logs del Sistema"])
    
    with tab1:
        # mostrar gráfico del símbolo seleccionado (bot.symbol), pero datos están llegando para BTC y ETH
        current_data = list(bot._current_tick_data.get(bot.symbol, []))
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
                
                # Mostrar posición actual del símbolo visible si existe
                if bot.get_symbol_position(bot.symbol) > 0 and bot.get_symbol_entry_price(bot.symbol) > 0:
                    fig.add_hline(
                        y=bot.get_symbol_entry_price(bot.symbol), 
                        line_dash="dash", 
                        line_color="yellow",
                        annotation_text=f"Entrada: ${bot.get_symbol_entry_price(bot.symbol):.2f}"
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
        
        # adicional: mostrar mini-últimos precios de ETH para visibilidad
        st.markdown("---")
        st.subheader("Últimos ticks ETHUSDT (resumen)")
        eth_ticks = list(bot._current_tick_data.get('ETHUSDT', []))[-5:]
        if eth_ticks:
            eth_df = pd.DataFrame([{'time': t['timestamp'], 'bid': t['bid'], 'ask': t['ask'], 'source': t.get('source','')} for t in eth_ticks])
            st.dataframe(eth_df, use_container_width=True)
        else:
            st.info("Sin ticks ETH aún.")
    
    with tab2:
        if bot.positions_history:
            # Crear DataFrame para futuros
            display_data = []
            for pos in bot.positions_history:
                row = {
                    'timestamp': pos['timestamp'].strftime('%H:%M:%S') if isinstance(pos['timestamp'], datetime) else str(pos['timestamp']),
                    'symbol': pos.get('symbol',''),
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
            st.info("No hay operaciones turbo registradas aún.")
    
    with tab3:
        log_container = st.container(height=400)
        with log_container:
            for log_entry in reversed(bot.log_messages[-100:]):
                if "ERROR" in log_entry or "CRITICAL" in log_entry:
                    st.error(log_entry)
                elif "TRADE" in log_entry:
                    if "ABRIR" in log_entry:
                        st.success(log_entry)
                    elif "CERRAR" in log_entry:
                        if "??" in log_entry:
                            st.error(log_entry)
                        else:
                            st.success(log_entry)
                    else:
                        st.info(log_entry)
                elif "SEÑAL" in log_entry or "PROFIT" in log_entry or "LOSS" in log_entry:
                    st.warning(log_entry)
                elif "SYSTEM" in log_entry:
                    st.info(log_entry)
                else:
                    st.info(log_entry)
    
    # Auto-refresh
    if bot.is_running:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
