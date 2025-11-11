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
    page_title="🤖 Bot HFT Futuros MEXC - ESTRATEGIA NUCLEAR",
    page_icon="🤖",
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
        
        # CONFIGURACIÓN HFT FUTUROS - ESTRATEGIA NUCLEAR
        self.leverage = 3  # Apalancamiento conservador
        self.position_size = 0.08  # 8% del capital por operación
        self.max_positions = 1     # 1 operación a la vez
        self.momentum_threshold = 0.0012
        self.mean_reversion_threshold = 0.001
        self.volatility_multiplier = 1.8
        
        # TARGETS OPTIMIZADOS - ESTRATEGIA NUCLEAR
        self.min_profit_target = 0.0025  # 0.25% target
        self.max_loss_stop = 0.0015     # 0.15% stop loss
        
        self.trading_thread = None
        self.last_save_time = time.time()
        
    def create_backup_state(self):
        """Crear estado de respaldo seguro"""
        self.bot_data = {
            'cash_balance': 255.0,
            'position': 0,
            'position_side': '',
            'entry_price': 0,
            'positions_history': [],
            'open_positions': 0,
            'log_messages': ["🔄 SISTEMA REINICIADO - Estado de respaldo activado"],
            'tick_data': deque(maxlen=100),
            'is_running': False,
            'total_profit': 0
        }
        self.save_state()
        
    def save_state(self):
        """Guardar estado con protección contra corrupción"""
        try:
            state = {
                'cash_balance': self.cash_balance,
                'position': self.position,
                'position_side': self.position_side,
                'entry_price': self.entry_price,
                'positions_history': [],
                'open_positions': self.open_positions,
                'log_messages': self.log_messages,
                'tick_data': [],
                'is_running': self.is_running,
                'total_profit': self.total_profit,
                'last_update': datetime.now().isoformat()
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
            
            # Guardar en archivo temporal primero
            temp_file = 'futures_bot_state_temp.json'
            with open(temp_file, 'w') as f:
                json.dump(state, f, default=str, indent=2)
            
            # Reemplazar archivo principal de forma atómica
            os.replace(temp_file, 'futures_bot_state.json')
            
            # Guardar backup cada 5 minutos
            current_time = time.time()
            if current_time - self.last_save_time > 300:  # 5 minutos
                backup_file = f'futures_bot_state_backup_{int(current_time)}.json'
                with open(backup_file, 'w') as f:
                    json.dump(state, f, default=str, indent=2)
                self.last_save_time = current_time
                
        except Exception as e:
            print(f"🚨 ERROR CRÍTICO guardando estado: {e}")
            # Intentar guardar en archivo de emergencia
            try:
                with open('futures_bot_state_emergency.json', 'w') as f:
                    json.dump({'error': str(e), 'timestamp': datetime.now().isoformat()}, f)
            except:
                pass
    
    def load_state(self):
        """Cargar estado con recuperación de errores"""
        try:
            if os.path.exists('futures_bot_state.json'):
                with open('futures_bot_state.json', 'r') as f:
                    state = json.load(f)
                
                # VALIDAR ESTRUCTURA CRÍTICA
                if 'cash_balance' not in state:
                    raise ValueError("Estado corrupto - falta cash_balance")
                
                # Convertir deque de tick_data
                tick_data = deque(maxlen=100)
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
                    'cash_balance': state.get('cash_balance', 255.0),
                    'position': state.get('position', 0),
                    'position_side': state.get('position_side', ''),
                    'entry_price': state.get('entry_price', 0),
                    'positions_history': positions_history,
                    'open_positions': state.get('open_positions', 0),
                    'log_messages': state.get('log_messages', []),
                    'tick_data': tick_data,
                    'is_running': state.get('is_running', False),
                    'total_profit': state.get('total_profit', 0)
                }
                
                self.log_message("✅ Estado cargado correctamente - ESTRATEGIA NUCLEAR ACTIVA", "SYSTEM")
                
            else:
                self.create_backup_state()
                
        except Exception as e:
            print(f"🚨 ERROR cargando estado: {e}")
            # Intentar cargar backup
            try:
                backup_files = [f for f in os.listdir('.') if f.startswith('futures_bot_state_backup_')]
                if backup_files:
                    latest_backup = max(backup_files)
                    with open(latest_backup, 'r') as f:
                        state = json.load(f)
                    self.log_message(f"🔄 Recuperado desde backup: {latest_backup}", "SYSTEM")
                else:
                    self.create_backup_state()
            except:
                self.create_backup_state()
    
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
        if len(self.log_messages) > 100:
            self.log_messages.pop(0)
        self.save_state()

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
            'BTCUSDT': 106000,
            'ETHUSDT': 3500,
            'ADAUSDT': 0.45,
            'DOTUSDT': 7.5,
            'LINKUSDT': 15.0
        }
        
        base_price = base_prices.get(self.symbol, 106000)
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
        if len(self.tick_data) < 15:
            return {}
        
        prices = [tick['bid'] for tick in self.tick_data]
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
        """ESTRATEGIA NUCLEAR HFT - Condiciones optimizadas"""
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
        
        # ESTRATEGIA NUCLEAR HFT - CONDICIONES OPTIMIZADAS
        
        # SEÑAL DE VENTA (SHORT) - Estrategia Nuclear
        short_conditions = [
            momentum > 0.0008,
            rsi > 35,  
            macd > macd_signal,         
            current_price > bb_lower * 0.999,
            volatility < 0.02
        ]
        
        # SEÑAL DE COMPRA (LONG) - Estrategia Nuclear
        long_conditions = [
            momentum < -0.0008,
            rsi < 65,
            macd < macd_signal,         
            current_price < bb_upper * 1.001,
            self.position == 0          
        ]
        
        # GESTIÓN DE POSICIÓN ABIERTA - ESTRATEGIA NUCLEAR
        if self.position > 0:
            if self.position_side == 'long':
                current_profit_pct = (current_price - self.entry_price) / self.entry_price
                current_loss_pct = (self.entry_price - current_price) / self.entry_price
            else:  # short
                current_profit_pct = (self.entry_price - current_price) / self.entry_price
                current_loss_pct = (current_price - self.entry_price) / self.entry_price
            
            # TOMA DE GANANCIAS NUCLEAR
            if current_profit_pct >= self.min_profit_target:
                self.log_message(f"🎯 GANANCIA NUCLEAR: {current_profit_pct:.3%} (+)", "PROFIT")
                return 'close'
            
            # STOP LOSS NUCLEAR
            if current_loss_pct >= self.max_loss_stop:
                self.log_message(f"🛑 STOP LOSS NUCLEAR: {current_loss_pct:.3%} (-)", "LOSS")
                return 'close'
        
        # SEÑALES PRINCIPALES NUCLEAR
        if sum(short_conditions) >= 2 and self.position == 0:
            self.log_message(f"⚡ SEÑAL VENTA NUCLEAR: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'sell'
        elif sum(long_conditions) >= 2 and self.position == 0:
            self.log_message(f"⚡ SEÑAL COMPRA NUCLEAR: momentum={momentum:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'buy'
        
        return 'hold'

    def execute_trade(self, action: str, price: float):
        """Ejecutar operación en FUTUROS - ESTRATEGIA NUCLEAR"""
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
                self.cash_balance -= (investment_amount / self.leverage)
                self.position = quantity
                self.entry_price = price
                self.position_side = 'long' if action == 'buy' else 'short'
                self.open_positions = 1
                
                side_emoji = "🟢" if action == 'buy' else "🔴"
                trade_info = f"{side_emoji} NUCLEAR {self.position_side.upper()}: {quantity:.6f} {self.symbol} @ ${price:.2f} | Margen: ${investment_amount/self.leverage:.2f} | Leverage: {self.leverage}x"
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
                trade_info = f"{profit_color} CERRAR NUCLEAR: {quantity_to_close:.6f} {self.symbol} @ ${price:.2f} | P/L: ${profit_loss:.4f} | Profit Total: ${self.total_profit:.2f}"
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
        self.tick_data.clear()
        self.total_profit = 0
        self.log_message("🔄 Cuenta reiniciada a $255.00 - ESTRATEGIA NUCLEAR", "INFO")
        self.save_state()

    def trading_cycle(self):
        """Ciclo principal de trading HFT con protección máxima"""
        self.log_message("🚀 INICIANDO ESTRATEGIA NUCLEAR HFT - PROTECCIÓN ACTIVADA")
        
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                tick_data = self.get_futures_price()
                if tick_data:
                    self.tick_data.append(tick_data)
                
                indicators = self.calculate_indicators()
                
                if indicators and len(self.tick_data) >= 15:
                    signal = self.trading_strategy(indicators)
                    if signal != 'hold':
                        price = tick_data['ask'] if signal == 'buy' else tick_data['bid']
                        self.execute_trade(signal, price)
                
                consecutive_errors = 0  # Resetear contador de errores
                time.sleep(0.8)
                
            except Exception as e:
                consecutive_errors += 1
                self.log_message(f"🚨 ERROR en ciclo #{consecutive_errors}: {e}", "ERROR")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.log_message("🛑 DEMASIADOS ERRORES - Cerrando todas las posiciones", "CRITICAL")
                    self.close_all_positions()
                    self.is_running = False
                    break
                
                time.sleep(10)  # Esperar 10 segundos antes de reintentar
                continue

    def start_trading(self):
        """Iniciar bot de trading HFT"""
        if not self.is_running:
            self.is_running = True
            self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
            self.trading_thread.start()
            self.log_message("✅ ESTRATEGIA NUCLEAR HFT ACTIVADA - SISTEMA ESTABLE", "SYSTEM")

    def stop_trading(self):
        """Detener bot de trading"""
        self.is_running = False
        self.log_message("⏹️ ESTRATEGIA NUCLEAR DETENIDA", "SYSTEM")

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

def main():
    st.title("🤖 Bot HFT Futuros MEXC - ESTRATEGIA NUCLEAR ⚡")
    st.markdown("**CAPITAL INICIAL: $255.00 | APALANCAMIENTO: 3x | SISTEMA ANTICRASH**")
    st.markdown("---")
    
    # Inicializar el bot
    if 'bot' not in st.session_state:
        st.session_state.bot = MexcFuturesTradingBot("", "", "BTCUSDT")
    
    bot = st.session_state.bot
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración Nuclear")
        
        api_key = st.text_input("API Key MEXC", type="password")
        secret_key = st.text_input("Secret Key MEXC", type="password")
        symbol = st.selectbox("Símbolo Futuros", ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"])
        
        bot.api_key = api_key
        bot.secret_key = secret_key
        bot.symbol = symbol
        
        st.markdown("---")
        st.header("🎯 Control Nuclear")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Activar Nuclear", use_container_width=True, type="primary"):
                bot.start_trading()
                st.rerun()
        
        with col2:
            if st.button("⏹️ Desactivar Nuclear", use_container_width=True):
                bot.stop_trading()
                st.rerun()
        
        if st.button("🧹 Cerrar Posición", use_container_width=True):
            bot.close_all_positions()
            st.rerun()
            
        if st.button("🔄 Reiniciar $255", use_container_width=True):
            bot.reset_account()
            st.rerun()
        
        st.markdown("---")
        st.header("💰 Configuración Nuclear")
        st.info(f"**Tamaño posición:** {bot.position_size*100}%")
        st.info(f"**Target ganancia:** {bot.min_profit_target*100}%")
        st.info(f"**Stop loss:** {bot.max_loss_stop*100}%")
        st.info(f"**Apalancamiento:** {bot.leverage}x")
        st.info(f"**Operaciones máx:** {bot.max_positions}")
        
        if bot.is_running:
            st.success("✅ NUCLEAR ACTIVO - SISTEMA ESTABLE")
        else:
            st.warning("⏸️ SISTEMA EN STANDBY")
            
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
    tab1, tab2, tab3 = st.tabs(["📈 Precios Futuros", "📋 Operaciones Nucleares", "📝 Logs del Sistema"])
    
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
                    title=f"Precio Futuros {bot.symbol} - ESTRATEGIA NUCLEAR",
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
            st.info("No hay operaciones nucleares registradas aún.")
    
    with tab3:
        log_container = st.container(height=400)
        with log_container:
            for log_entry in reversed(bot.log_messages[-30:]):
                if "ERROR" in log_entry or "CRITICAL" in log_entry:
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
                elif "SYSTEM" in log_entry:
                    st.info(log_entry)
                else:
                    st.info(log_entry)
    
    # Auto-refresh más rápido para HFT
    if bot.is_running:
        time.sleep(3)
        st.rerun()

if __name__ == "__main__":
    main()
