# app.py
import streamlit as st
import time
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import Dict, List, Optional
import requests

# Clase MEXC API proporcionada
class MEXCAPI:
    def __init__(self):
        self.base_url = "https://api.mexc.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def obtener_datos_mercado(self, symbol, interval='5m', limit=100):
        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Convertir a DataFrame
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convertir tipos de datos
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                return df
            else:
                print(f"Error API MEXC: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error conectando con MEXC: {e}")
            return None
    
    def obtener_precio_actual(self, symbol):
        try:
            url = f"{self.base_url}/ticker/price"
            params = {'symbol': symbol}
            
            response = self.session.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
            else:
                return None
                
        except Exception as e:
            print(f"Error obteniendo precio: {e}")
            return None

    def obtener_estado_servidor(self):
        try:
            url = f"{self.base_url}/ping"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except:
            return False

class MEXCMarketMaker:
    def __init__(self):
        self.initial_balance = 255.0
        self.available_balance = self.initial_balance
        self.positions = {}
        self.active = False
        self.leverage_range = (10, 20)
        self.mexc_api = MEXCAPI()
        
        # Símbolos de MEXC para perpetuals
        self.coins = {
            'BTC': 'BTCUSDT',
            'ETH': 'ETHUSDT', 
            'SOL': 'SOLUSDT',
            'XRP': 'XRPUSDT',
            'DOGE': 'DOGEUSDT',
            'BNB': 'BNBUSDT'
        }
        
        self.data_file = "mexc_trading_data.json"
        self.load_data()
        
        # Estadísticas de trading MEJORADAS
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_balance': self.initial_balance,
            'min_balance': self.initial_balance,
            'trades_today': 0,
            'last_trade_time': None,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }
        
        # Sistema de logs
        self.logs = []
        self.max_logs = 50
        
        # Interés compuesto automático
        self.compound_threshold = 0.02  # 2% de ganancia para reinvertir
        self.last_compound_balance = self.available_balance
        
    def add_log(self, message: str, level: str = "INFO"):
        """Agregar mensaje al log"""
        log_entry = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'level': level,
            'message': message
        }
        self.logs.insert(0, log_entry)
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[:self.max_logs]
        self.save_data()
        
    def load_data(self):
        """Cargar datos persistentes"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.available_balance = data.get('balance', self.initial_balance)
                    self.positions = data.get('positions', {})
                    self.stats = data.get('stats', self.stats)
                    self.logs = data.get('logs', [])
                    self.last_compound_balance = data.get('last_compound_balance', self.available_balance)
        except Exception as e:
            self.add_log(f"Error cargando datos: {e}", "ERROR")

    def save_data(self):
        """Guardar datos persistentes"""
        try:
            data = {
                'balance': self.available_balance,
                'positions': self.positions,
                'stats': self.stats,
                'logs': self.logs,
                'last_compound_balance': self.last_compound_balance,
                'last_update': datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            self.add_log(f"Error guardando datos: {e}", "ERROR")

    def apply_compound_interest(self):
        """Aplicar interés compuesto automáticamente cuando hay ganancias"""
        current_total = self.get_total_balance()
        
        # Calcular ganancia desde el último compound
        gain_since_last = current_total - self.last_compound_balance
        gain_percentage = gain_since_last / self.last_compound_balance if self.last_compound_balance > 0 else 0
        
        # Si la ganancia supera el threshold, reinvertir todo el capital
        if gain_percentage >= self.compound_threshold and gain_since_last > 1.0:
            old_balance = self.available_balance
            self.available_balance = current_total
            self.last_compound_balance = current_total
            
            self.add_log(f"🎯 INTERÉS COMPUESTO: ${old_balance:.2f} → ${current_total:.2f} (+${gain_since_last:.2f})")
            self.save_data()
            return True
        
        return False

    def get_total_balance(self):
        """Obtener balance total incluyendo PnL no realizado"""
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values() if pos['status'] == 'ACTIVE')
        return self.available_balance + unrealized_pnl

    def get_real_time_price(self, coin: str) -> float:
        """Obtener precio real desde MEXC"""
        symbol = self.coins[coin]
        price = self.mexc_api.obtener_precio_actual(symbol)
        
        if price is not None:
            return price
        
        # Solo fallback si la API falla completamente
        base_prices = {
            'BTC': 101260.0,
            'ETH': 2500.0, 
            'SOL': 100.0,
            'XRP': 0.60, 
            'DOGE': 0.08, 
            'BNB': 300.0
        }
        volatility = np.random.uniform(-0.002, 0.002)
        return base_prices[coin] * (1 + volatility)

    def get_market_data(self, coin: str) -> Optional[pd.DataFrame]:
        """Obtener datos de mercado desde MEXC"""
        symbol = self.coins[coin]
        return self.mexc_api.obtener_datos_mercado(symbol, interval='5m', limit=50)

    def calculate_position_size(self, coin: str, price: float) -> tuple:
        """Calcular tamaño de posición con interés compuesto"""
        # Usar el balance total para calcular posiciones más grandes
        total_balance = self.get_total_balance()
        max_position_value = total_balance * 0.15
        leverage = np.random.randint(self.leverage_range[0], self.leverage_range[1] + 1)
        notional = min(max_position_value * leverage, total_balance * 0.8)
        size = notional / price
            
        return size, notional, leverage

    def generate_market_signals(self, coin: str) -> dict:
        """Generar señales de market making con datos reales de MEXC"""
        current_price = self.get_real_time_price(coin)
        market_data = self.get_market_data(coin)
        
        spread = np.random.uniform(0.001, 0.005)
        bid_price = current_price * (1 - spread/2)
        ask_price = current_price * (1 + spread/2)
        
        return {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'mid_price': current_price,
            'spread': spread,
            'timestamp': datetime.now()
        }

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calcular stop loss"""
        stop_percent = 0.015
        
        if side == "LONG":
            stop_loss = entry_price * (1 - stop_percent)
        else:
            stop_loss = entry_price * (1 + stop_percent)
            
        return stop_loss

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calcular take profit"""
        tp_percent = 0.025
        
        if side == "LONG":
            take_profit = entry_price * (1 + tp_percent)
        else:
            take_profit = entry_price * (1 - tp_percent)
            
        return take_profit

    def execute_trade(self, coin: str, signals: dict):
        """Ejecutar operación de market making"""
        if coin in self.positions and self.positions[coin]['status'] == 'ACTIVE':
            return
            
        if signals['spread'] > 0.002:
            side = "LONG" if np.random.random() > 0.5 else "SHORT"
            
            entry_price = signals['bid_price'] if side == "LONG" else signals['ask_price']
            size, notional, leverage = self.calculate_position_size(coin, entry_price)
            
            # Verificar que tenemos suficiente balance disponible (no usar PnL no realizado)
            margin_required = notional / leverage
            if margin_required <= self.available_balance and notional > 10:
                stop_loss = self.calculate_stop_loss(entry_price, side)
                take_profit = self.calculate_take_profit(entry_price, side)
                
                self.positions[coin] = {
                    'side': side,
                    'entry_price': entry_price,
                    'size': size,
                    'notional': notional,
                    'leverage': leverage,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_time': datetime.now().isoformat(),
                    'status': 'ACTIVE',
                    'unrealized_pnl': 0.0,
                    'coin': coin,
                    'margin_used': margin_required
                }
                
                # Reservar margen (SOLO el margen real, no el notional completo)
                self.available_balance -= margin_required
                self.save_data()
                
                self.add_log(f"🔄 NUEVA POSICIÓN: {side} {coin} | Entrada: ${entry_price:.2f} | Tamaño: ${notional:.2f}")

    def update_positions(self):
        """Actualizar PnL y verificar condiciones de salida - CORREGIDO"""
        current_pnl = 0
        positions_to_close = []
        
        for coin, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = self.get_real_time_price(coin)
                
                # Calcular PnL no realizado CORRECTAMENTE
                if position['side'] == "LONG":
                    pnl_percent = (current_price - position['entry_price']) / position['entry_price']
                else:
                    pnl_percent = (position['entry_price'] - current_price) / position['entry_price']
                
                # PnL no realizado es sobre el notional con leverage
                unrealized_pnl = position['notional'] * pnl_percent
                position['unrealized_pnl'] = unrealized_pnl
                position['current_price'] = current_price
                
                current_pnl += unrealized_pnl
                
                # Verificar condiciones de salida
                if (position['side'] == "LONG" and current_price <= position['stop_loss']):
                    positions_to_close.append(coin)
                    position['exit_reason'] = 'STOP_LOSS'
                elif (position['side'] == "SHORT" and current_price >= position['stop_loss']):
                    positions_to_close.append(coin)
                    position['exit_reason'] = 'STOP_LOSS'
                elif (position['side'] == "LONG" and current_price >= position['take_profit']):
                    positions_to_close.append(coin)
                    position['exit_reason'] = 'TAKE_PROFIT'
                elif (position['side'] == "SHORT" and current_price <= position['take_profit']):
                    positions_to_close.append(coin)
                    position['exit_reason'] = 'TAKE_PROFIT'
        
        # Cerrar posiciones que cumplen condiciones
        for coin in positions_to_close:
            self.close_position(coin)
            
        return current_pnl

    def close_position(self, coin: str):
        """Cerrar posición específica - CORREGIDO"""
        if coin in self.positions:
            position = self.positions[coin]
            realized_pnl = position['unrealized_pnl']
            
            # Actualizar estadísticas
            self.stats['total_trades'] += 1
            self.stats['total_pnl'] += realized_pnl
            
            if realized_pnl > 0:
                self.stats['winning_trades'] += 1
                self.stats['consecutive_wins'] += 1
                self.stats['consecutive_losses'] = 0
            else:
                self.stats['losing_trades'] += 1
                self.stats['consecutive_losses'] += 1
                self.stats['consecutive_wins'] = 0
            
            # CORRECCIÓN: Liberar margen + agregar PnL realizado
            margin_return = position['margin_used']
            self.available_balance += margin_return + realized_pnl
            
            # Actualizar balances máximo y mínimo
            current_balance = self.get_total_balance()
            self.stats['max_balance'] = max(self.stats['max_balance'], current_balance)
            self.stats['min_balance'] = min(self.stats['min_balance'], current_balance)
            
            position['status'] = 'CLOSED'
            position['exit_time'] = datetime.now().isoformat()
            position['realized_pnl'] = realized_pnl
            position['closed_balance'] = self.available_balance
            
            log_color = "✅" if realized_pnl > 0 else "❌"
            self.add_log(f"{log_color} POSICIÓN CERRADA: {position['side']} {coin} | PnL: ${realized_pnl:.2f}")
            
            self.save_data()

    def close_all_positions(self):
        """Cerrar todas las posiciones activas - CORREGIDO"""
        active_positions = [coin for coin, pos in self.positions.items() if pos['status'] == 'ACTIVE']
        closed_count = 0
        
        for coin in active_positions:
            # Forzar cierre al precio actual (simulación de mercado)
            position = self.positions[coin]
            current_price = self.get_real_time_price(coin)
            
            # Calcular PnL final
            if position['side'] == "LONG":
                pnl_percent = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_percent = (position['entry_price'] - current_price) / position['entry_price']
            
            realized_pnl = position['notional'] * pnl_percent
            
            # Actualizar balance CORRECTAMENTE
            margin_return = position['margin_used']
            self.available_balance += margin_return + realized_pnl
            
            # Marcar como cerrada
            position['status'] = 'CLOSED'
            position['exit_time'] = datetime.now().isoformat()
            position['realized_pnl'] = realized_pnl
            position['exit_reason'] = 'MANUAL_CLOSE'
            
            closed_count += 1
            
            self.add_log(f"🛑 CIERRE MANUAL: {position['side']} {coin} | PnL: ${realized_pnl:.2f}")
        
        if closed_count > 0:
            self.save_data()
            
        return closed_count

    def get_api_status(self):
        """Verificar estado de la API de MEXC"""
        return self.mexc_api.obtener_estado_servidor()

def main():
    st.title("🤖 HFT Market Maker - MEXC")
    st.markdown("**Operador Autónomo - Perpetuos USDT con datos reales de MEXC**")
    
    # Inicializar bot
    if 'bot' not in st.session_state:
        st.session_state.bot = MEXCMarketMaker()
    
    bot = st.session_state.bot
    
    # Verificar estado de API
    api_status = bot.get_api_status()
    
    # Panel de control PRINCIPAL
    st.subheader("🎛️ Panel de Control")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Encender Bot", type="primary", disabled=bot.active, use_container_width=True):
            bot.active = True
            bot.add_log("Bot ENCENDIDO - Iniciando estrategia")
            st.rerun()
            
    with col2:
        if st.button("🛑 Apagar Bot", disabled=not bot.active, use_container_width=True):
            bot.active = False
            bot.add_log("Bot APAGADO - Deteniendo operaciones")
            st.rerun()
            
    with col3:
        if st.button("📊 Cerrar Todas", use_container_width=True):
            closed_count = bot.close_all_positions()
            st.success(f"✅ Cerradas {closed_count} posiciones")
            st.rerun()
    
    with col4:
        total_balance = bot.get_total_balance()
        st.metric("💰 Balance Total", f"${total_balance:.2f}", 
                 delta=f"${(total_balance - bot.initial_balance):.2f}")

    # Estado del sistema
    status_color = "🟢" if bot.active else "🔴"
    st.write(f"**Estado Bot:** {status_color} {'ACTIVO' if bot.active else 'INACTIVO'} | **API MEXC:** {'🟢 CONECTADO' if api_status else '🔴 DESCONECTADO'}")

    # Ejecutar bot si está activo
    if bot.active:
        with st.spinner("🤖 Ejecutando estrategia de Market Making..."):
            # Aplicar interés compuesto si hay ganancias
            compound_applied = bot.apply_compound_interest()
            if compound_applied:
                st.success("🎯 Interés compuesto aplicado automáticamente!")
            
            # Ejecutar trading
            for coin in bot.coins.keys():
                try:
                    signals = bot.generate_market_signals(coin)
                    bot.execute_trade(coin, signals)
                except Exception as e:
                    bot.add_log(f"Error en {coin}: {e}", "ERROR")
            
            # Actualizar posiciones
            current_pnl = bot.update_positions()

    # 📊 PANEL DE ESTADÍSTICAS EN TIEMPO REAL
    st.subheader("📈 Métricas de Rendimiento")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_balance = bot.get_total_balance()
    active_positions = len([p for p in bot.positions.values() if p['status'] == 'ACTIVE'])
    
    with col1:
        st.metric("Balance Total", f"${total_balance:.2f}", 
                 f"{((total_balance - bot.initial_balance) / bot.initial_balance * 100):.1f}%")
    
    with col2:
        win_rate = (bot.stats['winning_trades'] / bot.stats['total_trades'] * 100) if bot.stats['total_trades'] > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{bot.stats['total_trades']} trades")
    
    with col3:
        st.metric("Posiciones Activas", active_positions, 
                 f"Max: ${bot.stats['max_balance']:.2f}" if bot.stats['max_balance'] > bot.initial_balance else "")
    
    with col4:
        current_pnl = sum(pos['unrealized_pnl'] for pos in bot.positions.values() if pos['status'] == 'ACTIVE')
        st.metric("PnL No Realizado", f"${current_pnl:.2f}", 
                 f"Racha: {bot.stats['consecutive_wins']}W/{bot.stats['consecutive_losses']}L")

    # 📋 PANEL DE POSICIONES ACTIVAS
    st.subheader("📊 Posiciones Activas")
    
    active_positions = [pos for pos in bot.positions.values() if pos['status'] == 'ACTIVE']
    
    if active_positions:
        positions_data = []
        for pos in active_positions:
            current_price = pos.get('current_price', bot.get_real_time_price(pos['coin']))
            
            # Determinar plan de salida
            if pos['side'] == "LONG":
                exit_plan = "STOP_LOSS" if current_price <= pos['stop_loss'] else "TAKE_PROFIT" if current_price >= pos['take_profit'] else "MANTENER"
            else:
                exit_plan = "STOP_LOSS" if current_price >= pos['stop_loss'] else "TAKE_PROFIT" if current_price <= pos['take_profit'] else "MANTENER"
            
            positions_data.append({
                'MONEDA': pos['coin'],
                'LADO': pos['side'],
                'APALANCAMIENTO': pos['leverage'],
                'NOCIONAL': f"${pos['notional']:.2f}",
                'PRECIO ENTRADA': f"${pos['entry_price']:.2f}",
                'PRECIO ACTUAL': f"${current_price:.2f}",
                'STOP LOSS': f"${pos['stop_loss']:.2f}",
                'TAKE PROFIT': f"${pos['take_profit']:.2f}",
                'P&L NO REALIZADO': f"${pos['unrealized_pnl']:.2f}",
                'PLAN DE SALIDA': exit_plan
            })
        
        df = pd.DataFrame(positions_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("📭 No hay posiciones activas en este momento")

    # 📝 SISTEMA DE LOGS EN TIEMPO REAL
    st.subheader("📋 Logs de Operaciones")
    
    if bot.logs:
        # Mostrar logs en un contenedor con scroll
        log_container = st.container(height=200)
        with log_container:
            for log in bot.logs[:20]:  # Mostrar últimos 20 logs
                level_color = {
                    "INFO": "🔵",
                    "SUCCESS": "🟢", 
                    "ERROR": "🔴",
                    "WARNING": "🟡"
                }
                color = level_color.get(log['level'], "⚪")
                st.write(f"`{log['timestamp']}` {color} **{log['level']}:** {log['message']}")
    else:
        st.info("📝 No hay logs registrados aún")

    # 🔧 CONFIGURACIÓN E INFORMACIÓN
    with st.expander("🔧 Información del Sistema"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Estadísticas Detalladas:**")
            st.write(f"- Trades Totales: {bot.stats['total_trades']}")
            st.write(f"- Trades Ganadores: {bot.stats['winning_trades']}")
            st.write(f"- Trades Perdedores: {bot.stats['losing_trades']}")
            st.write(f"- PnL Total Realizado: ${bot.stats['total_pnl']:.2f}")
            st.write(f"- Balance Disponible: ${bot.available_balance:.2f}")
            
        with col2:
            st.write("**⚙️ Configuración:**")
            st.write(f"- Balance Inicial: ${bot.initial_balance}")
            st.write(f"- Apalancamiento: {bot.leverage_range[0]}x - {bot.leverage_range[1]}x")
            st.write(f"- Interés Compuesto: {bot.compound_threshold*100}% threshold")
            st.write(f"- Monedas: {', '.join(bot.coins.keys())}")

if __name__ == "__main__":
    main()
