# app.py
import streamlit as st
import asyncio
import threading
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
        
        # Símbolos de MEXC para perpetuals (ejemplo - verificar en documentación)
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
        
        # Estadísticas de trading
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0
        }
        
    def load_data(self):
        """Cargar datos persistentes"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.available_balance = data.get('balance', self.initial_balance)
                    self.positions = data.get('positions', {})
                    self.stats = data.get('stats', self.stats)
        except Exception as e:
            st.warning(f"Error cargando datos: {e}")

    def save_data(self):
        """Guardar datos persistentes"""
        try:
            data = {
                'balance': self.available_balance,
                'positions': self.positions,
                'stats': self.stats,
                'last_update': datetime.now().isoformat()
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            st.error(f"Error guardando datos: {e}")

    def get_real_time_price(self, coin: str) -> float:
        """Obtener precio real desde MEXC"""
        symbol = self.coins[coin]
        price = self.mexc_api.obtener_precio_actual(symbol)
        
        if price is None:
            # Fallback a precios simulados si la API falla
            base_prices = {
                'BTC': 45000.0, 'ETH': 2500.0, 'SOL': 100.0,
                'XRP': 0.60, 'DOGE': 0.08, 'BNB': 300.0
            }
            volatility = np.random.uniform(-0.002, 0.002)
            price = base_prices[coin] * (1 + volatility)
            st.warning(f"Usando precio simulado para {coin}")
        
        return price

    def get_market_data(self, coin: str) -> Optional[pd.DataFrame]:
        """Obtener datos de mercado desde MEXC"""
        symbol = self.coins[coin]
        return self.mexc_api.obtener_datos_mercado(symbol, interval='5m', limit=50)

    def calculate_position_size(self, coin: str, price: float) -> tuple:
        """Calcular tamaño de posición basado en el balance disponible"""
        max_position_value = self.available_balance * 0.15  # 15% por moneda
        leverage = np.random.randint(self.leverage_range[0], self.leverage_range[1] + 1)
        notional = min(max_position_value * leverage, self.available_balance * 0.8)
        size = notional / price
            
        return size, notional, leverage

    def generate_market_signals(self, coin: str) -> dict:
        """Generar señales de market making con datos reales de MEXC"""
        current_price = self.get_real_time_price(coin)
        market_data = self.get_market_data(coin)
        
        # Calcular spread basado en volatilidad reciente
        if market_data is not None and len(market_data) > 0:
            recent_high = market_data['high'].max()
            recent_low = market_data['low'].min()
            volatility = (recent_high - recent_low) / recent_low
            base_spread = max(0.001, min(0.01, volatility * 0.5))
        else:
            base_spread = np.random.uniform(0.001, 0.005)
        
        spread = base_spread
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
        """Calcular stop loss dinámico basado en volatilidad"""
        stop_percent = 0.015  # 1.5% base
        
        if side == "LONG":
            stop_loss = entry_price * (1 - stop_percent)
        else:  # SHORT
            stop_loss = entry_price * (1 + stop_percent)
            
        return stop_loss

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calcular take profit con ratio riesgo:beneficio 1:1.67"""
        tp_percent = 0.025  # 2.5%
        
        if side == "LONG":
            take_profit = entry_price * (1 + tp_percent)
        else:  # SHORT
            take_profit = entry_price * (1 - tp_percent)
            
        return take_profit

    def execute_trade(self, coin: str, signals: dict):
        """Ejecutar operación de market making"""
        if coin in self.positions and self.positions[coin]['status'] == 'ACTIVE':
            return  # Ya hay posición activa
            
        # Condiciones para entrar - solo si el spread es atractivo
        if signals['spread'] > 0.002:  # Spread mínimo de 0.2%
            # Decidir lado basado en momentum simple
            market_data = self.get_market_data(coin)
            if market_data is not None and len(market_data) > 1:
                current_close = market_data['close'].iloc[-1]
                previous_close = market_data['close'].iloc[-2]
                side = "LONG" if current_close > previous_close else "SHORT"
            else:
                side = "LONG" if np.random.random() > 0.5 else "SHORT"
            
            entry_price = signals['bid_price'] if side == "LONG" else signals['ask_price']
            size, notional, leverage = self.calculate_position_size(coin, entry_price)
            
            # Verificar que tenemos suficiente balance
            if notional > self.available_balance * 0.1 and notional > 10:  # Mínimo $10
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
                    'coin': coin
                }
                
                # Reservar margen
                margin_used = notional / leverage
                self.available_balance -= margin_used
                self.save_data()
                
                st.success(f"🔄 Nueva posición: {side} {coin} | Precio: ${entry_price:.4f}")

    def update_positions(self):
        """Actualizar PnL y verificar condiciones de salida con precios reales"""
        current_pnl = 0
        positions_to_close = []
        
        for coin, position in self.positions.items():
            if position['status'] == 'ACTIVE':
                current_price = self.get_real_time_price(coin)
                
                # Calcular PnL no realizado
                if position['side'] == "LONG":
                    pnl_percent = (current_price - position['entry_price']) / position['entry_price']
                else:  # SHORT
                    pnl_percent = (position['entry_price'] - current_price) / position['entry_price']
                
                unrealized_pnl = position['notional'] * pnl_percent * position['leverage']
                position['unrealized_pnl'] = unrealized_pnl
                position['current_price'] = current_price
                
                current_pnl += unrealized_pnl
                
                # Verificar stop loss
                stop_loss_triggered = False
                if position['side'] == "LONG" and current_price <= position['stop_loss']:
                    stop_loss_triggered = True
                    position['exit_reason'] = 'STOP_LOSS'
                elif position['side'] == "SHORT" and current_price >= position['stop_loss']:
                    stop_loss_triggered = True
                    position['exit_reason'] = 'STOP_LOSS'
                
                # Verificar take profit
                take_profit_triggered = False
                if position['side'] == "LONG" and current_price >= position['take_profit']:
                    take_profit_triggered = True
                    position['exit_reason'] = 'TAKE_PROFIT'
                elif position['side'] == "SHORT" and current_price <= position['take_profit']:
                    take_profit_triggered = True
                    position['exit_reason'] = 'TAKE_PROFIT'
                
                if stop_loss_triggered or take_profit_triggered:
                    positions_to_close.append(coin)
        
        # Cerrar posiciones que cumplen condiciones
        for coin in positions_to_close:
            self.close_position(coin)
            
        return current_pnl

    def close_position(self, coin: str):
        """Cerrar posición específica"""
        if coin in self.positions:
            position = self.positions[coin]
            realized_pnl = position['unrealized_pnl']
            
            # Actualizar estadísticas
            self.stats['total_trades'] += 1
            self.stats['total_pnl'] += realized_pnl
            
            if realized_pnl > 0:
                self.stats['winning_trades'] += 1
            else:
                self.stats['losing_trades'] += 1
            
            # Liberar margen + agregar PnL realizado
            margin_return = position['notional'] / position['leverage']
            self.available_balance += margin_return + realized_pnl
            
            position['status'] = 'CLOSED'
            position['exit_time'] = datetime.now().isoformat()
            position['realized_pnl'] = realized_pnl
            
            st.info(f"✅ Posición cerrada: {coin} | PnL: ${realized_pnl:.2f}")
            self.save_data()

    def close_all_positions(self):
        """Cerrar todas las posiciones activas"""
        active_positions = [coin for coin, pos in self.positions.items() if pos['status'] == 'ACTIVE']
        for coin in active_positions:
            self.close_position(coin)
        return len(active_positions)

    def get_api_status(self):
        """Verificar estado de la API de MEXC"""
        return self.mexc_api.obtener_estado_servidor()

def main():
    st.title("🤖 HFT Market Maker - MEXC")
    st.markdown("**Operador Autónomo - Perpetuos USDT con datos reales de MEXC**")
    
    # Inicializar o recuperar instancia del bot
    if 'bot' not in st.session_state:
        st.session_state.bot = MEXCMarketMaker()
    
    bot = st.session_state.bot
    
    # Verificar estado de API
    api_status = bot.get_api_status()
    status_color = "🟢" if api_status else "🔴"
    st.sidebar.markdown(f"**Estado API MEXC:** {status_color} {'CONECTADO' if api_status else 'DESCONECTADO'}")
    
    # Panel de control
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 Encender Bot", type="primary", disabled=bot.active):
            bot.active = True
            st.rerun()
            
    with col2:
        if st.button("🛑 Apagar Bot", disabled=not bot.active):
            bot.active = False
            st.rerun()
            
    with col3:
        if st.button("📊 Cerrar Todas", key="close_all"):
            closed_count = bot.close_all_positions()
            st.success(f"Cerradas {closed_count} posiciones")
            st.rerun()
    
    with col4:
        st.metric("Balance Disponible", f"${bot.available_balance:.2f}")
    
    # Mostrar estado del bot
    status_color = "🟢" if bot.active else "🔴"
    st.subheader(f"{status_color} Estado del Bot: {'ACTIVO' if bot.active else 'INACTIVO'}")
    
    # Ejecutar lógica del bot si está activo
    if bot.active:
        with st.spinner("Ejecutando estrategia de Market Making con datos reales de MEXC..."):
            # Ejecutar trading para cada moneda
            for coin in bot.coins.keys():
                try:
                    signals = bot.generate_market_signals(coin)
                    bot.execute_trade(coin, signals)
                except Exception as e:
                    st.error(f"Error procesando {coin}: {e}")
            
            # Actualizar posiciones
            current_pnl = bot.update_positions()
        
        # Mostrar PnL actualizado
        total_value = bot.available_balance + current_pnl
        st.metric("PnL Total No Realizado", f"${current_pnl:.2f}", 
                 delta=f"{((total_value - bot.initial_balance) / bot.initial_balance * 100):.2f}%")
    
    # Mostrar panel de posiciones
    st.subheader("📊 Panel de Posiciones Activas")
    
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
                'PRECIO ENTRADA': f"${pos['entry_price']:.4f}",
                'PRECIO ACTUAL': f"${current_price:.4f}",
                'STOP LOSS': f"${pos['stop_loss']:.4f}",
                'TAKE PROFIT': f"${pos['take_profit']:.4f}",
                'P&L NO REALIZADO': f"${pos['unrealized_pnl']:.2f}",
                'PLAN DE SALIDA': exit_plan
            })
        
        df = pd.DataFrame(positions_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No hay posiciones activas en este momento")
    
    # Mostrar estadísticas de rendimiento
    st.subheader("📈 Métricas de Rendimiento")
    col1, col2, col3, col4 = st.columns(4)
    
    total_value = bot.available_balance + sum(pos['unrealized_pnl'] for pos in bot.positions.values() if pos['status'] == 'ACTIVE')
    
    with col1:
        st.metric("Valor Total", f"${total_value:.2f}")
    with col2:
        st.metric("Rendimiento Total", f"{((total_value - bot.initial_balance) / bot.initial_balance * 100):.2f}%")
    with col3:
        win_rate = (bot.stats['winning_trades'] / bot.stats['total_trades'] * 100) if bot.stats['total_trades'] > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col4:
        st.metric("Trades Totales", bot.stats['total_trades'])
    
    # Historial de trades cerrados
    st.subheader("📋 Historial de Trades Cerrados")
    closed_trades = [pos for pos in bot.positions.values() if pos['status'] == 'CLOSED']
    if closed_trades:
        closed_data = []
        for trade in closed_trades[-10:]:  # Últimos 10 trades
            closed_data.append({
                'Moneda': trade['coin'],
                'Lado': trade['side'],
                'P&L': f"${trade['realized_pnl']:.2f}",
                'Entrada': f"${trade['entry_price']:.4f}",
                'Razón Salida': trade.get('exit_reason', 'MANUAL'),
                'Tiempo': trade.get('entry_time', '')[:16]
            })
        st.dataframe(pd.DataFrame(closed_data))
    else:
        st.info("No hay trades cerrados aún")
    
    # Información de conexión
    with st.expander("🔧 Información de Conexión y Configuración"):
        st.write(f"**Balance Inicial:** ${bot.initial_balance}")
        st.write(f"**Balance Actual:** ${bot.available_balance:.2f}")
        st.write(f"**Apalancamiento:** {bot.leverage_range[0]}x - {bot.leverage_range[1]}x")
        st.write(f"**Monedas Operadas:** {', '.join(bot.coins.keys())}")
        st.write(f"**Símbolos MEXC:** {', '.join(bot.coins.values())}")
        st.write("**Estrategia:** Market Making con stops dinámicos")
        st.write("**Persistencia:** Datos guardados automáticamente")
        st.write(f"**API Status:** {'🟢 CONECTADO' if api_status else '🔴 DESCONECTADO'}")

if __name__ == "__main__":
    main()
