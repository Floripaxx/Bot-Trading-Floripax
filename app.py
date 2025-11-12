import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime

# CONFIGURACIÓN ESPECÍFICA PARA BTC
SYMBOL = "BTCUSDT"  # ENFOQUE SOLO EN BTC
LEVERAGE = 3
MAX_DAILY_TRADES = 100  # ALTA FRECUENCIA
MIN_TIME_BETWEEN_TRADES = 2  # OPERACIONES RÁPIDAS
BASE_CAPITAL = 255.0

class TradingBot:
    def __init__(self):
        self.leverage = LEVERAGE
        self.cash_balance = BASE_CAPITAL
        self.total_equity = BASE_CAPITAL
        self.open_positions = []
        self.trade_count = 0
        self.last_trade_time = None
        self.daily_trades = 0
        self.last_daily_reset = datetime.now().date()
        self.compound_growth = 1.0  # FACTOR DE INTERÉS COMPUESTO
        self.initial_capital = BASE_CAPITAL
        self.trade_logs = []  # LOGS DE OPERACIONES
        
    def add_log(self, message, log_type="INFO"):
        """AGREGA LOGS DE OPERACIONES"""
        log_entry = {
            'timestamp': datetime.now(),
            'type': log_type,
            'message': message
        }
        self.trade_logs.append(log_entry)
        # Mantener solo últimos 50 logs
        if len(self.trade_logs) > 50:
            self.trade_logs.pop(0)
    
    def apply_compound_interest(self, profit):
        """APLICA INTERÉS COMPUESTO SOLO SOBRE GANANCIAS"""
        if profit > 0:
            # Reinvertir el 80% de las ganancias para crecimiento compuesto
            reinvest_amount = profit * 0.8
            self.compound_growth *= (1 + (reinvest_amount / self.total_equity))
            
    def calculate_dynamic_position(self, price):
        """CALCULA POSICIÓN DINÁMICA CON INTERÉS COMPUESTO"""
        # Base: 2% del capital ajustado por crecimiento compuesto
        base_risk = 0.02
        adjusted_capital = self.total_equity * self.compound_growth
        
        risk_amount = adjusted_capital * base_risk
        position_size = risk_amount / price
        
        # Limitar para alta frecuencia (posiciones más pequeñas)
        max_position = (adjusted_capital * 0.15) / price
        return min(position_size, max_position)
    
    def execute_btc_trade(self, action, side, price):
        """EJECUCIÓN ESPECÍFICA PARA BTC"""
        # Verificar reset diario
        current_date = datetime.now().date()
        if current_date != self.last_daily_reset:
            self.daily_trades = 0
            self.last_daily_reset = current_date
            self.add_log("✅ Reset diario de operaciones")
            
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Límite diario excedido"
        
        # Calcular cantidad dinámica con interés compuesto
        quantity = self.calculate_dynamic_position(price)
        
        # Verificar cantidad mínima de BTC
        if quantity < 0.0001:  # Mínimo operativo
            quantity = 0.0001
            
        cost = quantity * price / self.leverage
        
        if cost > self.cash_balance:
            return False, "Fondos insuficientes"
            
        trade = {
            'timestamp': datetime.now(),
            'symbol': SYMBOL,
            'action': action,
            'side': side,
            'leverage': self.leverage,
            'price': price,
            'quantity': quantity,
            'compound_factor': self.compound_growth
        }
        
        self.open_positions.append(trade)
        self.cash_balance -= cost
        self.trade_count += 1
        self.daily_trades += 1
        self.last_trade_time = datetime.now()
        
        log_msg = f"BTC {action.upper()} {quantity:.6f} @ ${price:.2f}"
        self.add_log(log_msg, "TRADE")
        
        return True, log_msg
    
    def close_btc_position(self, position, close_price):
        """CIERRE CON APLICACIÓN DE INTERÉS COMPUESTO"""
        if position['side'] == 'long':
            pnl = (close_price - position['price']) * position['quantity'] * position['leverage']
        else:
            pnl = (position['price'] - close_price) * position['quantity'] * position['leverage']
            
        initial_cost = position['quantity'] * position['price'] / position['leverage']
        self.cash_balance += initial_cost + pnl
        self.total_equity = self.cash_balance
        
        # APLICAR INTERÉS COMPUESTO SOBRE GANANCIAS
        self.apply_compound_interest(pnl)
        
        self.open_positions.remove(position)
        
        # LOG de cierre
        pnl_type = "GANANCIA" if pnl > 0 else "PÉRDIDA"
        log_msg = f"CIERRE {pnl_type}: ${pnl:.4f} | Precio: ${close_price:.2f}"
        self.add_log(log_msg, "CLOSE")
        
        return pnl
    
    def get_btc_signal(self):
        """SEÑAL DE ALTA FRECUENCIA PARA BTC"""
        # Simular señal de alta frecuencia (reemplazar con lógica real)
        current_time = datetime.now()
        seconds = current_time.second
        
        # Múltiples señales por minuto para alta frecuencia
        if seconds % 15 < 3:  # Señal cada ~15 segundos
            return 'short' if seconds % 30 < 15 else 'long'
        return None

def main():
    st.title("🤖 Bot BTC High-Frequency + Interés Compuesto")
    
    # Inicializar bot
    if 'bot' not in st.session_state:
        st.session_state.bot = TradingBot()
        st.session_state.running = False
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Configuración BTC")
        
        # Estado en línea/desconectado
        status_color = "🟢" if st.session_state.running else "🔴"
        status_text = "EN LÍNEA" if st.session_state.running else "DESCONECTADO"
        st.markdown(f"### {status_color} {status_text}")
        
        if st.button("🚀 Iniciar Bot" if not st.session_state.running else "⏸️ Detener Bot"):
            st.session_state.running = not st.session_state.running
            # Log de cambio de estado
            if st.session_state.running:
                st.session_state.bot.add_log("🟢 BOT INICIADO - En línea")
            else:
                st.session_state.bot.add_log("🔴 BOT DETENIDO - Desconectado")
            
        st.divider()
        st.subheader(f"Par: {SYMBOL}")
        st.write(f"Apalancamiento: {LEVERAGE}x")
        st.write(f"Frecuencia: {MAX_DAILY_TRADES} ops/día")
        st.write(f"Velocidad: {MIN_TIME_BETWEEN_TRADES}s entre trades")
    
    # Panel principal
    bot = st.session_state.bot
    
    # Estado de conexión prominente
    col_status, col1, col2, col3 = st.columns([1, 1, 1, 1])
    
    with col_status:
        if st.session_state.running:
            st.success("🟢 BOT EN LÍNEA")
        else:
            st.error("🔴 BOT DESCONECTADO")
    
    with col1:
        st.metric("💰 Capital BTC", f"${bot.total_equity:.2f}")
    
    with col2:
        growth_pct = (bot.compound_growth - 1) * 100
        st.metric("📈 Interés Compuesto", f"{growth_pct:.2f}%")
    
    with col3:
        st.metric("⚡ Ops Hoy", f"{bot.daily_trades}/{MAX_DAILY_TRADES}")
    
    # Sección de Logs en tiempo real
    st.subheader("📋 Logs de Operaciones en Tiempo Real")
    
    # Crear contenedor para logs
    log_container = st.container()
    
    with log_container:
        # Mostrar logs más recientes primero
        for log in reversed(bot.trade_logs[-20:]):  # Últimos 20 logs
            timestamp_str = log['timestamp'].strftime("%H:%M:%S")
            
            if log['type'] == "TRADE":
                st.success(f"🕒 {timestamp_str} | {log['message']}")
            elif log['type'] == "CLOSE":
                if "GANANCIA" in log['message']:
                    st.success(f"🕒 {timestamp_str} | {log['message']}")
                else:
                    st.error(f"🕒 {timestamp_str} | {log['message']}")
            else:
                st.info(f"🕒 {timestamp_str} | {log['message']}")
    
    # Ejecución de trading BTC
    if st.session_state.running:
        # Precio BTC simulado (reemplazar con API real)
        btc_price = 3448.07 + np.random.normal(0, 15)
        
        # Obtener señal de alta frecuencia
        signal = bot.get_btc_signal()
        
        # Verificar si puede operar (alta frecuencia)
        can_trade = True
        if bot.last_trade_time:
            time_diff = (datetime.now() - bot.last_trade_time).total_seconds()
            if time_diff < MIN_TIME_BETWEEN_TRADES:
                can_trade = False
        
        # Ejecutar operación BTC
        if can_trade and signal and bot.daily_trades < MAX_DAILY_TRADES:
            success, message = bot.execute_btc_trade(
                action='sell' if signal == 'short' else 'buy',
                side=signal,
                price=btc_price
            )
            
            if not success:
                bot.add_log(f"⚠️ {message}", "INFO")
        
        # Cerrar posiciones BTC (alta frecuencia)
        if bot.open_positions and len(bot.open_positions) > 0:
            # Cierre rápido basado en ganancias pequeñas (alta frecuencia)
            for position in bot.open_positions[:]:
                current_pnl = 0
                if position['side'] == 'long':
                    current_pnl = (btc_price - position['price']) * position['quantity'] * position['leverage']
                else:
                    current_pnl = (position['price'] - btc_price) * position['quantity'] * position['leverage']
                
                # Cierre rápido: 0.5% ganancia o 0.3% pérdida
                if current_pnl > position['quantity'] * position['price'] * 0.005 or \
                   current_pnl < -position['quantity'] * position['price'] * 0.003:
                    
                    pnl = bot.close_btc_position(position, btc_price)
                    break
        
        # Actualización rápida para alta frecuencia
        time.sleep(1)
        st.rerun()
    
    # Mostrar posiciones BTC
    if bot.open_positions:
        st.subheader("📊 Posiciones BTC Abiertas")
        for pos in bot.open_positions:
            st.write(f"- {pos['side'].upper()} {pos['quantity']:.6f} BTC @ ${pos['price']:.2f}")
    
    # Estadísticas avanzadas
    with st.expander("📈 Estadísticas BTC Avanzadas"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Capital Inicial", f"${BASE_CAPITAL:.2f}")
            st.metric("Crecimiento Real", f"${bot.total_equity - BASE_CAPITAL:.2f}")
            st.metric("Rendimiento Total", f"{((bot.total_equity - BASE_CAPITAL) / BASE_CAPITAL * 100):.2f}%")
        
        with col2:
            st.metric("Factor Compound", f"{bot.compound_growth:.4f}")
            st.metric("Ops Totales", bot.trade_count)
            st.metric("Eficiencia HF", f"{(bot.daily_trades / MAX_DAILY_TRADES * 100):.1f}%")

if __name__ == "__main__":
    main()
