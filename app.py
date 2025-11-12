import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime

# CONFIGURACIÓN MEJORADA - MÍNIMOS CAMBIOS
LEVERAGE = 2  # CAMBIO: De 3 a 2 para reducir riesgo
MAX_DAILY_TRADES = 25  # CAMBIO: Límite de operaciones diarias
MIN_TIME_BETWEEN_TRADES = 5  # CAMBIO: Segundos entre operaciones

class TradingBot:
    def __init__(self):
        self.leverage = LEVERAGE
        self.cash_balance = 255.0
        self.total_equity = 255.0
        self.open_positions = []
        self.trade_count = 0
        self.last_trade_time = None
        self.daily_trades = 0
        self.last_daily_reset = datetime.now().date()
        
    def improved_entry_signal(self, current_data, market_conditions):
        """MEJORA: Filtros básicos para mejores entradas"""
        # Filtro 1: Tiempo entre operaciones
        if self.last_trade_time:
            time_diff = (datetime.now() - self.last_trade_time).total_seconds()
            if time_diff < MIN_TIME_BETWEEN_TRADES:
                return False, "Esperando entre operaciones"
        
        # Filtro 2: Límite diario
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Límite diario alcanzado"
            
        # Filtro 3: Volatilidad mínima
        if 'volatility' in market_conditions and market_conditions['volatility'] < 0.003:
            return False, "Volatilidad muy baja"
            
        return True, "Condiciones OK"
    
    def execute_trade(self, action, side, price, quantity):
        """EJECUCIÓN CON MEJORAS MÍNIMAS"""
        # Verificar límite diario
        current_date = datetime.now().date()
        if current_date != self.last_daily_reset:
            self.daily_trades = 0
            self.last_daily_reset = current_date
            
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Límite diario excedido"
        
        # Ejecutar normalmente (mantener tu lógica original)
        cost = quantity * price / self.leverage
        
        if cost > self.cash_balance:
            return False, "Fondos insuficientes"
            
        trade = {
            'timestamp': datetime.now(),
            'action': action,
            'side': side,
            'leverage': self.leverage,
            'price': price,
            'quantity': quantity
        }
        
        self.open_positions.append(trade)
        self.cash_balance -= cost
        self.trade_count += 1
        self.daily_trades += 1
        self.last_trade_time = datetime.now()
        
        return True, "Operación exitosa"
    
    def calculate_safe_quantity(self, price, risk_percent=0.01):
        """MEJORA: Cálculo de cantidad más seguro"""
        risk_amount = self.total_equity * risk_percent
        base_quantity = risk_amount / price
        
        # Limitar a máximo 20% del capital
        max_quantity = (self.total_equity * 0.2) / price
        return min(base_quantity, max_quantity)
    
    def close_position(self, position, close_price):
        """MANTENER tu lógica original de cierre"""
        if position['side'] == 'long':
            pnl = (close_price - position['price']) * position['quantity'] * position['leverage']
        else:
            pnl = (position['price'] - close_price) * position['quantity'] * position['leverage']
            
        initial_cost = position['quantity'] * position['price'] / position['leverage']
        self.cash_balance += initial_cost + pnl
        self.total_equity = self.cash_balance
        
        self.open_positions.remove(position)
        return pnl

def main():
    st.title("🤖 Bot Trading - MEXC")
    
    # Inicializar bot
    if 'bot' not in st.session_state:
        st.session_state.bot = TradingBot()
        st.session_state.running = False
    
    # Sidebar
    with st.sidebar:
        st.header("Controles")
        
        if st.button("Iniciar Bot" if not st.session_state.running else "Detener Bot"):
            st.session_state.running = not st.session_state.running
            
        st.divider()
        st.subheader("Configuración Mejorada")
        st.write(f"Apalancamiento: {LEVERAGE}x")
        st.write(f"Límite diario: {MAX_DAILY_TRADES} operaciones")
        st.write(f"Tiempo entre trades: {MIN_TIME_BETWEEN_TRADES}s")
    
    # Panel principal
    col1, col2, col3, col4 = st.columns(4)
    
    bot = st.session_state.bot
    
    with col1:
        st.metric("💰 Capital", f"${bot.total_equity:.2f}")
    
    with col2:
        st.metric("📊 Operaciones Hoy", bot.daily_trades)
    
    with col3:
        st.metric("🎯 Apalancamiento", f"{bot.leverage}x")
    
    with col4:
        st.metric("📈 Posiciones Abiertas", len(bot.open_positions))
    
    # Simulación de trading
    if st.session_state.running:
        st.success("✅ Bot ejecutándose con configuración mejorada")
        
        # Simular datos de mercado (MANTENER tu lógica original)
        current_price = 3448.07 + np.random.normal(0, 8)
        
        # MEJORA: Usar estrategia mejorada
        market_conditions = {
            'volatility': np.random.uniform(0.002, 0.01),
            'trend': 'neutral'
        }
        
        can_trade, reason = bot.improved_entry_signal(None, market_conditions)
        
        if can_trade and np.random.random() > 0.6:  # 40% probabilidad
            quantity = bot.calculate_safe_quantity(current_price)
            
            success, message = bot.execute_trade(
                action='sell',
                side='short', 
                price=current_price,
                quantity=quantity
            )
            
            if success:
                st.info(f"📈 Nueva operación: {quantity:.6f} BTC @ ${current_price:.2f}")
            else:
                st.warning(f"❌ {message}")
        elif not can_trade:
            st.write(f"⏳ {reason}")
        
        # Cerrar posiciones (MANTENER tu lógica original)
        if bot.open_positions and np.random.random() > 0.7:
            position = bot.open_positions[0]
            pnl = bot.close_position(position, current_price)
            color = "green" if pnl > 0 else "red"
            st.markdown(f"<span style='color:{color}'>🔒 Posición cerrada: ${pnl:.4f}</span>", unsafe_allow_html=True)
        
        # Actualizar cada 3 segundos
        time.sleep(3)
        st.rerun()
    
    # Mostrar posiciones
    if bot.open_positions:
        st.subheader("Posiciones Abiertas")
        for pos in bot.open_positions:
            st.write(f"- {pos['side']} {pos['quantity']:.6f} @ ${pos['price']:.2f}")
    
    # Estadísticas
    with st.expander("📊 Estadísticas Detalladas"):
        st.write(f"Capital inicial: $255.00")
        st.write(f"Capital actual: ${bot.total_equity:.2f}")
        st.write(f"Rendimiento: {((bot.total_equity - 255) / 255 * 100):.2f}%")
        st.write(f"Operaciones totales: {bot.trade_count}")
        st.write(f"Operaciones hoy: {bot.daily_trades}/{MAX_DAILY_TRADES}")

if __name__ == "__main__":
    main()
