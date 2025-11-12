import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import json

# Configuración de página
st.set_page_config(
    page_title="Bot Trading Mejorado",
    page_icon="🤖",
    layout="wide"
)

class SafeTradingBot:
    def __init__(self, initial_capital=255.0):
        # CONFIGURACIÓN SEGURA - MÍNIMOS CAMBIOS
        self.leverage = 2  # Reducido de 3x a 2x
        self.risk_per_trade = 0.01  # 1% riesgo
        self.stop_loss_pct = 0.015  # 1.5% stop loss
        self.take_profit_pct = 0.020  # 2.0% take profit
        
        # Estado
        self.cash_balance = initial_capital
        self.total_equity = initial_capital
        self.open_positions = []
        self.trade_history = []
        
        # Stats
        self.winning_trades = 0
        self.total_trades = 0
        
    def calculate_safe_position(self, price):
        """Calcula posición segura"""
        risk_amount = self.total_equity * self.risk_per_trade
        position_size = risk_amount / price
        return min(position_size, (self.total_equity * 0.3) / price)
    
    def execute_trade(self, action, side, price, quantity):
        """Ejecuta operación con protección"""
        cost = quantity * price / self.leverage
        
        if cost > self.cash_balance:
            return False, "Fondos insuficientes"
            
        # Crear trade
        trade = {
            'timestamp': datetime.now(),
            'action': action,
            'side': side,
            'leverage': self.leverage,
            'price': price,
            'quantity': quantity,
            'stop_loss': price * (1 - self.stop_loss_pct) if side == 'long' else price * (1 + self.stop_loss_pct),
            'take_profit': price * (1 + self.take_profit_pct) if side == 'long' else price * (1 - self.take_profit_pct)
        }
        
        self.open_positions.append(trade)
        self.cash_balance -= cost
        self.total_trades += 1
        
        return True, "Operación exitosa"
    
    def check_stop_loss_take_profit(self, current_price):
        """Verifica condiciones de salida"""
        closed_positions = []
        
        for pos in self.open_positions[:]:
            if (pos['side'] == 'long' and current_price <= pos['stop_loss']) or \
               (pos['side'] == 'short' and current_price >= pos['stop_loss']):
                # Stop loss hit
                pnl = self.calculate_pnl(pos, current_price)
                closed_positions.append(('STOP_LOSS', pos, pnl))
                self.open_positions.remove(pos)
                
            elif (pos['side'] == 'long' and current_price >= pos['take_profit']) or \
                 (pos['side'] == 'short' and current_price <= pos['take_profit']):
                # Take profit hit
                pnl = self.calculate_pnl(pos, current_price)
                closed_positions.append(('TAKE_PROFIT', pos, pnl))
                self.open_positions.remove(pos)
                
        return closed_positions
    
    def calculate_pnl(self, position, close_price):
        """Calcula P&L"""
        if position['side'] == 'long':
            pnl = (close_price - position['price']) * position['quantity'] * position['leverage']
        else:
            pnl = (position['price'] - close_price) * position['quantity'] * position['leverage']
            
        # Actualizar balance
        initial_cost = position['quantity'] * position['price'] / position['leverage']
        self.cash_balance += initial_cost + pnl
        self.total_equity = self.cash_balance
        
        if pnl > 0:
            self.winning_trades += 1
            
        return pnl
    
    def get_stats(self):
        """Obtiene estadísticas"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        return {
            'win_rate': win_rate,
            'total_trades': self.total_trades,
            'equity': self.total_equity,
            'open_positions': len(self.open_positions)
        }

def main():
    st.title("🤖 Bot Trading Mejorado - MEXC")
    
    # Inicializar bot en session state
    if 'bot' not in st.session_state:
        st.session_state.bot = SafeTradingBot(initial_capital=255.0)
        st.session_state.running = False
        st.session_state.last_update = datetime.now()
    
    # Sidebar - Controles
    with st.sidebar:
        st.header("🎛 Controles")
        
        if st.button("▶️ Iniciar Bot" if not st.session_state.running else "⏸️ Detener Bot"):
            st.session_state.running = not st.session_state.running
            
        st.divider()
        
        # Configuración
        st.subheader("⚙️ Configuración")
        leverage = st.selectbox("Apalancamiento", [1, 2, 3], index=1)
        risk = st.slider("Riesgo por Operación (%)", 0.5, 5.0, 1.0)
        
        # Actualizar configuración
        st.session_state.bot.leverage = leverage
        st.session_state.bot.risk_per_trade = risk / 100
    
    # Panel principal
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "💰 Capital Actual", 
            f"${st.session_state.bot.total_equity:.2f}",
            delta=f"${st.session_state.bot.total_equity - 255.0:.2f}" if st.session_state.bot.total_equity != 255.0 else None
        )
    
    with col2:
        stats = st.session_state.bot.get_stats()
        st.metric("🎯 Tasa de Acierto", f"{stats['win_rate']:.1f}%")
    
    with col3:
        st.metric("📊 Operaciones Totales", stats['total_trades'])
    
    # Simulación de trading
    if st.session_state.running:
        st.info("🟢 Bot ejecutándose...")
        
        # Simular datos de mercado (reemplazar con API real)
        current_price = 3448.07 + np.random.normal(0, 10)
        
        # Verificar condiciones de salida
        closed_positions = st.session_state.bot.check_stop_loss_take_profit(current_price)
        
        for reason, pos, pnl in closed_positions:
            st.warning(f"🔒 {reason}: P&L ${pnl:.4f}")
        
        # Simular señal de trading (reemplazar con lógica real)
        if np.random.random() > 0.7:  # 30% probabilidad de señal
            position_size = st.session_state.bot.calculate_safe_position(current_price)
            success, msg = st.session_state.bot.execute_trade(
                action='sell',
                side='short',
                price=current_price,
                quantity=position_size
            )
            
            if success:
                st.success(f"✅ {msg} - Precio: ${current_price:.2f}")
        
        # Actualizar cada 2 segundos
        time.sleep(2)
        st.rerun()
    else:
        st.warning("⏸️ Bot detenido")
    
    # Mostrar posiciones abiertas
    if st.session_state.bot.open_positions:
        st.subheader("📈 Posiciones Abiertas")
        positions_df = pd.DataFrame(st.session_state.bot.open_positions)
        st.dataframe(positions_df[['timestamp', 'side', 'price', 'quantity', 'stop_loss', 'take_profit']])
    
    # Mostrar estadísticas detalladas
    with st.expander("📊 Estadísticas Detalladas"):
        stats = st.session_state.bot.get_stats()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Apalancamiento", f"{st.session_state.bot.leverage}x")
        with col2:
            st.metric("Stop Loss", f"{st.session_state.bot.stop_loss_pct*100:.1f}%")
        with col3:
            st.metric("Take Profit", f"{st.session_state.bot.take_profit_pct*100:.1f}%")
        with col4:
            st.metric("Posiciones Abiertas", stats['open_positions'])

if __name__ == "__main__":
    main()
