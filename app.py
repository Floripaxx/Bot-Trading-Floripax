import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime

# CONFIGURACIÓN BTC
SYMBOL = "BTCUSDT"
LEVERAGE = 3
MAX_DAILY_TRADES = 50
MIN_TIME_BETWEEN_TRADES = 3
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
        self.compound_growth = 1.0
        self.trade_logs = []
        
    def add_log(self, message, log_type="INFO"):
        """AGREGA LOGS"""
        log_entry = {
            'timestamp': datetime.now(),
            'type': log_type,
            'message': message
        }
        self.trade_logs.append(log_entry)
        if len(self.trade_logs) > 30:
            self.trade_logs.pop(0)
    
    def apply_compound_interest(self, profit):
        """INTERÉS COMPUESTO"""
        if profit > 0:
            reinvest_amount = profit * 0.8
            self.compound_growth *= (1 + (reinvest_amount / self.total_equity))
            
    def calculate_position(self, price):
        """CALCULA POSICIÓN"""
        base_risk = 0.02
        adjusted_capital = self.total_equity * self.compound_growth
        risk_amount = adjusted_capital * base_risk
        position_size = risk_amount / price
        max_position = (adjusted_capital * 0.15) / price
        return min(position_size, max_position)
    
    def execute_trade(self, action, side, price):
        """EJECUTA OPERACIÓN"""
        current_date = datetime.now().date()
        if current_date != self.last_daily_reset:
            self.daily_trades = 0
            self.last_daily_reset = current_date
            self.add_log("🔄 Reset diario completado")
            
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Límite diario alcanzado"
        
        quantity = self.calculate_position(price)
        if quantity < 0.0001:
            quantity = 0.0001
            
        cost = quantity * price / self.leverage
        
        if cost > self.cash_balance:
            return False, "Fondos insuficientes"
            
        trade = {
            'timestamp': datetime.now(),
            'symbol': SYMBOL,
            'action': action,
            'side': side,
            'price': price,
            'quantity': quantity,
        }
        
        self.open_positions.append(trade)
        self.cash_balance -= cost
        self.trade_count += 1
        self.daily_trades += 1
        self.last_trade_time = datetime.now()
        
        log_msg = f"💰 {action.upper()} {quantity:.6f} BTC @ ${price:.2f}"
        self.add_log(log_msg, "TRADE")
        return True, log_msg
    
    def close_position(self, position, close_price):
        """CIERRA POSICIÓN"""
        if position['side'] == 'long':
            pnl = (close_price - position['price']) * position['quantity'] * self.leverage
        else:
            pnl = (position['price'] - close_price) * position['quantity'] * self.leverage
            
        initial_cost = position['quantity'] * position['price'] / self.leverage
        self.cash_balance += initial_cost + pnl
        self.total_equity = self.cash_balance
        
        self.apply_compound_interest(pnl)
        self.open_positions.remove(position)
        
        pnl_type = "✅ GANANCIA" if pnl > 0 else "❌ PÉRDIDA"
        log_msg = f"{pnl_type}: ${pnl:.4f}"
        self.add_log(log_msg, "CLOSE")
        return pnl
    
    def check_positions_for_close(self, current_price):
        """VERIFICA POSICIONES PARA CERRAR"""
        positions_to_close = []
        
        for position in self.open_positions:
            if position['side'] == 'long':
                pnl = (current_price - position['price']) * position['quantity'] * self.leverage
            else:
                pnl = (position['price'] - current_price) * position['quantity'] * self.leverage
            
            # Cierre rápido: 0.4% de ganancia/pérdida
            if abs(pnl) > position['quantity'] * position['price'] * 0.004:
                positions_to_close.append(position)
        
        return positions_to_close
    
    def get_signal(self):
        """GENERA SEÑAL"""
        current_time = datetime.now()
        seconds = current_time.second
        if seconds % 20 < 4:
            return 'short' if seconds % 40 < 20 else 'long'
        return None

def main():
    st.set_page_config(page_title="Bot Trading BTC", layout="wide")
    st.title("🤖 Bot Trading BTC - MEXC")
    
    # Inicializar bot
    if 'bot' not in st.session_state:
        st.session_state.bot = TradingBot()
        st.session_state.running = False
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Control")
        
        # Estado
        status = "🟢 EN LÍNEA" if st.session_state.running else "🔴 DESCONECTADO"
        st.subheader(status)
        
        if st.button("🚀 Iniciar Bot" if not st.session_state.running else "⏸️ Detener Bot"):
            st.session_state.running = not st.session_state.running
            if st.session_state.running:
                st.session_state.bot.add_log("🚀 BOT INICIADO")
            else:
                st.session_state.bot.add_log("🛑 BOT DETENIDO")
        
        st.divider()
        st.write(f"**Par:** {SYMBOL}")
        st.write(f"**Apalancamiento:** {LEVERAGE}x")
        st.write(f"**Límite diario:** {MAX_DAILY_TRADES} ops")
        st.write(f"**Velocidad:** {MIN_TIME_BETWEEN_TRADES}s")
    
    # Métricas principales
    bot = st.session_state.bot
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Capital", f"${bot.total_equity:.2f}")
    with col2:
        growth = (bot.compound_growth - 1) * 100
        st.metric("📈 Compound", f"{growth:.2f}%")
    with col3:
        st.metric("⚡ Ops Hoy", f"{bot.daily_trades}/{MAX_DAILY_TRADES}")
    with col4:
        st.metric("🎯 Abiertas", len(bot.open_positions))
    
    # Logs en tiempo real
    st.subheader("📋 Logs de Operaciones")
    logs_container = st.container()
    
    with logs_container:
        if bot.trade_logs:
            for log in reversed(bot.trade_logs[-15:]):
                time_str = log['timestamp'].strftime("%H:%M:%S")
                if log['type'] == "TRADE":
                    st.success(f"🕒 {time_str} | {log['message']}")
                elif log['type'] == "CLOSE":
                    if "GANANCIA" in log['message']:
                        st.success(f"🕒 {time_str} | {log['message']}")
                    else:
                        st.error(f"🕒 {time_str} | {log['message']}")
                else:
                    st.info(f"🕒 {time_str} | {log['message']}")
        else:
            st.info("No hay logs aún...")
    
    # Ejecución del bot
    if st.session_state.running:
        # Precio simulado
        btc_price = 34450.25 + np.random.normal(0, 25)
        
        # Verificar tiempo entre operaciones
        can_trade = True
        if bot.last_trade_time:
            time_diff = (datetime.now() - bot.last_trade_time).total_seconds()
            if time_diff < MIN_TIME_BETWEEN_TRADES:
                can_trade = False
        
        # Generar señal y operar
        signal = bot.get_signal()
        if can_trade and signal and bot.daily_trades < MAX_DAILY_TRADES:
            success, message = bot.execute_trade(
                action='sell' if signal == 'short' else 'buy',
                side=signal,
                price=btc_price
            )
            if not success:
                bot.add_log(f"⚠️ {message}")
        
        # Verificar y cerrar posiciones
        positions_to_close = bot.check_positions_for_close(btc_price)
        for position in positions_to_close:
            if position in bot.open_positions:  # Verificar que aún existe
                bot.close_position(position, btc_price)
        
        time.sleep(2)
        st.rerun()
    
    # Posiciones abiertas
    if bot.open_positions:
        st.subheader("📊 Posiciones Abiertas")
        for pos in bot.open_positions:
            st.write(f"- {pos['side'].upper()} {pos['quantity']:.6f} BTC @ ${pos['price']:.2f}")
    else:
        st.info("No hay posiciones abiertas")
    
    # Estadísticas
    with st.expander("📈 Estadísticas Detalladas"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Capital inicial:** ${BASE_CAPITAL:.2f}")
            st.write(f"**Ganancia/Perdida:** ${bot.total_equity - BASE_CAPITAL:.2f}")
            st.write(f"**Rendimiento:** {((bot.total_equity - BASE_CAPITAL) / BASE_CAPITAL * 100):.2f}%")
        with col2:
            st.write(f"**Operaciones totales:** {bot.trade_count}")
            st.write(f"**Factor compound:** {bot.compound_growth:.4f}")
            st.write(f"**Eficiencia:** {(bot.daily_trades / MAX_DAILY_TRADES * 100):.1f}%" if MAX_DAILY_TRADES > 0 else "0%")

if __name__ == "__main__":
    main()
