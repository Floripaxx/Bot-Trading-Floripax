import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import hmac
import hashlib
import base64
from datetime import datetime
from urllib.parse import urlencode

# CONFIGURACIÓN MEXC API
MEXC_BASE_URL = "https://api.mexc.com"
API_KEY = "tu_api_key_aqui"  # 🔑 Configurar en secrets
SECRET_KEY = "tu_secret_key_aqui"  # 🔑 Configurar en secrets

SYMBOL = "BTCUSDT"
LEVERAGE = 3
MAX_DAILY_TRADES = 200  # 🔥 ALTA FRECUENCIA
MIN_TIME_BETWEEN_TRADES = 1  # 🔥 ALTA FRECUENCIA
BASE_CAPITAL = 255.0

class MexcTradingBot:
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
        self.profits_reinvested = 0.0  # 🔥 INTERÉS COMPUESTO
        self.initial_capital = BASE_CAPITAL
        
    def sign_request(self, params):
        """Firma la solicitud para MEXC API"""
        query_string = urlencode(params)
        signature = hmac.new(
            SECRET_KEY.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        return params
    
    def mexc_api_request(self, endpoint, params=None, method='GET'):
        """Realiza solicitud a MEXC API"""
        if params is None:
            params = {}
        
        params['timestamp'] = int(time.time() * 1000)
        params['recvWindow'] = 5000
        
        signed_params = self.sign_request(params)
        
        headers = {
            'X-MEXC-APIKEY': API_KEY,
            'Content-Type': 'application/json'
        }
        
        url = f"{MEXC_BASE_URL}{endpoint}"
        
        try:
            if method == 'GET':
                response = requests.get(url, params=signed_params, headers=headers, timeout=5)
            else:
                response = requests.post(url, params=signed_params, headers=headers, timeout=5)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.add_log(f"❌ Error API MEXC: {str(e)}", "ERROR")
            return None
    
    def get_btc_price(self):
        """Obtiene precio REAL de BTC desde MEXC"""
        try:
            endpoint = "/api/v3/ticker/price"
            params = {'symbol': SYMBOL}
            data = self.mexc_api_request(endpoint, params)
            if data:
                price = float(data['price'])
                self.add_log(f"📊 Precio BTC: ${price:.2f}", "INFO")
                return price
        except Exception as e:
            self.add_log(f"Error obteniendo precio: {e}", "ERROR")
        
        # Fallback a precio simulado si hay error
        return 34450.25 + np.random.normal(0, 25)
    
    def execute_real_trade(self, side, quantity):
        """Ejecuta operación REAL en MEXC"""
        try:
            endpoint = "/api/v3/order"
            params = {
                'symbol': SYMBOL,
                'side': side.upper(),
                'type': 'MARKET',
                'quantity': round(quantity, 6),
                'timestamp': int(time.time() * 1000)
            }
            
            data = self.mexc_api_request(endpoint, params, 'POST')
            if data:
                order_id = data.get('orderId', 'N/A')
                self.add_log(f"✅ Orden {order_id} ejecutada: {side} {quantity:.6f} BTC", "TRADE")
                return True, f"Orden {order_id} ejecutada"
            return False, "Error en ejecución"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
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
        """🔥 INTERÉS COMPUESTO REAL - Aumenta tamaño de posición con ganancias"""
        if profit > 0:
            # Reinvertir el 100% de las ganancias para crecimiento compuesto agresivo
            self.profits_reinvested += profit
            self.compound_growth = 1.0 + (self.profits_reinvested / self.initial_capital)
            self.add_log(f"💰 Interés Compuesto: +${profit:.4f} | Factor: {self.compound_growth:.4f}", "COMPOUND")
            
    def calculate_position(self, price):
        """CALCULA POSICIÓN CON INTERÉS COMPUESTO"""
        base_risk = 0.02
        # 🔥 Capital ajustado por interés compuesto
        adjusted_capital = self.total_equity * self.compound_growth
        risk_amount = adjusted_capital * base_risk
        position_size = risk_amount / price
        max_position = (adjusted_capital * 0.15) / price
        return min(position_size, max_position)
    
    def execute_trade(self, action, side, price):
        """EJECUTA OPERACIÓN CON MEXC"""
        current_date = datetime.now().date()
        if current_date != self.last_daily_reset:
            self.daily_trades = 0
            self.last_daily_reset = current_date
            self.add_log("🔄 Reset diario completado", "INFO")
            
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Límite diario alcanzado"
        
        quantity = self.calculate_position(price)
        if quantity < 0.0001:
            quantity = 0.0001
            
        cost = quantity * price / self.leverage
        
        if cost > self.cash_balance:
            return False, "Fondos insuficientes"
        
        # 🔥 EJECUCIÓN REAL EN MEXC
        side_map = {'buy': 'BUY', 'sell': 'SELL'}
        success, message = self.execute_real_trade(side_map[action], quantity)
        
        if success:
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
        else:
            return False, message
    
    def close_position(self, position, close_price):
        """CIERRA POSICIÓN CON INTERÉS COMPUESTO"""
        # Ejecutar orden de cierre en MEXC
        close_side = 'BUY' if position['side'] == 'short' else 'SELL'
        success, message = self.execute_real_trade(close_side, position['quantity'])
        
        if not success:
            return 0
        
        if position['side'] == 'long':
            pnl = (close_price - position['price']) * position['quantity'] * self.leverage
        else:
            pnl = (position['price'] - close_price) * position['quantity'] * self.leverage
            
        initial_cost = position['quantity'] * position['price'] / self.leverage
        self.cash_balance += initial_cost + pnl
        self.total_equity = self.cash_balance
        
        # 🔥 APLICAR INTERÉS COMPUESTO
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
            
            # Cierre rápido para alta frecuencia
            if abs(pnl) > position['quantity'] * position['price'] * 0.003:  # 0.3%
                positions_to_close.append(position)
        
        return positions_to_close
    
    def get_signal(self):
        """GENERA SEÑAL DE ALTA FRECUENCIA"""
        current_time = datetime.now()
        seconds = current_time.second
        # 🔥 ALTA FRECUENCIA: Señal cada 10 segundos
        if seconds % 10 < 2:
            return 'short' if seconds % 20 < 10 else 'long'
        return None

def main():
    st.set_page_config(page_title="Bot HF MEXC", layout="wide")
    st.title("🤖 Bot Alta Frecuencia BTC - MEXC API REAL")
    
    # Configurar API Keys (usar secrets en producción)
    if 'api_configured' not in st.session_state:
        st.session_state.api_configured = False
    
    with st.sidebar:
        st.header("🔑 Configuración MEXC API")
        api_key = st.text_input("API Key", type="password")
        secret_key = st.text_input("Secret Key", type="password")
        
        if st.button("🔗 Conectar con MEXC"):
            if api_key and secret_key:
                st.session_state.api_key = api_key
                st.session_state.secret_key = secret_key
                st.session_state.api_configured = True
                st.success("✅ API Configurada")
            else:
                st.error("❌ Ingresa ambas claves API")
    
    if not st.session_state.get('api_configured', False):
        st.warning("⚠️ Configura las API Keys de MEXC en el sidebar")
        return
    
    # Configurar API keys globales
    global API_KEY, SECRET_KEY
    API_KEY = st.session_state.api_key
    SECRET_KEY = st.session_state.secret_key
    
    # Inicializar bot
    if 'bot' not in st.session_state:
        st.session_state.bot = MexcTradingBot()
        st.session_state.running = False
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Control")
        
        status = "🟢 EN LÍNEA" if st.session_state.running else "🔴 DESCONECTADO"
        st.subheader(status)
        
        if st.button("🚀 Iniciar Bot HF" if not st.session_state.running else "⏸️ Detener Bot"):
            st.session_state.running = not st.session_state.running
            if st.session_state.running:
                st.session_state.bot.add_log("🚀 BOT HF INICIADO - MEXC API")
            else:
                st.session_state.bot.add_log("🛑 BOT DETENIDO")
        
        st.divider()
        st.write(f"**Par:** {SYMBOL}")
        st.write(f"**Apalancamiento:** {LEVERAGE}x")
        st.write(f"**Frecuencia:** {MAX_DAILY_TRADES} ops/día")
        st.write(f"**Velocidad:** {MIN_TIME_BETWEEN_TRADES}s")
        st.write(f"**Interés Compuesto:** ACTIVADO")
    
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
    st.subheader("📋 Logs MEXC en Tiempo Real")
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
                elif log['type'] == "COMPOUND":
                    st.info(f"🕒 {time_str} | {log['message']}")
                else:
                    st.info(f"🕒 {time_str} | {log['message']}")
        else:
            st.info("No hay logs aún...")
    
    # Ejecución del bot
    if st.session_state.running:
        # 🔥 Obtener precio REAL de MEXC
        btc_price = bot.get_btc_price()
        
        # Verificar tiempo entre operaciones (ALTA FRECUENCIA)
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
            if position in bot.open_positions:
                bot.close_position(position, btc_price)
        
        time.sleep(1)  # 🔥 ALTA FRECUENCIA: 1 segundo
        st.rerun()
    
    # Posiciones abiertas
    if bot.open_positions:
        st.subheader("📊 Posiciones Abiertas MEXC")
        for pos in bot.open_positions:
            st.write(f"- {pos['side'].upper()} {pos['quantity']:.6f} BTC @ ${pos['price']:.2f}")
    else:
        st.info("No hay posiciones abiertas")
    
    # Estadísticas
    with st.expander("📈 Estadísticas HF MEXC"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Capital inicial:** ${BASE_CAPITAL:.2f}")
            st.write(f"**Ganancia/Perdida:** ${bot.total_equity - BASE_CAPITAL:.2f}")
            st.write(f"**Rendimiento:** {((bot.total_equity - BASE_CAPITAL) / BASE_CAPITAL * 100):.2f}%")
        with col2:
            st.write(f"**Operaciones totales:** {bot.trade_count}")
            st.write(f"**Factor compound:** {bot.compound_growth:.4f}")
            st.write(f"**Ganancias reinvertidas:** ${bot.profits_reinvested:.4f}")

if __name__ == "__main__":
    main()
