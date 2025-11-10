import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# Configuración de la página
st.set_page_config(
    page_title="Bot de Trading Avanzado",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TradingBot:
    def __init__(self):
        self.initial_balance = 250.0
        self.current_balance = self.initial_balance
        self.trade_history = []
        self.price_data = []
        self.indicators_history = []
        
        self.config = {
            'risk_per_trade': 0.02,
            'target_profit': 0.004,
            'stop_loss': 0.002,
            'max_daily_trades': 12,
        }
        
        self.daily_trades_count = 0
        self.is_running = False
        
    def initialize_price_data(self):
        """Inicializa datos de precio simulados"""
        base_price = 50000
        self.price_data = []
        for i in range(200):
            change = np.random.normal(0, 0.015)
            price = base_price * (1 + change)
            self.price_data.append(price)
            base_price = price
    
    def calculate_indicators(self, prices: list) -> dict:
        """Calcula indicadores técnicos"""
        if len(prices) < 20:
            return {}
            
        df = pd.DataFrame(prices, columns=['close'])
        
        # Medias móviles
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD simple
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        current = df.iloc[-1]
        
        return {
            'price': current['close'],
            'sma_20': current['sma_20'],
            'sma_50': current['sma_50'],
            'rsi': current['rsi'],
            'macd': current['macd'],
            'macd_signal': current['macd_signal'],
            'trend': 'ALCISTA' if current['sma_20'] > current['sma_50'] else 'BAJISTA',
            'momentum': 'ALCISTA' if current['macd'] > current['macd_signal'] else 'BAJISTA'
        }
    
    def generate_signal(self, indicators: dict) -> dict:
        """Genera señal de trading"""
        if not indicators:
            return {'action': 'MANTENER', 'confidence': 0}
        
        bullish_conditions = 0
        bearish_conditions = 0
        
        # Condiciones alcistas
        if indicators['trend'] == 'ALCISTA':
            bullish_conditions += 1
        if indicators['momentum'] == 'ALCISTA':
            bullish_conditions += 1
        if 40 < indicators['rsi'] < 65:
            bullish_conditions += 1
        if indicators['price'] > indicators['sma_20']:
            bullish_conditions += 1
            
        # Condiciones bajistas  
        if indicators['trend'] == 'BAJISTA':
            bearish_conditions += 1
        if indicators['momentum'] == 'BAJISTA':
            bearish_conditions += 1
        if 35 < indicators['rsi'] < 60:
            bearish_conditions += 1
        if indicators['price'] < indicators['sma_20']:
            bearish_conditions += 1
        
        confidence = max(bullish_conditions, bearish_conditions) / 4.0
        
        if bullish_conditions >= 3 and confidence > 0.65:
            return {
                'action': 'COMPRAR',
                'confidence': confidence,
                'price': indicators['price'],
                'conditions': f'Alcistas: {bullish_conditions}/4'
            }
        elif bearish_conditions >= 3 and confidence > 0.65:
            return {
                'action': 'VENDER', 
                'confidence': confidence,
                'price': indicators['price'],
                'conditions': f'Bajistas: {bearish_conditions}/4'
            }
        else:
            return {
                'action': 'MANTENER', 
                'confidence': confidence,
                'conditions': f'Alc: {bullish_conditions}/4, Baj: {bearish_conditions}/4'
            }
    
    def calculate_position_size(self, price: float) -> float:
        """Calcula tamaño de posición"""
        risk_amount = self.current_balance * self.config['risk_per_trade']
        position_value = risk_amount / self.config['stop_loss']
        quantity = position_value / price
        
        max_position_value = self.current_balance * 0.6
        if position_value > max_position_value:
            quantity = (max_position_value / price)
        
        return quantity
    
    def execute_trade(self, signal: dict):
        """Ejecuta operación de trading simulada"""
        if self.daily_trades_count >= self.config['max_daily_trades']:
            return
        
        action = signal['action']
        price = signal['price']
        quantity = self.calculate_position_size(price)
        
        # Simular orden
        trade_value = quantity * price
        fee = trade_value * 0.001
        
        if action == 'COMPRAR':
            self.current_balance -= trade_value + fee
            pnl = 0  # Se calculará al cerrar
        else:  # VENDER
            self.current_balance += trade_value - fee
            pnl = 0
        
        # Registrar trade
        trade = {
            'timestamp': datetime.now(),
            'action': action,
            'price': price,
            'quantity': quantity,
            'value': trade_value,
            'fee': fee,
            'confidence': signal['confidence'],
            'conditions': signal['conditions']
        }
        
        self.trade_history.append(trade)
        self.daily_trades_count += 1

def main():
    # Título principal
    st.title("🤖 Bot de Trading Avanzado - MEXC")
    st.markdown("---")
    
    # Inicializar bot en session state
    if 'bot' not in st.session_state:
        st.session_state.bot = TradingBot()
        st.session_state.bot.initialize_price_data()
    
    bot = st.session_state.bot
    
    # Sidebar para controles
    with st.sidebar:
        st.header("🎛️ Controles del Bot")
        
        st.subheader("Configuración de Riesgo")
        bot.config['risk_per_trade'] = st.slider(
            "Riesgo por Operación (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5
        ) / 100
        
        bot.config['target_profit'] = st.slider(
            "Take Profit (%)",
            min_value=0.1,
            max_value=2.0,
            value=0.4,
            step=0.1
        ) / 100
        
        bot.config['stop_loss'] = st.slider(
            "Stop Loss (%)", 
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.1
        ) / 100
        
        st.subheader("Control de Ejecución")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Iniciar Bot", type="primary"):
                bot.is_running = True
                st.rerun()
        
        with col2:
            if st.button("⏹️ Detener Bot"):
                bot.is_running = False
                st.rerun()
        
        if st.button("🔄 Reiniciar Simulación"):
            bot.__init__()
            bot.initialize_price_data()
            st.rerun()
        
        st.markdown("---")
        st.subheader("📊 Estado Actual")
        st.metric("Balance", f"${bot.current_balance:.2f}")
        st.metric("Trades Hoy", bot.daily_trades_count)
        st.metric("Trades Totales", len(bot.trade_history))
        
        if bot.trade_history:
            last_trade = bot.trade_history[-1]
            st.metric("Última Operación", 
                     f"{last_trade['action']} @ ${last_trade['price']:.2f}")
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Gráfico de Precios y Señales")
        
        # Actualizar datos si el bot está corriendo
        if bot.is_running:
            # Agregar nuevo precio
            last_price = bot.price_data[-1] if bot.price_data else 50000
            change = np.random.normal(0, 0.01)
            new_price = last_price * (1 + change)
            bot.price_data.append(new_price)
            
            # Mantener solo últimos 200 puntos
            if len(bot.price_data) > 200:
                bot.price_data.pop(0)
            
            # Calcular indicadores y señal
            indicators = bot.calculate_indicators(bot.price_data)
            if indicators:
                bot.indicators_history.append(indicators)
                signal = bot.generate_signal(indicators)
                
                # Ejecutar trade si hay señal
                if signal['action'] in ['COMPRAR', 'VENDER']:
                    bot.execute_trade(signal)
        
        # Crear gráfico de precios
        if bot.price_data:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Precio BTC y Medias Móviles', 'RSI'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Precios y medias móviles
            x_axis = list(range(len(bot.price_data)))
            fig.add_trace(
                go.Scatter(x=x_axis, y=bot.price_data, name='Precio BTC', line=dict(color='blue')),
                row=1, col=1
            )
            
            if bot.indicators_history:
                sma_20 = [ind.get('sma_20', 0) for ind in bot.indicators_history]
                sma_50 = [ind.get('sma_50', 0) for ind in bot.indicators_history]
                
                fig.add_trace(
                    go.Scatter(x=x_axis[-len(sma_20):], y=sma_20, name='SMA 20', line=dict(color='orange')),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(x=x_axis[-len(sma_50):], y=sma_50, name='SMA 50', line=dict(color='red')),
                    row=1, col=1
                )
            
            # RSI
            if bot.indicators_history:
                rsi_values = [ind.get('rsi', 50) for ind in bot.indicators_history]
                fig.add_trace(
                    go.Scatter(x=x_axis[-len(rsi_values):], y=rsi_values, name='RSI', line=dict(color='purple')),
                    row=2, col=1
                )
                
                # Líneas RSI
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Métricas en Tiempo Real")
        
        if bot.indicators_history:
            current = bot.indicators_history[-1]
            signal = bot.generate_signal(current)
            
            # Métricas principales
            col_met1, col_met2 = st.columns(2)
            with col_met1:
                st.metric("Precio Actual", f"${current['price']:.2f}")
                st.metric("RSI", f"{current['rsi']:.1f}")
            with col_met2:
                st.metric("Tendencia", current['trend'])
                st.metric("Momentum", current['momentum'])
            
            st.subheader("🎯 Señal Actual")
            
            # Mostrar señal con color
            if signal['action'] == 'COMPRAR':
                st.success(f"**{signal['action']}** - Confianza: {signal['confidence']:.1%}")
            elif signal['action'] == 'VENDER':
                st.error(f"**{signal['action']}** - Confianza: {signal['confidence']:.1%}")
            else:
                st.info(f"**{signal['action']}** - Confianza: {signal['confidence']:.1%}")
            
            st.write(f"Condiciones: {signal['conditions']}")
            
            # Performance
            st.subheader("💰 Performance")
            total_pnl = bot.current_balance - bot.initial_balance
            roi = (total_pnl / bot.initial_balance) * 100
            
            st.metric("Ganancia/Pérdida Total", f"${total_pnl:.2f}")
            st.metric("ROI Total", f"{roi:.2f}%")
            
            if bot.trade_history:
                winning_trades = len([t for t in bot.trade_history if t.get('pnl', 0) > 0])
                win_rate = (winning_trades / len(bot.trade_history)) * 100
                st.metric("Tasa de Acierto", f"{win_rate:.1f}%")
        
        st.subheader("📋 Últimas Operaciones")
        if bot.trade_history:
            recent_trades = bot.trade_history[-5:]  # Últimos 5 trades
            for trade in reversed(recent_trades):
                time_str = trade['timestamp'].strftime("%H:%M:%S")
                if trade['action'] == 'COMPRAR':
                    st.success(f"🟢 {time_str} - {trade['action']} ${trade['price']:.2f}")
                else:
                    st.error(f"🔴 {time_str} - {trade['action']} ${trade['price']:.2f}")
        else:
            st.info("No hay operaciones aún")
    
    # Auto-refresh si el bot está corriendo
    if bot.is_running:
        time.sleep(1)  # Espera 1 segundo entre actualizaciones
        st.rerun()

if __name__ == "__main__":
    main()
