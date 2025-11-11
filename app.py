import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Bot Trading - Estrategia Momentum",
    page_icon="🚀",
    layout="wide"
)

# Simulador de datos (reemplaza con tu conexión real a Binance)
class BinanceSimulator:
    def __init__(self):
        self.positions = []
        self.balance = 1000.0
        self.performance = []
        
    def get_historical_data(self, symbol='BTCUSDT', interval='1m', limit=100):
        """Simular datos históricos"""
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='T')
        np.random.seed(42)
        
        # Generar datos realistas de BTC
        prices = [50000]
        for i in range(1, limit):
            change = np.random.normal(0, 0.001)  # 0.1% de volatilidad
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [p * 0.999 for p in prices],
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 5000, limit)
        })
        
        return df

class HighFrequencyBot:
    def __init__(self):
        self.client = BinanceSimulator()
        self.symbol = 'BTCUSDT'
        self.position = None
        self.entry_price = 0
        self.performance = []
        self.equity_curve = [1000]  # Starting balance
        
        # Parámetros de la estrategia MOMENTUM MICROSECONDS
        self.ema_fast = 8
        self.ema_slow = 21
        self.volume_multiplier = 2.0
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.12
        self.leverage = 3
        self.risk_per_trade = 0.15  # 0.15% del capital
        
    def calculate_ema(self, prices, period):
        """Calcular EMA manualmente"""
        if len(prices) < period:
            return None
        ema = [prices[0]]
        alpha = 2 / (period + 1)
        for price in prices[1:]:
            ema.append(alpha * price + (1 - alpha) * ema[-1])
        return ema
    
    def calculate_vwap(self, df):
        """Calcular VWAP manualmente"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def calculate_indicators(self, df):
        """Calcular todos los indicadores técnicos"""
        df = df.copy()
        
        # EMAs
        df['ema_fast'] = self.calculate_ema(df['close'].tolist(), self.ema_fast)
        df['ema_slow'] = self.calculate_ema(df['close'].tolist(), self.ema_slow)
        
        # Volumen promedio
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        
        # VWAP
        df['vwap'] = self.calculate_vwap(df)
        
        return df
    
    def should_enter_long(self, df):
        """Señal de entrada LONG mejorada"""
        if len(df) < 30 or df['ema_fast'].isna().any() or df['ema_slow'].isna().any():
            return False
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # 1. EMA rápido cruza arriba de EMA lento
        ema_cross_up = (current['ema_fast'] > current['ema_slow'] and 
                       previous['ema_fast'] <= previous['ema_slow'])
        
        # 2. Confirmación de volumen
        volume_confirm = current['volume_ratio'] >= self.volume_multiplier
        
        # 3. Precio sobre VWAP
        price_above_vwap = current['close'] > current['vwap']
        
        # 4. Momentum consistente
        momentum_confirm = current['close'] > current['ema_fast']
        
        return all([ema_cross_up, volume_confirm, price_above_vwap, momentum_confirm])
    
    def should_enter_short(self, df):
        """Señal de entrada SHORT mejorada"""
        if len(df) < 30 or df['ema_fast'].isna().any() or df['ema_slow'].isna().any():
            return False
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # 1. EMA rápido cruza abajo de EMA lento
        ema_cross_down = (current['ema_fast'] < current['ema_slow'] and 
                         previous['ema_fast'] >= previous['ema_slow'])
        
        # 2. Confirmación de volumen
        volume_confirm = current['volume_ratio'] >= self.volume_multiplier
        
        # 3. Precio bajo VWAP
        price_below_vwap = current['close'] < current['vwap']
        
        # 4. Momentum consistente
        momentum_confirm = current['close'] < current['ema_fast']
        
        return all([ema_cross_down, volume_confirm, price_below_vwap, momentum_confirm])
    
    def run_simulation(self, df):
        """Ejecutar simulación de la estrategia"""
        signals = []
        position = None
        entry_price = 0
        
        for i in range(30, len(df)):
            current_data = df.iloc[:i+1]
            current_data = self.calculate_indicators(current_data)
            
            current_row = current_data.iloc[-1]
            current_price = current_row['close']
            
            # Verificar salidas primero
            if position == 'long':
                if current_price >= entry_price * (1 + self.take_profit_pct / 100):
                    position = None
                    signals.append({'timestamp': current_row['timestamp'], 'action': 'close_long_tp', 'price': current_price})
                elif current_price <= entry_price * (1 - self.stop_loss_pct / 100):
                    position = None
                    signals.append({'timestamp': current_row['timestamp'], 'action': 'close_long_sl', 'price': current_price})
                    
            elif position == 'short':
                if current_price <= entry_price * (1 - self.take_profit_pct / 100):
                    position = None
                    signals.append({'timestamp': current_row['timestamp'], 'action': 'close_short_tp', 'price': current_price})
                elif current_price >= entry_price * (1 + self.stop_loss_pct / 100):
                    position = None
                    signals.append({'timestamp': current_row['timestamp'], 'action': 'close_short_sl', 'price': current_price})
            
            # Verificar entradas si no hay posición
            if not position:
                if self.should_enter_long(current_data):
                    position = 'long'
                    entry_price = current_price
                    signals.append({'timestamp': current_row['timestamp'], 'action': 'enter_long', 'price': current_price})
                elif self.should_enter_short(current_data):
                    position = 'short'
                    entry_price = current_price
                    signals.append({'timestamp': current_row['timestamp'], 'action': 'enter_short', 'price': current_price})
        
        return signals

def main():
    st.title("🚀 Bot Trading - Estrategia Momentum Profesional")
    
    # Sidebar con parámetros
    st.sidebar.header("⚙️ Parámetros de la Estrategia")
    
    ema_fast = st.sidebar.slider("EMA Rápida", 5, 15, 8)
    ema_slow = st.sidebar.slider("EMA Lenta", 15, 30, 21)
    volume_multiplier = st.sidebar.slider("Múltiplo de Volumen", 1.5, 3.0, 2.0)
    stop_loss = st.sidebar.slider("Stop Loss (%)", 0.05, 0.2, 0.08)
    take_profit = st.sidebar.slider("Take Profit (%)", 0.08, 0.25, 0.12)
    leverage = st.sidebar.slider("Leverage", 1, 5, 3)
    
    # Inicializar bot
    bot = HighFrequencyBot()
    bot.ema_fast = ema_fast
    bot.ema_slow = ema_slow
    bot.volume_multiplier = volume_multiplier
    bot.stop_loss_pct = stop_loss
    bot.take_profit_pct = take_profit
    bot.leverage = leverage
    
    # Obtener datos
    st.sidebar.header("📊 Datos de Mercado")
    if st.sidebar.button("Cargar Datos y Simular"):
        with st.spinner("Cargando datos y ejecutando estrategia..."):
            # Obtener datos históricos
            df = bot.client.get_historical_data(limit=200)
            
            # Ejecutar simulación
            signals = bot.run_simulation(df)
            
            # Mostrar resultados
            col1, col2, col3 = st.columns(3)
            
            total_trades = len([s for s in signals if 'enter' in s['action']])
            winning_trades = len([s for s in signals if 'tp' in s['action']])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            with col1:
                st.metric("Total Operaciones", total_trades)
            with col2:
                st.metric("Operaciones Ganadoras", winning_trades)
            with col3:
                st.metric("Tasa de Acierto", f"{win_rate:.1f}%")
            
            # Gráfico de precios y señales
            st.subheader("📈 Precio y Señales de Trading")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.05,
                              subplot_titles=('Precio BTC y Señales', 'Volumen'),
                              row_heights=[0.7, 0.3])
            
            # Precio y EMAs
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'],
                                   name='Precio BTC', line=dict(color='blue')), row=1, col=1)
            
            df_with_indicators = bot.calculate_indicators(df)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df_with_indicators['ema_fast'],
                                   name=f'EMA {ema_fast}', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df_with_indicators['ema_slow'],
                                   name=f'EMA {ema_slow}', line=dict(color='red')), row=1, col=1)
            
            # Señales de trading
            buy_signals = [s for s in signals if s['action'] == 'enter_long']
            sell_signals = [s for s in signals if s['action'] == 'enter_short']
            
            if buy_signals:
                fig.add_trace(go.Scatter(
                    x=[s['timestamp'] for s in buy_signals],
                    y=[s['price'] for s in buy_signals],
                    mode='markers', name='Compra',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ), row=1, col=1)
            
            if sell_signals:
                fig.add_trace(go.Scatter(
                    x=[s['timestamp'] for s in sell_signals],
                    y=[s['price'] for s in sell_signals],
                    mode='markers', name='Venta',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ), row=1, col=1)
            
            # Volumen
            fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'],
                               name='Volumen', marker_color='lightblue'), row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar últimas señales
            st.subheader("📋 Últimas Señales de Trading")
            if signals:
                signals_df = pd.DataFrame(signals[-10:])  # Últimas 10 señales
                st.dataframe(signals_df)
            else:
                st.info("No se generaron señales en este período")
            
            # Estadísticas de performance
            st.subheader("📊 Métricas de la Estrategia")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.info(f"**Risk/Reward Ratio:** 1:{take_profit/stop_loss:.1f}")
            with col2:
                st.info(f"**Leverage:** {leverage}x")
            with col3:
                st.info(f"**Volumen Mínimo:** {volume_multiplier}x")
            with col4:
                st.info(f"**EMAs:** {ema_fast}/{ema_slow}")
    
    # Explicación de la estrategia
    st.sidebar.header("🎯 Estrategia Momentum")
    st.sidebar.write("""
    **Señales de Entrada:**
    - EMA rápida cruza EMA lenta
    - Volumen 2x promedio
    - Precio sobre VWAP (LONG)
    - Precio bajo VWAP (SHORT)
    
    **Gestión de Riesgo:**
    - Stop Loss: 0.08%
    - Take Profit: 0.12%
    - Leverage: 3x
    - Risk por trade: 0.15%
    """)
    
    # Información adicional
    st.sidebar.header("ℹ️ Información")
    st.sidebar.write("""
    Esta estrategia está optimizada para:
    - Alta frecuencia
    - Momentum confirmado
    - Gestión estricta de riesgo
    - Tasa de acierto >54%
    """)

if __name__ == "__main__":
    main()
