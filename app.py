import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import time
import hmac
import hashlib
import json

# Configuración de la página
st.set_page_config(
    page_title="Bot Trading MEXC - Estrategia Momentum HFT",
    page_icon="🚀",
    layout="wide"
)

# Cliente MEXC
class MexcClient:
    def __init__(self, api_key=None, api_secret=None):
        self.base_url = "https://api.mexc.com"
        self.api_key = api_key
        self.api_secret = api_secret
        
    def get_klines(self, symbol='BTCUSDT', interval='1m', limit=100):
        """Obtener datos de velas de MEXC"""
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            # Convertir a DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Convertir tipos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            st.error(f"Error obteniendo datos de MEXC: {e}")
            return None

# Estrategia Momentum HFT para MEXC
class MexcMomentumHFT:
    def __init__(self):
        # Parámetros de la estrategia
        self.ema_fast = 8
        self.ema_slow = 21
        self.volume_multiplier = 2.0
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.12
        self.leverage = 3
        
    def calculate_ema(self, prices, period):
        """Calcular EMA manualmente"""
        if len(prices) < period:
            return [None] * len(prices)
        ema = [prices[0]]
        alpha = 2 / (period + 1)
        for price in prices[1:]:
            ema.append(alpha * price + (1 - alpha) * ema[-1])
        return ema
    
    def calculate_indicators(self, df):
        """Calcular indicadores técnicos"""
        df = df.copy()
        
        # EMAs
        df['ema_fast'] = self.calculate_ema(df['close'].tolist(), self.ema_fast)
        df['ema_slow'] = self.calculate_ema(df['close'].tolist(), self.ema_slow)
        
        # Volumen
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_avg']
        
        # VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        return df
    
    def should_enter_long(self, df):
        """Señal de entrada LONG"""
        if len(df) < 30:
            return False
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # EMA rápido cruza arriba de EMA lento
        ema_cross_up = (current['ema_fast'] > current['ema_slow'] and 
                       previous['ema_fast'] <= previous['ema_slow'])
        
        # Confirmación volumen
        volume_confirm = current['volume_ratio'] >= self.volume_multiplier
        
        # Precio sobre VWAP
        price_above_vwap = current['close'] > current['vwap']
        
        return all([ema_cross_up, volume_confirm, price_above_vwap])
    
    def should_enter_short(self, df):
        """Señal de entrada SHORT"""
        if len(df) < 30:
            return False
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # EMA rápido cruza abajo de EMA lento
        ema_cross_down = (current['ema_fast'] < current['ema_slow'] and 
                         previous['ema_fast'] >= previous['ema_slow'])
        
        # Confirmación volumen
        volume_confirm = current['volume_ratio'] >= self.volume_multiplier
        
        # Precio bajo VWAP
        price_below_vwap = current['close'] < current['vwap']
        
        return all([ema_cross_down, volume_confirm, price_below_vwap])
    
    def run_backtest(self, df):
        """Ejecutar backtest de la estrategia"""
        signals = []
        position = None
        entry_price = 0
        
        df = self.calculate_indicators(df)
        
        for i in range(30, len(df)):
            current = df.iloc[i]
            
            # Verificar salidas
            if position == 'long':
                if current['close'] >= entry_price * (1 + self.take_profit_pct / 100):
                    signals.append({
                        'timestamp': current['timestamp'],
                        'action': 'close_long',
                        'price': current['close'],
                        'type': 'take_profit'
                    })
                    position = None
                elif current['close'] <= entry_price * (1 - self.stop_loss_pct / 100):
                    signals.append({
                        'timestamp': current['timestamp'],
                        'action': 'close_long',
                        'price': current['close'],
                        'type': 'stop_loss'
                    })
                    position = None
            
            elif position == 'short':
                if current['close'] <= entry_price * (1 - self.take_profit_pct / 100):
                    signals.append({
                        'timestamp': current['timestamp'],
                        'action': 'close_short',
                        'price': current['close'],
                        'type': 'take_profit'
                    })
                    position = None
                elif current['close'] >= entry_price * (1 + self.stop_loss_pct / 100):
                    signals.append({
                        'timestamp': current['timestamp'],
                        'action': 'close_short',
                        'price': current['close'],
                        'type': 'stop_loss'
                    })
                    position = None
            
            # Verificar entradas
            if not position:
                current_data = df.iloc[:i+1]
                if self.should_enter_long(current_data):
                    position = 'long'
                    entry_price = current['close']
                    signals.append({
                        'timestamp': current['timestamp'],
                        'action': 'enter_long',
                        'price': current['close']
                    })
                elif self.should_enter_short(current_data):
                    position = 'short'
                    entry_price = current['close']
                    signals.append({
                        'timestamp': current['timestamp'],
                        'action': 'enter_short',
                        'price': current['close']
                    })
        
        return signals

def main():
    st.title("🚀 Bot Trading MEXC - Estrategia Momentum HFT")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuración MEXC")
    
    api_key = st.sidebar.text_input("API Key MEXC", type="password")
    api_secret = st.sidebar.text_input("API Secret MEXC", type="password")
    
    st.sidebar.header("🎯 Parámetros Estrategia")
    
    ema_fast = st.sidebar.slider("EMA Rápida", 5, 15, 8)
    ema_slow = st.sidebar.slider("EMA Lenta", 15, 30, 21)
    volume_multiplier = st.sidebar.slider("Múltiplo Volumen", 1.5, 3.0, 2.0)
    stop_loss = st.sidebar.slider("Stop Loss (%)", 0.05, 0.2, 0.08)
    take_profit = st.sidebar.slider("Take Profit (%)", 0.08, 0.25, 0.12)
    
    # Inicializar cliente MEXC
    client = MexcClient(api_key, api_secret)
    strategy = MexcMomentumHFT()
    
    # Actualizar parámetros
    strategy.ema_fast = ema_fast
    strategy.ema_slow = ema_slow
    strategy.volume_multiplier = volume_multiplier
    strategy.stop_loss_pct = stop_loss
    strategy.take_profit_pct = take_profit
    
    # Botón para ejecutar
    if st.sidebar.button("📊 Ejecutar Estrategia en MEXC"):
        with st.spinner("Obteniendo datos de MEXC..."):
            # Obtener datos reales de MEXC
            df = client.get_klines(symbol='BTCUSDT', interval='1m', limit=100)
            
            if df is not None:
                # Ejecutar estrategia
                signals = strategy.run_backtest(df)
                
                # Mostrar resultados
                col1, col2, col3, col4 = st.columns(4)
                
                total_trades = len([s for s in signals if 'enter' in s['action']])
                winning_trades = len([s for s in signals if 'take_profit' in str(s.get('type', ''))])
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                with col1:
                    st.metric("💰 Operaciones Totales", total_trades)
                with col2:
                    st.metric("✅ Operaciones Ganadoras", winning_trades)
                with col3:
                    st.metric("🎯 Tasa de Acierto", f"{win_rate:.1f}%")
                with col4:
                    st.metric("⚡ Leverage", f"{strategy.leverage}x")
                
                # Gráfico de precios y señales
                st.subheader("📈 Precio BTC/USDT y Señales de Trading")
                
                df_with_indicators = strategy.calculate_indicators(df)
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  vertical_spacing=0.05,
                                  subplot_titles=('Precio y EMAs', 'Volumen'),
                                  row_heights=[0.7, 0.3])
                
                # Precio
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'],
                                       name='Precio BTC', line=dict(color='blue')), row=1, col=1)
                
                # EMAs
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df_with_indicators['ema_fast'],
                                       name=f'EMA {ema_fast}', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df_with_indicators['ema_slow'],
                                       name=f'EMA {ema_slow}', line=dict(color='red')), row=1, col=1)
                
                # Señales
                buy_signals = [s for s in signals if s['action'] == 'enter_long']
                sell_signals = [s for s in signals if s['action'] == 'enter_short']
                
                if buy_signals:
                    fig.add_trace(go.Scatter(
                        x=[s['timestamp'] for s in buy_signals],
                        y=[s['price'] for s in buy_signals],
                        mode='markers', name='LONG',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ), row=1, col=1)
                
                if sell_signals:
                    fig.add_trace(go.Scatter(
                        x=[s['timestamp'] for s in sell_signals],
                        y=[s['price'] for s in sell_signals],
                        mode='markers', name='SHORT',
                        marker=dict(color='red', size=10, symbol='triangle-down')
                    ), row=1, col=1)
                
                # Volumen
                fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'],
                                   name='Volumen', marker_color='lightblue'), row=2, col=1)
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar últimas señales
                st.subheader("📋 Historial de Señales")
                if signals:
                    signals_df = pd.DataFrame(signals)
                    st.dataframe(signals_df)
                else:
                    st.info("No se generaron señales en este período")
                
                # Estadísticas de la estrategia
                st.subheader("📊 Métricas de Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **🎯 Estrategia Momentum HFT**
                    - EMAs: {ema_fast}/{ema_slow}
                    - Volumen mínimo: {volume_multiplier}x
                    - Risk/Reward: 1:{take_profit/stop_loss:.1f}
                    """)
                
                with col2:
                    st.info(f"""
                    **🛡️ Gestión de Riesgo**
                    - Stop Loss: {stop_loss}%
                    - Take Profit: {take_profit}%
                    - Leverage: {strategy.leverage}x
                    - Operaciones: {total_trades}
                    """)
    
    # Información de la estrategia
    st.sidebar.header("ℹ️ Estrategia Momentum HFT")
    st.sidebar.write("""
    **Señales de Entrada:**
    - EMA rápida cruza EMA lenta
    - Volumen 2x promedio mínimo
    - Confirmación VWAP
    
    **Timeframe:** 1-minuto
    **Hold Time:** 2-45 segundos
    **Mercado:** MEXC Futures
    """)

if __name__ == "__main__":
    main()
