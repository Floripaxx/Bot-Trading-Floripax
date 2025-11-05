import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configurar página de Streamlit
st.set_page_config(
    page_title="Bot de Trading FloripaX - MEXC",
    page_icon="📈",
    layout="wide"
)

# Título principal
st.title("🤖 Bot de Trading FloripaX - MEXC")
st.markdown("---")

# Función para obtener datos de MEXC
def obtener_datos_mexc(symbol='BTCUSDT', interval='1m', limit=100):
    """
    Obtener datos de MEXC API
    """
    try:
        # Mapeo de intervalos de Streamlit a MEXC
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        mexc_interval = interval_map.get(interval, '1m')
        
        url = f"https://api.mexc.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': mexc_interval,
            'limit': limit
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code != 200:
            st.error(f"Error API MEXC: {response.status_code}")
            return None
            
        data = response.json()
        
        if not data or len(data) == 0:
            st.error("No se recibieron datos de MEXC")
            return None
        
        # Crear DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convertir tipos de datos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        if len(df) == 0:
            st.error("No hay datos válidos después de limpiar NaN")
            return None
            
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except requests.exceptions.Timeout:
        st.error("⏰ Timeout conectando a MEXC API")
        return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Error de conexión con MEXC API")
        return None
    except Exception as e:
        st.error(f"❌ Error obteniendo datos de MEXC: {e}")
        return None

# Función para calcular RSI de manera robusta
def calcular_rsi(df, period=14):
    """Calcular RSI con manejo de errores"""
    try:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Reemplazar infinitos y NaN
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
        return rsi
    except Exception as e:
        st.error(f"Error calculando RSI: {e}")
        return pd.Series([50] * len(df))

# Función para calcular Bandas de Bollinger
def calcular_bb(df, period=20, std=2):
    """Calcular Bandas de Bollinger"""
    try:
        df = df.copy()
        df['bb_middle'] = df['close'].rolling(window=period, min_periods=1).mean()
        bb_std = df['close'].rolling(window=period, min_periods=1).std()
        
        df['bb_upper'] = df['bb_middle'] + (bb_std * std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std)
        
        return df['bb_upper'], df['bb_middle'], df['bb_lower']
    except Exception as e:
        st.error(f"Error calculando BB: {e}")
        return None, None, None

# Función para calcular Estocástico
def calcular_estocastico(df, k_period=14, d_period=3):
    """Calcular Estocástico"""
    try:
        low_min = df['low'].rolling(window=k_period, min_periods=1).min()
        high_max = df['high'].rolling(window=k_period, min_periods=1).max()
        
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_k'] = df['stoch_k'].replace([np.inf, -np.inf], 50).fillna(50)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period, min_periods=1).mean()
        
        return df['stoch_k'], df['stoch_d']
    except Exception as e:
        st.error(f"Error calculando Estocástico: {e}")
        return None, None

# Función principal de señales
def obtener_senal_compra_venta(df):
    """
    Obtener señal de compra o venta basada en múltiples indicadores
    """
    try:
        if df is None or len(df) < 20:
            return False, False, "Datos insuficientes"
        
        # Calcular indicadores
        df['rsi'] = calcular_rsi(df)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calcular_bb(df)
        df['stoch_k'], df['stoch_d'] = calcular_estocastico(df)
        
        # Obtener últimos valores
        ultimo = df.iloc[-1]
        
        # CONDICIONES DE COMPRA
        condicion_compra_rsi = (ultimo['rsi'] < 35)
        condicion_compra_bb = (ultimo['close'] < ultimo['bb_lower'])
        condicion_compra_stoch = (ultimo['stoch_k'] < 20 and 
                                ultimo['stoch_k'] > ultimo['stoch_d'])
        
        # Señal de compra (más estricta - requiere 2 de 3 condiciones)
        condiciones_compra = [condicion_compra_rsi, condicion_compra_bb, condicion_compra_stoch]
        senal_compra = sum(condiciones_compra) >= 2
        
        # CONDICIONES DE VENTA
        condicion_venta_rsi = (ultimo['rsi'] > 65)
        condicion_venta_bb = (ultimo['close'] > ultimo['bb_upper'])
        condicion_venta_stoch = (ultimo['stoch_k'] > 80 and 
                               ultimo['stoch_k'] < ultimo['stoch_d'])
        
        # Señal de venta (más estricta - requiere 2 de 3 condiciones)
        condiciones_venta = [condicion_venta_rsi, condicion_venta_bb, condicion_venta_stoch]
        senal_venta = sum(condiciones_venta) >= 2
        
        # Determinar mensaje de estado
        if senal_compra:
            mensaje = "🔵 SEÑAL DE COMPRA FUERTE"
        elif senal_venta:
            mensaje = "🔴 SEÑAL DE VENTA FUERTE"
        else:
            mensaje = "⚪ SIN SEÑAL - ESPERANDO"
            
        return senal_compra, senal_venta, mensaje
        
    except Exception as e:
        st.error(f"Error en señales: {e}")
        return False, False, f"Error: {e}"

# Función para crear gráfico
def crear_grafico(df, senal_compra, senal_venta):
    """Crear gráfico interactivo con Plotly"""
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Precio y Bandas de Bollinger', 'RSI'),
            row_heights=[0.7, 0.3]
        )
        
        # Gráfico de precio y BB
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['close'], 
                      name='Precio', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        if 'bb_upper' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['bb_upper'], 
                          name='BB Superior', line=dict(color='red', width=1, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['bb_lower'], 
                          name='BB Inferior', line=dict(color='green', width=1, dash='dash')),
                row=1, col=1
            )
            
            # Destacar señales en el gráfico
            if senal_compra:
                fig.add_annotation(x=df['timestamp'].iloc[-1], y=df['close'].iloc[-1],
                                 text="COMPRA", showarrow=True, arrowhead=2,
                                 bgcolor="green", font=dict(color="white"),
                                 row=1, col=1)
            elif senal_venta:
                fig.add_annotation(x=df['timestamp'].iloc[-1], y=df['close'].iloc[-1],
                                 text="VENTA", showarrow=True, arrowhead=2,
                                 bgcolor="red", font=dict(color="white"),
                                 row=1, col=1)
        
        # Gráfico de RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['rsi'], 
                          name='RSI', line=dict(color='purple', width=1)),
                row=2, col=1
            )
        
        # Líneas de referencia RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        # Actualizar layout
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Análisis Técnico en Tiempo Real - MEXC"
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creando gráfico: {e}")
        return None

# Interfaz principal
def main():
    try:
        # Sidebar para controles
        st.sidebar.title("⚙️ Configuración MEXC")
        
        # Selector de símbolo (pares populares en MEXC)
        simbolo = st.sidebar.selectbox(
            "Seleccionar Par",
            ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'ATOMUSDT', 'NEARUSDT', 'APTUSDT']
        )
        
        # Selector de intervalo
        intervalo = st.sidebar.selectbox(
            "Intervalo",
            ['1m', '5m', '15m', '1h', '4h']
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info("**ℹ️ Información MEXC**\n\n- API pública sin necesidad de key\n- Límite: 1200 requests por minuto\n- Datos en tiempo real")
        
        # Botón para actualizar
        if st.sidebar.button("🔄 Actualizar Datos MEXC"):
            st.rerun()
        
        # Obtener datos de MEXC
        with st.spinner('Obteniendo datos de MEXC...'):
            df = obtener_datos_mexc(symbol=simbolo, interval=intervalo)
        
        if df is not None and len(df) > 0:
            # Calcular señales
            senal_compra, senal_venta, mensaje = obtener_senal_compra_venta(df)
            
            # Mostrar estado principal
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Precio Actual", f"${df['close'].iloc[-1]:.2f}")
            
            with col2:
                if senal_compra:
                    st.success("SEÑAL: COMPRA")
                elif senal_venta:
                    st.error("SEÑAL: VENTA")
                else:
                    st.info("SEÑAL: NEUTRAL")
            
            with col3:
                st.metric("Última Actualización", 
                         df['timestamp'].iloc[-1].strftime('%H:%M:%S'))
            
            st.markdown(f"### 📊 {mensaje}")
            st.markdown("---")
            
            # Mostrar gráfico
            fig = crear_grafico(df, senal_compra, senal_venta)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar datos técnicos
            st.subheader("📈 Datos Técnicos MEXC")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'rsi' in df.columns:
                    rsi_actual = df['rsi'].iloc[-1]
                    color_rsi = "red" if rsi_actual > 70 else "green" if rsi_actual < 30 else "gray"
                    st.metric("RSI", f"{rsi_actual:.2f}", delta=None, delta_color=color_rsi)
            
            with col2:
                if 'stoch_k' in df.columns:
                    stoch_actual = df['stoch_k'].iloc[-1]
                    color_stoch = "red" if stoch_actual > 80 else "green" if stoch_actual < 20 else "gray"
                    st.metric("Estocástico K", f"{stoch_actual:.2f}", delta=None, delta_color=color_stoch)
            
            with col3:
                if 'bb_upper' in df.columns:
                    precio_actual = df['close'].iloc[-1]
                    bb_upper = df['bb_upper'].iloc[-1]
                    bb_lower = df['bb_lower'].iloc[-1]
                    if bb_upper != bb_lower:  # Evitar división por cero
                        pos_bb = ((precio_actual - bb_lower) / (bb_upper - bb_lower)) * 100
                        st.metric("Posición BB", f"{pos_bb:.1f}%")
                    else:
                        st.metric("Posición BB", "N/A")
            
            with col4:
                volumen_actual = df['volume'].iloc[-1]
                volumen_promedio = df['volume'].tail(20).mean()
                if volumen_promedio > 0:
                    cambio_volumen = ((volumen_actual - volumen_promedio) / volumen_promedio) * 100
                    st.metric("Volumen vs Promedio", f"{cambio_volumen:.1f}%")
                else:
                    st.metric("Volumen", f"{volumen_actual:.2f}")
            
            # Mostrar últimos datos
            st.subheader("📋 Últimos Precios MEXC")
            st.dataframe(df.tail(10)[['timestamp', 'open', 'high', 'low', 'close', 'volume']].round(4), 
                        use_container_width=True)
            
        else:
            st.error("❌ No se pudieron obtener datos de MEXC.")
            st.info("💡 Soluciones posibles:")
            st.info("1. Verifica tu conexión a internet")
            st.info("2. El par seleccionado podría no existir en MEXC")
            st.info("3. La API de MEXC podría estar temporalmente saturada")
            st.info("4. Intenta con un intervalo diferente")
            
    except Exception as e:
        st.error(f"❌ Error en la aplicación: {e}")
        st.info("💡 Intenta recargar la página")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
