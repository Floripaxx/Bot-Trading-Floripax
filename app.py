import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import json

# Configurar logging para diagnóstico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurar página de Streamlit
st.set_page_config(
    page_title="Bot de Trading FloripaX - MEXC",
    page_icon="📈",
    layout="wide"
)

# Título principal
st.title("🤖 Bot de Trading FloripaX - MEXC")
st.markdown("---")

# Estado inicial del bot con persistencia mejorada
if 'bot_initialized' not in st.session_state:
    st.session_state.bot_initialized = True
    st.session_state.capital = 250.0
    st.session_state.operaciones_activas = []
    st.session_state.historial_operaciones = []
    st.session_state.bot_activo = False
    st.session_state.ultima_actualizacion = datetime.now()
    st.session_state.errores_conexion = 0
    st.session_state.debug_info = []

# Función para guardar estado del bot
def guardar_estado_bot():
    """Guardar estado del bot para persistencia entre sesiones"""
    try:
        estado = {
            'capital': st.session_state.capital,
            'operaciones_activas': st.session_state.operaciones_activas,
            'historial_operaciones': st.session_state.historial_operaciones,
            'bot_activo': st.session_state.bot_activo,
            'ultima_actualizacion': st.session_state.ultima_actualizacion.isoformat()
        }
        with open('estado_bot.json', 'w') as f:
            json.dump(estado, f, default=str)
    except Exception as e:
        logger.error(f"Error guardando estado: {e}")

# Función para cargar estado del bot
def cargar_estado_bot():
    """Cargar estado del bot desde archivo"""
    try:
        with open('estado_bot.json', 'r') as f:
            estado = json.load(f)
            st.session_state.capital = estado['capital']
            st.session_state.operaciones_activas = estado['operaciones_activas']
            st.session_state.historial_operaciones = estado['historial_operaciones']
            st.session_state.bot_activo = estado['bot_activo']
            st.session_state.ultima_actualizacion = datetime.fromisoformat(estado['ultima_actualizacion'])
    except FileNotFoundError:
        logger.info("No se encontró archivo de estado, usando valores por defecto")
    except Exception as e:
        logger.error(f"Error cargando estado: {e}")

# Cargar estado al inicio
cargar_estado_bot()

# Función mejorada para obtener datos de MEXC
def obtener_datos_mexc(symbol='BTCUSDT', interval='1m', limit=100):
    """
    Obtener datos de MEXC API con manejo robusto de errores
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
        
        url = "https://api.mexc.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': mexc_interval,
            'limit': limit
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        
        # Timeout más agresivo y reintentos
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code != 200:
            error_msg = f"Error API MEXC: {response.status_code}"
            logger.error(error_msg)
            st.session_state.errores_conexion += 1
            return None
            
        data = response.json()
        
        if not data or len(data) == 0:
            error_msg = "No se recibieron datos de MEXC"
            logger.warning(error_msg)
            return None
        
        # El formato de MEXC tiene 8 columnas:
        columnas_mexc = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume'
        ]
        
        # Crear DataFrame con el formato correcto de MEXC
        df = pd.DataFrame(data, columns=columnas_mexc)
        
        # Convertir tipos de datos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        if len(df) == 0:
            error_msg = "No hay datos válidos después de limpiar NaN"
            logger.warning(error_msg)
            return None
        
        logger.info(f"Datos obtenidos exitosamente: {symbol} - {len(df)} registros")
        st.session_state.errores_conexion = 0  # Resetear contador de errores
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
    except requests.exceptions.Timeout:
        error_msg = "⏰ Timeout conectando a MEXC API"
        logger.error(error_msg)
        st.session_state.errores_conexion += 1
        return None
    except requests.exceptions.ConnectionError:
        error_msg = "🔌 Error de conexión con MEXC API"
        logger.error(error_msg)
        st.session_state.errores_conexion += 1
        return None
    except Exception as e:
        error_msg = f"❌ Error obteniendo datos de MEXC: {str(e)}"
        logger.error(error_msg)
        st.session_state.errores_conexion += 1
        return None

# Función para calcular RSI de manera robusta
def calcular_rsi(df, period=14):
    """Calcular RSI con manejo de errores mejorado"""
    try:
        if len(df) < period:
            return pd.Series([50] * len(df))
            
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Reemplazar infinitos y NaN
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
        
        # Asegurar que RSI esté entre 0 y 100
        rsi = rsi.clip(0, 100)
        
        return rsi
    except Exception as e:
        logger.error(f"Error calculando RSI: {e}")
        return pd.Series([50] * len(df))

# Función para calcular Bandas de Bollinger mejorada
def calcular_bb(df, period=20, std=2):
    """Calcular Bandas de Bollinger con mejor manejo de bordes"""
    try:
        df = df.copy()
        
        if len(df) < period:
            # Si no hay suficientes datos, usar valores por defecto
            middle = df['close'].mean() if len(df) > 0 else 0
            return (pd.Series([middle + middle * 0.1] * len(df)), 
                    pd.Series([middle] * len(df)), 
                    pd.Series([middle - middle * 0.1] * len(df)))
        
        df['bb_middle'] = df['close'].rolling(window=period, min_periods=1).mean()
        bb_std = df['close'].rolling(window=period, min_periods=1).std()
        
        df['bb_upper'] = df['bb_middle'] + (bb_std * std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * std)
        
        # Rellenar NaN con valores extrapolados
        df['bb_upper'] = df['bb_upper'].fillna(method='bfill').fillna(method='ffill')
        df['bb_lower'] = df['bb_lower'].fillna(method='bfill').fillna(method='ffill')
        df['bb_middle'] = df['bb_middle'].fillna(method='bfill').fillna(method='ffill')
        
        return df['bb_upper'], df['bb_middle'], df['bb_lower']
    except Exception as e:
        logger.error(f"Error calculando BB: {e}")
        middle = df['close'].mean() if len(df) > 0 else 0
        return (pd.Series([middle] * len(df)), 
                pd.Series([middle] * len(df)), 
                pd.Series([middle] * len(df)))

# Función para calcular Estocástico
def calcular_estocastico(df, k_period=14, d_period=3):
    """Calcular Estocástico con manejo de errores"""
    try:
        if len(df) < k_period:
            return (pd.Series([50] * len(df)), 
                    pd.Series([50] * len(df)))
            
        low_min = df['low'].rolling(window=k_period, min_periods=1).min()
        high_max = df['high'].rolling(window=k_period, min_periods=1).max()
        
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_k'] = df['stoch_k'].replace([np.inf, -np.inf], 50).fillna(50)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period, min_periods=1).mean()
        
        # Asegurar que esté entre 0 y 100
        df['stoch_k'] = df['stoch_k'].clip(0, 100)
        df['stoch_d'] = df['stoch_d'].clip(0, 100)
        
        return df['stoch_k'], df['stoch_d']
    except Exception as e:
        logger.error(f"Error calculando Estocástico: {e}")
        return (pd.Series([50] * len(df)), 
                pd.Series([50] * len(df)))

# Función mejorada para ejecutar operación de compra
def ejecutar_compra(symbol, precio, cantidad_usdt):
    """Ejecutar operación de compra con logging"""
    try:
        if cantidad_usdt > st.session_state.capital:
            logger.warning(f"Capital insuficiente: {cantidad_usdt} > {st.session_state.capital}")
            return False
            
        cantidad_cripto = cantidad_usdt / precio
        operacion = {
            'id': f"compra_{int(time.time())}_{np.random.randint(1000, 9999)}",
            'symbol': symbol,
            'tipo': 'COMPRA',
            'precio_entrada': precio,
            'cantidad_cripto': cantidad_cripto,
            'cantidad_usdt': cantidad_usdt,
            'timestamp': datetime.now(),
            'estado': 'ACTIVA'
        }
        st.session_state.operaciones_activas.append(operacion)
        st.session_state.capital -= cantidad_usdt
        st.session_state.ultima_actualizacion = datetime.now()
        
        logger.info(f"Compra ejecutada: {symbol} - ${precio:.4f} - ${cantidad_usdt:.2f}")
        guardar_estado_bot()
        return True
    except Exception as e:
        logger.error(f"Error ejecutando compra: {e}")
        return False

# Función mejorada para ejecutar operación de venta
def ejecutar_venta(operacion, precio_venta):
    """Ejecutar operación de venta con logging"""
    try:
        ganancia_perdida = (precio_venta - operacion['precio_entrada']) * operacion['cantidad_cripto']
        
        # Mover de activas a historial
        st.session_state.operaciones_activas = [op for op in st.session_state.operaciones_activas if op['id'] != operacion['id']]
        
        operacion_cerrada = operacion.copy()
        operacion_cerrada.update({
            'precio_salida': precio_venta,
            'ganancia_perdida': ganancia_perdida,
            'timestamp_cierre': datetime.now(),
            'estado': 'CERRADA'
        })
        
        st.session_state.historial_operaciones.append(operacion_cerrada)
        st.session_state.capital += operacion['cantidad_usdt'] + ganancia_perdida
        st.session_state.ultima_actualizacion = datetime.now()
        
        logger.info(f"Venta ejecutada: {operacion['symbol']} - Ganancia: ${ganancia_perdida:.2f}")
        guardar_estado_bot()
        return True
    except Exception as e:
        logger.error(f"Error ejecutando venta: {e}")
        return False

# Función principal de señales MEJORADA
def obtener_senal_compra_venta(df):
    """
    Obtener señal de compra o venta con criterios más flexibles
    """
    try:
        if df is None or len(df) < 20:
            return False, False, "Datos insuficientes", {}
        
        # Calcular indicadores
        df['rsi'] = calcular_rsi(df)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calcular_bb(df)
        df['stoch_k'], df['stoch_d'] = calcular_estocastico(df)
        
        # Obtener últimos valores
        ultimo = df.iloc[-1]
        
        # DEBUG: Información de diagnóstico
        debug_info = {
            'precio': ultimo['close'],
            'rsi': ultimo['rsi'],
            'bb_upper': ultimo['bb_upper'],
            'bb_lower': ultimo['bb_lower'],
            'stoch_k': ultimo['stoch_k'],
            'stoch_d': ultimo['stoch_d']
        }
        
        # CONDICIONES DE COMPRA MÁS FLEXIBLES
        condicion_compra_rsi = (ultimo['rsi'] < 40)  # Cambiado de 35 a 40
        condicion_compra_bb = (ultimo['close'] <= ultimo['bb_lower'] * 1.02)  # 2% de tolerancia
        condicion_compra_stoch = (ultimo['stoch_k'] < 25 and  # Cambiado de 20 a 25
                                ultimo['stoch_k'] > ultimo['stoch_d'])
        
        # Señal de compra (más flexible - requiere solo 2 de 3 condiciones)
        condiciones_compra = [condicion_compra_rsi, condicion_compra_bb, condicion_compra_stoch]
        senal_compra = sum(condiciones_compra) >= 2
        
        # CONDICIONES DE VENTA MÁS FLEXIBLES
        condicion_venta_rsi = (ultimo['rsi'] > 60)  # Cambiado de 65 a 60
        condicion_venta_bb = (ultimo['close'] >= ultimo['bb_upper'] * 0.98)  # 2% de tolerancia
        condicion_venta_stoch = (ultimo['stoch_k'] > 75 and  # Cambiado de 80 a 75
                               ultimo['stoch_k'] < ultimo['stoch_d'])
        
        # Señal de venta (más flexible - requiere solo 2 de 3 condiciones)
        condiciones_venta = [condicion_venta_rsi, condicion_venta_bb, condicion_venta_stoch]
        senal_venta = sum(condiciones_venta) >= 2
        
        # Determinar mensaje de estado
        if senal_compra:
            mensaje = "🔵 SEÑAL DE COMPRA DETECTADA"
        elif senal_venta:
            mensaje = "🔴 SEÑAL DE VENTA DETECTADA"
        else:
            mensaje = "⚪ SIN SEÑAL - ESPERANDO"
            
        return senal_compra, senal_venta, mensaje, debug_info
        
    except Exception as e:
        error_msg = f"Error en señales: {e}"
        logger.error(error_msg)
        return False, False, f"Error: {e}", {}

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
                                 text="🎯 COMPRA", showarrow=True, arrowhead=2,
                                 bgcolor="green", font=dict(color="white", size=12),
                                 row=1, col=1)
            elif senal_venta:
                fig.add_annotation(x=df['timestamp'].iloc[-1], y=df['close'].iloc[-1],
                                 text="🎯 VENTA", showarrow=True, arrowhead=2,
                                 bgcolor="red", font=dict(color="white", size=12),
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

# Función para alternar el estado del bot con persistencia
def toggle_bot():
    """Alternar entre ejecución y detención del bot"""
    st.session_state.bot_activo = not st.session_state.bot_activo
    st.session_state.ultima_actualizacion = datetime.now()
    
    estado = "ACTIVADO" if st.session_state.bot_activo else "DETENIDO"
    logger.info(f"Bot {estado}")
    guardar_estado_bot()

# Función para forzar sincronización
def sincronizar_estado():
    """Forzar sincronización del estado del bot"""
    cargar_estado_bot()
    st.session_state.ultima_actualizacion = datetime.now()
    st.success("✅ Estado sincronizado correctamente")
    logger.info("Estado del bot sincronizado")

# Interfaz principal MEJORADA
def main():
    try:
        # Sidebar para controles
        st.sidebar.title("⚙️ Configuración MEXC")
        
        # Botón de sincronización
        if st.sidebar.button("🔄 Sincronizar Estado", use_container_width=True):
            sincronizar_estado()
        
        # Botón de ejecución/detención MEJORADO
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎮 Control del Bot")
        
        boton_texto = "⏹️ Detener Bot" if st.session_state.bot_activo else "▶️ Ejecutar Bot"
        boton_color = "secondary" if st.session_state.bot_activo else "primary"
        
        if st.sidebar.button(boton_texto, type=boton_color, use_container_width=True):
            toggle_bot()
        
        # Indicador de estado del bot MEJORADO
        if st.session_state.bot_activo:
            st.sidebar.success("✅ Bot EJECUTÁNDOSE")
            st.sidebar.info(f"🕒 Última actualización: {st.session_state.ultima_actualizacion.strftime('%H:%M:%S')}")
        else:
            st.sidebar.warning("⏸️ Bot DETENIDO")
        
        # Selector de símbolo
        simbolo = st.sidebar.selectbox(
            "Seleccionar Par",
            ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'ATOMUSDT', 'NEARUSDT', 'APTUSDT']
        )
        
        # Selector de intervalo
        intervalo = st.sidebar.selectbox(
            "Intervalo",
            ['1m', '5m', '15m', '1h', '4h']
        )
        
        # Configuración de trading MEJORADA
        st.sidebar.markdown("---")
        st.sidebar.subheader("💰 Configuración de Trading")
        
        cantidad_operacion = st.sidebar.slider(
            "Cantidad por operación (USDT)",
            min_value=10.0,
            max_value=float(st.session_state.capital),
            value=min(50.0, st.session_state.capital),
            step=5.0
        )
        
        # Configuración de estrategia MEJORADA
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Estrategia de Trading")
        
        st.sidebar.info("""
        **Condiciones de Compra:**
        - RSI < 40
        - Precio cerca de BB Inferior
        - Estocástico K < 25 y creciente
        """)
        
        st.sidebar.info("""
        **Condiciones de Venta:**
        - RSI > 60  
        - Precio cerca de BB Superior
        - Estocástico K > 75 y decreciente
        """)
        
        # Información de diagnóstico
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Diagnóstico")
        
        if st.session_state.errores_conexion > 0:
            st.sidebar.error(f"Errores de conexión: {st.session_state.errores_conexion}")
        
        st.sidebar.info(f"Operaciones activas: {len(st.session_state.operaciones_activas)}")
        
        # Botón para actualizar
        if st.sidebar.button("🔄 Actualizar Datos MEXC", use_container_width=True):
            st.rerun()
        
        # Obtener datos de MEXC
        with st.spinner('Obteniendo datos de MEXC...'):
            df = obtener_datos_mexc(symbol=simbolo, interval=intervalo, limit=50)
        
        if df is not None and len(df) > 0:
            # Calcular señales
            senal_compra, senal_venta, mensaje, debug_info = obtener_senal_compra_venta(df)
            precio_actual = df['close'].iloc[-1]
            
            # Mostrar información de capital y operaciones MEJORADA
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "💰 Capital Disponible", 
                    f"${st.session_state.capital:.2f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "📈 Operaciones Activas", 
                    f"{len(st.session_state.operaciones_activas)}",
                    delta=None
                )
            
            with col3:
                ganancia_total = sum(op.get('ganancia_perdida', 0) for op in st.session_state.historial_operaciones)
                color_ganancia = "normal" if ganancia_total >= 0 else "inverse"
                st.metric(
                    "🎯 Ganancia/Pérdida Total", 
                    f"${ganancia_total:.2f}",
                    delta=None,
                    delta_color=color_ganancia
                )
            
            with col4:
                estado_bot = "🟢 EJECUTANDO" if st.session_state.bot_activo else "🔴 DETENIDO"
                st.metric("Estado del Bot", estado_bot)
            
            # Mostrar estado de señales MEJORADO
            st.markdown("---")
            col5, col6 = st.columns(2)
            
            with col5:
                if senal_compra:
                    st.success("🎯 SEÑAL: COMPRA DETECTADA")
                    # Ejecutar compra automáticamente solo si el bot está activo
                    if (st.session_state.bot_activo and 
                        st.session_state.capital >= cantidad_operacion and 
                        len(st.session_state.operaciones_activas) < 5):  # Aumentado límite a 5
                        
                        if ejecutar_compra(simbolo, precio_actual, cantidad_operacion):
                            st.success(f"✅ Compra ejecutada: {cantidad_operacion} USDT en {simbolo}")
                            st.balloons()
                        else:
                            st.error("❌ Error ejecutando compra")
                    elif st.session_state.bot_activo:
                        st.info("ℹ️ Señal de compra detectada, pero el bot no puede ejecutar (sin capital o límite alcanzado)")
                elif senal_venta:
                    st.error("🎯 SEÑAL: VENTA DETECTADA")
                    # Cerrar operaciones activas si hay señal de venta y el bot está activo
                    if st.session_state.bot_activo:
                        operaciones_cerradas = 0
                        for operacion in st.session_state.operaciones_activas[:]:
                            if operacion['symbol'] == simbolo:
                                if ejecutar_venta(operacion, precio_actual):
                                    operaciones_cerradas += 1
                                    st.success(f"✅ Venta ejecutada para {operacion['symbol']}")
                        if operaciones_cerradas == 0:
                            st.info("ℹ️ Señal de venta detectada, pero no hay operaciones activas para cerrar")
                else:
                    st.info("🎯 SEÑAL: NEUTRAL - Esperando condiciones óptimas")
            
            with col6:
                st.metric("Precio Actual", f"${precio_actual:.4f}")
                st.metric("Última Actualización", 
                         df['timestamp'].iloc[-1].strftime('%H:%M:%S'))
            
            st.markdown(f"### 📊 {mensaje}")
            
            # Mostrar información de diagnóstico
            with st.expander("🔍 Información de Diagnóstico"):
                st.write("**Valores actuales de los indicadores:**")
                st.write(f"- RSI: {debug_info.get('rsi', 'N/A'):.2f}")
                st.write(f"- Precio: ${debug_info.get('precio', 'N/A'):.4f}")
                st.write(f"- BB Superior: ${debug_info.get('bb_upper', 'N/A'):.4f}")
                st.write(f"- BB Inferior: ${debug_info.get('bb_lower', 'N/A'):.4f}")
                st.write(f"- Estocástico K: {debug_info.get('stoch_k', 'N/A'):.2f}")
                st.write(f"- Estocástico D: {debug_info.get('stoch_d', 'N/A'):.2f}")
                
                if st.session_state.bot_activo:
                    st.success("✅ Bot listo para operar")
                else:
                    st.warning("⏸️ Bot en pausa - Actívalo para operar")
            
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
                    delta_color_rsi = "inverse" if rsi_actual > 70 else "normal" if rsi_actual < 30 else "off"
                    st.metric("RSI", f"{rsi_actual:.2f}", delta=None, delta_color=delta_color_rsi)
            
            with col2:
                if 'stoch_k' in df.columns:
                    stoch_actual = df['stoch_k'].iloc[-1]
                    delta_color_stoch = "inverse" if stoch_actual > 80 else "normal" if stoch_actual < 20 else "off"
                    st.metric("Estocástico K", f"{stoch_actual:.2f}", delta=None, delta_color=delta_color_stoch)
            
            with col3:
                if 'bb_upper' in df.columns:
                    precio_actual = df['close'].iloc[-1]
                    bb_upper = df['bb_upper'].iloc[-1]
                    bb_lower = df['bb_lower'].iloc[-1]
                    if bb_upper != bb_lower:
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
            
            # Mostrar operaciones activas
            if st.session_state.operaciones_activas:
                st.subheader("📊 Operaciones Activas")
                for operacion in st.session_state.operaciones_activas:
                    if operacion['symbol'] == simbolo:
                        with st.expander(f"{operacion['tipo']} - {operacion['symbol']} - {operacion['timestamp'].strftime('%H:%M:%S')}"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Precio entrada:** ${operacion['precio_entrada']:.4f}")
                                st.write(f"**Precio actual:** ${precio_actual:.4f}")
                            with col2:
                                st.write(f"**Cantidad:** {operacion['cantidad_cripto']:.6f}")
                                st.write(f"**Invertido:** ${operacion['cantidad_usdt']:.2f}")
                            with col3:
                                ganancia_actual = (precio_actual - operacion['precio_entrada']) * operacion['cantidad_cripto']
                                porcentaje_ganancia = (ganancia_actual / operacion['cantidad_usdt']) * 100
                                color_ganancia = "green" if ganancia_actual >= 0 else "red"
                                st.write(f"**Ganancia actual:** <span style='color:{color_ganancia}'>${ganancia_actual:.2f} ({porcentaje_ganancia:.1f}%)</span>", 
                                        unsafe_allow_html=True)
            
            # Mostrar historial de operaciones
            st.subheader("📋 Historial de Operaciones del Bot")
            if st.session_state.historial_operaciones:
                # Crear DataFrame del historial
                historial_df = pd.DataFrame(st.session_state.historial_operaciones)
                historial_df['timestamp'] = historial_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                historial_df['timestamp_cierre'] = historial_df['timestamp_cierre'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Mostrar tabla formateada
                st.dataframe(
                    historial_df[['symbol', 'tipo', 'precio_entrada', 'precio_salida', 
                                 'ganancia_perdida', 'timestamp', 'timestamp_cierre']].round(4),
                    use_container_width=True
                )
                
                # Resumen del historial
                st.subheader("📊 Resumen del Historial")
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_operaciones = len(st.session_state.historial_operaciones)
                    st.metric("Total Operaciones", total_operaciones)
                with col2:
                    operaciones_ganadoras = len([op for op in st.session_state.historial_operaciones if op['ganancia_perdida'] > 0])
                    st.metric("Operaciones Ganadoras", operaciones_ganadoras)
                with col3:
                    if total_operaciones > 0:
                        tasa_exito = (operaciones_ganadoras / total_operaciones) * 100
                        st.metric("Tasa de Éxito", f"{tasa_exito:.1f}%")
                    else:
                        st.metric("Tasa de Éxito", "0%")
            else:
                st.info("📝 Aún no hay operaciones en el historial")
            
        else:
            st.error("❌ No se pudieron obtener datos de MEXC.")
            st.info("💡 Soluciones posibles:")
            st.info("1. Verifica tu conexión a internet")
            st.info("2. El par seleccionado podría no existir en MEXC")
            st.info("3. La API de MEXC podría estar temporalmente saturada")
            st.info("4. Intenta con un intervalo diferente")
            
            # Botón de reintento
            if st.button("🔄 Reintentar Conexión"):
                st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error en la aplicación: {e}")
        logger.error(f"Error crítico en main: {e}")
        st.info("💡 Intenta recargar la página o sincronizar el estado")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
