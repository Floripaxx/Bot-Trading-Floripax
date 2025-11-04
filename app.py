import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from trading_bot import TradingBot
import requests

# Configuración de la página
st.set_page_config(
    page_title="Bot Trading MEXC",
    page_icon="🤖",
    layout="wide"
)

# Título principal
st.title("🤖 Bot de Trading MEXC")
st.markdown("---")

# Inicializar el bot en session_state
if 'trading_bot' not in st.session_state:
    st.session_state.trading_bot = TradingBot()

# Sidebar - Configuración
st.sidebar.header("⚙️ Configuración")

# Modo de Trading
trading_mode = st.sidebar.radio(
    "Modo de Trading",
    ["Paper Trading (Simulación)", "Trading Real"]
)

# Capital inicial
capital = st.sidebar.number_input(
    "Capital Inicial ($)",
    min_value=10.0,
    max_value=10000.0,
    value=250.0,
    step=50.0
)

# Parámetros de estrategia
st.sidebar.header("📊 Parámetros de Estrategia")

ema_corta = st.sidebar.slider("EMA Corta", 5, 20, 9)
ema_larga = st.sidebar.slider("EMA Larga", 15, 50, 21)
rsi_periodo = st.sidebar.slider("RSI Periodo", 5, 21, 14)
rsi_sobrecompra = st.sidebar.slider("RSI Sobrecompra", 60, 80, 65)
rsi_sobreventa = st.sidebar.slider("RSI Sobreventa", 20, 40, 35)
volumen_minimo = st.sidebar.slider("Mínimo Volumen", 1.0, 2.0, 1.1)

# Actualizar parámetros del bot
st.session_state.trading_bot.update_parameters(
    ema_corta=ema_corta,
    ema_larga=ema_larga,
    rsi_periodo=rsi_periodo,
    rsi_sobrecompra=rsi_sobrecompra,
    rsi_sobreventa=rsi_sobreventa,
    volumen_minimo=volumen_minimo,
    capital=capital,
    trading_real=(trading_mode == "Trading Real")
)

# Layout principal
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.header("📈 Análisis de Mercado")
    
    # Botón para analizar mercado
    if st.button("🔄 Analizar Mercado", type="primary"):
        with st.spinner("Analizando mercado..."):
            resultados = st.session_state.trading_bot.analizar_mercado()
            
            if resultados:
                for resultado in resultados:
                    with st.expander(f"📊 {resultado['par']} - {resultado['estado']}", expanded=True):
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Precio Actual", f"${resultado['precio_actual']:.2f}")
                        with col_b:
                            st.metric("RSI", f"{resultado['rsi']:.1f}")
                        with col_c:
                            st.metric("Volumen Ratio", f"{resultado['volumen_ratio']:.2f}")
                        
                        # Mostrar gráfico
                        if resultado['datos_grafico']:
                            fig = go.Figure()
                            
                            # Precio
                            fig.add_trace(go.Scatter(
                                x=resultado['datos_grafico']['timestamp'],
                                y=resultado['datos_grafico']['close'],
                                name='Precio',
                                line=dict(color='blue')
                            ))
                            
                            # EMA Corta
                            fig.add_trace(go.Scatter(
                                x=resultado['datos_grafico']['timestamp'],
                                y=resultado['datos_grafico']['ema_corta'],
                                name=f'EMA {ema_corta}',
                                line=dict(color='orange')
                            ))
                            
                            # EMA Larga
                            fig.add_trace(go.Scatter(
                                x=resultado['datos_grafico']['timestamp'],
                                y=resultado['datos_grafico']['ema_larga'],
                                name=f'EMA {ema_larga}',
                                line=dict(color='red')
                            ))
                            
                            fig.update_layout(
                                title=f"Gráfico {resultado['par']}",
                                xaxis_title="Tiempo",
                                yaxis_title="Precio (USDT)",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar señal si existe
                        if resultado['senal']:
                            st.success(f"🚨 SEÑAL: {resultado['senal']}")
                            if st.button(f"Ejecutar {resultado['senal']}", key=resultado['par']):
                                st.session_state.trading_bot.ejecutar_orden(
                                    resultado['par'], 
                                    resultado['senal']
                                )
                                st.success(f"Orden {resultado['senal']} ejecutada para {resultado['par']}")

with col2:
    st.header("💼 Estado Actual")
    
    estado = st.session_state.trading_bot.obtener_estado()
    
    st.metric("Capital Actual", f"${estado['capital_actual']:.2f}")
    st.metric("Señales Compra", estado['senales_compra'])
    st.metric("Señales Venta", estado['senales_venta'])
    st.metric("Órdenes Activas", estado['ordenes_activas'])
    
    st.metric("Par Actual", estado['par_actual'])
    st.metric("Próximo Par", estado['proximo_par'])
    st.metric("Cambio en", estado['tiempo_restante'])

with col3:
    st.header("📊 Rendimiento")
    
    if st.button("📋 Historial de Operaciones"):
        historial = st.session_state.trading_bot.obtener_historial()
        if historial:
            st.dataframe(historial)
        else:
            st.info("No hay operaciones registradas")
    
    if st.button("🔄 Reiniciar Capital"):
        st.session_state.trading_bot.reiniciar_capital()
        st.success("Capital reiniciado a $" + str(capital))

# Auto-actualización
if st.sidebar.checkbox("🔄 Auto-actualizar cada 30s", value=True):
    st.sidebar.write("Próxima actualización automática en 30 segundos")
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**⚠️ Advertencia:** El trading de criptomonedas implica riesgos. Usa bajo tu responsabilidad.")
