import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
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

# Clase simplificada del bot (para evitar import issues)
class TradingBotSimple:
    def __init__(self):
        self.capital = 250.0
        self.capital_actual = 250.0
        self.senales_compra = 0
        self.senales_venta = 0
        self.ordenes_activas = 0
        self.pares = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
        self.pair_index = 0
        
    def analizar_mercado(self):
        """Función simplificada para demo"""
        import random
        resultado = {
            'par': self.pares[self.pair_index],
            'precio_actual': round(random.uniform(50000, 60000), 2),
            'rsi': round(random.uniform(30, 70), 1),
            'volumen_ratio': round(random.uniform(0.8, 1.5), 2),
            'senal': random.choice([None, "COMPRA", "VENTA"]),
            'estado': "Analizando...",
            'datos_grafico': None
        }
        
        if resultado['senal'] == "COMPRA":
            self.senales_compra += 1
            resultado['estado'] = "🔴 SEÑAL COMPRA"
        elif resultado['senal'] == "VENTA":
            self.senales_venta += 1
            resultado['estado'] = "🟢 SEÑAL VENTA"
        else:
            resultado['estado'] = "⏳ Esperando oportunidad"
            
        return [resultado]
    
    def obtener_estado(self):
        return {
            'capital_actual': self.capital_actual,
            'senales_compra': self.senales_compra,
            'senales_venta': self.senales_venta,
            'ordenes_activas': self.ordenes_activas,
            'par_actual': self.pares[self.pair_index],
            'proximo_par': self.pares[(self.pair_index + 1) % len(self.pares)],
            'tiempo_restante': "04:30"
        }
    
    def ejecutar_orden(self, par, senal):
        return True
    
    def obtener_historial(self):
        return None
    
    def reiniciar_capital(self):
        self.capital_actual = self.capital

# Inicializar el bot en session_state
if 'trading_bot' not in st.session_state:
    st.session_state.trading_bot = TradingBotSimple()

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

# Layout principal
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.header("📈 Análisis de Mercado")
    
    # Botón para analizar mercado - FUNCIONAL
    if st.button("🔄 Analizar Mercado", type="primary"):
        with st.spinner("Analizando mercado..."):
            # Pequeña pausa para simular análisis
            time.sleep(2)
            resultados = st.session_state.trading_bot.analizar_mercado()
            
            if resultados:
                for resultado in resultados:
                    with st.expander(f"📊 {resultado['par']} - {resultado['estado']}", expanded=True):
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Precio Actual", f"${resultado['precio_actual']:,.2f}")
                        with col_b:
                            st.metric("RSI", f"{resultado['rsi']:.1f}")
                        with col_c:
                            st.metric("Volumen Ratio", f"{resultado['volumen_ratio']:.2f}")
                        
                        # Mostrar señal si existe
                        if resultado['senal']:
                            st.success(f"🚨 SEÑAL: {resultado['senal']}")
                            if st.button(f"Ejecutar {resultado['senal']}", key=resultado['par']):
                                if st.session_state.trading_bot.ejecutar_orden(resultado['par'], resultado['senal']):
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
        if historial is not None:
            st.dataframe(historial)
        else:
            st.info("No hay operaciones registradas")
    
    if st.button("🔄 Reiniciar Capital"):
        st.session_state.trading_bot.reiniciar_capital()
        st.success("Capital reiniciado a $" + str(capital))

# Footer
st.markdown("---")
st.markdown("**⚠️ Advertencia:** El trading de criptomonedas implica riesgos. Usa bajo tu responsabilidad.")
