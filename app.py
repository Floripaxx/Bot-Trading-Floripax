import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random

# Configuración de la página
st.set_page_config(
    page_title="Bot Trading MEXC - AUTOMÁTICO",
    page_icon="🤖",
    layout="wide"
)

# Título principal
st.title("🤖 Bot de Trading MEXC - MODO AUTOMÁTICO")
st.markdown("---")

# Clase del bot AUTOMÁTICO
class TradingBotAuto:
    def __init__(self):
        self.capital = 250.0
        self.capital_actual = 250.0
        self.senales_compra = 0
        self.senales_venta = 0
        self.ordenes_activas = 0
        self.operaciones_abiertas = []
        self.historial = []
        self.pares = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
        self.pair_index = 0
        self.ultima_analisis = None
        
    def analizar_y_ejecutar(self):
        """Analiza el mercado y ejecuta órdenes AUTOMÁTICAMENTE"""
        resultados_analisis = self._analizar_mercado()
        self._ejecutar_ordenes_automaticas(resultados_analisis)
        self._gestionar_operaciones_abiertas()
        
        return resultados_analisis
    
    def _analizar_mercado(self):
        """Análisis de mercado mejorado"""
        import random
        par_actual = self.pares[self.pair_index]
        
        # Simular análisis técnico más realista
        precio = round(random.uniform(50000, 60000), 2)
        rsi = round(random.uniform(20, 80), 1)
        volumen = round(random.uniform(0.5, 2.0), 2)
        
        # Lógica de señales más sofisticada
        senal = None
        if rsi < 35 and volumen > 1.2:  # Condiciones para COMPRA
            senal = "COMPRA"
            self.senales_compra += 1
        elif rsi > 65 and volumen > 1.1:  # Condiciones para VENTA
            senal = "VENTA" 
            self.senales_venta += 1
        
        resultado = {
            'par': par_actual,
            'precio_actual': precio,
            'rsi': rsi,
            'volumen_ratio': volumen,
            'senal': senal,
            'estado': "🔴 SEÑAL COMPRA" if senal == "COMPRA" else 
                     "🟢 SEÑAL VENTA" if senal == "VENTA" else 
                     "⏳ Esperando oportunidad"
        }
        
        self.ultima_analisis = resultado
        return [resultado]
    
    def _ejecutar_ordenes_automaticas(self, resultados):
        """Ejecuta órdenes AUTOMÁTICAMENTE cuando hay señales"""
        for resultado in resultados:
            if resultado['senal'] and self.capital_actual > 25:  # Capital mínimo
                
                # EJECUCIÓN AUTOMÁTICA - sin confirmación manual
                orden = {
                    'par': resultado['par'],
                    'tipo': resultado['senal'],
                    'precio_entrada': resultado['precio_actual'],
                    'cantidad': self.capital_actual * 0.1,  # 10% del capital
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'estado': 'ABIERTA',
                    'stop_loss': resultado['precio_actual'] * 0.95,  # -5%
                    'take_profit': resultado['precio_actual'] * 1.08  # +8%
                }
                
                self.operaciones_abiertas.append(orden)
                self.historial.append(orden)
                self.ordenes_activas += 1
                self.capital_actual -= orden['cantidad']
                
                # Rotar al siguiente par después de operar
                self.pair_index = (self.pair_index + 1) % len(self.pares)
                
    def _gestionar_operaciones_abiertas(self):
        """Cierra operaciones automáticamente cuando se cumplen condiciones"""
        operaciones_cerradas = []
        
        for operacion in self.operaciones_abiertas[:]:
            # Simular movimiento de precio
            precio_actual = operacion['precio_entrada'] * random.uniform(0.9, 1.15)
            
            # Verificar si se activa STOP LOSS o TAKE PROFIT
            if precio_actual <= operacion['stop_loss']:
                # Cierre por STOP LOSS
                operacion['estado'] = 'CERRADA - STOP LOSS'
                operacion['precio_salida'] = operacion['stop_loss']
                operacion['profit_loss'] = -operacion['cantidad'] * 0.05  # -5%
                self.capital_actual += operacion['cantidad'] + operacion['profit_loss']
                operaciones_cerradas.append(operacion)
                self.operaciones_abiertas.remove(operacion)
                self.ordenes_activas -= 1
                
            elif precio_actual >= operacion['take_profit']:
                # Cierre por TAKE PROFIT  
                operacion['estado'] = 'CERRADA - TAKE PROFIT'
                operacion['precio_salida'] = operacion['take_profit']
                operacion['profit_loss'] = operacion['cantidad'] * 0.08  # +8%
                self.capital_actual += operacion['cantidad'] + operacion['profit_loss']
                operaciones_cerradas.append(operacion)
                self.operaciones_abiertas.remove(operacion)
                self.ordenes_activas -= 1
    
    def obtener_estado(self):
        return {
            'capital_actual': round(self.capital_actual, 2),
            'senales_compra': self.senales_compra,
            'senales_venta': self.senales_venta,
            'ordenes_activas': self.ordenes_activas,
            'par_actual': self.pares[self.pair_index],
            'proximo_par': self.pares[(self.pair_index + 1) % len(self.pares)],
            'tiempo_restante': "03:15",
            'operaciones_abiertas': len(self.operaciones_abiertas)
        }
    
    def obtener_historial(self):
        if self.historial:
            return pd.DataFrame(self.historial)
        return None
    
    def reiniciar_capital(self):
        self.capital_actual = self.capital
        self.senales_compra = 0
        self.senales_venta = 0
        self.ordenes_activas = 0
        self.operaciones_abiertas = []
        self.historial = []

# Inicializar el bot en session_state
if 'trading_bot' not in st.session_state:
    st.session_state.trading_bot = TradingBotAuto()

# Sidebar - Configuración
st.sidebar.header("⚙️ Configuración - MODO AUTOMÁTICO")

st.sidebar.warning("""
**🔴 MODO AUTOMÁTICO ACTIVADO**
- El bot ejecutará órdenes automáticamente
- Gestionará Stop-Loss y Take-Profit
- Sin intervención manual requerida
""")

# Capital inicial
capital = st.sidebar.number_input(
    "Capital Inicial ($)",
    min_value=10.0,
    max_value=10000.0,
    value=250.0,
    step=50.0
)

# Layout principal
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.header("📈 Análisis y Ejecución AUTOMÁTICA")
    
    # Botón único - analiza y ejecuta automáticamente
    if st.button("🔄 ANALIZAR Y OPERAR AUTOMÁTICAMENTE", type="primary", use_container_width=True):
        with st.spinner("Analizando y ejecutando órdenes automáticamente..."):
            time.sleep(2)  # Simular análisis
            resultados = st.session_state.trading_bot.analizar_y_ejecutar()
            
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
                        
                        # Mostrar si se ejecutó orden automáticamente
                        if resultado['senal']:
                            st.success(f"✅ ORDEN AUTOMÁTICA: {resultado['senal']} EJECUTADA")
                            st.info("El bot gestionará Stop-Loss y Take-Profit automáticamente")

with col2:
    st.header("💼 Estado Actual")
    
    estado = st.session_state.trading_bot.obtener_estado()
    
    st.metric("Capital Actual", f"${estado['capital_actual']:.2f}")
    st.metric("Señales Compra", estado['senales_compra'])
    st.metric("Señales Venta", estado['senales_venta'])
    st.metric("Órdenes Activas", estado['ordenes_activas'])
    
    st.metric("Par Actual", estado['par_actual'])
    st.metric("Próximo Par", estado['proximo_par'])
    st.metric("Ops. Abiertas", estado['operaciones_abiertas'])

with col3:
    st.header("📊 Rendimiento")
    
    if st.button("📋 Ver Historial Completo"):
        historial = st.session_state.trading_bot.obtener_historial()
        if historial is not None:
            st.dataframe(historial)
        else:
            st.info("No hay operaciones registradas")
    
    if st.button("🔄 Reiniciar Sistema"):
        st.session_state.trading_bot.reiniciar_capital()
        st.success("✅ Sistema reiniciado - Capital: $" + str(capital))
        st.rerun()

# Mostrar operaciones abiertas
if st.session_state.trading_bot.operaciones_abiertas:
    st.header("🔓 Operaciones Abiertas")
    for op in st.session_state.trading_bot.operaciones_abiertas:
        st.info(f"""
        **{op['par']}** - {op['tipo']}
        • Entrada: ${op['precio_entrada']:.2f}
        • Stop Loss: ${op['stop_loss']:.2f} 
        • Take Profit: ${op['take_profit']:.2f}
        • Cantidad: ${op['cantidad']:.2f}
        """)

# Auto-actualización automática
if st.sidebar.checkbox("🔄 Auto-ejecutar cada 45s", value=True):
    st.sidebar.write("Próxima ejecución automática en 45 segundos")
    time.sleep(45)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**🤖 MODO AUTOMÁTICO ACTIVADO** - El bot analiza, ejecuta y cierra operaciones automáticamente")
st.markdown("**⚠️ Advertencia:** Trading simulado - Para trading real se requiere API Key y configuración adicional")
