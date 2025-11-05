import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
import json
import os

# Configuración de la página
st.set_page_config(
    page_title="BTC Scalping EMA Momentum 3M",
    page_icon="⚡",
    layout="wide"
)

# Título principal
st.title("⚡ BTC Scalping EMA Momentum (3Min)")
st.markdown("---")

class BTCScalpingBot:
    def __init__(self):
        self.capital = 250.0
        self.capital_actual = 250.0
        self.senales_compra = 0
        self.senales_venta = 0
        self.ordenes_activas = []
        self.historial = []
        
        # ✅ ESTRATEGIA BTC SCALPING 3M
        self.pares = ["BTCUSDT"]
        self.pares_mostrar = ["BTC/USDT"]
        self.timeframe = "3m"
        self.limit_velas = 50  # Más velas para EMAs
        
        # Parámetros estrategia
        self.ema_rapida = 9
        self.ema_lenta = 21
        self.rsi_periodo = 7
        self.volumen_periodo = 20
        self.rsi_compra = 55
        self.rsi_venta = 45
        self.rsi_salida = 50
        
        # Gestión riesgo
        self.riesgo_por_operacion = 0.01  # 1%
        self.stop_loss_porcentaje = 0.0025  # 0.25%
        self.take_profit_porcentaje = 0.005  # 0.5%
        self.trailing_trigger = 0.003  # 0.3%
        self.apalancamiento = 5
        
        self.ultima_analisis = None
        self.ultima_actualizacion = None
        self.auto_trading = False
        
        self._cargar_estado_persistente()
    
    def _guardar_estado_persistente(self):
        """GUARDADO DEFINITIVO"""
        try:
            estado = {
                'capital_actual': self.capital_actual,
                'senales_compra': self.senales_compra,
                'senales_venta': self.senales_venta,
                'ordenes_activas': self.ordenes_activas,
                'historial': self.historial,
                'ultima_actualizacion': self.ultima_actualizacion.isoformat() if self.ultima_actualizacion else None,
                'auto_trading': self.auto_trading
            }
            
            with open('/tmp/btc_scalping_state.json', 'w') as f:
                json.dump(estado, f, indent=2)
            
            if 'btc_scalping_state' not in st.session_state:
                st.session_state.btc_scalping_state = {}
            st.session_state.btc_scalping_state = estado
            
        except Exception as e:
            st.error(f"❌ Error guardando estado: {e}")
    
    def _cargar_estado_persistente(self):
        """CARGA DEFINITIVA"""
        estado_cargado = None
        
        try:
            if os.path.exists('/tmp/btc_scalping_state.json'):
                with open('/tmp/btc_scalping_state.json', 'r') as f:
                    estado_cargado = json.load(f)
            
            elif 'btc_scalping_state' in st.session_state and st.session_state.btc_scalping_state:
                estado_cargado = st.session_state.btc_scalping_state
            
            if estado_cargado:
                self.capital_actual = estado_cargado.get('capital_actual', 250.0)
                self.senales_compra = estado_cargado.get('senales_compra', 0)
                self.senales_venta = estado_cargado.get('senales_venta', 0)
                self.ordenes_activas = estado_cargado.get('ordenes_activas', [])
                self.historial = estado_cargado.get('historial', [])
                
                ultima_act = estado_cargado.get('ultima_actualizacion')
                if ultima_act:
                    self.ultima_actualizacion = datetime.fromisoformat(ultima_act)
                self.auto_trading = estado_cargado.get('auto_trading', False)
                
        except Exception as e:
            self.capital_actual = 250.0
            self.auto_trading = False
    
    def obtener_datos_mercado(self, simbolo):
        """Obtiene datos OHLCV de MEXC para 3min"""
        try:
            url = f"https://api.mexc.com/api/v3/klines"
            params = {
                'symbol': simbolo,
                'interval': self.timeframe,
                'limit': self.limit_velas
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy', 'taker_quote', 'ignore'
                ])
                
                # Convertir tipos de datos
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            else:
                # Fallback: datos de ejemplo para testing
                return self._generar_datos_testing()
                
        except Exception as e:
            st.error(f"Error obteniendo datos: {e}")
            return self._generar_datos_testing()
    
    def _generar_datos_testing(self):
        """Genera datos de testing cuando API falla"""
        dates = pd.date_range(end=datetime.now(), periods=self.limit_velas, freq='3min')
        np.random.seed(42)
        
        # Precio alrededor de $100,000 con algo de volatilidad
        prices = [100000]
        for i in range(1, self.limit_velas):
            change = np.random.normal(0, 0.002)  # 0.2% de volatilidad
            prices.append(prices[-1] * (1 + change))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(100, 1000) for _ in range(self.limit_velas)]
        })
        
        return df
    
    def calcular_ema(self, datos, periodo):
        return datos.ewm(span=periodo, adjust=False).mean()
    
    def calcular_rsi(self, precios, periodo=14):
        delta = precios.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def analizar_mercado(self):
        """✅ EJECUTA ESTRATEGIA SCALPING EMA MOMENTUM"""
        df = self.obtener_datos_mercado(self.pares[0])
        
        if df is None or len(df) < max(self.ema_lenta, self.volumen_periodo):
            return {'error': 'Datos insuficientes'}
        
        # Calcular indicadores
        df['ema_rapida'] = self.calcular_ema(df['close'], self.ema_rapida)
        df['ema_lenta'] = self.calcular_ema(df['close'], self.ema_lenta)
        df['rsi'] = self.calcular_rsi(df['close'], self.rsi_periodo)
        df['volumen_ma'] = df['volume'].rolling(window=self.volumen_periodo).mean()
        
        # Últimos valores
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        precio_actual = current['close']
        ema_rapida_actual = current['ema_rapida']
        ema_lenta_actual = current['ema_lenta']
        ema_rapida_prev = prev['ema_rapida']
        ema_lenta_prev = prev['ema_lenta']
        rsi_actual = current['rsi']
        rsi_prev = prev['rsi']
        volumen_actual = current['volume']
        volumen_promedio = current['volumen_ma']
        
        # ✅ SEÑAL COMPRA: EMA9 > EMA21 Y RSI > 55 Y Volumen > Promedio
senal_compra = (
    ema_rapida_actual > ema_lenta_actual and  # EMA9 sobre EMA21
    ema_rapida_prev <= ema_lenta_prev and     # Cruzó hacia arriba
    rsi_actual > self.rsi_compra and          # RSI sobre 55
    rsi_prev <= self.rsi_compra and           # Cruzó hacia arriba
    volumen_actual > volumen_promedio         # Volumen sobre promedio
)

        # ✅ SEÑAL VENTA: EMA9 < EMA21 Y RSI < 45 Y Volumen > Promedio
        senal_venta = (
            ema_rapida_actual < ema_lenta_actual and  # EMA9 bajo EMA21
            ema_rapida_prev >= ema_lenta_prev and     # Cruzó hacia abajo
            rsi_actual < self.rsi_venta and           # RSI bajo 45
            rsi_prev >= self.rsi_venta and            # Cruzó hacia abajo
            volumen_actual > volumen_promedio         # Volumen sobre promedio
        )
        
        senal = None
        if senal_compra:
            senal = "COMPRA"
            self.senales_compra += 1
        elif senal_venta:
            senal = "VENTA"
            self.senales_venta += 1
        
        resultado = {
            'par': self.pares_mostrar[0],
            'precio_actual': precio_actual,
            'ema_rapida': ema_rapida_actual,
            'ema_lenta': ema_lenta_actual,
            'rsi': rsi_actual,
            'volumen_actual': volumen_actual,
            'volumen_promedio': volumen_promedio,
            'senal': senal,
            'estado': "🔴 SEÑAL COMPRA" if senal == "COMPRA" else 
                     "🟢 SEÑAL VENTA" if senal == "VENTA" else 
                     "⏳ Esperando señal",
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'datos_grafico': {
                'precio': df['close'].tolist(),
                'ema_rapida': df['ema_rapida'].tolist(),
                'ema_lenta': df['ema_lenta'].tolist(),
                'timestamp': df['timestamp'].dt.strftime('%H:%M').tolist()
            }
        }
        
        self.ultima_analisis = resultado
        self.ultima_actualizacion = datetime.now()
        return resultado
    
    def ejecutar_orden(self, senal, precio_entrada):
        """Ejecuta orden con gestión de riesgo de scalping"""
        if len(self.ordenes_activas) >= 1:  # Máximo 1 operación
            return None
        
        # Calcular tamaño posición con apalancamiento
        riesgo_dolares = self.capital_actual * self.riesgo_por_operacion
        stop_loss_pips = precio_entrada * self.stop_loss_porcentaje
        
        # Tamaño posición (apalancamiento x5)
        tamaño_posicion = (riesgo_dolares / stop_loss_pips) * self.apalancamiento
        tamaño_posicion = min(tamaño_posicion, self.capital_actual * 0.2)  # Máximo 20% capital
        
        if senal == "COMPRA":
            stop_loss = precio_entrada * (1 - self.stop_loss_porcentaje)
            take_profit = precio_entrada * (1 + self.take_profit_porcentaje)
        else:
            stop_loss = precio_entrada * (1 + self.stop_loss_porcentaje)
            take_profit = precio_entrada * (1 - self.take_profit_porcentaje)
        
        orden_id = len(self.historial) + 1
        orden = {
            'id': orden_id,
            'par': self.pares_mostrar[0],
            'tipo': senal,
            'precio_entrada': precio_entrada,
            'tamaño_posicion': round(tamaño_posicion, 6),
            'valor_dolares': round(tamaño_posicion * precio_entrada / self.apalancamiento, 2),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'estado': 'ABIERTA',
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2),
            'stop_original': round(stop_loss, 2),
            'trailing_activado': False,
            'riesgo_dolares': round(riesgo_dolares, 2)
        }
        
        self.ordenes_activas.append(orden)
        self.historial.append(orden.copy())
        self.capital_actual -= orden['valor_dolares']
        
        return orden
    
    def gestionar_operaciones(self):
        """Gestiona operaciones con TRAILING STOP"""
        operaciones_cerradas = []
        precio_actual = self.obtener_precio_actual()
        
        for operacion in self.ordenes_activas[:]:
            # ✅ SALIDA POR RSI (si está configurada)
            if self._debe_salir_por_rsi(operacion):
                profit_loss = self._calcular_profit_loss(operacion, operacion['precio_entrada'])
                operacion.update({
                    'estado': 'CERRADA - RSI SALIDA',
                    'precio_salida': operacion['precio_entrada'],
                    'profit_loss': round(profit_loss, 2),
                    'timestamp_cierre': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'razon_cierre': 'RSI alcanzó nivel de salida'
                })
                self._cerrar_operacion(operacion, operaciones_cerradas)
                continue
            
            # ✅ TRAILING STOP
            if operacion['tipo'] == "COMPRA":
                profit_actual = (precio_actual - operacion['precio_entrada']) / operacion['precio_entrada']
                
                # Activar trailing stop después de +0.3%
                if profit_actual >= self.trailing_trigger and not operacion['trailing_activado']:
                    operacion['stop_loss'] = operacion['precio_entrada']
                    operacion['trailing_activado'] = True
                
                # Mover trailing stop
                if operacion['trailing_activado']:
                    nuevo_stop = precio_actual * (1 - self.stop_loss_porcentaje)
                    if nuevo_stop > operacion['stop_loss']:
                        operacion['stop_loss'] = nuevo_stop
                
                # Verificar stop loss
                if precio_actual <= operacion['stop_loss']:
                    profit_loss = self._calcular_profit_loss(operacion, operacion['stop_loss'])
                    operacion.update({
                        'estado': 'CERRADA - STOP LOSS',
                        'precio_salida': operacion['stop_loss'],
                        'profit_loss': round(profit_loss, 2),
                        'timestamp_cierre': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'razon_cierre': f"Stop Loss alcanzado {operacion['stop_loss']:,.2f}"
                    })
                    self._cerrar_operacion(operacion, operaciones_cerradas)
                
                # Verificar take profit
                elif precio_actual >= operacion['take_profit']:
                    profit_loss = self._calcular_profit_loss(operacion, operacion['take_profit'])
                    operacion.update({
                        'estado': 'CERRADA - TAKE PROFIT',
                        'precio_salida': operacion['take_profit'],
                        'profit_loss': round(profit_loss, 2),
                        'timestamp_cierre': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'razon_cierre': f"Take Profit alcanzado {operacion['take_profit']:,.2f}"
                    })
                    self._cerrar_operacion(operacion, operaciones_cerradas)
            
            else:  # VENTA
                profit_actual = (operacion['precio_entrada'] - precio_actual) / operacion['precio_entrada']
                
                # Activar trailing stop después de +0.3%
                if profit_actual >= self.trailing_trigger and not operacion['trailing_activado']:
                    operacion['stop_loss'] = operacion['precio_entrada']
                    operacion['trailing_activado'] = True
                
                # Mover trailing stop
                if operacion['trailing_activado']:
                    nuevo_stop = precio_actual * (1 + self.stop_loss_porcentaje)
                    if nuevo_stop < operacion['stop_loss']:
                        operacion['stop_loss'] = nuevo_stop
                
                # Verificar stop loss
                if precio_actual >= operacion['stop_loss']:
                    profit_loss = self._calcular_profit_loss(operacion, operacion['stop_loss'])
                    operacion.update({
                        'estado': 'CERRADA - STOP LOSS',
                        'precio_salida': operacion['stop_loss'],
                        'profit_loss': round(profit_loss, 2),
                        'timestamp_cierre': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'razon_cierre': f"Stop Loss alcanzado {operacion['stop_loss']:,.2f}"
                    })
                    self._cerrar_operacion(operacion, operaciones_cerradas)
                
                # Verificar take profit
                elif precio_actual <= operacion['take_profit']:
                    profit_loss = self._calcular_profit_loss(operacion, operacion['take_profit'])
                    operacion.update({
                        'estado': 'CERRADA - TAKE PROFIT',
                        'precio_salida': operacion['take_profit'],
                        'profit_loss': round(profit_loss, 2),
                        'timestamp_cierre': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'razon_cierre': f"Take Profit alcanzado {operacion['take_profit']:,.2f}"
                    })
                    self._cerrar_operacion(operacion, operaciones_cerradas)
        
        return operaciones_cerradas
    
    def _debe_salir_por_rsi(self, operacion):
        """Verifica si debe salir por condición RSI"""
        # Por simplicidad, en esta versión no implementamos salida por RSI
        # Pero la estructura está lista para agregarla
        return False
    
    def _calcular_profit_loss(self, operacion, precio_salida):
        """Calcula profit/loss en dólares"""
        if operacion['tipo'] == "COMPRA":
            return (precio_salida - operacion['precio_entrada']) * operacion['tamaño_posicion']
        else:
            return (operacion['precio_entrada'] - precio_salida) * operacion['tamaño_posicion']
    
    def _cerrar_operacion(self, operacion, operaciones_cerradas):
        """Cierra operación y actualiza capital"""
        self.capital_actual += operacion['valor_dolares'] + operacion['profit_loss']
        self.ordenes_activas.remove(operacion)
        operaciones_cerradas.append(operacion)
    
    def obtener_precio_actual(self):
        """Precio actual rápido"""
        try:
            url = f"https://api.mexc.com/api/v3/ticker/price?symbol=BTCUSDT"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
        except:
            pass
        return 100000  # Fallback
    
    def analizar_y_ejecutar(self):
        """Análisis completo + ejecución"""
        # Gestionar operaciones abiertas primero
        self.gestionar_operaciones()
        
        # Solo analizar si no hay operaciones abiertas
        if len(self.ordenes_activas) == 0:
            resultado = self.analizar_mercado()
            if resultado.get('senal'):
                self.ejecutar_orden(resultado['senal'], resultado['precio_actual'])
        else:
            resultado = {
                'par': self.pares_mostrar[0],
                'estado': '⏳ Operación activa - Esperando cierre',
                'senal': None,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
        
        self._guardar_estado_persistente()
        return resultado
    
    def obtener_estado(self):
        return {
            'capital_actual': round(self.capital_actual, 2),
            'senales_compra': self.senales_compra,
            'senales_venta': self.senales_venta,
            'ordenes_activas': len(self.ordenes_activas),
            'par_actual': self.pares_mostrar[0],
            'ultima_actualizacion': self.ultima_actualizacion.strftime("%H:%M:%S") if self.ultima_actualizacion else "Nunca",
            'total_operaciones': len(self.historial),
            'auto_trading': self.auto_trading
        }
    
    def obtener_historial(self):
        if self.historial:
            df = pd.DataFrame(self.historial)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False)
            return df
        return None
    
    def reiniciar_sistema(self):
        self.capital_actual = self.capital
        self.senales_compra = 0
        self.senales_venta = 0
        self.ordenes_activas = []
        self.historial = []
        self.ultima_actualizacion = datetime.now()
        self.auto_trading = False
        self._guardar_estado_persistente()

# Inicializar el bot
if 'scalping_bot' not in st.session_state:
    st.session_state.scalping_bot = BTCScalpingBot()

# AUTO-INICIO del contador si el Auto-Trading estaba activo
if st.session_state.scalping_bot.auto_trading and 'auto_trading_counter' not in st.session_state:
    st.session_state.auto_trading_counter = 0
    st.session_state.last_auto_execution = time.time()

# Sidebar - Configuración
st.sidebar.header("⚙️ BTC Scalping 3M")

st.sidebar.success("""
**✅ ESTRATEGIA AVANZADA:**
- EMA 9/21 + RSI 7
- Timeframe 3min
- RR 1:2 + Trailing Stop
- Apalancamiento x5
""")

# Auto-trading toggle
auto_trading = st.sidebar.toggle("🔄 Auto-Trading 3M", 
                                value=st.session_state.scalping_bot.auto_trading,
                                help="Ejecuta automáticamente cada 60 segundos")

if auto_trading != st.session_state.scalping_bot.auto_trading:
    st.session_state.scalping_bot.auto_trading = auto_trading
    st.session_state.scalping_bot._guardar_estado_persistente()
    
    if auto_trading:
        st.session_state.auto_trading_counter = 0
        st.session_state.last_auto_execution = time.time()
        st.sidebar.success("✅ Auto-Trading ACTIVADO")
    else:
        if 'auto_trading_counter' in st.session_state:
            del st.session_state.auto_trading_counter
        st.sidebar.info("⏸️ Auto-Trading PAUSADO")
    st.rerun()

# SISTEMA AUTO-TRADING
if st.session_state.scalping_bot.auto_trading:
    st.sidebar.success("✅ AUTO-TRADING ACTIVO")
    
    if 'auto_trading_counter' not in st.session_state:
        st.session_state.auto_trading_counter = 0
        st.session_state.last_auto_execution = time.time()
    
    st.session_state.auto_trading_counter += 1
    tiempo_desde_ultima = time.time() - st.session_state.last_auto_execution
    
    st.sidebar.write(f"🔄 Ejecuciones: {st.session_state.auto_trading_counter}")
    st.sidebar.write(f"⏰ Última ejecución: {int(tiempo_desde_ultima)}s")
    
    # EJECUCIÓN CADA 60 SEGUNDOS
    if tiempo_desde_ultima >= 60:
        try:
            with st.sidebar:
                with st.spinner("🤖 ANALIZANDO 3M..."):
                    resultado = st.session_state.scalping_bot.analizar_y_ejecutar()
                    st.session_state.last_auto_execution = time.time()
                    
                    if resultado.get('senal'):
                        st.success(f"✅ {resultado['par']} - {resultado['senal']} EJECUTADA")
                    else:
                        if len(st.session_state.scalping_bot.ordenes_activas) > 0:
                            st.info("⏳ Operación activa")
                        else:
                            st.info("📊 Esperando señal")
            
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")

# Layout principal
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Análisis BTC/USDT 3Min")
    
    if st.button("🔄 ANALIZAR Y OPERAR", type="primary", use_container_width=True):
        with st.spinner("Analizando mercado 3M..."):
            resultado = st.session_state.scalping_bot.analizar_y_ejecutar()
            
            if resultado:
                with st.expander(f"📈 {resultado['par']} - {resultado['estado']} ({resultado['timestamp']})", expanded=True):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.metric("Precio", f"${resultado['precio_actual']:,.2f}")
                    with col_b:
                        st.metric("EMA9", f"${resultado['ema_rapida']:,.2f}")
                    with col_c:
                        st.metric("EMA21", f"${resultado['ema_lenta']:,.2f}")
                    with col_d:
                        st.metric("RSI7", f"{resultado['rsi']:.1f}")
                    
                    st.metric("Volumen", f"{resultado['volumen_actual']:,.0f} vs {resultado['volumen_promedio']:,.0f}")
                    
                    if resultado['senal']:
                        st.success(f"✅ SEÑAL {resultado['senal']} - Orden ejecutada")
                    
                    # Gráfico
                    if resultado.get('datos_grafico'):
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=resultado['datos_grafico']['timestamp'],
                            y=resultado['datos_grafico']['precio'],
                            name='Precio BTC',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            x=resultado['datos_grafico']['timestamp'],
                            y=resultado['datos_grafico']['ema_rapida'],
                            name='EMA 9',
                            line=dict(color='orange')
                        ))
                        fig.add_trace(go.Scatter(
                            x=resultado['datos_grafico']['timestamp'],
                            y=resultado['datos_grafico']['ema_lenta'],
                            name='EMA 21',
                            line=dict(color='red')
                        ))
                        fig.update_layout(height=400, title="BTC/USDT 3Min - EMA Momentum")
                        st.plotly_chart(fig, use_container_width=True)
    
    # Operaciones activas
    if st.session_state.scalping_bot.ordenes_activas:
        st.subheader("🔓 Operación Activa")
        for op in st.session_state.scalping_bot.ordenes_activas:
            precio_actual = st.session_state.scalping_bot.obtener_precio_actual()
            
            if op['tipo'] == "COMPRA":
                profit_actual = ((precio_actual - op['precio_entrada']) / op['precio_entrada']) * 100
                color = "🟢" if profit_actual > 0 else "🔴"
            else:
                profit_actual = ((op['precio_entrada'] - precio_actual) / op['precio_entrada']) * 100
                color = "🟢" if profit_actual > 0 else "🔴"
            
            trailing_status = "✅" if op['trailing_activado'] else "⏳"
            
            st.info(f"""
            **{op['par']}** - {op['tipo']} | ID: {op['id']}
            • **Entrada:** ${op['precio_entrada']:,.2f}
            • **Actual:** ${precio_actual:,.2f} {color} ({profit_actual:+.2f}%)
            • **Stop Loss:** ${op['stop_loss']:,.2f} {trailing_status}
            • **Take Profit:** ${op['take_profit']:,.2f}
            • **Invertido:** ${op['valor_dolares']:.2f}
            • **Riesgo:** ${op['riesgo_dolares']:.2f}
            """)

with col2:
    st.header("💼 Rendimiento")
    
    estado = st.session_state.scalping_bot.obtener_estado()
    
    st.metric("Capital Actual", f"${estado['capital_actual']:.2f}")
    st.metric("Señales Compra", estado['senales_compra'])
    st.metric("Señales Venta", estado['senales_venta'])
    st.metric("Operaciones Activas", estado['ordenes_activas'])
    
    st.subheader("📋 Historial")
    historial = st.session_state.scalping_bot.obtener_historial()
    if historial is not None and not historial.empty:
        st.dataframe(historial, use_container_width=True, height=300)
        
        if 'profit_loss' in historial.columns:
            total_ganancias = historial['profit_loss'].sum()
            st.metric("Ganancias/Pérdidas", f"${total_ganancias:.2f}")
    else:
        st.info("📈 El historial aparecerá aquí")
    
    if st.button("🔄 Reiniciar Sistema", type="secondary"):
        st.session_state.scalping_bot.reiniciar_sistema()
        if 'auto_trading_counter' in st.session_state:
            del st.session_state.auto_trading_counter
        st.success("✅ Sistema reiniciado")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**⚡ ESTRATEGIA BTC SCALPING 3M:** EMA9/21 + RSI7 + Volumen | RR 1:2 + Trailing Stop")
st.markdown("**🎯 PARÁMETROS:** SL 0.25% | TP 0.5% | Riesgo 1% | Apalancamiento x5")

# Debug
with st.expander("🔍 Estado del Sistema"):
    estado = st.session_state.scalping_bot.obtener_estado()
    st.json(estado)
