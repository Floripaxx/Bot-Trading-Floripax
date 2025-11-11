import pandas as pd
import asyncio
import logging
from datetime import datetime
import streamlit as st
import time

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== NUEVAS VARIABLES GLOBALES ==========
BOT_ACTIVE = False  # Estado del bot
LAST_SYNC_TIME = None

# ========== VARIABLES DE POSICIÓN (SI NO EXISTEN) ==========
has_open_position = False
entry_price = 0.0
position_size = 0.0
position_side = None

# ========== INDICADOR VISUAL EN STREAMLIT ==========
def display_bot_status():
    """Muestra el estado del bot de forma visible en la interfaz"""
    st.sidebar.markdown("---")
    
    if BOT_ACTIVE:
        st.sidebar.markdown(
            """
            <div style="background-color: #4CAF50; padding: 10px; border-radius: 5px; text-align: center; color: white;">
                <h3>🟢 BOT ACTIVO</h3>
                <p>Ejecutando operaciones</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div style="background-color: #ff4444; padding: 10px; border-radius: 5px; text-align: center; color: white;">
                <h3>🔴 BOT DETENIDO</h3>
                <p>No ejecuta operaciones</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Mostrar última sincronización
    if LAST_SYNC_TIME:
        st.sidebar.info(f"🕐 Última sincronización: {LAST_SYNC_TIME}")

# ========== FUNCIÓN DE SINCRONIZACIÓN FORZADA ==========
async def force_sync_with_exchange(exchange):
    """
    SINCRONIZACIÓN FORZADA: Verifica el estado REAL del exchange y ajusta el estado interno
    """
    global LAST_SYNC_TIME, has_open_position, entry_price, position_size, position_side
    
    try:
        logger.info("🔄 INICIANDO SINCRONIZACIÓN FORZADA CON EXCHANGE...")
        
        # Obtener posiciones reales del exchange
        positions = await exchange.fetch_positions(['BTC/USDT:USDT'])
        open_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]
        
        # DEBUG: Logear lo que encontró
        logger.info(f"📊 Exchange reporta {len(open_positions)} posiciones abiertas")
        
        if len(open_positions) == 0:
            # EXCHANGE DICE: No hay posiciones → RESETEAR estado interno
            if hasattr(st, 'session_state'):
                if hasattr(st.session_state, 'has_open_position') and st.session_state.has_open_position:
                    logger.warning("🚨 CORRECCIÓN: Bot tenía posición fantasma. Reseteando estado.")
                    st.session_state.has_open_position = False
                    st.session_state.entry_price = 0.0
                    st.session_state.position_size = 0.0
                    st.session_state.position_side = None
            else:
                # Resetear variables globales
                if has_open_position:
                    logger.warning("🚨 CORRECCIÓN: Bot tenía posición fantasma. Reseteando estado.")
                    has_open_position = False
                    entry_price = 0.0
                    position_size = 0.0
                    position_side = None
            
            logger.info("✅ Sincronización completada: Estado reseteado a NEUTRAL")
            LAST_SYNC_TIME = datetime.now().strftime("%H:%M:%S")
            return "NEUTRAL"
        else:
            # EXCHANGE DICE: Hay posición abierta → Actualizar estado interno
            pos = open_positions[0]
            logger.info(f"✅ Sincronización: Posición real encontrada - {pos['side']} {pos['contracts']} contratos")
            LAST_SYNC_TIME = datetime.now().strftime("%H:%M:%S")
            return "POSITION_OPEN"
            
    except Exception as e:
        logger.error(f"❌ Error en sincronización: {e}")
        return "ERROR"

# ========== FUNCIONES DE CONTROL DEL BOT ==========
def start_bot():
    """Inicia el bot"""
    global BOT_ACTIVE
    BOT_ACTIVE = True
    logger.info("🚀 Bot iniciado")
    st.success("Bot iniciado correctamente")

def stop_bot():
    """Detiene el bot"""
    global BOT_ACTIVE
    BOT_ACTIVE = False
    logger.info("🛑 Bot detenido")
    st.warning("Bot detenido")

# ========== INTERFAZ DE CONTROL EN STREAMLIT ==========
def create_control_panel():
    """Crea el panel de control en Streamlit"""
    st.sidebar.title("🤖 Panel de Control")
    
    # Botones de control
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🚀 Iniciar Bot", type="primary"):
            start_bot()
    
    with col2:
        if st.button("🛑 Detener Bot", type="secondary"):
            stop_bot()
    
    # Botón de sincronización manual
    if st.sidebar.button("🔄 Sincronizar Ahora"):
        st.sidebar.info("Sincronización iniciada...")
        # La sincronización se ejecutará en el loop principal
    
    # Mostrar estado del bot
    display_bot_status()

# ========== FUNCIÓN MAIN MODIFICADA ==========
async def main():
    logger.info("🤖 INICIANDO BOT DE TRADING...")
    
    # ==== INICIALIZAR EXCHANGE (AGREGAR ESTO) ====
    try:
        # REEMPLAZA ESTO CON TU CÓDIGO DE INICIALIZACIÓN DEL EXCHANGE
        from ccxt import binanceusdm  # o el exchange que uses
        
        exchange = binanceusdm({
            'apiKey': 'tu_api_key',
            'secret': 'tu_secret',
            'enableRateLimit': True,
            'sandbox': False,  # Cambia a True para testing
        })
        
        logger.info("✅ Exchange inicializado correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error al inicializar exchange: {e}")
        return
    
    # ==== CAMBIO CRÍTICO: Sincronización ANTES de cualquier operación ====
    sync_result = await force_sync_with_exchange(exchange)
    if sync_result == "ERROR":
        logger.error("NO SE PUEDE INICIAR - Error de sincronización")
        return
    
    logger.info("✅ Bot sincronizado con exchange. Iniciando operaciones...")
    
    # ========== TU CÓDIGO ORIGINAL CONTINÚA AQUÍ ==========
    while True:
        try:
            # Solo ejecutar operaciones si el bot está activo
            if BOT_ACTIVE:
                # [MANTENER TODO TU CÓDIGO EXISTENTE DE TRADING]
                # Ejemplo:
                # await check_signals()
                # await execute_trades()
                
                # Tu lógica de trading aquí
                pass
            else:
                # Bot detenido - esperar
                await asyncio.sleep(1)
                continue
                
            await asyncio.sleep(1)  # Ajustar según tu intervalo
            
        except Exception as e:
            logger.error(f"Error en loop principal: {e}")
            await asyncio.sleep(5)

# ========== FUNCIÓN DE CIERRE MEJORADA ==========
async def close_position(exchange, price):
    """
    Función de cierre con verificación de confirmación
    """
    global has_open_position, entry_price, position_size, position_side
    
    try:
        # [MANTENER TU CÓDIGO ORIGINAL DE CERRAR POSICIÓN]
        
        # EJEMPLO de tu código actual:
        # order = await exchange.create_order(symbol, 'market', 'sell', quantity)
        # logger.info(f"Orden de cierre ejecutada: {order}")
        
        # ==== CAMBIO CRÍTICO: Verificación de cierre real ====
        await asyncio.sleep(2)  # Esperar que el exchange procese
        
        # Verificar que realmente se cerró
        positions = await exchange.fetch_positions(['BTC/USDT:USDT'])
        open_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]
        
        if len(open_positions) == 0:
            logger.info("✅ Posición cerrada confirmada por exchange")
            # Resetear estado interno
            if hasattr(st, 'session_state'):
                st.session_state.has_open_position = False
            else:
                has_open_position = False
                entry_price = 0.0
                position_size = 0.0
                position_side = None
        else:
            logger.warning("⚠️ Posición podría no haberse cerrado completamente")
            
    except Exception as e:
        logger.error(f"❌ Error al cerrar posición: {e}")

# ========== CONFIGURACIÓN STREAMLIT ==========
def main_streamlit():
    """Función principal de Streamlit"""
    st.set_page_config(page_title="Trading Bot", layout="wide")
    
    st.title("🤖 Bot de Trading Automatizado")
    
    # Panel de control
    create_control_panel()
    
    # [MANTENER TU INTERFAZ ACTUAL DE DATOS Y MÉTRICAS]
    st.header("📊 Métricas en Tiempo Real")
    
    # Aquí va tu código actual de visualización de datos
    # ...

# Punto de entrada
if __name__ == "__main__":
    # Iniciar interfaz Streamlit
    main_streamlit()
    
    # Iniciar bot de trading en segundo plano
    # Usar esta línea si Streamlit no bloquea el event loop
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Error al ejecutar bot: {e}")
