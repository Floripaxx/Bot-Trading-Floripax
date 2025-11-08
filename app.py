import streamlit as st
import time
import threading
from collections import deque
from datetime import datetime
import pandas as pd
import numpy as np
import hmac
import hashlib
import requests
import json
import plotly.graph_objects as go

# Configurar la página de Streamlit
st.set_page_config(
    page_title="🤖 Bot HFT MEXC",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MexcHighFrequencyTradingBot:
    def __init__(self, api_key: str, secret_key: str, symbol: str = 'BTCUSDT'):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.base_url = 'https://api.mexc.com'
        
        # Estado del trading - SALDO INICIAL 250 USD
        self.position = 0
        self.entry_price = 0
        self.balance = 250.0  # Saldo inicial de 250 USD
        self.positions_history = []
        self.is_running = False
        self.trading_thread = None
        
        # Configuración HFT - AJUSTADO PARA PRECIOS REALES ~$100k
        self.tick_window = 50
        self.tick_data = deque(maxlen=self.tick_window)
        self.position_size = 0.08  # 8% del balance por operación (más agresivo)
        self.max_positions = 3
        self.open_positions = 0
        
        # Estrategias - AJUSTADO PARA PRECIOS REALES
        self.momentum_threshold = 0.0015  # Reducido para mayor sensibilidad
        self.mean_reversion_threshold = 0.001  # Reducido para mayor sensibilidad
        self.volatility_multiplier = 2.0  # Multiplicador para ajustar por volatilidad
        
        # Logging
        self.log_messages = []
        
    def log_message(self, message: str, level: str = "INFO"):
        """Agregar mensaje al log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.log_messages.append(log_entry)
        # Mantener solo los últimos 50 mensajes
        if len(self.log_messages) > 50:
            self.log_messages.pop(0)

    def get_real_price_from_api(self) -> dict:
        """Obtener precio REAL de MEXC sin necesidad de API keys"""
        try:
            # Usar endpoint público que no requiere autenticación
            url = f"https://api.mexc.com/api/v3/ticker/price"
            params = {'symbol': self.symbol}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    current_price = float(data['price'])
                    # Para el bid/ask, usar el precio actual con un pequeño spread
                    spread = current_price * 0.0001  # 0.01% de spread
                    
                    return {
                        'timestamp': datetime.now(),
                        'bid': current_price - spread,
                        'ask': current_price + spread,
                        'symbol': self.symbol,
                        'simulated': False,
                        'source': 'MEXC Real'
                    }
            
            # Si falla la API principal, intentar con Binance como backup
            return self.get_binance_price()
            
        except Exception as e:
            self.log_message(f"Error obteniendo precio real: {e}", "ERROR")
            # Si todo falla, usar Binance
            return self.get_binance_price()

    def get_binance_price(self) -> dict:
        """Obtener precio de Binance como backup"""
        try:
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {'symbol': self.symbol}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    current_price = float(data['price'])
                    spread = current_price * 0.0001
                    
                    return {
                        'timestamp': datetime.now(),
                        'bid': current_price - spread,
                        'ask': current_price + spread,
                        'symbol': self.symbol,
                        'simulated': False,
                        'source': 'Binance Backup'
                    }
            
            # Último recurso: datos realistas basados en precio actual aproximado
            return self.get_realistic_price()
            
        except Exception as e:
            self.log_message(f"Error obteniendo precio de Binance: {e}", "ERROR")
            return self.get_realistic_price()

    def get_realistic_price(self) -> dict:
        """Generar precio realista basado en el mercado actual"""
        # Precios base realistas para diferentes símbolos
        base_prices = {
            'BTCUSDT': 100000,  # Precio realista de BTC alrededor de 100k
            'ETHUSDT': 3500,    # Precio realista de ETH
            'ADAUSDT': 0.45,    # Precio realista de ADA
            'DOTUSDT': 7.5,     # Precio realista de DOT
            'LINKUSDT': 15.0    # Precio realista de LINK
        }
        
        base_price = base_prices.get(self.symbol, 50000)
        # Pequeña variación para simular mercado real
        variation = np.random.uniform(-0.02, 0.02)  # ±2%
        current_price = base_price * (1 + variation)
        spread = current_price * 0.0001
        
        return {
            'timestamp': datetime.now(),
            'bid': current_price - spread,
            'ask': current_price + spread,
            'symbol': self.symbol,
            'simulated': True,
            'source': 'Realistic Simulation'
        }

    def get_ticker_price(self) -> dict:
        """Obtener precio actual - SIEMPRE intentar precio real primero"""
        try:
            # Siempre intentar obtener precio real primero
            real_data = self.get_real_price_from_api()
            
            if real_data and not real_data.get('simulated', True):
                source_msg = real_data.get('source', 'API')
                # No loguear cada precio para no saturar
                if len(self.tick_data) % 20 == 0:
                    self.log_message(f"✅ Precio real: ${real_data['bid']:.2f} desde {source_msg}", "INFO")
            else:
                if len(self.tick_data) % 20 == 0:
                    self.log_message(f"⚠️ Usando datos simulados: ${real_data['bid']:.2f}", "INFO")
                
            return real_data
            
        except Exception as e:
            self.log_message(f"Error crítico obteniendo precio: {e}", "ERROR")
            # Fallback a datos realistas
            return self.get_realistic_price()

    def calculate_indicators(self) -> dict:
        """Calcular indicadores técnicos - OPTIMIZADO PARA PRECIOS REALES"""
        if len(self.tick_data) < 10:
            return {}
        
        prices = [tick['bid'] for tick in self.tick_data]
        df = pd.DataFrame(prices, columns=['price'])
        
        # Indicadores básicos
        df['returns'] = df['price'].pct_change()
        df['momentum'] = df['returns'].rolling(5).mean()
        df['sma_5'] = df['price'].rolling(5).mean()
        df['sma_10'] = df['price'].rolling(10).mean()
        df['price_deviation'] = (df['price'] - df['sma_5']) / df['sma_5']
        
        # Volatilidad ajustada para precios altos
        df['volatility'] = df['returns'].rolling(10).std() * self.volatility_multiplier
        
        # RSI más sensible
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()  # Ventana más corta
        loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD rápido para HFT
        exp12 = df['price'].ewm(span=6, adjust=False).mean()
        exp26 = df['price'].ewm(span=12, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=5, adjust=False).mean()
        
        latest = df.iloc[-1]
        
        return {
            'momentum': latest['momentum'],
            'price_deviation': latest['price_deviation'],
            'current_price': latest['price'],
            'sma_5': latest['sma_5'],
            'sma_10': latest['sma_10'],
            'rsi': latest['rsi'],
            'volatility': latest['volatility'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal']
        }

    def trading_strategy(self, indicators: dict) -> str:
        """Estrategia de trading HFT - OPTIMIZADA PARA PRECIOS REALES"""
        if not indicators:
            return 'hold'
        
        momentum = indicators['momentum']
        deviation = indicators['price_deviation']
        rsi = indicators['rsi']
        volatility = indicators['volatility']
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        
        # Estrategia más agresiva y sensible
        buy_conditions = [
            momentum > self.momentum_threshold,
            deviation < -0.0008,  # Más sensible
            rsi < 40,  # Menos restrictivo
            macd > macd_signal,  # Tendencia alcista
            volatility < 0.025  # Control de volatilidad
        ]
        
        sell_conditions = [
            momentum < -self.momentum_threshold,
            deviation > 0.0008,  # Más sensible
            rsi > 60,  # Menos restrictivo
            macd < macd_signal,  # Tendencia bajista
            volatility < 0.025  # Control de volatilidad
        ]
        
        # Estrategia de scalping rápido
        if sum(buy_conditions) >= 3:  # Solo necesita 3 de 5 condiciones
            self.log_message(f"🔔 SEÑAL COMPRA: momentum={momentum:.4f}, dev={deviation:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'buy'
        elif sum(sell_conditions) >= 3:  # Solo necesita 3 de 5 condiciones
            self.log_message(f"🔔 SEÑAL VENTA: momentum={momentum:.4f}, dev={deviation:.4f}, RSI={rsi:.1f}", "SIGNAL")
            return 'sell'
        
        # Estrategia adicional: tomar ganancias rápidas
        if self.position > 0:
            current_profit_pct = (indicators['current_price'] - self.entry_price) / self.entry_price
            if current_profit_pct > 0.002:  # Tomar ganancias en 0.2%
                self.log_message(f"🎯 TOMANDO GANANCIAS: {current_profit_pct:.3%}", "PROFIT")
                return 'sell'
        
        return 'hold'

    def execute_trade(self, action: str, price: float):
        """Ejecutar operación"""
        try:
            quantity = (self.balance * self.position_size) / price
            
            if action == 'buy':
                if self.open_positions < self.max_positions:
                    self.position += quantity
                    self.entry_price = price
                    self.open_positions += 1
                    trade_info = f"🟢 COMPRA: {quantity:.6f} {self.symbol} @ ${price:.2f}"
                    self.log_message(trade_info, "TRADE")
                    
            elif action == 'sell' and self.position >= quantity:
                profit = (price - self.entry_price) * quantity
                self.position -= quantity
                self.balance += profit
                self.open_positions = max(0, self.open_positions - 1)
                profit_color = "🟢" if profit > 0 else "🔴"
                trade_info = f"{profit_color} VENTA: {quantity:.6f} {self.symbol} @ ${price:.2f} | Profit: ${profit:.4f}"
                self.log_message(trade_info, "TRADE")
            
            # Registrar posición
            self.positions_history.append({
                'timestamp': datetime.now(),
                'action': action,
                'price': price,
                'quantity': quantity,
                'balance': self.balance,
                'position': self.position,
                'open_positions': self.open_positions
            })
            
        except Exception as e:
            self.log_message(f"Error ejecutando trade: {e}", "ERROR")

    def trading_cycle(self):
        """Ciclo principal de trading"""
        self.log_message("🚀 Iniciando ciclo de trading HFT - ESTRATEGIA OPTIMIZADA")
        
        while self.is_running:
            try:
                # Obtener datos de mercado REALES
                tick_data = self.get_ticker_price()
                if tick_data:
                    self.tick_data.append(tick_data)
                
                # Calcular indicadores y ejecutar estrategia
                indicators = self.calculate_indicators()
                
                if indicators and len(self.tick_data) >= 10:
                    signal = self.trading_strategy(indicators)
                    if signal != 'hold':
                        price = tick_data['bid'] if signal == 'buy' else tick_data['ask']
                        self.execute_trade(signal, price)
                
                time.sleep(2)  # Intervalo de 2 segundos para HFT
                
            except Exception as e:
                self.log_message(f"Error en ciclo de trading: {e}", "ERROR")
                time.sleep(5)

    def start_trading(self):
        """Iniciar bot de trading"""
        if not self.is_running:
            self.is_running = True
            self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
            self.trading_thread.start()
            self.log_message("✅ Bot de trading iniciado - ESTRATEGIA AGRESIVA ACTIVADA")

    def stop_trading(self):
        """Detener bot de trading"""
        self.is_running = False
        self.log_message("⏹️ Bot de trading detenido")

    def get_performance_stats(self):
        """Obtener estadísticas de performance"""
        current_price = self.tick_data[-1]['bid'] if self.tick_data else 0
        
        stats = {
            'total_trades': len(self.positions_history),
            'win_rate': 0,
            'current_balance': self.balance,
            'open_positions': self.open_positions,
            'current_price': current_price,
            'total_profit': self.balance - 250.0,
            'equity': self.balance,
            'position_size': self.position
        }
        
        if not self.positions_history:
            return stats
        
        # Calcular win rate
        sell_trades = [t for t in self.positions_history if t['action'] == 'sell']
        
        if sell_trades:
            profitable_trades = 0
            for i in range(1, len(self.positions_history)):
                current_trade = self.positions_history[i]
                prev_trade = self.positions_history[i-1]
                
                if (current_trade['action'] == 'sell' and 
                    prev_trade['action'] == 'buy' and
                    current_trade['price'] > prev_trade['price']):
                    profitable_trades += 1
            
            stats['win_rate'] = (profitable_trades / len(sell_trades)) * 100 if sell_trades else 0
        
        # Calcular equity actual
        stats['equity'] = self.balance + (self.position * current_price) if self.position > 0 else self.balance
        
        return stats

def main():
    # Título principal
    st.title("🤖 Bot de Trading de Alta Frecuencia - MEXC")
    st.markdown("---")
    
    # Inicializar el bot en session_state si no existe
    if 'bot' not in st.session_state:
        st.session_state.bot = MexcHighFrequencyTradingBot("", "", "BTCUSDT")
    
    bot = st.session_state.bot
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Campos de configuración
        api_key = st.text_input("API Key MEXC", type="password", 
                               help="Opcional - El bot funciona sin API keys usando datos reales de mercado")
        secret_key = st.text_input("Secret Key MEXC", type="password",
                                  help="Opcional - El bot funciona sin API keys usando datos reales de mercado")
        symbol = st.selectbox("Símbolo", ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"])
        
        # Actualizar configuración del bot
        bot.api_key = api_key
        bot.secret_key = secret_key
        bot.symbol = symbol
        
        st.markdown("---")
        st.header("🎯 Control del Bot")
        
        # Botones de control
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Iniciar Bot", use_container_width=True):
                bot.start_trading()
                st.rerun()
        
        with col2:
            if st.button("⏹️ Detener Bot", use_container_width=True):
                bot.stop_trading()
                st.rerun()
        
        st.markdown("---")
        st.header("💰 Información de Cuenta")
        st.info(f"**Saldo Inicial:** $250.00 USD")
        
        # Mostrar estado del bot
        if bot.is_running:
            st.success("✅ Bot Ejecutándose - ESTRATEGIA AGRESIVA")
        else:
            st.warning("⏸️ Bot Detenido")
            
        # Información de precios reales
        st.markdown("---")
        st.header("📊 Fuente de Precios")
        if bot.tick_data:
            latest_tick = bot.tick_data[-1]
            source = latest_tick.get('source', 'Unknown')
            st.info(f"**Fuente actual:** {source}")
    
    # Layout principal
    col1, col2, col3, col4 = st.columns(4)
    
    # Obtener estadísticas actuales
    stats = bot.get_performance_stats()
    
    with col1:
        st.metric(
            label="💰 Balance Actual",
            value=f"${stats['current_balance']:.2f}",
            delta=f"${stats['total_profit']:.2f}"
        )
    
    with col2:
        st.metric(
            label="📈 Precio Actual",
            value=f"${stats['current_price']:.2f}"
        )
    
    with col3:
        st.metric(
            label="🎯 Tasa de Acierto",
            value=f"{stats['win_rate']:.1f}%"
        )
    
    with col4:
        st.metric(
            label="📊 Total Operaciones",
            value=f"{stats['total_trades']}"
        )
    
    # Segunda fila de métricas
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            label="🔓 Posiciones Abiertas",
            value=f"{stats['open_positions']}"
        )
    
    with col6:
        st.metric(
            label="📦 Tamaño Posición",
            value=f"{stats['position_size']:.6f}"
        )
    
    with col7:
        st.metric(
            label="💹 Equity Total",
            value=f"${stats['equity']:.2f}"
        )
    
    with col8:
        status = "🟢 LIVE" if bot.is_running else "🔴 STOP"
        st.metric(
            label="🔴 Estado",
            value=status
        )
    
    st.markdown("---")
    
    # Gráficos y datos
    tab1, tab2, tab3 = st.tabs(["📈 Gráfico de Precios", "📋 Historial de Operaciones", "📝 Logs del Sistema"])
    
    with tab1:
        # Gráfico de precios
        if bot.tick_data:
            prices = [tick['bid'] for tick in list(bot.tick_data)]
            timestamps = [tick['timestamp'] for tick in list(bot.tick_data)]
            sources = [tick.get('source', 'Unknown') for tick in list(bot.tick_data)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps, 
                y=prices,
                mode='lines',
                name=f'Precio {bot.symbol}',
                line=dict(color='#00ff88', width=2),
                hovertemplate='<b>Precio:</b> $%{y:.2f}<br><b>Hora:</b> %{x}<br><b>Fuente:</b> ' + sources[-1] + '<extra></extra>'
            ))
            
            # Agregar SMA si hay suficientes datos
            if len(prices) >= 10:
                df = pd.DataFrame(prices, columns=['price'])
                df['sma_10'] = df['price'].rolling(10).mean()
                fig.add_trace(go.Scatter(
                    x=timestamps[9:],
                    y=df['sma_10'].dropna(),
                    mode='lines',
                    name='SMA 10',
                    line=dict(color='#ffaa00', width=1, dash='dash')
                ))
            
            fig.update_layout(
                title=f"Precio de {bot.symbol} en Tiempo Real - Fuente: {sources[-1] if sources else 'Unknown'}",
                xaxis_title="Tiempo",
                yaxis_title="Precio (USD)",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar información de la fuente
            latest_source = sources[-1] if sources else "Unknown"
            if "MEXC" in latest_source:
                st.success(f"✅ Usando datos en tiempo real de MEXC")
            elif "Binance" in latest_source:
                st.warning(f"⚠️ Usando datos de Binance (backup)")
            else:
                st.info(f"ℹ️ Usando datos simulados realistas")
        else:
            st.info("Esperando datos de mercado...")
    
    with tab2:
        # Historial de operaciones
        if bot.positions_history:
            # Crear DataFrame para mostrar
            df = pd.DataFrame(bot.positions_history)
            df['timestamp'] = df['timestamp'].dt.strftime('%H:%M:%S')
            df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
            df['quantity'] = df['quantity'].apply(lambda x: f"{x:.6f}")
            df['balance'] = df['balance'].apply(lambda x: f"${x:.2f}")
            
            # Mostrar tabla
            st.dataframe(
                df[['timestamp', 'action', 'price', 'quantity', 'balance', 'open_positions']],
                use_container_width=True,
                height=400
            )
            
            # Botón para limpiar historial
            if st.button("🗑️ Limpiar Historial", key="clear_history"):
                bot.positions_history.clear()
                st.rerun()
        else:
            st.info("No hay operaciones registradas aún.")
    
    with tab3:
        # Logs del sistema
        log_container = st.container(height=400)
        with log_container:
            for log_entry in reversed(bot.log_messages[-20:]):  # Mostrar últimos 20 logs
                if "ERROR" in log_entry:
                    st.error(log_entry)
                elif "TRADE" in log_entry:
                    if "COMPRA" in log_entry:
                        st.success(log_entry)
                    elif "VENTA" in log_entry:
                        if "🔴" in log_entry:
                            st.error(log_entry)
                        else:
                            st.success(log_entry)
                    else:
                        st.info(log_entry)
                elif "SEÑAL" in log_entry or "PROFIT" in log_entry:
                    st.warning(log_entry)
                elif "Precio actual:" in log_entry:
                    st.info(log_entry)
                else:
                    st.info(log_entry)
        
        # Botones de control de logs
        col_log1, col_log2 = st.columns(2)
        with col_log1:
            if st.button("🔄 Actualizar Logs", use_container_width=True, key="refresh_logs"):
                st.rerun()
        with col_log2:
            if st.button("🗑️ Limpiar Logs", use_container_width=True, key="clear_logs"):
                bot.log_messages.clear()
                st.rerun()
    
    # Auto-refresh cada 5 segundos si el bot está corriendo
    if bot.is_running:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
