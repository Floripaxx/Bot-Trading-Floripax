# ... código anterior ...

def obtener_senal_compra_venta(df, rsi_periodo=14, bb_periodo=20, stoch_k=14, stoch_d=3, adx_periodo=14):
    """
    Obtener señal de compra o venta basada en múltiples indicadores
    """
    try:
        # Calcular RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_periodo).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_periodo).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calcular Bandas de Bollinger
        bb_ma = df['close'].rolling(window=bb_periodo).mean()
        bb_std = df['close'].rolling(window=bb_periodo).std()
        df['bb_upper'] = bb_ma + (bb_std * 2)
        df['bb_lower'] = bb_ma - (bb_std * 2)
        df['bb_middle'] = bb_ma
        
        # Calcular Estocástico
        low_min = df['low'].rolling(window=stoch_k).min()
        high_max = df['high'].rolling(window=stoch_k).max()
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=stoch_d).mean()
        
        # Calcular ADX
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['plus_dm'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0),
            0
        )
        df['minus_dm'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0),
            0
        )
        
        tr_smooth = df['tr'].rolling(window=adx_periodo).mean()
        plus_dm_smooth = df['plus_dm'].rolling(window=adx_periodo).mean()
        minus_dm_smooth = df['minus_dm'].rolling(window=adx_periodo).mean()
        
        df['plus_di'] = 100 * (plus_dm_smooth / tr_smooth)
        df['minus_di'] = 100 * (minus_dm_smooth / tr_smooth)
        dx = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
        df['adx'] = dx.rolling(window=adx_periodo).mean()
        
        # Obtener últimos valores
        ultimo = df.iloc[-1]
        penultimo = df.iloc[-2]
        
        # CONDICIONES DE COMPRA (más estrictas)
        condicion_compra_rsi = (ultimo['rsi'] < 35 and penultimo['rsi'] >= 35)
        condicion_compra_bb = ultimo['close'] < ultimo['bb_lower']
        condicion_compra_stoch = (ultimo['stoch_k'] < 20 and ultimo['stoch_d'] < 20 and 
                                ultimo['stoch_k'] > ultimo['stoch_d'])
        condicion_compra_adx = ultimo['adx'] > 25
        condicion_tendencia = ultimo['plus_di'] > ultimo['minus_di']
        
        # Señal de compra (requiere múltiples condiciones)
        senal_compra = (
            condicion_compra_rsi and 
            condicion_compra_bb and 
            condicion_compra_stoch and
            condicion_tendencia and
            condicion_compra_adx
        )
        
        # CONDICIONES DE VENTA (más estrictas)
        condicion_venta_rsi = (ultimo['rsi'] > 65 and penultimo['rsi'] <= 65)
        condicion_venta_bb = ultimo['close'] > ultimo['bb_upper']
        condicion_venta_stoch = (ultimo['stoch_k'] > 80 and ultimo['stoch_d'] > 80 and 
                               ultimo['stoch_k'] < ultimo['stoch_d'])
        condicion_venta_adx = ultimo['adx'] > 25
        condicion_tendencia_venta = ultimo['minus_di'] > ultimo['plus_di']
        
        # Señal de venta (requiere múltiples condiciones)
        senal_venta = (
            condicion_venta_rsi and 
            condicion_venta_bb and 
            condicion_venta_stoch and
            condicion_tendencia_venta and
            condicion_venta_adx
        )
        
        return senal_compra, senal_venta
        
    except Exception as e:
        print(f"Error calculando señales: {e}")
        return False, False

# Función para verificar si ya tenemos una operación abierta
def hay_operacion_abierta(symbol):
    """
    Verificar si ya tenemos una operación abierta para evitar duplicados
    """
    try:
        # Aquí implementarías la lógica para verificar operaciones abiertas
        # Por ahora, asumimos que no hay operaciones abiertas
        return False
    except Exception as e:
        print(f"Error verificando operaciones abiertas: {e}")
        return False

# Función principal mejorada
def ejecutar_bot():
    """
    Función principal del bot con controles para reducir operaciones
    """
    try:
        # Obtener datos
        df = obtener_datos_binance()
        if df is None or len(df) < 50:
            print("No hay suficientes datos")
            return
        
        # Obtener señales
        senal_compra, senal_venta = obtener_senal_compra_venta(df)
        
        # Verificar si ya hay operación abierta
        if hay_operacion_abierta('BTCUSDT'):
            print("Ya hay una operación abierta, esperando...")
            return
        
        # Ejecutar órdenes solo si las señales son fuertes
        if senal_compra:
            print("🔵 SEÑAL DE COMPRA DETECTADA")
            # Aquí iría la lógica de compra
            
        elif senal_venta:
            print("🔴 SEÑAL DE VENTA DETECTADA")
            # Aquí iría la lógica de venta
            
        else:
            print("⚪ Sin señal clara, esperando...")
            
    except Exception as e:
        print(f"Error en ejecutar_bot: {e}")

# Configurar el intervalo de ejecución (más largo para reducir operaciones)
INTERVALO_EJECUCION = 300  # 5 minutos en lugar de 1 minuto

# ... resto del código ...
