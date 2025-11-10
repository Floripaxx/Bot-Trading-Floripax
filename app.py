# EN LA CLASE MexcHighFrequencyTradingBot, ACTUALIZAR:

def __init__(self, api_key: str, secret_key: str, symbol: str = 'BTCUSDT'):
    # ... código existente ...
    
    # 🚨 MEJORAS CRÍTICAS - AJUSTAR PARÁMETROS:
    self.required_conditions = 4    # AUMENTADO: De 3 a 4 condiciones mínimas
    self.rsi_upper_bound = 65       # BAJADO: De 68 a 65 (más conservador)
    self.rsi_lower_bound = 35       # SUBIDO: De 32 a 35 (mejor zona)
    
    # Nuevo filtro para evitar señales consecutivas
    self.last_signal_time = None
    self.min_signal_interval = 10   # segundos entre señales

def trading_strategy(self, indicators: dict) -> str:
    """Estrategia OPTIMIZADA con mejores filtros"""
    if not indicators:
        return 'hold'
    
    # 🚨 MEJORA 1: Validar que los indicadores sean numéricos
    if any(np.isnan(indicators.get(key, 0)) or not isinstance(indicators.get(key, 0), (int, float)) 
           for key in ['rsi', 'momentum', 'macd']):
        return 'hold'
    
    momentum = indicators['momentum']
    rsi = indicators['rsi']
    macd = indicators['macd']
    macd_signal = indicators['macd_signal']
    current_price = indicators['current_price']
    bb_upper = indicators['bb_upper']
    bb_lower = indicators['bb_lower']
    volume_ratio = indicators['volume_ratio']
    sma_10 = indicators['sma_10']
    sma_20 = indicators['sma_20']
    
    # 🚨 MEJORA 2: Filtro de intervalo entre señales
    current_time = datetime.now()
    if self.last_signal_time:
        time_since_last = (current_time - self.last_signal_time).total_seconds()
        if time_since_last < self.min_signal_interval:
            return 'hold'
    
    # 🚨 MEJORA 3: Condiciones MÁS ESTRICTAS
    buy_conditions = [
        momentum > self.momentum_threshold,
        self.rsi_lower_bound < rsi < self.rsi_upper_bound,  # RSI en zona ÓPTIMA
        macd > macd_signal,
        current_price > sma_10 > sma_20,
        volume_ratio > self.min_volume_threshold,
        current_price < bb_upper * 0.95,  # MÁS alejado de resistencia
        current_price > sma_10  # Precio sobre SMA 10
    ]
    
    sell_conditions = [
        momentum < -self.momentum_threshold * 0.8,
        rsi > 60,  # BAJADO: Vender antes de sobrecompra extrema
        macd < macd_signal,
        current_price < sma_10,
        self.position > 0,
        current_price > self.entry_price * (1 + self.min_profit_target * 0.5)  # Si tenemos ganancia parcial
    ]
    
    if not self.can_trade():
        return 'hold'
    
    # TOMA DE GANANCIAS MEJORADA
    if self.position > 0:
        current_profit_pct = (current_price - self.entry_price) / self.entry_price
        
        # 🚨 MEJORA 4: Take profit más inteligente
        if current_profit_pct >= self.min_profit_target:
            # Tomar ganancias si RSI está alto O momentum se debilita
            if rsi > 62 or momentum < 0:
                self.log_message(f"🎯 TOMANDO GANANCIAS: {current_profit_pct:.3%}, RSI={rsi:.1f}", "PROFIT")
                self.last_signal_time = current_time
                return 'sell'
        
        # STOP LOSS MEJORADO
        current_loss_pct = (self.entry_price - current_price) / self.entry_price
        if current_loss_pct >= self.max_loss_stop:
            self.log_message(f"🛑 STOP LOSS: {current_loss_pct:.3%}, RSI={rsi:.1f}", "LOSS")
            self.last_signal_time = current_time
            return 'sell'
    
    # 🚨 MEJORA 5: SEÑALES MÁS SELECTIVAS
    buy_signal_strength = sum(buy_conditions)
    sell_signal_strength = sum(sell_conditions)
    
    # Solo señales FUERTES (no "débiles")
    if buy_signal_strength >= self.required_conditions:
        # FILTRO EXTRA: RSI no puede estar en sobrecompra para comprar
        if rsi < 60:  
            self.log_message(f"✅ SEÑAL COMPRA CONFIRMADA: condiciones={buy_signal_strength}/7, RSI={rsi:.1f}", "SIGNAL")
            self.last_signal_time = current_time
            return 'buy'
    
    elif sell_signal_strength >= 4:  # Aumentado de 3 a 4
        self.log_message(f"🎯 SEÑAL VENTA CONFIRMADA: condiciones={sell_signal_strength}/6, RSI={rsi:.1f}", "SIGNAL")
        self.last_signal_time = current_time
        return 'sell'
    
    return 'hold'
