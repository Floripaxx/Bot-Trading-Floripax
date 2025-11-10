def trading_strategy(self, indicators: dict) -> str:
    """Estrategia con corrección RSI - MÍNIMOS CAMBIOS"""
    if not indicators:
        return 'hold'
    
    # 🚨 SOLO ESTA VALIDACIÓN CRÍTICA (evitar NaN)
    if np.isnan(indicators.get('rsi', 0)):
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
    
    # 🚨 CORRECCIÓN PRINCIPAL: No comprar si RSI > 60
    buy_conditions = [
        momentum > self.momentum_threshold,
        self.rsi_lower_bound < rsi < self.rsi_upper_bound,
        macd > macd_signal,
        current_price > sma_10 > sma_20,
        volume_ratio > self.min_volume_threshold,
        current_price < bb_upper * 0.98,
        rsi < 60  # 🚨 NUEVA CONDICIÓN: RSI menor a 60 para comprar
    ]
    
    sell_conditions = [
        momentum < -self.momentum_threshold * 0.8,
        rsi > 65,  # 🚨 AJUSTADO: Vender con RSI > 65
        macd < macd_signal,
        current_price < sma_10,
        self.position > 0
    ]
    
    if not self.can_trade():
        return 'hold'
    
    # TOMA DE GANANCIAS (igual que antes)
    if self.position > 0:
        current_profit_pct = (current_price - self.entry_price) / self.entry_price
        current_loss_pct = (self.entry_price - current_price) / self.entry_price
        
        if current_profit_pct >= self.min_profit_target:
            self.log_message(f"🎯 TOMANDO GANANCIAS: {current_profit_pct:.3%}", "PROFIT")
            return 'sell'
        
        if current_loss_pct >= self.max_loss_stop:
            self.log_message(f"🛑 STOP LOSS: {current_loss_pct:.3%}", "LOSS")
            return 'sell'
    
    # SEÑALES (igual que antes, solo cambian las condiciones internas)
    buy_signal_strength = sum(buy_conditions)
    sell_signal_strength = sum(sell_conditions)
    
    if buy_signal_strength >= self.required_conditions:
        self.log_message(f"✅ SEÑAL COMPRA: condiciones={buy_signal_strength}/7, RSI={rsi:.1f}", "SIGNAL")
        return 'buy'
    elif sell_signal_strength >= 3:
        self.log_message(f"🎯 SEÑAL VENTA: condiciones={sell_signal_strength}/6, RSI={rsi:.1f}", "SIGNAL")
        return 'sell'
    
    return 'hold'
