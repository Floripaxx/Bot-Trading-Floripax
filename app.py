def get_real_time_price(self, coin: str) -> float:
    """Obtener precio real desde MEXC - CORREGIDO"""
    symbol = self.coins[coin]
    price = self.mexc_api.obtener_precio_actual(symbol)
    
    if price is not None:
        return price  # Usar el precio real directamente de MEXC
    
    # Solo fallback si la API falla completamente
    st.warning(f"API MEXC no disponible, usando precio simulado para {coin}")
    base_prices = {
        'BTC': 101260.0,  # Precio real aproximado
        'ETH': 2500.0, 
        'SOL': 100.0,
        'XRP': 0.60, 
        'DOGE': 0.08, 
        'BNB': 300.0
    }
    volatility = np.random.uniform(-0.002, 0.002)
    return base_prices[coin] * (1 + volatility)
