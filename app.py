import time
import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
import threading
from collections import deque
import hmac
import hashlib
import requests
import json

class MexcHighFrequencyTradingBot:
    def __init__(self, api_key: str, secret_key: str, symbol: str = 'BTCUSDT'):
        self.api_key = api_key
        self.secret_key = secret_key
        self.symbol = symbol
        self.base_url = 'https://api.mexc.com'
        self.ws_url = 'wss://wbs.mexc.com/ws'
        
        # Estado del trading
        self.position = 0
        self.entry_price = 0
        self.balance = 1000  # Balance inicial en USDT
        self.positions_history = []
        
        # Configuración HFT para MEXC
        self.tick_window = 50
        self.tick_data = deque(maxlen=self.tick_window)
        self.min_spread = 0.01  # 1%
        self.position_size = 0.05  # 5% del balance por operación
        self.max_positions = 3
        self.open_positions = 0
        
        # Estrategias HFT
        self.momentum_threshold = 0.003  # 0.3%
        self.mean_reversion_threshold = 0.002  # 0.2%
        self.volatility_lookback = 15
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def generate_signature(self, params: dict) -> str:
        """Generar firma para API de MEXC"""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def mexc_request(self, endpoint: str, method: str = 'GET', params: dict = None) -> dict:
        """Realizar petición a la API de MEXC"""
        try:
            url = f"{self.base_url}{endpoint}"
            headers = {
                'X-MEXC-APIKEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            if params is None:
                params = {}
                
            if method == 'GET':
                params['timestamp'] = int(time.time() * 1000)
                params['signature'] = self.generate_signature(params)
                response = requests.get(url, headers=headers, params=params)
            else:
                params['timestamp'] = int(time.time() * 1000)
                params['signature'] = self.generate_signature(params)
                response = requests.post(url, headers=headers, data=json.dumps(params))
            
            return response.json()
        except Exception as e:
            self.logger.error(f"Error en petición MEXC: {e}")
            return {}

    def get_account_balance(self) -> float:
        """Obtener balance de la cuenta en MEXC"""
        try:
            data = self.mexc_request('/api/v3/account')
            if 'balances' in data:
                for balance in data['balances']:
                    if balance['asset'] == 'USDT':
                        return float(balance['free'])
            return 0.0
        except Exception as e:
            self.logger.error(f"Error obteniendo balance: {e}")
            return 0.0

    def get_ticker_price(self) -> Dict:
        """Obtener precio actual de MEXC"""
        try:
            endpoint = '/api/v3/ticker/bookTicker'
            params = {'symbol': self.symbol}
            data = self.mexc_request(endpoint, params=params)
            
            if data and 'bidPrice' in data and 'askPrice' in data:
                return {
                    'timestamp': datetime.now(),
                    'bid': float(data['bidPrice']),
                    'ask': float(data['askPrice']),
                    'symbol': self.symbol
                }
            return None
        except Exception as e:
            self.logger.error(f"Error obteniendo ticker: {e}")
            return None

    def place_order(self, side: str, quantity: float, price: float = None) -> bool:
        """Colocar orden en MEXC"""
        try:
            endpoint = '/api/v3/order'
            params = {
                'symbol': self.symbol,
                'side': side.upper(),
                'type': 'LIMIT',
                'quantity': round(quantity, 6),
                'price': round(price, 2) if price else None,
                'timeInForce': 'IOC'  # Immediate or Cancel para HFT
            }
            
            # Para market orders
            if price is None:
                params['type'] = 'MARKET'
                params.pop('price')
                params.pop('timeInForce')
            
            result = self.mexc_request(endpoint, 'POST', params)
            
            if 'orderId' in result:
                self.logger.info(f"Orden {side} ejecutada: {quantity} {self.symbol} @ {price}")
                return True
            else:
                self.logger.error(f"Error en orden: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error colocando orden: {e}")
            return False

    def calculate_indicators(self) -> Dict:
        """Calcular indicadores para estrategias HFT"""
        if len(self.tick_data) < self.volatility_lookback:
            return {}
        
        prices = [tick['bid'] for tick in self.tick_data]
        df = pd.DataFrame(prices, columns=['price'])
        
        # Indicadores de momentum
        df['returns'] = df['price'].pct_change()
        df['momentum'] = df['returns'].rolling(5).mean()
        df['volatility'] = df['returns'].rolling(self.volatility_lookback).std()
        
        # Indicadores de mean reversion
        df['sma_10'] = df['price'].rolling(10).mean()
        df['sma_5'] = df['price'].rolling(5).mean()
        df['price_deviation'] = (df['price'] - df['sma_5']) / df['sma_5']
        
        # RSI rápido
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=8).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=8).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        latest = df.iloc[-1]
        
        return {
            'momentum': latest['momentum'],
            'volatility': latest['volatility'],
            'price_deviation': latest['price_deviation'],
            'current_price': latest['price'],
            'sma_5': latest['sma_5'],
            'sma_10': latest['sma_10'],
            'rsi': latest['rsi']
        }

    def scalping_strategy(self, indicators: Dict) -> str:
        """Estrategia de scalping para HFT en MEXC"""
        if not indicators:
            return 'hold'
        
        momentum = indicators['momentum']
        deviation = indicators['price_deviation']
        rsi = indicators['rsi']
        volatility = indicators['volatility']
        
        # Condiciones de compra (scalping largo)
        buy_conditions = [
            momentum > self.momentum_threshold,
            deviation < -0.001,  # Precio por debajo de SMA
            rsi < 40,  # RSI oversold
            volatility < 0.015  # Baja volatilidad
        ]
        
        # Condiciones de venta (scalping corto)
        sell_conditions = [
            momentum < -self.momentum_threshold,
            deviation > 0.001,  # Precio por encima de SMA
            rsi > 60,  # RSI overbought
            volatility < 0.015
        ]
        
        if sum(buy_conditions) >= 3:
            return 'buy'
        elif sum(sell_conditions) >= 3:
            return 'sell'
        
        return 'hold'

    def market_making_strategy(self, indicators: Dict, current_bid: float, current_ask: float) -> str:
        """Estrategia de market making"""
        if not indicators or self.open_positions >= self.max_positions:
            return 'hold'
        
        spread = (current_ask - current_bid) / current_bid
        
        if spread > self.min_spread:
            # Oportunidad de market making
            if self.position == 0:
                return 'buy'
            elif self.position > 0 and indicators['price_deviation'] > 0.002:
                return 'sell'
        
        return 'hold'

    def execute_trade(self, action: str, price: float):
        """Ejecutar operación en MEXC"""
        try:
            if self.open_positions >= self.max_positions and action == 'buy':
                self.logger.warning("Máximo de posiciones alcanzado")
                return
            
            # Calcular cantidad basada en balance y tamaño de posición
            quantity = (self.balance * self.position_size) / price
            
            if action == 'buy':
                if self.place_order('BUY', quantity, price * 0.999):  # Precio ligeramente mejor
                    self.position += quantity
                    self.entry_price = price
                    self.open_positions += 1
                    self.logger.info(f"COMPRA HFT: {quantity:.6f} {self.symbol} @ {price:.2f}")
                    
            elif action == 'sell' and self.position >= quantity:
                if self.place_order('SELL', quantity, price * 1.001):  # Precio ligeramente mejor
                    profit = (price - self.entry_price) * quantity
                    self.position -= quantity
                    self.balance += profit
                    self.open_positions = max(0, self.open_positions - 1)
                    self.logger.info(f"VENTA HFT: {quantity:.6f} {self.symbol} @ {price:.2f} | Profit: {profit:.4f} USDT")
            
            # Actualizar balance periódicamente
            if len(self.positions_history) % 10 == 0:
                self.balance = self.get_account_balance()
            
            # Registrar posición
            self.positions_history.append({
                'timestamp': datetime.now(),
                'action': action,
                'price': price,
                'quantity': quantity,
                'balance': self.balance,
                'position': self.position
            })
            
        except Exception as e:
            self.logger.error(f"Error ejecutando trade: {e}")

    def risk_management(self):
        """Gestión de riesgo específica para MEXC"""
        if len(self.positions_history) < 5:
            return
        
        # Calcular drawdown reciente
        recent_trades = self.positions_history[-10:]
        profits = []
        for i in range(1, len(recent_trades)):
            if recent_trades[i]['action'] == 'sell':
                profit = (recent_trades[i]['price'] - recent_trades[i-1]['price']) * recent_trades[i]['quantity']
                profits.append(profit)
        
        if profits and sum(profits) < -self.balance * 0.03:  # 3% drawdown
            self.logger.warning("Drawdown del 3% detectado - reduciendo exposición")
            self.position_size = max(0.01, self.position_size * 0.5)  # Reducir tamaño de posición

    def trading_cycle(self):
        """Ciclo principal de trading HFT para MEXC"""
        self.logger.info(f"Iniciando ciclo HFT para {self.symbol} en MEXC")
        
        while True:
            try:
                # Obtener datos de mercado
                tick_data = self.get_ticker_price()
                if not tick_data:
                    time.sleep(0.2)
                    continue
                
                self.tick_data.append(tick_data)
                
                # Calcular indicadores
                indicators = self.calculate_indicators()
                
                # Aplicar estrategias si tenemos suficientes datos
                if len(self.tick_data) >= self.volatility_lookback:
                    # Estrategia principal de scalping
                    signal = self.scalping_strategy(indicators)
                    
                    # Estrategia secundaria de market making
                    if signal == 'hold':
                        signal = self.market_making_strategy(
                            indicators, 
                            tick_data['bid'], 
                            tick_data['ask']
                        )
                    
                    # Ejecutar trade si hay señal
                    if signal != 'hold':
                        price = tick_data['bid'] if signal == 'buy' else tick_data['ask']
                        self.execute_trade(signal, price)
                
                # Gestión de riesgo
                self.risk_management()
                
                # Intervalo entre operaciones (ajustable para HFT)
                time.sleep(0.3)  # ~3 operaciones por segundo
                
            except Exception as e:
                self.logger.error(f"Error en ciclo de trading: {e}")
                time.sleep(1)

    def start_trading(self):
        """Iniciar bot de trading HFT en MEXC"""
        self.logger.info("Iniciando Bot de Alta Frecuencia en MEXC")
        
        # Verificar conexión y balance
        balance = self.get_account_balance()
        if balance > 0:
            self.balance = balance
            self.logger.info(f"Balance inicial: {balance} USDT")
        else:
            self.logger.warning("Balance insuficiente o error de conexión")
        
        # Iniciar ciclo de trading en hilo separado
        trading_thread = threading.Thread(target=self.trading_cycle)
        trading_thread.daemon = True
        trading_thread.start()

    def get_performance_report(self) -> str:
        """Generar reporte de desempeño"""
        if not self.positions_history:
            return "No hay operaciones realizadas aún"
        
        df = pd.DataFrame(self.positions_history)
        total_trades = len(df)
        
        # Calcular métricas
        buy_trades = df[df['action'] == 'buy']
        sell_trades = df[df['action'] == 'sell']
        
        win_rate = 0
        if len(sell_trades) > 0:
            profitable_trades = len([1 for i in range(1, len(self.positions_history)) 
                                  if self.positions_history[i]['action'] == 'sell' and
                                  self.positions_history[i]['price'] > self.positions_history[i-1]['price']])
            win_rate = (profitable_trades / len(sell_trades)) * 100
        
        return f"""
        🤖 REPORTE HFT MEXC 🤖
        ───────────────────
        • Símbolo: {self.symbol}
        • Total Órdenes: {total_trades}
        • Tasa de Acierto: {win_rate:.1f}%
        • Balance: {self.balance:.2f} USDT
        • Posición Actual: {self.position:.6f}
        • Posiciones Abiertas: {self.open_positions}
        • Última Actualización: {datetime.now().strftime('%H:%M:%S')}
        ───────────────────
        """

# Configuración y uso
if __name__ == "__main__":
    # Credenciales de MEXC (¡MANTENER SEGURAS!)
    MEXC_API_KEY = "tu_api_key_de_mexc"
    MEXC_SECRET_KEY = "tu_secret_key_de_mexc"
    
    # Símbolos populares en MEXC
    SYMBOL = "BTCUSDT"  # También: ETHUSDT, SOLUSDT, etc.
    
    # Inicializar bot
    bot = MexcHighFrequencyTradingBot(
        api_key=MEXC_API_KEY,
        secret_key=MEXC_SECRET_KEY,
        symbol=SYMBOL
    )
    
    # Iniciar trading
    bot.start_trading()
    
    # Loop principal para monitoreo
    try:
        while True:
            time.sleep(30)  # Reporte cada 30 segundos
            print(bot.get_performance_report())
            print("⏳ Ejecutando estrategias HFT...")
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido por el usuario")
