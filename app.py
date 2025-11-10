import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import ta
import requests
import json
import os
import hmac
import hashlib
import base64
from urllib.parse import urlencode

# Configuración logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class MexcAPI:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.mexc.com"
        
    def _generate_signature(self, params: dict) -> str:
        """Genera firma para autenticación MEXC"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        """Realiza petición a la API de MEXC"""
        url = f"{self.base_url}{endpoint}"
        headers = {
            'Content-Type': 'application/json',
            'X-MEXC-APIKEY': self.api_key
        }
        
        try:
            if signed and params:
                params['timestamp'] = int(time.time() * 1000)
                params['signature'] = self._generate_signature(params)
            
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=params, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, params=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"Método HTTP no soportado: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error en petición API: {e}")
            return {}
    
    def get_klines(self, symbol: str, interval: str = '15m', limit: int = 100) -> list:
        """Obtiene datos de velas de MEXC"""
        endpoint = '/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        return self._request('GET', endpoint, params)
    
    def get_account_info(self) -> dict:
        """Obtiene información de la cuenta"""
        endpoint = '/api/v3/account'
        params = {}
        return self._request('GET', endpoint, params, signed=True)
    
    def create_order(self, symbol: str, side: str, order_type: str, quantity: float) -> dict:
        """Crea una orden MARKET en MEXC"""
        endpoint = '/api/v3/order'
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
            'quantity': round(quantity, 6)  # Redondear a 6 decimales
        }
        
        return self._request('POST', endpoint, params, signed=True)

class EnhancedTradingBot:
    def __init__(self, api_key: str, api_secret: str, symbol: str = 'BTCUSDT'):
        self.client = MexcAPI(api_key, api_secret)
        self.symbol = symbol
        self.initial_balance = 250.0
        self.current_balance = self.initial_balance
        self.trade_history = []
        
        # Configuración mejorada
        self.config = {
            'timeframe': '15m',
            'risk_per_trade': 0.02,  # 2% del capital por operación
            'target_profit': 0.004,  # 0.4% profit target
            'stop_loss': 0.002,  # 0.2% stop loss
            'max_daily_trades': 12,
            'volatility_filter': 0.008,
            'volume_multiplier': 1.5,
            'cooldown_minutes': 30,
        }
        
        self.daily_trades_count = 0
        self.last_trade_time = None
        self.daily_pnl = 0.0
        
    def get_klines(self, limit: int = 100) -> pd.DataFrame:
        """Obtiene datos de velas con indicadores técnicos"""
        try:
            klines_data = self.client.get_klines(
                symbol=self.symbol,
                interval=self.config['timeframe'],
                limit=limit
            )
            
            if not klines_data:
                return pd.DataFrame()
            
            # Convertir a DataFrame
            df = pd.DataFrame(klines_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            # Convertir tipos de datos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Calcular indicadores técnicos
            df = self._calculate_indicators(df)
            return df
            
        except Exception as e:
            logging.error(f"Error obteniendo klines: {e}")
            return pd.DataFrame()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos mejorados"""
        if df.empty:
            return df
            
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Medias móviles
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # Volumen
        df['volume_sma'] = ta.trend.SMAIndicator(df['volume'], window=20).sma_indicator()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatilidad
        df['price_range'] = (df['high'] - df['low']) / df['low']
        
        return df
    
    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """Analiza condiciones del mercado para filtrar señales"""
        if len(df) < 20:
            return {'valid': False, 'reason': 'Datos insuficientes'}
        
        current = df.iloc[-1]
        
        # Filtro de volatilidad
        avg_volatility = df['price_range'].tail(10).mean()
        if avg_volatility < self.config['volatility_filter']:
            return {'valid': False, 'reason': 'Volatilidad insuficiente'}
        
        # Filtro de volumen
        if current['volume_ratio'] < self.config['volume_multiplier']:
            return {'valid': False, 'reason': 'Volumen insuficiente'}
        
        # Filtro RSI
        if current['rsi'] < 25 or current['rsi'] > 75:
            return {'valid': False, 'reason': 'RSI en zona extrema'}
        
        return {
            'valid': True,
            'volatility': avg_volatility,
            'volume_ratio': current['volume_ratio'],
            'rsi': current['rsi']
        }
    
    def generate_signal(self, df: pd.DataFrame) -> Dict:
        """Genera señal de trading mejorada"""
        if len(df) < 3:
            return {'action': 'hold', 'confidence': 0}
        
        market_analysis = self.analyze_market_conditions(df)
        if not market_analysis['valid']:
            return {'action': 'hold', 'confidence': 0, 'reason': market_analysis['reason']}
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Estrategia Momentum-Breakout
        bullish_conditions = 0
        bearish_conditions = 0
        
        # Condiciones alcistas
        if current['close'] > current['ema_9'] > current['ema_21']:
            bullish_conditions += 1
        if current['macd'] > current['macd_signal']:
            bullish_conditions += 1
        if 50 < current['rsi'] < 65:
            bullish_conditions += 1
        if current['close'] > current['bb_upper'] * 0.98:
            bullish_conditions += 1
        
        # Condiciones bajistas
        if current['close'] < current['ema_9'] < current['ema_21']:
            bearish_conditions += 1
        if current['macd'] < current['macd_signal']:
            bearish_conditions += 1
        if 35 < current['rsi'] < 50:
            bearish_conditions += 1
        if current['close'] < current['bb_lower'] * 1.02:
            bearish_conditions += 1
        
        # Determinar señal
        confidence = max(bullish_conditions, bearish_conditions) / 4.0
        
        if bullish_conditions >= 3 and confidence > 0.6:
            return {
                'action': 'buy',
                'confidence': confidence,
                'price': current['close'],
                'conditions_met': bullish_conditions
            }
        elif bearish_conditions >= 3 and confidence > 0.6:
            return {
                'action': 'sell',
                'confidence': confidence,
                'price': current['close'],
                'conditions_met': bearish_conditions
            }
        else:
            return {
                'action': 'hold',
                'confidence': confidence,
                'reason': f'Condiciones insuficientes'
            }
    
    def can_trade(self) -> bool:
        """Verifica si se puede realizar trading"""
        if self.daily_trades_count >= self.config['max_daily_trades']:
            return False
        
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds() / 60
            if time_since_last < self.config['cooldown_minutes'] and self.daily_pnl < 0:
                return False
        
        return True
    
    def calculate_position_size(self, price: float) -> float:
        """Calcula tamaño de posición"""
        risk_amount = self.current_balance * self.config['risk_per_trade']
        position_value = risk_amount / self.config['stop_loss']
        quantity = position_value / price
        
        # Máximo 50% del capital
        max_position_value = self.current_balance * 0.5
        if position_value > max_position_value:
            position_value = max_position_value
            quantity = position_value / price
        
        return quantity
    
    def execute_trade(self, signal: Dict):
        """Ejecuta operación de trading (SIMULACIÓN)"""
        if not self.can_trade():
            return
        
        try:
            action = signal['action'].upper()
            price = signal['price']
            quantity = self.calculate_position_size(price)
            
            # Simular orden (para testing)
            trade_value = quantity * price
            fee = trade_value * 0.001  # 0.1% fee
            
            if action == 'BUY':
                self.current_balance -= trade_value + fee
            else:  # SELL
                self.current_balance += trade_value - fee
            
            # Registrar trade
            trade = {
                'timestamp': datetime.now(),
                'action': action,
                'price': price,
                'quantity': quantity,
                'value': trade_value,
                'confidence': signal['confidence']
            }
            
            self.trade_history.append(trade)
            self.daily_trades_count += 1
            self.last_trade_time = datetime.now()
            
            logging.info(f"✅ TRADE SIMULADO: {action} {quantity:.6f} {self.symbol} @ ${price:.2f}")
            
        except Exception as e:
            logging.error(f"Error ejecutando orden: {e}")
    
    def run_strategy(self):
        """Ejecuta la estrategia principal"""
        logging.info("🤖 INICIANDO BOT MEJORADO - Estrategia Momentum-Breakout")
        logging.info(f"🔗 Exchange: MEXC | Símbolo: {self.symbol}")
        
        iteration = 0
        while True:
            try:
                iteration += 1
                
                # Obtener datos de mercado
                df = self.get_klines(50)
                if df.empty:
                    logging.warning("No se pudieron obtener datos de mercado")
                    time.sleep(60)
                    continue
                
                # Generar señal
                signal = self.generate_signal(df)
                
                # Log de análisis cada 5 iteraciones
                if iteration % 5 == 0:
                    current_price = df.iloc[-1]['close']
                    current_rsi = df.iloc[-1]['rsi']
                    logging.info(f"📊 Precio: ${current_price:.2f} | RSI: {current_rsi:.1f}")
                
                if signal['action'] != 'hold':
                    logging.info(f"🎯 SEÑAL: {signal['action'].upper()} | Confianza: {signal['confidence']:.2f}")
                    
                    # Ejecutar trade si hay señal válida
                    if signal['confidence'] > 0.65:
                        self.execute_trade(signal)
                
                # Mostrar resumen cada 10 iteraciones
                if iteration % 10 == 0:
                    self.show_performance_summary()
                
                # Esperar entre iteraciones
                time.sleep(60)  # 1 minuto para testing
                
            except KeyboardInterrupt:
                logging.info("Bot detenido por usuario")
                break
            except Exception as e:
                logging.error(f"Error en ejecución: {e}")
                time.sleep(30)
    
    def show_performance_summary(self):
        """Muestra resumen de performance"""
        total_trades = len(self.trade_history)
        if total_trades == 0:
            return
        
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = self.current_balance - self.initial_balance
        roi = (total_pnl / self.initial_balance) * 100
        
        logging.info("📈 RESUMEN DE PERFORMANCE")
        logging.info(f"   Balance: ${self.current_balance:.2f}")
        logging.info(f"   ROI: {roi:.2f}%")
        logging.info(f"   Total Trades: {total_trades}")
        logging.info(f"   Tasa Acierto: {win_rate:.1f}%")
        logging.info(f"   Trades Hoy: {self.daily_trades_count}")

def main():
    # Configuración - usa variables de entorno en producción
    API_KEY = os.getenv('MEXC_API_KEY', 'TU_API_KEY_MEXC')
    API_SECRET = os.getenv('MEXC_API_SECRET', 'TU_API_SECRET_MEXC')
    
    # Inicializar bot
    bot = EnhancedTradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol='BTCUSDT'
    )
    
    # Ejecutar estrategia
    bot.run_strategy()

if __name__ == "__main__":
    main()
