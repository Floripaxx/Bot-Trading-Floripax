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
        self.testnet_url = "https://api.mexc.com"  # MEXC no tiene testnet público
        
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
    
    def create_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> dict:
        """Crea una orden en MEXC"""
        endpoint = '/api/v3/order'
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity
        }
        
        if price and order_type == 'LIMIT':
            params['price'] = price
            params['timeInForce'] = 'GTC'
        else:
            params['type'] = 'MARKET'
        
        return self._request('POST', endpoint, params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancela una orden"""
        endpoint = '/api/v3/order'
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        return self._request('DELETE', endpoint, params, signed=True)

class EnhancedTradingBot:
    def __init__(self, api_key: str, api_secret: str, symbol: str = 'BTCUSDT', testnet: bool = False):
        self.client = MexcAPI(api_key, api_secret)
        self.symbol = symbol
        self.initial_balance = 250.0
        self.current_balance = self.initial_balance
        self.positions = {}
        self.trade_history = []
        
        # Configuración mejorada para Spot (preparación Futures)
        self.config = {
            'timeframe': '15m',  # Aumentado de 5m para menos ruido
            'max_positions': 2,  # Reducido de 4 para mejor gestión
            'risk_per_trade': 0.02,  # 2% del capital por operación
            'target_profit': 0.004,  # 0.4% profit target
            'stop_loss': 0.002,  # 0.2% stop loss
            'max_daily_trades': 12,
            'volatility_filter': 0.008,  # Mínimo 0.8% de volatilidad
            'volume_multiplier': 1.5,  # Volumen 1.5x promedio
            'cooldown_minutes': 30,  # Cooldown después de pérdida
        }
        
        # Configuración preparación Futures
        self.futures_config = {
            'leverage': 3,
            'position_mode': 'hedge',
            'margin_mode': 'isolated',
            'profit_targets': [0.008, 0.015],  # 0.8% y 1.5% con leverage
            'futures_stop_loss': 0.004,
        }
        
        self.daily_trades_count = 0
        self.last_trade_time = None
        self.daily_pnl = 0.0
        self.last_kline_time = None
        
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
        df['macd_histogram'] = macd.macd_diff()
        
        # Medias móviles
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        
        # Volumen
        df['volume_sma'] = ta.trend.SMAIndicator(df['volume'], window=20).sma_indicator()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatilidad
        df['price_range'] = (df['high'] - df['low']) / df['low']
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        return df
    
    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """Analiza condiciones del mercado para filtrar señales"""
        if len(df) < 50:
            return {'valid': False, 'reason': 'Datos insuficientes'}
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Filtro de volatilidad
        avg_volatility = df['price_range'].tail(20).mean()
        if avg_volatility < self.config['volatility_filter']:
            return {'valid': False, 'reason': 'Volatilidad insuficiente'}
        
        # Filtro de volumen
        if current['volume_ratio'] < self.config['volume_multiplier']:
            return {'valid': False, 'reason': 'Volumen insuficiente'}
        
        # Filtro RSI (evitar sobrecompra/sobreventa extrema)
        if current['rsi'] < 25 or current['rsi'] > 75:
            return {'valid': False, 'reason': 'RSI en zona extrema'}
        
        # Tendencia (alineación EMAs)
        ema_trend = current['ema_9'] > current['ema_21'] > current['sma_50']
        
        return {
            'valid': True,
            'volatility': avg_volatility,
            'volume_ratio': current['volume_ratio'],
            'rsi_zone': 'neutral' if 40 <= current['rsi'] <= 60 else 'extreme',
            'trend_aligned': ema_trend,
            'price_action': 'strong' if current['price_range'] > avg_volatility else 'weak'
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
        
        # Estrategia Momentum-Breakout Mejorada
        bullish_conditions = 0
        bearish_conditions = 0
        
        # Condiciones alcistas
        if current['close'] > current['ema_9'] > current['ema_21']:
            bullish_conditions += 1
        if current['macd'] > current['macd_signal'] and previous['macd'] <= previous['macd_signal']:
            bullish_conditions += 1
        if current['rsi'] > 50 and current['rsi'] < 65:
            bullish_conditions += 1
        if current['close'] > current['bb_upper'] * 0.98:  # Breakout suave
            bullish_conditions += 1
        
        # Condiciones bajistas
        if current['close'] < current['ema_9'] < current['ema_21']:
            bearish_conditions += 1
        if current['macd'] < current['macd_signal'] and previous['macd'] >= previous['macd_signal']:
            bearish_conditions += 1
        if current['rsi'] < 50 and current['rsi'] > 35:
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
                'reason': f'Condiciones insuficientes (Bull:{bullish_conditions}, Bear:{bearish_conditions})'
            }
    
    def can_trade(self) -> bool:
        """Verifica si se puede realizar trading"""
        # Límite diario de trades
        if self.daily_trades_count >= self.config['max_daily_trades']:
            logging.info("Límite diario de trades alcanzado")
            return False
        
        # Cooldown después de pérdida
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds() / 60
            if time_since_last < self.config['cooldown_minutes'] and self.daily_pnl < 0:
                logging.info(f"En cooldown después de pérdida. Tiempo restante: {self.config['cooldown_minutes'] - time_since_last:.1f} min")
                return False
        
        # Verificar balance
        if self.current_balance < self.initial_balance * 0.1:
            logging.warning("Balance insuficiente para trading")
            return False
        
        return True
    
    def calculate_position_size(self, price: float) -> float:
        """Calcula tamaño de posición basado en gestión de riesgo"""
        risk_amount = self.current_balance * self.config['risk_per_trade']
        position_value = risk_amount / self.config['stop_loss']
        quantity = position_value / price
        
        # Ajustar para no exceder el balance disponible
        max_position_value = self.current_balance * 0.7  # Máximo 70% del capital
        if position_value > max_position_value:
            position_value = max_position_value
            quantity = position_value / price
        
        return quantity
    
    def execute_trade(self, signal: Dict):
        """Ejecuta operación de trading"""
        if not self.can_trade():
            return
        
        try:
            action = signal['action'].upper()
            price = signal['price']
            quantity = self.calculate_position_size(price)
            
            # Verificar cantidad mínima para MEXC
            if quantity * price < 10:  # MEXC mínimo aproximado
                logging.warning("Posición muy pequeña, skip")
                return
            
            # Ejecutar orden en MEXC
            order = self.client.create_order(
                symbol=self.symbol,
                side=action,
                order_type='MARKET',
                quantity=quantity
            )
            
            if order and 'orderId' in order:
                # Registrar trade
                trade = {
                    'timestamp': datetime.now(),
                    'action': action,
                    'price': price,
                    'quantity': quantity,
                    'value': quantity * price,
                    'confidence': signal['confidence'],
                    'order_id': order['orderId']
                }
                
                self.trade_history.append(trade)
                self.daily_trades_count += 1
                self.last_trade_time = datetime.now()
                
                logging.info(f"✅ TRADE EJECUTADO: {action} {quantity:.6f} {self.symbol} @ ${price:.2f}")
            else:
                logging.error("Error: No se pudo crear la orden")
            
        except Exception as e:
            logging.error(f"Error ejecutando orden: {e}")
    
    def monitor_and_close_positions(self, df: pd.DataFrame):
        """Monitorea y cierra posiciones según estrategia"""
        if not self.positions:
            return
        
        current_price = df.iloc[-1]['close']
        
        for pos_id, position in list(self.positions.items()):
            entry_price = position['entry_price']
            current_pnl = (current_price - entry_price) / entry_price if position['side'] == 'BUY' else (entry_price - current_price) / entry_price
            
            # Verificar take profit
            if current_pnl >= self.config['target_profit']:
                self.close_position(pos_id, current_price, 'TP')
            
            # Verificar stop loss
            elif current_pnl <= -self.config['stop_loss']:
                self.close_position(pos_id, current_price, 'SL')
    
    def close_position(self, position_id: str, price: float, reason: str):
        """Cierra posición"""
        try:
            position = self.positions[position_id]
            
            # Ejecutar orden de cierre en MEXC
            close_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
            order = self.client.create_order(
                symbol=self.symbol,
                side=close_side,
                order_type='MARKET',
                quantity=position['quantity']
            )
            
            if order and 'orderId' in order:
                # Calcular P&L
                pnl = (price - position['entry_price']) * position['quantity'] if position['side'] == 'BUY' else (position['entry_price'] - price) * position['quantity']
                self.current_balance += pnl
                self.daily_pnl += pnl
                
                logging.info(f"🔒 POSICIÓN CERRADA: {position_id} @ ${price:.2f} | P&L: ${pnl:.2f} | Razón: {reason}")
                
                # Eliminar posición
                del self.positions[position_id]
            else:
                logging.error("Error: No se pudo cerrar la posición")
            
        except Exception as e:
            logging.error(f"Error cerrando posición: {e}")
    
    def run_strategy(self):
        """Ejecuta la estrategia principal"""
        logging.info("🤖 INICIANDO BOT MEJORADO - Estrategia Momentum-Breakout")
        logging.info(f"⚙️  Configuración: {json.dumps(self.config, indent=2)}")
        logging.info(f"🔗 Exchange: MEXC | Símbolo: {self.symbol}")
        
        while True:
            try:
                # Obtener datos de mercado
                df = self.get_klines(100)
                if df.empty:
                    logging.warning("No se pudieron obtener datos de mercado")
                    time.sleep(60)
                    continue
                
                # Generar señal
                signal = self.generate_signal(df)
                
                # Log de análisis
                market_analysis = self.analyze_market_conditions(df)
                current_price = df.iloc[-1]['close']
                
                logging.info(f"📊 MARKET: Price=${current_price:.2f} | "
                           f"Volatility={market_analysis.get('volatility', 0):.3%} | "
                           f"Volume={market_analysis.get('volume_ratio', 0):.1f}x | "
                           f"RSI={df.iloc[-1]['rsi']:.1f}")
                
                if signal['action'] != 'hold':
                    logging.info(f"🎯 SEÑAL: {signal['action'].upper()} | Confianza: {signal['confidence']:.2f} | {signal.get('reason', '')}")
                    
                    # Ejecutar trade si hay señal válida
                    if signal['confidence'] > 0.65:
                        self.execute_trade(signal)
                
                # Monitorear posiciones abiertas
                self.monitor_and_close_positions(df)
                
                # Mostrar resumen cada 10 ciclos
                if len(self.trade_history) % 10 == 0:
                    self.show_performance_summary()
                
                # Esperar entre iteraciones (15 minutos para timeframe 15m)
                time.sleep(900)
                
            except KeyboardInterrupt:
                logging.info("Bot detenido por usuario")
                break
            except Exception as e:
                logging.error(f"Error en ejecución: {e}")
                time.sleep(60)
    
    def show_performance_summary(self):
        """Muestra resumen de performance"""
        if not self.trade_history:
            return
        
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = self.current_balance - self.initial_balance
        roi = (total_pnl / self.initial_balance) * 100
        
        logging.info("📈 RESUMEN DE PERFORMANCE")
        logging.info(f"   Balance: ${self.current_balance:.2f}")
        logging.info(f"   ROI: {roi:.2f}%")
        logging.info(f"   Total Trades: {total_trades}")
        logging.info(f"   Tasa Acierto: {win_rate:.1f}%")
        logging.info(f"   Trades Hoy: {self.daily_trades_count}")
        logging.info(f"   P&L Diario: ${self.daily_pnl:.2f}")
    
    def prepare_for_futures(self):
        """Prepara la estrategia para migración a Futures"""
        logging.info("🔄 PREPARANDO MIGRACIÓN A FUTUROS")
        
        # Ajustar parámetros para Futures
        self.config.update({
            'target_profit': self.futures_config['profit_targets'][0],
            'stop_loss': self.futures_config['futures_stop_loss'],
            'risk_per_trade': 0.015,  # 1.5% para leverage
        })
        
        logging.info(f"🎯 Configuración Futures Ready: {json.dumps(self.config, indent=2)}")
        logging.info(f"⚡ Configuración Futures: {json.dumps(self.futures_config, indent=2)}")

def main():
    # Configuración (reemplaza con tus claves API de MEXC)
    API_KEY = "TU_API_KEY_MEXC"
    API_SECRET = "TU_API_SECRET_MEXC"
    
    # Inicializar bot
    bot = EnhancedTradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol='BTCUSDT',
        testnet=False  # MEXC no tiene testnet público
    )
    
    # Preparar para Futures
    bot.prepare_for_futures()
    
    # Ejecutar estrategia
    bot.run_strategy()

if __name__ == "__main__":
    main()
