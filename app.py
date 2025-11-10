import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import json
import os

# Configuración logging básica
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingBot:
    def __init__(self, symbol: str = 'BTCUSDT'):
        self.symbol = symbol
        self.initial_balance = 250.0
        self.current_balance = self.initial_balance
        self.trade_history = []
        
        # Configuración mejorada
        self.config = {
            'timeframe': '15m',
            'risk_per_trade': 0.02,
            'target_profit': 0.004,
            'stop_loss': 0.002,
            'max_daily_trades': 12,
        }
        
        self.daily_trades_count = 0
        self.last_trade_time = None
        
    def calculate_indicators(self, prices: list) -> dict:
        """Calcula indicadores técnicos simples"""
        if len(prices) < 20:
            return {}
            
        df = pd.DataFrame(prices, columns=['close'])
        
        # SMA simple
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI simple
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        return {
            'price': current['close'],
            'sma_20': current['sma_20'],
            'sma_50': current['sma_50'],
            'rsi': current['rsi'],
            'trend': 'bullish' if current['sma_20'] > current['sma_50'] else 'bearish'
        }
    
    def generate_signal(self, indicators: dict) -> dict:
        """Genera señal de trading"""
        if not indicators:
            return {'action': 'hold', 'confidence': 0}
        
        # Estrategia simple: tendencia + RSI
        bullish_conditions = 0
        bearish_conditions = 0
        
        # Condiciones alcistas
        if indicators['trend'] == 'bullish':
            bullish_conditions += 1
        if 30 < indicators['rsi'] < 70:
            bullish_conditions += 1
        if indicators['price'] > indicators['sma_20']:
            bullish_conditions += 1
            
        # Condiciones bajistas  
        if indicators['trend'] == 'bearish':
            bearish_conditions += 1
        if 30 < indicators['rsi'] < 70:
            bearish_conditions += 1
        if indicators['price'] < indicators['sma_20']:
            bearish_conditions += 1
        
        confidence = max(bullish_conditions, bearish_conditions) / 3.0
        
        if bullish_conditions >= 2 and confidence > 0.6:
            return {
                'action': 'buy',
                'confidence': confidence,
                'price': indicators['price']
            }
        elif bearish_conditions >= 2 and confidence > 0.6:
            return {
                'action': 'sell', 
                'confidence': confidence,
                'price': indicators['price']
            }
        else:
            return {'action': 'hold', 'confidence': confidence}
    
    def can_trade(self) -> bool:
        """Verifica si se puede realizar trading"""
        if self.daily_trades_count >= self.config['max_daily_trades']:
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
            quantity = (max_position_value / price)
        
        return quantity
    
    def execute_trade(self, signal: dict):
        """Ejecuta operación de trading (SIMULACIÓN)"""
        if not self.can_trade():
            return
        
        try:
            action = signal['action'].upper()
            price = signal['price']
            quantity = self.calculate_position_size(price)
            
            # Simular orden
            trade_value = quantity * price
            fee = trade_value * 0.001  # 0.1% fee
            
            if action == 'BUY':
                self.current_balance -= trade_value + fee
                log_msg = f"COMPRA simulada"
            else:  # SELL
                self.current_balance += trade_value - fee
                log_msg = f"VENTA simulada"
            
            # Registrar trade
            trade = {
                'timestamp': datetime.now(),
                'action': action,
                'price': price,
                'quantity': quantity,
                'value': trade_value
            }
            
            self.trade_history.append(trade)
            self.daily_trades_count += 1
            self.last_trade_time = datetime.now()
            
            logging.info(f"✅ {log_msg}: {quantity:.6f} {self.symbol} @ ${price:.2f}")
            
        except Exception as e:
            logging.error(f"Error ejecutando orden: {e}")
    
    def simulate_price_data(self) -> list:
        """Simula datos de precio para testing"""
        base_price = 50000
        prices = []
        for i in range(100):
            # Simular precio con algo de volatilidad
            change = np.random.normal(0, 0.02)  # 2% de volatilidad
            price = base_price * (1 + change)
            prices.append(price)
            base_price = price
        return prices
    
    def run_strategy(self):
        """Ejecuta la estrategia principal"""
        logging.info("🤖 INICIANDO BOT DE TRADING - Modo Simulación")
        logging.info(f"💰 Balance inicial: ${self.initial_balance:.2f}")
        
        # Simular datos de precio
        price_data = self.simulate_price_data()
        
        iteration = 0
        while True:
            try:
                iteration += 1
                
                # Usar datos simulados (en producción aquí conectarías con MEXC)
                if iteration >= len(price_data):
                    iteration = 20  # Reiniciar
                
                current_prices = price_data[:iteration + 20]
                indicators = self.calculate_indicators(current_prices)
                
                if indicators:
                    # Log cada 10 iteraciones
                    if iteration % 10 == 0:
                        logging.info(f"📊 Precio: ${indicators['price']:.2f} | RSI: {indicators['rsi']:.1f} | Tendencia: {indicators['trend']}")
                    
                    # Generar y ejecutar señal
                    signal = self.generate_signal(indicators)
                    
                    if signal['action'] != 'hold':
                        logging.info(f"🎯 Señal: {signal['action'].upper()} | Confianza: {signal['confidence']:.2f}")
                        
                        if signal['confidence'] > 0.6:
                            self.execute_trade(signal)
                
                # Mostrar resumen cada 20 iteraciones
                if iteration % 20 == 0:
                    self.show_performance_summary()
                
                # Esperar entre iteraciones
                time.sleep(5)  # 5 segundos para testing rápido
                
            except KeyboardInterrupt:
                logging.info("Bot detenido por usuario")
                break
            except Exception as e:
                logging.error(f"Error en ejecución: {e}")
                time.sleep(10)
    
    def show_performance_summary(self):
        """Muestra resumen de performance"""
        total_trades = len(self.trade_history)
        if total_trades == 0:
            logging.info("📈 Aún no hay trades ejecutados")
            return
        
        total_pnl = self.current_balance - self.initial_balance
        roi = (total_pnl / self.initial_balance) * 100
        
        logging.info("📈 RESUMEN DE PERFORMANCE")
        logging.info(f"   Balance actual: ${self.current_balance:.2f}")
        logging.info(f"   Ganancia/Pérdida: ${total_pnl:.2f}")
        logging.info(f"   ROI: {roi:.2f}%")
        logging.info(f"   Total trades: {total_trades}")
        logging.info(f"   Trades hoy: {self.daily_trades_count}")

def main():
    """Función principal"""
    logging.info("🚀 Iniciando aplicación de trading...")
    
    # Inicializar bot
    bot = TradingBot(symbol='BTCUSDT')
    
    # Ejecutar estrategia
    bot.run_strategy()

if __name__ == "__main__":
    main()
