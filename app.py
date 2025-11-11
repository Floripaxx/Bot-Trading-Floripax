import pandas as pd
import numpy as np
import time
from binance.client import Client
from binance.enums import *
import logging
from datetime import datetime
import talib

# Configuración
API_KEY = 'tu_api_key'
API_SECRET = 'tu_api_secret'
SYMBOL = 'BTCUSDT'
QUANTITY = 0.0005
LEVERAGE = 3

# Inicializar cliente Binance
client = Client(API_KEY, API_SECRET)

class HighFrequencyBot:
    def __init__(self):
        self.client = client
        self.symbol = SYMBOL
        self.quantity = QUANTITY
        self.leverage = LEVERAGE
        self.position = None
        self.entry_price = 0
        self.performance = []
        
        # *** SOLO MODIFICACIÓN: NUEVOS PARÁMETROS ESTRATEGIA ***
        self.ema_fast = 8
        self.ema_slow = 21
        self.volume_multiplier = 2.0
        self.stop_loss_pct = 0.08
        self.take_profit_pct = 0.12
        # *** FIN MODIFICACIÓN ***
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_leverage(self):
        """Configurar leverage"""
        try:
            self.client.futures_change_leverage(
                symbol=self.symbol, 
                leverage=self.leverage
            )
            self.logger.info(f"Leverage configurado a {self.leverage}x")
        except Exception as e:
            self.logger.error(f"Error configurando leverage: {e}")

    def get_historical_data(self, interval='1m', limit=50):
        """Obtener datos históricos para análisis técnico"""
        try:
            klines = self.client.futures_klines(
                symbol=self.symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            return df
        except Exception as e:
            self.logger.error(f"Error obteniendo datos históricos: {e}")
            return None

    # *** SOLO MODIFICACIÓN: NUEVA FUNCIÓN DE INDICADORES ***
    def calculate_indicators(self, df):
        """Calcular indicadores para estrategia momentum"""
        try:
            # EMAs para momentum
            df['ema_fast'] = talib.EMA(df['close'], timeperiod=self.ema_fast)
            df['ema_slow'] = talib.EMA(df['close'], timeperiod=self.ema_slow)
            
            # Volumen promedio
            df['volume_avg'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_avg']
            
            # VWAP
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            return df
        except Exception as e:
            self.logger.error(f"Error calculando indicadores: {e}")
            return df

    # *** SOLO MODIFICACIÓN: NUEVAS SEÑALES DE TRADING ***
    def should_enter_long(self, df):
        """Señal de entrada LONG - Momentum con volumen"""
        if len(df) < 30:
            return False
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # EMA rápido cruza arriba de EMA lento
        ema_cross_up = (current['ema_fast'] > current['ema_slow'] and 
                       previous['ema_fast'] <= previous['ema_slow'])
        
        # Confirmación volumen 2x promedio
        volume_confirm = current['volume_ratio'] >= self.volume_multiplier
        
        # Precio sobre VWAP (tendencia alcista)
        price_above_vwap = current['close'] > current['vwap']
        
        return all([ema_cross_up, volume_confirm, price_above_vwap])

    def should_enter_short(self, df):
        """Señal de entrada SHORT - Momentum con volumen"""
        if len(df) < 30:
            return False
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # EMA rápido cruza abajo de EMA lento
        ema_cross_down = (current['ema_fast'] < current['ema_slow'] and 
                         previous['ema_fast'] >= previous['ema_slow'])
        
        # Confirmación volumen 2x promedio
        volume_confirm = current['volume_ratio'] >= self.volume_multiplier
        
        # Precio bajo VWAP (tendencia bajista)
        price_below_vwap = current['close'] < current['vwap']
        
        return all([ema_cross_down, volume_confirm, price_below_vwap])

    def get_current_price(self):
        """Obtener precio actual"""
        try:
            ticker = self.client.futures_symbol_ticker(symbol=self.symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.error(f"Error obteniendo precio: {e}")
            return None

    def enter_long(self):
        """Abrir posición long"""
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=self.quantity
            )
            
            self.position = 'long'
            self.entry_price = self.get_current_price()
            self.logger.info(f"LONG abierta - Precio: {self.entry_price}, Cantidad: {self.quantity}")
            return True
        except Exception as e:
            self.logger.error(f"Error abriendo LONG: {e}")
            return False

    def enter_short(self):
        """Abrir posición short"""
        try:
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=self.quantity
            )
            
            self.position = 'short'
            self.entry_price = self.get_current_price()
            self.logger.info(f"SHORT abierta - Precio: {self.entry_price}, Cantidad: {self.quantity}")
            return True
        except Exception as e:
            self.logger.error(f"Error abriendo SHORT: {e}")
            return False

    # *** SOLO MODIFICACIÓN: NUEVAS CONDICIONES DE SALIDA ***
    def should_exit_long(self, current_price):
        """Verificar condiciones de salida para LONG"""
        if not self.position == 'long':
            return False
            
        # Take profit 0.12%
        if current_price >= self.entry_price * (1 + self.take_profit_pct / 100):
            return 'take_profit'
        
        # Stop loss 0.08%
        if current_price <= self.entry_price * (1 - self.stop_loss_pct / 100):
            return 'stop_loss'
            
        return False

    def should_exit_short(self, current_price):
        """Verificar condiciones de salida para SHORT"""
        if not self.position == 'short':
            return False
            
        # Take profit 0.12%
        if current_price <= self.entry_price * (1 - self.take_profit_pct / 100):
            return 'take_profit'
        
        # Stop loss 0.08%
        if current_price >= self.entry_price * (1 + self.stop_loss_pct / 100):
            return 'stop_loss'
            
        return False

    def close_position(self, reason="manual"):
        """Cerrar posición actual"""
        try:
            if not self.position:
                return True
                
            side = SIDE_SELL if self.position == 'long' else SIDE_BUY
            
            order = self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=self.quantity
            )
            
            exit_price = self.get_current_price()
            pnl = ((exit_price - self.entry_price) / self.entry_price * 100 * self.leverage 
                  if self.position == 'long' else 
                  (self.entry_price - exit_price) / self.entry_price * 100 * self.leverage)
            
            self.performance.append({
                'entry': self.entry_price,
                'exit': exit_price,
                'side': self.position,
                'pnl': pnl,
                'reason': reason,
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"Posición {self.position} cerrada - Razón: {reason}, PnL: {pnl:.4f}%")
            
            self.position = None
            self.entry_price = 0
            return True
        except Exception as e:
            self.logger.error(f"Error cerrando posición: {e}")
            return False

    def check_exit_conditions(self):
        """Verificar condiciones de salida para posición actual"""
        current_price = self.get_current_price()
        if not current_price:
            return
            
        if self.position == 'long':
            exit_signal = self.should_exit_long(current_price)
        elif self.position == 'short':
            exit_signal = self.should_exit_short(current_price)
        else:
            return
            
        if exit_signal:
            self.close_position(reason=exit_signal)

    # *** SOLO MODIFICACIÓN: NUEVA LÓGICA PRINCIPAL ***
    def run_strategy(self):
        """Ejecutar la estrategia momentum mejorada"""
        try:
            # Obtener datos
            df = self.get_historical_data()
            if df is None or len(df) < 30:
                self.logger.warning("Datos insuficientes para análisis")
                return
                
            # Calcular indicadores
            df = self.calculate_indicators(df)
            
            # Verificar salidas primero
            self.check_exit_conditions()
            
            # Si no hay posición, verificar entradas
            if not self.position:
                if self.should_enter_long(df):
                    self.enter_long()
                elif self.should_enter_short(df):
                    self.enter_short()
                    
        except Exception as e:
            self.logger.error(f"Error en ejecución de estrategia: {e}")

    def get_performance_stats(self):
        """Obtener estadísticas de performance"""
        if not self.performance:
            return "No hay operaciones aún"
            
        pnls = [trade['pnl'] for trade in self.performance]
        wins = [pnl for pnl in pnls if pnl > 0]
        
        stats = {
            'total_trades': len(self.performance),
            'win_rate': len(wins) / len(pnls) * 100,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean([pnl for pnl in pnls if pnl <= 0]) if len(pnls) > len(wins) else 0,
            'total_pnl': sum(pnls)
        }
        
        return stats

def main():
    bot = HighFrequencyBot()
    bot.setup_leverage()
    
    print("🤖 Bot de Trading Iniciado")
    print("🎯 Estrategia: MOMENTUM CON VOLUMEN")
    print(f"⚡ EMA{bot.ema_fast}/{bot.ema_slow} | Vol {bot.volume_multiplier}x")
    print(f"📊 SL: {bot.stop_loss_pct}% | TP: {bot.take_profit_pct}%")
    print("=" * 50)
    
    while True:
        try:
            bot.run_strategy()
            
            # Mostrar stats cada 10 ciclos
            if len(bot.performance) > 0 and len(bot.performance) % 10 == 0:
                stats = bot.get_performance_stats()
                if isinstance(stats, dict):
                    print(f"\n📈 Estadísticas: {stats['total_trades']} trades")
                    print(f"   Win Rate: {stats['win_rate']:.1f}%")
                    print(f"   PnL Total: {stats['total_pnl']:.2f}%")
            
            time.sleep(10)  # Verificar cada 10 segundos
            
        except KeyboardInterrupt:
            print("\n🛑 Bot detenido manualmente")
            if bot.position:
                bot.close_position("shutdown")
            break
        except Exception as e:
            print(f"Error en loop principal: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
