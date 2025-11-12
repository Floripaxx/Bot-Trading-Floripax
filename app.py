import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class ImprovedTradingBot:
    def __init__(self, initial_capital=255.0):
        # CONFIGURACIÓN MEJORADA DE RIESGO
        self.leverage = 2  # REDUCIDO de 3x a 2x
        self.risk_per_trade = 0.01  # 1% riesgo por operación
        self.stop_loss_pct = 0.018  # 1.8% stop loss
        self.take_profit_pct = 0.025  # 2.5% take profit
        self.max_daily_trades = 15
        self.min_time_between_trades = 10  # segundos entre operaciones
        
        # Estado del bot
        self.cash_balance = initial_capital
        self.total_equity = initial_capital
        self.open_positions = []
        self.daily_trades_count = 0
        self.last_trade_time = None
        
        # Filtros de estrategia mejorados
        self.rsi_overbought = 72
        self.rsi_oversold = 28
        self.min_volume_factor = 1.3
        
        # Seguimiento de rendimiento
        self.winning_trades = 0
        self.total_trades = 0
        
    def calculate_position_size(self, current_price):
        """Calcula tamaño de posición basado en riesgo"""
        risk_amount = self.total_equity * self.risk_per_trade
        stop_loss_distance = current_price * self.stop_loss_pct
        position_size = risk_amount / stop_loss_distance
        
        # Limitar tamaño máximo (20% del capital)
        max_position = (self.total_equity * 0.2) / current_price
        return min(position_size, max_position)
    
    def should_enter_trade(self, current_data, market_conditions):
        """Filtros mejorados para entrada"""
        # Filtro 1: Tiempo entre operaciones
        if self.last_trade_time and (datetime.now() - self.last_trade_time).seconds < self.min_time_between_trades:
            return False, "Esperando entre operaciones"
            
        # Filtro 2: Límite diario de operaciones
        if self.daily_trades_count >= self.max_daily_trades:
            return False, "Límite diario alcanzado"
            
        # Filtro 3: Verificar señales técnicas mejoradas
        if not self.check_improved_conditions(current_data, market_conditions):
            return False, "Condiciones no óptimas"
            
        # Filtro 4: Verificar que hay capital suficiente
        position_size = self.calculate_position_size(current_data['price'])
        if position_size * current_data['price'] > self.cash_balance * 0.8:
            return False, "Capital insuficiente"
            
        return True, "Condiciones óptimas"
    
    def check_improved_conditions(self, current_data, market_conditions):
        """Condiciones de entrada mejoradas"""
        # Filtro RSI más estricto
        if 'rsi' in current_data:
            if current_data['rsi'] > self.rsi_overbought and market_conditions['trend'] == 'downtrend':
                return True  # Señal short en sobrecompra con tendencia bajista
            elif current_data['rsi'] < self.rsi_oversold and market_conditions['trend'] == 'uptrend':
                return True  # Señal long en sobreventa con tendencia alcista
        
        # Filtro de volumen
        if 'volume' in current_data and current_data['volume'] < market_conditions['avg_volume'] * self.min_volume_factor:
            return False
            
        # Evitar rangos laterales estrechos
        if market_conditions['volatility'] < 0.005:  # 0.5% de volatilidad mínima
            return False
            
        return True
    
    def execute_trade(self, action, side, price, quantity, leverage=None):
        """Ejecutar operación con gestión de riesgo mejorada"""
        if leverage is None:
            leverage = self.leverage
            
        # Verificar cantidad mínima
        if quantity * price < 10:  # Mínimo $10 por operación
            return False, "Cantidad muy pequeña"
            
        cost = quantity * price / leverage
        
        if cost > self.cash_balance:
            return False, "Fondos insuficientes"
            
        # Ejecutar la operación
        self.cash_balance -= cost
        trade = {
            'timestamp': datetime.now(),
            'action': action,
            'side': side,
            'leverage': leverage,
            'price': price,
            'quantity': quantity,
            'stop_loss': price * (1 - self.stop_loss_pct) if side == 'long' else price * (1 + self.stop_loss_pct),
            'take_profit': price * (1 + self.take_profit_pct) if side == 'long' else price * (1 - self.take_profit_pct)
        }
        
        self.open_positions.append(trade)
        self.last_trade_time = datetime.now()
        self.daily_trades_count += 1
        self.total_trades += 1
        
        return True, "Operación ejecutada"
    
    def check_exit_conditions(self, current_price):
        """Verificar condiciones de salida mejoradas"""
        positions_to_close = []
        
        for position in self.open_positions:
            # Verificar stop loss
            if (position['side'] == 'long' and current_price <= position['stop_loss']) or \
               (position['side'] == 'short' and current_price >= position['stop_loss']):
                positions_to_close.append((position, 'stop_loss'))
                
            # Verificar take profit  
            elif (position['side'] == 'long' and current_price >= position['take_profit']) or \
                 (position['side'] == 'short' and current_price <= position['take_profit']):
                positions_to_close.append((position, 'take_profit'))
                
        return positions_to_close
    
    def close_position(self, position, close_price, exit_reason):
        """Cerrar posición con cálculo de P&L"""
        # Calcular P&L
        if position['side'] == 'long':
            pnl = (close_price - position['price']) * position['quantity'] * position['leverage']
        else:
            pnl = (position['price'] - close_price) * position['quantity'] * position['leverage']
            
        # Actualizar balances
        initial_cost = position['quantity'] * position['price'] / position['leverage']
        self.cash_balance += initial_cost + pnl
        self.total_equity = self.cash_balance
        
        # Actualizar estadísticas
        if pnl > 0:
            self.winning_trades += 1
            
        # Remover posición
        self.open_positions.remove(position)
        
        return pnl
    
    def get_performance_metrics(self):
        """Métricas de rendimiento mejoradas"""
        if self.total_trades == 0:
            return {
                'win_rate': 0,
                'total_trades': 0,
                'current_equity': self.total_equity,
                'daily_trades': self.daily_trades_count
            }
            
        return {
            'win_rate': (self.winning_trades / self.total_trades) * 100,
            'total_trades': self.total_trades,
            'current_equity': self.total_equity,
            'daily_trades': self.daily_trades_count,
            'open_positions': len(self.open_positions)
        }
    
    def reset_daily_count(self):
        """Reiniciar contador diario"""
        self.daily_trades_count = 0

# FUNCIÓN DE EJECUCIÓN MEJORADA
def execute_improved_strategy():
    bot = ImprovedTradingBot(initial_capital=255.0)
    
    print("=== BOT MEJORADO INICIADO ===")
    print(f"Apalancamiento: {bot.leverage}x")
    print(f"Riesgo por operación: {bot.risk_per_trade * 100}%")
    print(f"Stop Loss: {bot.stop_loss_pct * 100}%")
    print(f"Take Profit: {bot.take_profit_pct * 100}%")
    print("=============================")
    
    # Simular ciclo de trading (adaptar a tu implementación actual)
    while True:
        try:
            # Obtener datos de mercado (implementar según tu API)
            market_data = get_market_data()
            conditions = analyze_market_conditions(market_data)
            
            # Verificar condiciones de entrada
            should_enter, reason = bot.should_enter_trade(market_data, conditions)
            
            if should_enter:
                # Calcular posición
                position_size = bot.calculate_position_size(market_data['price'])
                
                # Ejecutar operación
                success, message = bot.execute_trade(
                    action='sell' if conditions['signal'] == 'short' else 'buy',
                    side='short' if conditions['signal'] == 'short' else 'long',
                    price=market_data['price'],
                    quantity=position_size
                )
                
                if success:
                    print(f"✅ OPERACIÓN EJECUTADA: {message}")
                else:
                    print(f"❌ Operación rechazada: {message}")
            
            # Verificar condiciones de salida
            positions_to_close = bot.check_exit_conditions(market_data['price'])
            for position, reason in positions_to_close:
                pnl = bot.close_position(position, market_data['price'], reason)
                print(f"🔒 POSICIÓN CERRADA: {reason} | P&L: ${pnl:.4f}")
            
            # Mostrar métricas cada 10 operaciones
            if bot.total_trades % 10 == 0:
                metrics = bot.get_performance_metrics()
                print(f"📊 MÉTRICAS: Win Rate {metrics['win_rate']:.1f}% | Equity: ${metrics['current_equity']:.2f}")
            
            time.sleep(1)  # Control de frecuencia
            
        except Exception as e:
            print(f"Error en ejecución: {e}")
            time.sleep(5)

# FUNCIONES AUXILIARES (adaptar a tu implementación actual)
def get_market_data():
    """Obtener datos de mercado actualizados"""
    # Implementar según tu API/exchange
    return {
        'price': 3448.07,  # Precio actual
        'rsi': 65,         # RSI actual
        'volume': 1000,     # Volumen actual
        # ... otros indicadores
    }

def analyze_market_conditions(market_data):
    """Analizar condiciones del mercado"""
    # Implementar tu lógica de análisis
    return {
        'trend': 'downtrend',
        'volatility': 0.012,
        'avg_volume': 800,
        'signal': 'short'  # o 'long'
    }

# INICIAR BOT MEJORADO
if __name__ == "__main__":
    execute_improved_strategy()
