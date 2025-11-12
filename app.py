def execute_trade(self, action: str, price: float):
    """Ejecutar operación en FUTUROS - MODO TURBO CORREGIDO"""
    try:
        # ========== CORRECCIÓN CRÍTICA: TAMAÑO MÁXIMO SEGURO ==========
        dynamic_position_size = self.calculate_dynamic_position_size()
        
        # LIMITAR el tamaño de posición a un MÁXIMO SEGURO
        safe_position_size = min(dynamic_position_size, 0.05)  # MÁXIMO 5% del capital
        
        investment_amount = 0
        quantity = 0
        quantity_to_close = 0
        close_amount = 0
        profit_loss = 0
        
        if action in ['buy', 'sell'] and self.open_positions < self.max_positions:
            # VERIFICACIÓN DE SEGURIDAD EXTRA
            max_safe_investment = self.cash_balance * 0.8  # Nunca usar más del 80% del cash
            
            investment_amount = self.cash_balance * safe_position_size * self.leverage
            
            # CORRECCIÓN: Verificar que no exceda el límite seguro
            if investment_amount > max_safe_investment:
                investment_amount = max_safe_investment
                self.log_message("⚠️ AJUSTE: Tamaño de posición reducido por seguridad", "WARNING")
            
            quantity = investment_amount / price
            
            # VERIFICACIÓN FINAL: ¿Tenemos suficiente margen?
            required_margin = investment_amount / self.leverage
            if required_margin > self.cash_balance:
                self.log_message(f"❌ MARGEN INSUFICIENTE: Se necesita ${required_margin:.2f}, disponible ${self.cash_balance:.2f}", "ERROR")
                return
            
            # Actualizar balances
            self.cash_balance -= (investment_amount / self.leverage)
            self.position += quantity
            self.entry_price = price if self.position == quantity else ((self.entry_price * (self.position - quantity)) + (price * quantity)) / self.position
            self.position_side = 'long' if action == 'buy' else 'short'
            self.open_positions += 1
            
            side_emoji = "🟢" if action == 'buy' else "🔴"
            trade_info = f"{side_emoji} TURBO {self.position_side.upper()} {self.symbol}: {quantity:.6f} @ ${price:.2f} | Size: {safe_position_size*100:.1f}% | Margen: ${investment_amount/self.leverage:.2f} | Leverage: {self.leverage}x"
            self.log_message(trade_info, "TRADE")
            
        elif action == 'close' and self.position > 0:
            # CERRAR POSICIÓN
            quantity_to_close = self.position
            
            if self.position_side == 'long':
                close_amount = quantity_to_close * price
                profit_loss = (close_amount - (self.position * self.entry_price)) * self.leverage
            else:  # short
                close_amount = quantity_to_close * price
                profit_loss = ((self.position * self.entry_price) - close_amount) * self.leverage
            
            # Actualizar balances
            self.cash_balance += (self.position * self.entry_price / self.leverage) + profit_loss
            self.position = 0
            self.open_positions = 0
            self.position_side = ''
            self.total_profit += profit_loss
            
            profit_color = "💰" if profit_loss > 0 else "💸"
            trade_info = f"{profit_color} CERRAR TURBO {self.symbol}: {quantity_to_close:.6f} @ ${price:.2f} | P/L: ${profit_loss:.4f} | Profit Total: ${self.total_profit:.2f}"
            self.log_message(trade_info, "TRADE")
        
        # Registrar posición
        current_equity = self.cash_balance + (self.position * self.entry_price / self.leverage if self.position > 0 else 0)
        self.positions_history.append({
            'timestamp': datetime.now(),
            'action': action,
            'side': self.position_side if action != 'close' else '',
            'leverage': self.leverage if action != 'close' else '',
            'price': price,
            'quantity': quantity_to_close if action == 'close' else quantity,
            'cash_balance': self.cash_balance,
            'total_equity': current_equity,
            'profit_loss': profit_loss if action == 'close' else 0
        })
        
        self._auto_save()
        
    except Exception as e:
        self.log_message(f"❌ ERROR ejecutando trade: {e}", "ERROR")
