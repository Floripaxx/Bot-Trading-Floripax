import time
import logging
import threading
from collections import deque
from datetime import datetime
import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd
import numpy as np
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
        
        # Estado del trading
        self.position = 0
        self.entry_price = 0
        self.balance = 1000
        self.positions_history = []
        self.is_running = False
        self.trading_thread = None
        
        # Configuración HFT
        self.tick_window = 50
        self.tick_data = deque(maxlen=self.tick_window)
        self.position_size = 0.05
        self.max_positions = 3
        self.open_positions = 0
        
        # Estrategias
        self.momentum_threshold = 0.003
        self.mean_reversion_threshold = 0.002
        
        # Logging
        logging.basicConfig(level=logging.INFO)
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
                response = requests.get(url, headers=headers, params=params, timeout=5)
            else:
                params['timestamp'] = int(time.time() * 1000)
                params['signature'] = self.generate_signature(params)
                response = requests.post(url, headers=headers, data=json.dumps(params), timeout=5)
            
            return response.json()
        except Exception as e:
            self.logger.error(f"Error en petición MEXC: {e}")
            return {}

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
            else:
                # Datos de simulación si la API falla
                return {
                    'timestamp': datetime.now(),
                    'bid': 50000 + np.random.uniform(-100, 100),
                    'ask': 50001 + np.random.uniform(-100, 100),
                    'symbol': self.symbol
                }
        except Exception as e:
            self.logger.error(f"Error obteniendo ticker: {e}")
            # Retornar datos simulados
            return {
                'timestamp': datetime.now(),
                'bid': 50000 + np.random.uniform(-100, 100),
                'ask': 50001 + np.random.uniform(-100, 100),
                'symbol': self.symbol
            }

    def calculate_indicators(self) -> Dict:
        """Calcular indicadores técnicos"""
        if len(self.tick_data) < 10:
            return {}
        
        prices = [tick['bid'] for tick in self.tick_data]
        df = pd.DataFrame(prices, columns=['price'])
        
        # Indicadores básicos
        df['returns'] = df['price'].pct_change()
        df['momentum'] = df['returns'].rolling(5).mean()
        df['sma_5'] = df['price'].rolling(5).mean()
        df['price_deviation'] = (df['price'] - df['sma_5']) / df['sma_5']
        
        latest = df.iloc[-1]
        
        return {
            'momentum': latest['momentum'],
            'price_deviation': latest['price_deviation'],
            'current_price': latest['price'],
            'sma_5': latest['sma_5']
        }

    def trading_strategy(self, indicators: Dict) -> str:
        """Estrategia de trading simplificada"""
        if not indicators:
            return 'hold'
        
        momentum = indicators['momentum']
        deviation = indicators['price_deviation']
        
        if momentum > self.momentum_threshold and deviation < -0.001:
            return 'buy'
        elif momentum < -self.momentum_threshold and deviation > 0.001:
            return 'sell'
        
        return 'hold'

    def execute_trade(self, action: str, price: float):
        """Ejecutar operación (simulada)"""
        try:
            quantity = (self.balance * self.position_size) / price
            
            if action == 'buy':
                self.position += quantity
                self.entry_price = price
                self.open_positions += 1
                trade_info = f"COMPRA: {quantity:.6f} {self.symbol} @ {price:.2f}"
                
            elif action == 'sell' and self.position >= quantity:
                profit = (price - self.entry_price) * quantity
                self.position -= quantity
                self.balance += profit
                self.open_positions = max(0, self.open_positions - 1)
                trade_info = f"VENTA: {quantity:.6f} {self.symbol} @ {price:.2f} | Profit: {profit:.4f} USDT"
            else:
                return
            
            self.logger.info(trade_info)
            
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

    def trading_cycle(self, gui_update_callback=None):
        """Ciclo principal de trading"""
        self.logger.info("Iniciando ciclo de trading HFT")
        
        while self.is_running:
            try:
                # Obtener datos de mercado
                tick_data = self.get_ticker_price()
                if tick_data:
                    self.tick_data.append(tick_data)
                
                # Calcular indicadores y ejecutar estrategia
                indicators = self.calculate_indicators()
                
                if indicators:
                    signal = self.trading_strategy(indicators)
                    if signal != 'hold':
                        price = tick_data['bid'] if signal == 'buy' else tick_data['ask']
                        self.execute_trade(signal, price)
                
                # Actualizar GUI si hay callback
                if gui_update_callback:
                    gui_update_callback()
                
                time.sleep(1)  # Intervalo de 1 segundo
                
            except Exception as e:
                self.logger.error(f"Error en ciclo de trading: {e}")
                time.sleep(2)

    def start_trading(self, gui_update_callback=None):
        """Iniciar bot de trading"""
        if not self.is_running:
            self.is_running = True
            self.trading_thread = threading.Thread(
                target=self.trading_cycle, 
                args=(gui_update_callback,),
                daemon=True
            )
            self.trading_thread.start()
            self.logger.info("Bot de trading iniciado")

    def stop_trading(self):
        """Detener bot de trading"""
        self.is_running = False
        self.logger.info("Bot de trading detenido")

    def get_performance_stats(self):
        """Obtener estadísticas de performance"""
        if not self.positions_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'current_balance': self.balance,
                'open_positions': self.open_positions,
                'current_price': self.tick_data[-1]['bid'] if self.tick_data else 0
            }
        
        df = pd.DataFrame(self.positions_history)
        sell_trades = df[df['action'] == 'sell']
        
        win_rate = 0
        if len(sell_trades) > 0:
            profitable = 0
            for i in range(1, len(self.positions_history)):
                if (self.positions_history[i]['action'] == 'sell' and 
                    self.positions_history[i]['price'] > self.positions_history[i-1]['price']):
                    profitable += 1
            win_rate = (profitable / len(sell_trades)) * 100
        
        current_price = self.tick_data[-1]['bid'] if self.tick_data else 0
        
        return {
            'total_trades': len(self.positions_history),
            'win_rate': win_rate,
            'current_balance': self.balance,
            'open_positions': self.open_positions,
            'current_price': current_price,
            'position_size': self.position
        }


class TradingBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Bot de Trading HFT - MEXC")
        self.root.geometry("1000x700")
        
        # Inicializar bot (con claves vacías para demo)
        self.bot = MexcHighFrequencyTradingBot("", "", "BTCUSDT")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="🤖 BOT DE TRADING DE ALTA FRECUENCIA - MEXC", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Sección de configuración
        config_frame = ttk.LabelFrame(main_frame, text="Configuración", padding="10")
        config_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # API Key
        ttk.Label(config_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W)
        self.api_key_entry = ttk.Entry(config_frame, width=40, show="*")
        self.api_key_entry.grid(row=0, column=1, padx=(5, 0))
        
        # Secret Key
        ttk.Label(config_frame, text="Secret Key:").grid(row=1, column=0, sticky=tk.W)
        self.secret_key_entry = ttk.Entry(config_frame, width=40, show="*")
        self.secret_key_entry.grid(row=1, column=1, padx=(5, 0))
        
        # Símbolo
        ttk.Label(config_frame, text="Símbolo:").grid(row=2, column=0, sticky=tk.W)
        self.symbol_entry = ttk.Entry(config_frame, width=20)
        self.symbol_entry.insert(0, "BTCUSDT")
        self.symbol_entry.grid(row=2, column=1, sticky=tk.W, padx=(5, 0))
        
        # Botones de control
        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))
        
        self.start_button = ttk.Button(button_frame, text="▶ INICIAR BOT", 
                                      command=self.start_bot, style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="⏹ DETENER BOT", 
                                     command=self.stop_bot, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        
        # Frame para información en tiempo real
        info_frame = ttk.LabelFrame(main_frame, text="Información en Tiempo Real", padding="10")
        info_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Precio actual
        self.price_label = ttk.Label(info_frame, text="Precio Actual: --", font=('Arial', 12, 'bold'))
        self.price_label.grid(row=0, column=0, sticky=tk.W)
        
        # Balance
        self.balance_label = ttk.Label(info_frame, text="Balance: --", font=('Arial', 12))
        self.balance_label.grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        
        # Posiciones abiertas
        self.positions_label = ttk.Label(info_frame, text="Posiciones Abiertas: --", font=('Arial', 12))
        self.positions_label.grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        
        # Tasa de acierto
        self.winrate_label = ttk.Label(info_frame, text="Tasa de Acierto: --", font=('Arial', 12))
        self.winrate_label.grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        
        # Total de operaciones
        self.trades_label = ttk.Label(info_frame, text="Total Operaciones: --", font=('Arial', 12))
        self.trades_label.grid(row=1, column=1, sticky=tk.W, padx=(20, 0), pady=(10, 0))
        
        # Frame para historial de operaciones
        trades_frame = ttk.LabelFrame(main_frame, text="Historial de Operaciones", padding="10")
        trades_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        trades_frame.columnconfigure(0, weight=1)
        trades_frame.rowconfigure(0, weight=1)
        
        # Treeview para operaciones
        columns = ('timestamp', 'action', 'price', 'quantity', 'balance')
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show='headings', height=8)
        
        # Definir columnas
        self.trades_tree.heading('timestamp', text='Fecha/Hora')
        self.trades_tree.heading('action', text='Acción')
        self.trades_tree.heading('price', text='Precio')
        self.trades_tree.heading('quantity', text='Cantidad')
        self.trades_tree.heading('balance', text='Balance')
        
        # Ajustar anchos de columna
        self.trades_tree.column('timestamp', width=150)
        self.trades_tree.column('action', width=80)
        self.trades_tree.column('price', width=100)
        self.trades_tree.column('quantity', width=120)
        self.trades_tree.column('balance', width=100)
        
        # Scrollbar para treeview
        scrollbar = ttk.Scrollbar(trades_frame, orient=tk.VERTICAL, command=self.trades_tree.yview)
        self.trades_tree.configure(yscrollcommand=scrollbar.set)
        
        self.trades_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Frame para logs
        log_frame = ttk.LabelFrame(main_frame, text="Logs del Sistema", padding="10")
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Área de texto para logs
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=100)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar logging para mostrar en GUI
        self.setup_gui_logging()
        
        # Configurar pesos para expansión
        main_frame.rowconfigure(3, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
    def setup_gui_logging(self):
        """Configurar logging para mostrar en la GUI"""
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.insert(tk.END, msg + '\n')
                self.text_widget.see(tk.END)
        
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.bot.logger.addHandler(text_handler)
        
    def start_bot(self):
        """Iniciar el bot de trading"""
        try:
            # Obtener configuración de la GUI
            api_key = self.api_key_entry.get()
            secret_key = self.secret_key_entry.get()
            symbol = self.symbol_entry.get()
            
            # Actualizar bot con nueva configuración
            self.bot.api_key = api_key
            self.bot.secret_key = secret_key
            self.bot.symbol = symbol
            
            # Iniciar bot
            self.bot.start_trading(gui_update_callback=self.update_gui)
            
            # Actualizar estado de botones
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            self.bot.logger.info("Bot iniciado correctamente")
            
            # Iniciar actualización periódica de la GUI
            self.schedule_gui_update()
            
        except Exception as e:
            self.bot.logger.error(f"Error al iniciar bot: {e}")
            
    def stop_bot(self):
        """Detener el bot de trading"""
        self.bot.stop_trading()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.bot.logger.info("Bot detenido")
        
    def update_gui(self):
        """Actualizar la interfaz gráfica con información actual"""
        stats = self.bot.get_performance_stats()
        
        # Actualizar labels
        self.price_label.config(text=f"Precio Actual: ${stats['current_price']:.2f}")
        self.balance_label.config(text=f"Balance: ${stats['current_balance']:.2f}")
        self.positions_label.config(text=f"Posiciones Abiertas: {stats['open_positions']}")
        self.winrate_label.config(text=f"Tasa de Acierto: {stats['win_rate']:.1f}%")
        self.trades_label.config(text=f"Total Operaciones: {stats['total_trades']}")
        
        # Actualizar treeview de operaciones
        self.update_trades_tree()
        
    def update_trades_tree(self):
        """Actualizar el treeview con las operaciones recientes"""
        # Limpiar treeview
        for item in self.trades_tree.get_children():
            self.trades_tree.delete(item)
            
        # Agregar operaciones (mostrar las últimas 20)
        recent_trades = self.bot.positions_history[-20:]
        for trade in recent_trades:
            self.trades_tree.insert('', 0, 
                values=(
                    trade['timestamp'].strftime('%H:%M:%S'),
                    trade['action'].upper(),
                    f"${trade['price']:.2f}",
                    f"{trade['quantity']:.6f}",
                    f"${trade['balance']:.2f}"
                ))
                
    def schedule_gui_update(self):
        """Programar actualización periódica de la GUI"""
        if self.bot.is_running:
            self.update_gui()
            self.root.after(1000, self.schedule_gui_update)  # Actualizar cada segundo

def main():
    root = tk.Tk()
    app = TradingBotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
