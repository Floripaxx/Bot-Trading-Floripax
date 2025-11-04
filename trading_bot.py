import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from mexc_api import MEXCAPI

class TradingBot:
    def __init__(self):
        # Parámetros por defecto
        self.ema_corta = 9
        self.ema_larga = 21
        self.rsi_periodo = 14
        self.rsi_sobrecompra = 65
        self.rsi_sobreventa = 35
        self.volumen_minimo = 1.1
        self.capital = 250.0
        self.capital_actual = 250.0
        self.trading_real = False
        
        # Pares de trading
        self.pares = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        self.pares_mostrar = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "DOT/USDT", "LINK/USDT"]
        self.pair_index = 0
        self.last_rotation = time.time()
        self.rotation_interval = 300  # 5 minutos
        
        # API
        self.api = MEXCAPI()
        
        # Estado
        self.senales_compra = 0
        self.senales_venta = 0
        self.ordenes_activas = 0
        self.historial = []
    
    def update_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def calcular_ema(self, datos, periodo):
        return datos.ewm(span=periodo, adjust=False).mean()
    
    def calcular_rsi(self, precios, periodo=14):
        delta = precios.diff()
        ganancia = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
        perdida = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
        rs = ganancia / perdida
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def rotar_par(self):
        current_time = time.time()
        if current_time - self.last_rotation >= self.rotation_interval:
            self.pair_index = (self.pair_index + 1) % len(self.pares)
            self.last_rotation = current_time
            return True
        return False
    
    def obtener_par_actual(self):
        return self.pares[self.pair_index]
    
    def obtener_proximo_par(self):
        next_index = (self.pair_index + 1) % len(self.pares)
        return self.pares_mostrar[next_index]
    
    def obtener_tiempo_restante(self):
        current_time = time.time()
        tiempo_transcurrido = current_time - self.last_rotation
        tiempo_restante = self.rotation_interval - tiempo_transcurrido
        minutos = int(tiempo_restante // 60)
        segundos = int(tiempo_restante % 60)
        return f"{minutos:02d}:{segundos:02d}"
    
    def analizar_par(self, par):
        try:
            # Obtener datos del mercado
            datos = self.api.obtener_datos_mercado(par)
            if datos is None or len(datos) < max(self.ema_larga, self.rsi_periodo):
                return None
            
            df = pd.DataFrame(datos)
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Calcular indicadores
            df['ema_corta'] = self.calcular_ema(df['close'], self.ema_corta)
            df['ema_larga'] = self.calcular_ema(df['close'], self.ema_larga)
            df['rsi'] = self.calcular_rsi(df['close'], self.rsi_periodo)
            
            # Calcular volumen promedio
            volumen_promedio = df['volume'].tail(20).mean()
            volumen_actual = df['volume'].iloc[-1]
            volumen_ratio = volumen_actual / volumen_promedio if volumen_promedio > 0 else 1
            
            # Últimos valores
            precio_actual = df['close'].iloc[-1]
            ema_corta_actual = df['ema_corta'].iloc[-1]
            ema_larga_actual = df['ema_larga'].iloc[-1]
            rsi_actual = df['rsi'].iloc[-1]
            
            # Determinar señal
            senal = None
            estado = "Sin señal clara"
            
            if (ema_corta_actual > ema_larga_actual and 
                rsi_actual < self.rsi_sobreventa and 
                volumen_ratio > self.volumen_minimo):
                senal = "COMPRA"
                estado = "🔴 SEÑAL COMPRA"
                self.senales_compra += 1
                
            elif (ema_corta_actual < ema_larga_actual and 
                  rsi_actual > self.rsi_sobrecompra and 
                  volumen_ratio > self.volumen_minimo):
                senal = "VENTA"
                estado = "🟢 SEÑAL VENTA"
                self.senales_venta += 1
            
            # Preparar datos para gráfico
            datos_grafico = {
                'timestamp': df.index.tolist(),
                'close': df['close'].tolist(),
                'ema_corta': df['ema_corta'].tolist(),
                'ema_larga': df['ema_larga'].tolist()
            }
            
            return {
                'par': self.pares_mostrar[self.pair_index],
                'precio_actual': precio_actual,
                'ema_corta': ema_corta_actual,
                'ema_larga': ema_larga_actual,
                'rsi': rsi_actual,
                'volumen_ratio': volumen_ratio,
                'senal': senal,
                'estado': estado,
                'datos_grafico': datos_grafico
            }
            
        except Exception as e:
            print(f"Error analizando {par}: {e}")
            return None
    
    def analizar_mercado(self):
        # Rotar par si es necesario
        self.rotar_par()
        
        resultados = []
        par_actual = self.obtener_par_actual()
        resultado = self.analizar_par(par_actual)
        
        if resultado:
            resultados.append(resultado)
        
        return resultados
    
    def ejecutar_orden(self, par, senal):
        try:
            # Simular ejecución (en paper trading)
            if not self.trading_real:
                orden = {
                    'par': par,
                    'tipo': senal,
                    'precio': 'Simulado',
                    'cantidad': self.capital_actual * 0.1,  # 10% del capital
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'estado': 'EJECUTADA'
                }
                self.historial.append(orden)
                self.ordenes_activas += 1
                return True
            else:
                # Aquí iría la lógica para trading real con API
                pass
                
        except Exception as e:
            print(f"Error ejecutando orden: {e}")
            return False
    
    def obtener_estado(self):
        return {
            'capital_actual': self.capital_actual,
            'senales_compra': self.senales_compra,
            'senales_venta': self.senales_venta,
            'ordenes_activas': self.ordenes_activas,
            'par_actual': self.pares_mostrar[self.pair_index],
            'proximo_par': self.obtener_proximo_par(),
            'tiempo_restante': self.obtener_tiempo_restante()
        }
    
    def obtener_historial(self):
        return pd.DataFrame(self.historial) if self.historial else None
    
    def reiniciar_capital(self):
        self.capital_actual = self.capital
        self.senales_compra = 0
        self.senales_venta = 0
        self.ordenes_activas = 0
        self.historial = []
