import streamlit as st 
import time
import threading
from collections import deque
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go
import os

# Configurar la página de Streamlit
st.set_page_config(
    page_title="🤖 Bot HFT Futuros MEXC - ESTRATEGIA ULTRA AGRESIVA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PersistentStateManager:
    """Gestor de estado INDEPENDIENTE de Streamlit - VERSIÓN SIMPLIFICADA"""
    
    def __init__(self, state_file='bot_persistent_state.json'):
        self.state_file = state_file
