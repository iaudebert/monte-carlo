import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import timedelta
from arch import arch_model
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Terminal Quant", layout="wide")

# --- INTERFACE ---
st.sidebar.title("📈 Configuration")
ticker = st.sidebar.selectbox("Actif", ['BTC-USD', 'ETH-USD', 'NVDA', 'AAPL', '^GSPC', 'GC=F'])
tf = st.sidebar.selectbox("Timeframe", ['1d', '1h', '15m'])
sims = st.sidebar.select_slider("Simulations", options=[1000, 5000, 10000, 20000], value=5000)
proj = st.sidebar.slider("Projection", 10, 200, 60)
calc_m = st.sidebar.slider("Mémoire (Mois)", 1, 36, 12)

if st.sidebar.button("Lancer la Simulation"):
    try:
        # 1. Download avec gestion d'erreur
        period_map = {"1d": "5y", "1h": "730d", "15m": "60d"}
        df = yf.download(ticker, period=period_map[tf], interval=tf, progress=False)
        
        if df.empty:
            st.error("Yahoo Finance bloque la requête. Réessayez dans 1 minute.")
        else:
            # Nettoyage colonnes MultiIndex si besoin
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # --- CALCULS ---
            last_price = float(df['Close'].iloc[-1])
            returns = np.log(df['Close'] / df['Close'].shift(1)).dropna().values
            
            # Simulation Monte Carlo (NumPy)
            mu = np.mean(returns)
            sigma = np.std(returns)
            
            # Correction GARCH simplifiée pour éviter les crashs serveurs
            shocks = np.random.normal(mu, sigma, (proj, sims))
            price_paths = last_price * np.exp(np.cumsum(shocks, axis=0))
            
            p_levels = [1, 5, 50, 95, 99]
            res = np.percentile(price_paths, p_levels, axis=1)
            
            # --- RENDU ---
            plt.style.use('dark_background')
            fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Main Chart
            ax1.plot(price_paths[:, :500], color='white', alpha=0.05) # Quelques chemins
            ax1.plot(res[2], color='orange', lw=2, label="Médiane")
            ax1.plot(res[4], color='#00FF00', lw=1.5, ls='--', label="Top 1%")
            ax1.plot(res[0], color='#FF0000', lw=1.5, ls='--', label="Flop 1%")
            ax1.set_title(f"Simulation Monte Carlo : {ticker}")
            ax1.legend()
            
            # Global View
            ax3.plot(df['Close'].values, color='cyan', alpha=0.6)
            ax3.set_title("Historique Global")
            
            st.pyplot(fig)
            st.success(f"Prix actuel : {last_price:.2f} | Probabilité de profit : {np.mean(price_paths[-1] > last_price)*100:.1f}%")

    except Exception as e:
        st.error(f"Erreur technique : {e}")
else:
    st.write("Cliquez sur le bouton pour générer les scénarios.")
