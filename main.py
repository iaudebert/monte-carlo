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

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Terminal Quant Monte-Carlo", layout="wide")

# --- CACHE DES DONNÉES ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, interval, period):
    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    return data

def render_terminal_streamlit(ticker, n_sims, forecast_days, zoom_months, calc_months, ma_period, interval, show_ma, mode):
    # On s'assure de nettoyer toute figure résiduelle
    plt.clf()
    plt.close('all')

    try:
        # 1. Acquisition Données
        period_map = {"1d": "5y", "1h": "730d", "15m": "60d", "5m": "60d"}
        target_period = period_map.get(interval, "5y")
        df = load_data(ticker, interval, target_period)
        
        if df.empty:
            st.error(f"Données indisponibles pour {ticker}")
            return
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        last_date = df.index[-1]
        calc_start = last_date - timedelta(days=calc_months * 30)
        df_calc = df[df.index >= calc_start]
        prices_calc = df_calc['Close'].dropna().values.astype(np.float32)
        last_price = float(prices_calc[-1])

        # 2. Régression Linéaire
        X_reg = np.arange(len(prices_calc)).reshape(-1, 1)
        y_reg = prices_calc.reshape(-1, 1)
        reg = LinearRegression().fit(X_reg, y_reg)
        trend_forecast = reg.predict([[len(prices_calc) + forecast_days]])[0][0]
        trend_change_pct = ((trend_forecast / last_price) - 1) * 100

        if show_ma:
            df['MA'] = df['Close'].rolling(window=ma_period).mean()

        do_sim = (mode == "FULL")
        p1, p99 = None, None
        
        # 3. Moteur Statistique
        if do_sim and len(prices_calc) > 50:
            returns = np.log(prices_calc[1:] / prices_calc[:-1])
            returns = returns[~np.isnan(returns)]

            n_days_mu = min(252, len(prices_calc))
            rend_geo = (prices_calc[-1] / prices_calc[-n_days_mu]) ** (365/n_days_mu) - 1
            mu_annual = np.clip(0.7 * rend_geo + 0.3 * (np.mean(returns) * 365), -0.4, 0.6)

            try:
                garch = arch_model(returns * 100, vol='Garch', p=1, q=1)
                res = garch.fit(disp='off', show_warning=False)
                sigma_annual = float(np.mean(np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.params['beta[1]'] * res.conditional_volatility**2)) / 100 * np.sqrt(365))
            except:
                sigma_annual = float(np.std(returns) * np.sqrt(365))

            shocks = np.random.normal(mu_annual/365, sigma_annual/np.sqrt(365), (forecast_days, n_sims))
            price_paths = last_price * np.exp(np.cumsum(shocks, axis=0))
            
            p_res = np.percentile(price_paths, [1, 5, 50, 95, 99], axis=1)
            p1, p5, p50, p95, p99 = p_res[0], p_res[1], p_res[2], p_res[3], p_res[4]
            prob_profit = float(np.mean(price_paths[-1, :] > last_price) * 100)

        # --- GRAPHISME ---
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[3.5, 1.5], height_ratios=[3, 1], wspace=0.01, hspace=0.3)

        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
        ax3 = fig.add_subplot(gs[1, :])

        delta = {"1d": timedelta(days=1), "1h": timedelta(hours=1), "15m": timedelta(minutes=15), "5m": timedelta(minutes=5)}
        z_start = last_date - (delta[interval] * (zoom_months * (30 if interval=="1d" else 24)))

        # Tracé Historique
        ax.plot(df[df.index >= z_start].index, df[df.index >= z_start]['Close'], color='white', alpha=0.15, lw=1)
        
        if do_sim and p1 is not None:
            fut_dates = [last_date + (delta[interval] * i) for i in range(1, forecast_days + 1)]
            ax.hist2d(np.repeat(mdates.date2num(fut_dates), min(n_sims, 3000)), price_paths[:, :min(n_sims, 3000)].flatten(), bins=[forecast_days, 100], cmap='gray', alpha=0.4, cmin=1)
            ax.plot(fut_dates, p99, '#00FF00', lw=1.2)
            ax.plot(fut_dates, p50, 'orange', lw=2)
            ax.plot(fut_dates, p1, '#FF0000', lw=1.2)
            ax.set_ylim(p1[-1]*0.8, p99[-1]*1.2)
            
        # Titre & Cosmétique
        ax.set_title(f"TERMINAL QUANT | {ticker} | Trend: {trend_change_pct:+.1f}%", loc='left', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m' if interval != "1d" else '%y'))

        # Histogramme Miroir
        if do_sim and p1 is not None:
            ax2.hist(price_paths[-1, :], bins=100, orientation='horizontal', density=True, color='white', alpha=0.1)
            ax2.axhline(last_price, color='cyan', lw=2)
            ax2.set_xlim(ax2.get_xlim()[1], 0)
            ax2.invert_xaxis()
        ax2.axis('off')

        # Vue Globale
        ax3.plot(df.index, df['Close'], color='gray', alpha=0.4, lw=1)
        ax3.axvspan(z_start, last_date, color='cyan', alpha=0.1)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Affichage sans retour
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    except Exception as e:
        st.error(f"Erreur technique : {e}")

# --- INTERFACE SIDEBAR ---
st.sidebar.title("📈 Paramètres")
t_list = [('Bitcoin', 'BTC-USD'), ('Ethereum', 'ETH-USD'), ('NVIDIA', 'NVDA'), ('S&P 500', '^GSPC'), ('Gold', 'GC=F')]
s_ticker = st.sidebar.selectbox("Actif", [t[1] for t in t_list])
s_tf = st.sidebar.selectbox("Timeframe", ['1d', '1h', '15m'], index=0)
s_sims = st.sidebar.selectbox("Simulations", [1000, 10000, 50000], index=1)
s_days = st.sidebar.selectbox("Projection", [30, 90, 180, 365], index=1)
s_calc = st.sidebar.slider("Calcul (Mois)", 1, 60, 12)
s_zoom = st.sidebar.slider("Zoom (Mois)", 1, 24, 6)
s_ma_show = st.sidebar.checkbox("Afficher MA", value=True)

# Déclencheur
if st.sidebar.button("Simuler"):
    render_terminal_streamlit(s_ticker, s_sims, s_days, s_zoom, s_calc, 50, s_tf, s_ma_show, mode="FULL")
elif st.sidebar.button("Afficher"):
    render_terminal_streamlit(s_ticker, s_sims, s_days, s_zoom, s_calc, 50, s_tf, s_ma_show, mode="CHECK")
else:
    st.info("Sélectionnez un actif et cliquez sur Simuler.")
