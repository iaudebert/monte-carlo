import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np  # On utilise NumPy (CPU) au lieu de CuPy
from datetime import timedelta
from arch import arch_model
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Terminal Quant Monte-Carlo", layout="wide")

def render_terminal_streamlit(ticker, n_sims, forecast_days, zoom_months, calc_months, ma_period, interval, show_ma):
    try:
        # 1. Acquisition Données
        period_map = {"1d": "5y", "1h": "730d", "15m": "60d", "5m": "60d"}
        df = yf.download(ticker, period=period_map.get(interval, "5y"), interval=interval, progress=False, auto_adjust=True)
        
        if df.empty:
            st.error(f"Aucune donnée trouvée pour {ticker}")
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

        # 3. Calculs Statistiques (CPU avec NumPy)
        returns = np.log(prices_calc[1:] / prices_calc[:-1])
        returns = returns[~np.isnan(returns)]
        
        # Drift et Volatilité
        rendement_geo = (prices_calc[-1] / prices_calc[-min(252, len(prices_calc))]) ** (365/min(252, len(prices_calc))) - 1
        mu_annual = np.clip(0.7 * rendement_geo + 0.3 * (np.mean(returns) * 365), -0.4, 0.6)
        
        try:
            garch = arch_model(returns * 100, vol='Garch', p=1, q=1)
            garch_fit = garch.fit(disp='off', show_warning=False)
            garch_forecast = garch_fit.forecast(horizon=forecast_days)
            sigma_annual = float(np.mean(np.sqrt(garch_forecast.variance.iloc[-1].values) / 100 * np.sqrt(365)))
        except:
            sigma_annual = float(np.std(returns) * np.sqrt(365))

        # 4. Simulation Monte-Carlo (NumPy)
        mu_daily = mu_annual / 365
        sigma_daily = sigma_annual / np.sqrt(365)
        
        # Génération des chocs (NumPy remplace CuPy ici)
        shocks = np.random.normal(mu_daily, sigma_daily, (forecast_days, n_sims)).astype(np.float32)
        price_paths = last_price * np.exp(np.cumsum(shocks, axis=0))
        
        p_levels = [1, 5, 50, 95, 99]
        res = np.percentile(price_paths, p_levels, axis=1)
        p1, p5, p50, p95, p99 = res[0], res[1], res[2], res[3], res[4]
        prob_profit = float(np.mean(price_paths[-1, :] > last_price) * 100)

        # 5. Graphisme
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(2, 2, width_ratios=[3.5, 1.5], height_ratios=[3, 1], wspace=0.01, hspace=0.3)
        
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
        ax3 = fig.add_subplot(gs[1, :])

        delta_time = {"1d": timedelta(days=1), "1h": timedelta(hours=1), "15m": timedelta(minutes=15), "5m": timedelta(minutes=5)}
        zoom_start = last_date - (delta_time[interval] * (zoom_months * (30 if interval=="1d" else 24)))
        
        # Axe 1 : Zoom & Sim
        ax.plot(df[df.index >= zoom_start].index, df[df.index >= zoom_start]['Close'], color='white', alpha=0.1, lw=1)
        
        future_dates = [last_date + (delta_time[interval] * i) for i in range(1, forecast_days + 1)]
        future_dates_num = mdates.date2num(future_dates)
        
        # Heatmap (échantillonnée pour le CPU)
        ax.hist2d(np.repeat(future_dates_num, min(n_sims, 2000)), 
                  price_paths[:, :min(n_sims, 2000)].flatten(), 
                  bins=[forecast_days, 100], cmap='gray', alpha=0.5, cmin=1)

        ax.plot(future_dates, p99, '#00FF00', lw=1.2, label="P99")
        ax.plot(future_dates, p50, 'orange', lw=2, label="Médiane")
        ax.plot(future_dates, p1, '#FF0000', lw=1.2, label="P1")
        
        ax.set_ylim(p1[-1]*0.8, p99[-1]*1.2)
        ax.set_title(f"{ticker} | Prob. Profit: {prob_profit:.1f}% | Trend: {trend_change_pct:+.1f}%")

        # Axe 2 : Miroir
        n, bins, patches = ax2.hist(price_paths[-1, :], bins=100, orientation='horizontal', density=True, alpha=0.2)
        ax2.invert_xaxis()
        ax2.axhline(last_price, color='cyan', lw=2)
        ax2.axis('off')

        # Axe 3 : Global
        ax3.plot(df.index, df['Close'], color='gray', alpha=0.4)
        ax3.axvspan(zoom_start, last_date, color='cyan', alpha=0.1)
        
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur lors du rendu : {e}")

# --- INTERFACE STREAMLIT ---
st.sidebar.title("📊 Paramètres")
ticker = st.sidebar.selectbox("Actif", ['BTC-USD', 'ETH-USD', 'NVDA', 'AAPL', 'TSLA', '^GSPC', '^IXIC', 'GC=F'])
tf = st.sidebar.selectbox("Timeframe", ['1d', '1h', '15m', '5m'])
sims = st.sidebar.select_slider("Simulations (CPU)", options=[1000, 5000, 10000, 50000], value=10000)
proj = st.sidebar.slider("Projection (Unités)", 10, 365, 90)
hist_calc = st.sidebar.slider("Mémoire Calcul (Mois)", 1, 60, 12)
zoom = st.sidebar.slider("Zoom Visuel (Mois)", 1, 24, 6)
ma_show = st.sidebar.checkbox("Afficher MA", value=True)

if st.sidebar.button("Lancer la Simulation"):
    render_terminal_streamlit(ticker, sims, proj, zoom, hist_calc, 50, tf, ma_show)
else:
    st.info("Ajustez les paramètres et cliquez sur 'Lancer la Simulation' dans la barre latérale.")
