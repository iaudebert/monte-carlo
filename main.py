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
@st.cache_data(ttl=3600)
def load_data(ticker, interval, period):
    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    return data

def render_terminal_streamlit(ticker, n_sims, forecast_days, zoom_months, calc_months, ma_period, interval, show_ma, mode):
    plt.close('all')

    try:
        # 1. Acquisition Données
        period_map = {"1d": "5y", "1h": "730d", "15m": "60d", "5m": "60d"}
        target_period = period_map.get(interval, "5y")
        df = load_data(ticker, interval, target_period)
        
        if df is None or df.empty:
            st.error(f"Aucune donnée pour {ticker}")
            return
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        last_date = df.index[-1]
        calc_start = last_date - timedelta(days=calc_months * 30)
        df_calc = df[df.index >= calc_start]
        prices_calc = df_calc['Close'].dropna().values.astype(np.float32)
        last_price = float(prices_calc[-1])

        # RÉGRESSION LINÉAIRE
        if len(prices_calc) > 10:
            X_reg = np.arange(len(prices_calc)).reshape(-1, 1)
            y_reg = prices_calc.reshape(-1, 1)
            reg = LinearRegression().fit(X_reg, y_reg)
            trend_forecast = reg.predict([[len(prices_calc) + forecast_days]])[0][0]
            trend_change_pct = ((trend_forecast / last_price) - 1) * 100
        else:
            trend_forecast, trend_change_pct = last_price, 0

        if show_ma:
            df['MA'] = df['Close'].rolling(window=ma_period).mean()

        do_sim = (mode == "FULL")
        prob_profit, p1, p99 = 0, None, None
        model_info = ""

        if do_sim and len(prices_calc) > 50:
            returns = np.log(prices_calc[1:] / prices_calc[:-1])
            returns = returns[~np.isnan(returns)]

            if len(returns) > 30:
                n_days_mu = min(252, len(prices_calc))
                rendement_geo = (prices_calc[-1] / prices_calc[-n_days_mu]) ** (365/n_days_mu) - 1
                mu_annual = np.clip(0.7 * rendement_geo + 0.3 * (np.mean(returns) * 365), -0.4, 0.6)

                try:
                    garch = arch_model(returns * 100, vol='Garch', p=1, q=1)
                    garch_fit = garch.fit(disp='off', show_warning=False)
                    garch_forecast = garch_fit.forecast(horizon=forecast_days)
                    sigma_annual = float(np.mean(np.sqrt(garch_forecast.variance.iloc[-1].values) / 100 * np.sqrt(365)))
                except:
                    sigma_annual = float(np.std(returns) * np.sqrt(365))

                mu_daily, sigma_daily = mu_annual / 365, sigma_annual / np.sqrt(365)
                shocks = np.random.normal(mu_daily, sigma_daily, (forecast_days, n_sims))
                price_paths = last_price * np.exp(np.cumsum(shocks, axis=0))

                res = np.percentile(price_paths, [1, 5, 50, 95, 99], axis=1)
                p1, p5, p50, p95, p99 = res[0], res[1], res[2], res[3], res[4]
                prob_profit = float(np.mean(price_paths[-1, :] > last_price) * 100)
                model_info = f"Drift: {mu_annual:.0f}%/an | Vol: {sigma_annual:.0f}%/an"

        # --- GRAPHISME ---
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[3.5, 1.5], height_ratios=[3, 1], wspace=0.01, hspace=0.3)

        ax, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
        ax2.get_shared_y_axes().join(ax2, ax)

        delta_time = {"1d": timedelta(days=1), "1h": timedelta(hours=1), "15m": timedelta(minutes=15), "5m": timedelta(minutes=5)}
        zoom_start = last_date - (delta_time[interval] * (zoom_months * (30 if interval=="1d" else 24)))

        ax.plot(df[df.index >= zoom_start].index, df[df.index >= zoom_start]['Close'], color='white', alpha=0.15, lw=1)

        if do_sim and p1 is not None:
            future_dates = [last_date + (delta_time[interval] * i) for i in range(1, forecast_days + 1)]
            future_dates_num = mdates.date2num(future_dates)
            ax.hist2d(np.repeat(future_dates_num, min(n_sims, 2500)), price_paths[:, :min(n_sims, 2500)].flatten(), bins=[forecast_days, 100], cmap='gray', alpha=0.5, cmin=1)
            ax.plot(future_dates, p99, '#00FF00', lw=1.2)
            ax.plot(future_dates, p50, 'orange', lw=2)
            ax.plot(future_dates, p1, '#FF0000', lw=1.2)
            ax.set_ylim(p1[-1]*0.8, p99[-1]*1.2)

        ax.set_title(f"QUANT TERMINAL | {ticker} | Profit: {prob_profit:.0f}% | Trend: {trend_change_pct:+.1f}%", loc='left', fontsize=10)
        
        if do_sim and p1 is not None:
            n, bins, patches = ax2.hist(price_paths[-1, :], bins=120, orientation='horizontal', density=True, alpha=0.1)
            ax2.invert_xaxis()
            ax2.axhline(last_price, color='cyan', lw=2)
        ax2.axis('off')

        ax3.plot(df.index, df['Close'], color='gray', alpha=0.4, lw=1)
        ax3.axvspan(zoom_start, last_date, color='cyan', alpha=0.1)
        
        st.pyplot(fig)
        plt.close(fig)

    except Exception as e:
        st.error(f"Erreur: {e}")
    
    return None # Crucial pour éviter les sorties 'null'

# --- INTERFACE ---
st.sidebar.title("📊 Params")
s_ticker = st.sidebar.selectbox("Actif", ['BTC-USD', 'ETH-USD', 'NVDA', 'AAPL', '^GSPC'])
s_tf = st.sidebar.selectbox("TF", ['1d', '1h', '15m', '5m'])
s_sims = st.sidebar.selectbox("Sims", [1000, 10000, 50000])
s_days = st.sidebar.selectbox("Proj", [30, 90, 180, 365])
s_calc = st.sidebar.slider("Calcul (Mois)", 1, 60, 12)
s_zoom = st.sidebar.slider("Zoom (Mois)", 1, 24, 6)
s_ma_val = st.sidebar.selectbox("MA", [20, 50, 200])
s_ma_show = st.sidebar.checkbox("Show MA", value=True)

if st.sidebar.button("Afficher"):
    render_terminal_streamlit(s_ticker, s_sims, s_days, s_zoom, s_calc, s_ma_val, s_tf, s_ma_show, mode="CHECK")
elif st.sidebar.button("Simuler"):
    render_terminal_streamlit(s_ticker, s_sims, s_days, s_zoom, s_calc, s_ma_val, s_tf, s_ma_show, mode="FULL")
