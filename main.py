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

# --- CACHE DES DONNÉES (Anti Rate-Limit) ---
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
        
        if df.empty:
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
            trend_slope = reg.coef_[0][0]
            trend_pct = (trend_slope / last_price * len(prices_calc)) * 100
            trend_forecast = reg.predict([[len(prices_calc) + forecast_days]])[0][0]
            trend_change_pct = ((trend_forecast / last_price) - 1) * 100
        else:
            trend_slope, trend_pct, trend_forecast, trend_change_pct = 0, 0, last_price, 0

        if show_ma:
            df['MA'] = df['Close'].rolling(window=ma_period).mean()

        do_sim = (mode == "FULL")
        prob_profit = 0
        p1, p5, p50, p95, p99 = None, None, None, None, None
        model_info = ""
        price_paths = None

        if do_sim and len(prices_calc) > 50:
            returns = np.log(prices_calc[1:] / prices_calc[:-1])
            returns = returns[~np.isnan(returns)]

            if len(returns) > 30:
                # Drift réaliste
                n_days_mu = min(252, len(prices_calc))
                rendement_geo = (prices_calc[-1] / prices_calc[-n_days_mu]) ** (365/n_days_mu) - 1
                rendement_moyen = np.mean(returns) * 365
                mu_annual = np.clip(0.7 * rendement_geo + 0.3 * rendement_moyen, -0.4, 0.6)

                try:
                    garch = arch_model(returns * 100, vol='Garch', p=1, q=1)
                    garch_fit = garch.fit(disp='off', show_warning=False)
                    garch_forecast = garch_fit.forecast(horizon=forecast_days)
                    sigma_annual = float(np.mean(np.sqrt(garch_forecast.variance.iloc[-1].values) / 100 * np.sqrt(365)))
                except:
                    sigma_annual = float(np.std(returns) * np.sqrt(365))

                mu_daily = mu_annual / 365
                sigma_daily = sigma_annual / np.sqrt(365)

                # Simulation NumPy (Optimisé CPU)
                shocks = np.random.normal(mu_daily, sigma_daily, (forecast_days, n_sims))
                price_paths = last_price * np.exp(np.cumsum(shocks, axis=0))

                p_levels = [1, 5, 50, 95, 99]
                res = np.percentile(price_paths, p_levels, axis=1)
                p1, p5, p50, p95, p99 = res[0], res[1], res[2], res[3], res[4]
                prob_profit = float(np.mean(price_paths[-1, :] > last_price) * 100)
                median_change = ((p50[-1] / last_price) - 1) * 100
                model_info = f"Drift: {mu_annual:.0f}%/an | Vol: {sigma_annual:.0f}%/an | Médiane: {median_change:+.1f}%"

        # --- GRAPHISME ---
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[3.5, 1.5], height_ratios=[3, 1], wspace=0.01, hspace=0.3)

        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
        ax3 = fig.add_subplot(gs[1, :])

        delta_time = {"1d": timedelta(days=1), "1h": timedelta(hours=1), "15m": timedelta(minutes=15), "5m": timedelta(minutes=5)}
        zoom_start = last_date - (delta_time[interval] * (zoom_months * (30 if interval=="1d" else 24)))

        ax.plot(df[df.index >= zoom_start].index, df[df.index >= zoom_start]['Close'],
                color='white', alpha=0.1 if do_sim else 0.4, lw=1)

        if len(prices_calc) > 10:
            calc_dates = df_calc.index
            X_plot = np.arange(len(prices_calc)).reshape(-1, 1)
            trend_line = reg.predict(X_plot).flatten()
            ax.plot(calc_dates, trend_line, color='white', lw=2, alpha=0.8, label="Trend Linéaire")

            future_dates_reg = [last_date + (delta_time[interval] * i) for i in range(1, forecast_days + 1)]
            future_x_reg = np.arange(len(prices_calc), len(prices_calc) + forecast_days).reshape(-1, 1)
            future_trend = reg.predict(future_x_reg).flatten()
            ax.plot(future_dates_reg, future_trend, color='white', lw=2, ls='--', alpha=0.6)

        if show_ma and 'MA' in df.columns:
            ax.plot(df[df.index >= zoom_start].index, df['MA'].loc[df.index >= zoom_start],
                   color='cyan', lw=1.5, label=f"MA {ma_period}", alpha=0.8)

        if do_sim and p1 is not None:
            future_dates = [last_date + (delta_time[interval] * i) for i in range(1, forecast_days + 1)]
            future_dates_num = mdates.date2num(future_dates)
            sample_idx = min(n_sims, 3000)
            
            ax.hist2d(np.repeat(future_dates_num, sample_idx),
                     price_paths[:, :sample_idx].flatten(),
                     bins=[forecast_days, 100], cmap='gray', alpha=0.5, cmin=1)

            ax.plot(future_dates, p99, '#00FF00', lw=1.2, label="P99")
            ax.plot(future_dates, p95, '#A2FF00', lw=0.8, ls='--', label="P95")
            ax.plot(future_dates, p50, 'orange', lw=2, label="Médiane")
            ax.plot(future_dates, p5, '#FF5E00', lw=0.8, ls='--', label="P5")
            ax.plot(future_dates, p1, '#FF0000', lw=1.2, label="P1")

            ax.set_ylim(p1[-1]*0.8, p99[-1]*1.2)
            ax.set_xlim(mdates.date2num(zoom_start), future_dates_num[-1])

        title_str = f"TERMINAL QUANT REALISTE | {ticker}"
        if do_sim and p1 is not None:
            title_str += f"\nProbabilité de profit: {prob_profit:.0f}% | {model_info} | Trend: {trend_change_pct:+.1f}%"
        ax.set_title(title_str, loc='left', fontsize=11, pad=20)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m' if interval != "1d" else '%m/%y'))

        if do_sim and p1 is not None:
            n, bins, patches = ax2.hist(price_paths[-1, :], bins=120, orientation='horizontal', density=True, alpha=0.1)
            for i in range(len(patches)):
                bc = (bins[i] + bins[i+1]) / 2
                if bc > p99[-1]: patches[i].set_facecolor('#00FF00'); patches[i].set_alpha(0.7)
                elif bc > p95[-1]: patches[i].set_facecolor('#A2FF00'); patches[i].set_alpha(0.5)
                elif bc > p50[-1]: patches[i].set_facecolor('#FFAA00'); patches[i].set_alpha(0.3)
                elif bc > p5[-1]: patches[i].set_facecolor('#FF6600'); patches[i].set_alpha(0.5)
                else: patches[i].set_facecolor('#FF0000'); patches[i].set_alpha(0.7)

            ax2.set_xlim(ax2.get_xlim()[1], 0); ax2.invert_xaxis()
            ax2.axhline(last_price, color='cyan', lw=2.5, alpha=0.9)
            ax2.axhline(trend_forecast, color='white', lw=2, alpha=0.8, ls='--')
            ax2.text(0.95, last_price, f"{last_price:,.0f}", transform=ax2.get_yaxis_transform(), color='cyan', ha='right', fontweight='bold')

        ax2.set_xticks([]); ax2.set_yticks([]); [s.set_visible(False) for s in ax2.spines.values()]

        ax3.plot(df.index, df['Close'], color='gray', alpha=0.4, lw=1)
        if len(prices_calc) > 10:
            ax3.plot(calc_dates, trend_line, color='white', lw=2, alpha=0.8)
        ax3.axvspan(calc_start, last_date, color='yellow', alpha=0.08)
        ax3.axvspan(zoom_start, last_date, color='cyan', alpha=0.1)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Erreur: {e}")

# --- INTERFACE STREAMLIT SIDEBAR ---
st.sidebar.title("📈 Paramètres")
tickers_list = [('Bitcoin', 'BTC-USD'), ('Ethereum', 'ETH-USD'), ('Solana', 'SOL-USD'), ('NVIDIA', 'NVDA'), ('Apple', 'AAPL'), ('Tesla', 'TSLA'), ('S&P 500', '^GSPC'), ('Nasdaq 100', '^IXIC'), ('CAC 40', '^FCHI'), ('Gold', 'GC=F')]
s_ticker = st.sidebar.selectbox("Actif", [t[1] for t in sorted(tickers_list)])
s_tf = st.sidebar.selectbox("Timeframe", ['1d', '1h', '15m', '5m'], index=0)
s_sims = st.sidebar.selectbox("Nombre Simulations", [1000, 10000, 50000, 100000], index=1)
s_days = st.sidebar.selectbox("Projection (Périodes)", [10, 30, 90, 180, 365], index=2)
s_calc = st.sidebar.slider("Historique Calcul (Mois)", 1, 60, 12)
s_zoom = st.sidebar.slider("Zoom Visuel (Mois)", 1, 24, 6)
s_ma_val = st.sidebar.selectbox("Période MA", [10, 20, 50, 100, 200], index=2)
s_ma_show = st.sidebar.checkbox("Afficher MA", value=True)

col1, col2 = st.sidebar.columns(2)
btn_check = col1.button("Afficher")
btn_sim = col2.button("Simuler")

if btn_check:
    render_terminal_streamlit(s_ticker, s_sims, s_days, s_zoom, s_calc, s_ma_val, s_tf, s_ma_show, mode="CHECK")
elif btn_sim:
    render_terminal_streamlit(s_ticker, s_sims, s_days, s_zoom, s_calc, s_ma_val, s_tf, s_ma_show, mode="FULL")
else:
    st.info("Ajustez les paramètres et cliquez sur 'Afficher' ou 'Simuler'.")
