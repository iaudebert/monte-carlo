# @title TERMINAL
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import cupy as cp
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import timedelta
from arch import arch_model
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


def render_terminal_v3(ticker, n_sims, forecast_days, zoom_months, calc_months, ma_period, interval, show_ma, mode="FULL"):
    plt.close('all')

    try:
        period_map = {"1d": "5y", "1h": "730d", "15m": "60d", "5m": "60d"}
        df = yf.download(ticker, period=period_map.get(interval, "5y"), interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            print(f"Aucune donnée pour {ticker}")
            return
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        last_date = df.index[-1]
        calc_start = last_date - timedelta(days=calc_months * 30)
        df_calc = df[df.index >= calc_start]
        prices_calc = df_calc['Close'].dropna().values.astype(np.float32)
        last_price = float(prices_calc[-1])

        # RÉGRESSION LINÉAIRE SUR HISTORIQUE CALC
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
        price_paths_gpu = None

        if do_sim and len(prices_calc) > 50:
            returns = np.log(prices_calc[1:] / prices_calc[:-1])
            returns = returns[~np.isnan(returns)]

            if len(returns) > 30:
                def drift_realiste(returns, prices):
                    n_days = min(252, len(prices))
                    rendement_geo = (prices[-1] / prices[-n_days]) ** (365/n_days) - 1
                    rendement_moyen = np.mean(returns) * 365
                    mu_annualise = 0.7 * rendement_geo + 0.3 * rendement_moyen
                    return np.clip(mu_annualise, -0.4, 0.6)

                mu_annual = drift_realiste(returns, prices_calc)

                try:
                    garch = arch_model(returns * 100, vol='Garch', p=1, q=1)
                    garch_fit = garch.fit(disp='off', show_warning=False)
                    garch_forecast = garch_fit.forecast(horizon=forecast_days)
                    sigma_annual = float(np.mean(np.sqrt(garch_forecast.variance.iloc[-1].values) / 100 * np.sqrt(365)))
                except:
                    sigma_annual = float(np.std(returns) * np.sqrt(365))

                mu_daily = mu_annual / 365
                sigma_daily = sigma_annual / np.sqrt(365)

                mu_gpu = cp.float32(mu_daily)
                sigma_gpu = cp.float32(sigma_daily)

                shocks_gpu = cp.random.normal(mu_gpu, sigma_gpu, (forecast_days, n_sims), dtype=cp.float32)
                price_paths_gpu = last_price * cp.exp(cp.cumsum(shocks_gpu, axis=0))

                final_prices_gpu = price_paths_gpu[-1, :]
                p_levels = [1, 5, 50, 95, 99]
                res_gpu = cp.percentile(price_paths_gpu, p_levels, axis=1)
                p1, p5, p50, p95, p99 = [cp.asnumpy(res_gpu[i]) for i in range(len(p_levels))]
                prob_profit = float(cp.mean(final_prices_gpu > last_price) * 100)

                median_change = ((p50[-1] / last_price) - 1) * 100
                model_info = f"Drift: {mu_annual:.0f}%/an | Vol: {sigma_annual:.0f}%/an | Médiane: {median_change:+.1f}%"

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(22, 11))
        gs = fig.add_gridspec(2, 2, width_ratios=[3.5, 1.5], height_ratios=[3, 1], wspace=0.01, hspace=0.3)

        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
        ax3 = fig.add_subplot(gs[1, :])

        delta_time = {"1d": timedelta(days=1), "1h": timedelta(hours=1), "15m": timedelta(minutes=15), "5m": timedelta(minutes=5)}
        zoom_start = last_date - (delta_time[interval] * (zoom_months * (30 if interval=="1d" else 24)))

        ax.plot(df[df.index >= zoom_start].index, df[df.index >= zoom_start]['Close'],
                color='white', alpha=0.1 if do_sim else 0.4, lw=1)

        # AFFICHAGE RÉGRESSION LINÉAIRE SUR HISTORIQUE CALC
        if len(prices_calc) > 10:
            calc_dates = df_calc.index
            X_plot = np.arange(len(prices_calc)).reshape(-1, 1)
            trend_line = reg.predict(X_plot).flatten()
            ax.plot(calc_dates, trend_line, color='white', lw=2, alpha=0.8, label=f"Trend Linéaire")

            # Projection future de la régression
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

            sample_idx = min(n_sims, 5000)
            ax.hist2d(np.repeat(future_dates_num, sample_idx),
                     cp.asnumpy(price_paths_gpu[:, :sample_idx]).flatten(),
                     bins=[forecast_days, 100], cmap='gray', alpha=0.5, cmin=1)

            ax.plot(future_dates, p99, '#00FF00', lw=1.2, label="P99")
            ax.plot(future_dates, p95, '#A2FF00', lw=0.8, ls='--', label="P95")
            ax.plot(future_dates, p50, 'orange', lw=2, label="Médiane")
            ax.plot(future_dates, p5, '#FF5E00', lw=0.8, ls='--', label="P5")
            ax.plot(future_dates, p1, '#FF0000', lw=1.2, label="P1")

            ax.set_ylim(p1[-1]*0.85, p99[-1]*1.15)
            ax.set_xlim(mdates.date2num(zoom_start), future_dates_num[-1])

        title_str = f"TERMINAL QUANT REALISTE | {ticker}"
        if do_sim and p1 is not None:
            title_str += f"\nProbabilité de profit: {prob_profit:.0f}% | {model_info}"
            title_str += f" | Trend: {trend_change_pct:+.1f}%"
        else:
            title_str += f" | Trend Hist Calc: {trend_pct:+.1f}%"
        ax.set_title(title_str, loc='left', fontsize=11, pad=20)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m' if interval != "1d" else '%m/%y'))
        if do_sim: ax.legend(fontsize=7)

        if do_sim and p1 is not None:
            final_prices_cpu = cp.asnumpy(price_paths_gpu[-1, :])
            n, bins, patches = ax2.hist(final_prices_cpu, bins=120, orientation='horizontal', density=True, alpha=0.1)

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
            ax2.text(0.95, last_price, f"{last_price:,.0f}", transform=ax2.get_yaxis_transform(),
                    color='cyan', ha='right', fontweight='bold', fontsize=10)
            ax2.text(0.95, trend_forecast, f"Trend: {trend_forecast:,.0f}", transform=ax2.get_yaxis_transform(),
                    color='white', ha='right', fontweight='bold', fontsize=9)

        ax2.set_xticks([]); ax2.set_yticks([]); [s.set_visible(False) for s in ax2.spines.values()]

        ax3.plot(df.index, df['Close'], color='gray', alpha=0.4, lw=1)
        # Ligne de tendance sur l'historique complet
        if len(prices_calc) > 10:
            ax3.plot(calc_dates, trend_line, color='white', lw=2, alpha=0.8)
        ax3.axvspan(calc_start, last_date, color='yellow', alpha=0.08)
        ax3.axvspan(zoom_start, last_date, color='cyan', alpha=0.1)
        ax3.set_title("Historique + Trend Linéaire", fontsize=9, alpha=0.6)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erreur: {e}")


def trigger(mode):
    with out:
        clear_output(wait=True)
        render_terminal_v3(w_ticker.value, w_sims.value, w_days.value, w_zoom.value, w_calc.value, w_ma_val.value, w_tf.value, w_ma_show.value, mode=mode)


tickers_list = [
    ('Bitcoin', 'BTC-USD'),
    ('Ethereum', 'ETH-USD'),
    ('Solana', 'SOL-USD'),
    ('NVIDIA', 'NVDA'),
    ('Apple', 'AAPL'),
    ('Tesla', 'TSLA'),
    ('S&P 500', '^GSPC'),
    ('Nasdaq 100', '^IXIC'),
    ('CAC 40', '^FCHI'),
    ('Gold', 'GC=F')
]


w_ticker = widgets.Dropdown(options=sorted(tickers_list), value='BTC-USD', description='Actif:')
w_tf = widgets.Dropdown(options=[('Daily', '1d'), ('Hourly', '1h'), ('15 Min', '15m'), ('5 Min', '5m')], value='1d', description='TF:')
w_calc = widgets.Dropdown(options=[('1 mois', 1), ('3 mois', 3), ('6 mois', 6), ('1 an', 12), ('2 ans', 24), ('5 ans', 60)], value=12, description='Hist. Calc:')
w_zoom = widgets.Dropdown(options=[('1 mois', 1), ('3 mois', 3), ('6 mois', 6), ('1 an', 12), ('2 ans', 24)], value=6, description='Zoom:')
w_days = widgets.Dropdown(options=[('10j', 10), ('30j', 30), ('90j', 90), ('180j', 180), ('1 an', 365)], value=90, description='Proj:')
w_sims = widgets.Dropdown(options=[('1k', 1000), ('10k', 10000), ('100k', 100000), ('1M', 1000000)], value=100000, description='Sims:')
w_ma_show = widgets.Checkbox(value=False, description='MA')
w_ma_val = widgets.Dropdown(options=[('MA10', 10), ('MA20', 20), ('MA50', 50), ('MA100', 100), ('MA200', 200)], value=50, description='MA:')


btn_check = widgets.Button(description="Afficher", button_style='info', layout=widgets.Layout(width='49%'))
btn_sim = widgets.Button(description="Simuler", button_style='success', layout=widgets.Layout(width='49%'))
out = widgets.Output()


btn_check.on_click(lambda b: trigger("CHECK"))
btn_sim.on_click(lambda b: trigger("FULL"))


display(widgets.VBox([
    widgets.HBox([w_ticker, w_tf, w_sims]),
    widgets.HBox([w_calc, w_zoom, w_days]),
    widgets.HBox([w_ma_val, w_ma_show]),
    widgets.HBox([btn_check, btn_sim]),
    out
]))
