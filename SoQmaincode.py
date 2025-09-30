import pandas as pd
import numpy as np
import pandas_ta as ta
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from backtester import BackTester
import warnings


def process_data(data):
    df = data.copy()
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['EMA_50'] = ta.ema(df['close'], length=50)
    df['MACD'] = ta.macd(df['close'])['MACD_12_26_9']
    df['MACD_signal'] = ta.macd(df['close'])['MACDs_12_26_9']
    df['RSI'] = ta.rsi(df['close'], length=14)

    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_upper'] = bb['BBU_20_2.0']
    df['BB_lower'] = bb['BBL_20_2.0']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['close']
    df['BB_pos'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['volume_sma'] = ta.sma(df['volume'], length=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['vol_forecast'] = df['log_ret'].rolling(100).apply(
        lambda x: arch_model(100 * x.dropna(), vol='Garch', p=1, q=1, rescale=False)
        .fit(disp="off").forecast(horizon=1).variance.values[-1, 0] / (100 ** 2)
        if len(x.dropna()) > 50 else np.nan
    )

    def arima_trend(series):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                smoothed = pd.Series(series).ewm(span=5).mean()
                model = ARIMA(smoothed, order=(1, 1, 1)).fit()
                forecast = model.forecast(steps=1)[0]
                return 1 if forecast > smoothed.iloc[-1] else -1
        except:
            return 0

    df['arima_trend'] = df['close'].rolling(100).apply(lambda x: arima_trend(x))

    return df.dropna()


def strat(df):
    df = df.copy()
    df['signals'] = 0
    df['trade_type'] = "HOLD"

    current_pos = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        signal = 0

        dynamic_rsi_low = 40 - (row['vol_forecast'] if not np.isnan(row['vol_forecast']) else 0) * 5
        dynamic_rsi_high = 60 + (row['vol_forecast'] if not np.isnan(row['vol_forecast']) else 0) * 5
        hyperparamlist = [0.935,0.935,1.055,1.065,1.065,0.945]
        if row['EMA_20'] > row['EMA_50'] * hyperparamlist[0] and row['MACD'] > row['MACD_signal']*hyperparamlist[1]:
            momentum = 1
        elif row['EMA_20'] < row['EMA_50'] * hyperparamlist[2] and row['MACD'] < row['MACD_signal']*hyperparamlist[3]:
            momentum = -1
        else:
            momentum = 0

        if row['RSI'] < dynamic_rsi_low and row['close'] < row['BB_lower'] * hyperparamlist[4]:
            meanrev = 1
        elif row['RSI'] > dynamic_rsi_high and row['close'] > row['BB_upper'] * hyperparamlist[5]:
            meanrev = -1
        else:
            meanrev = 0

        arima_dir = row['arima_trend']

        if (momentum == 1 or meanrev == 1) and arima_dir == 1:
            if current_pos == -1:
                signal = 2
                current_pos = 1
            elif current_pos == 0:
                signal = 1
                current_pos = 1
            else:
                signal = 0
        elif (momentum == -1 or meanrev == -1) and arima_dir == -1:
            if current_pos == 1:
                signal = -2
                current_pos = -1
            elif current_pos == 0:
                signal = -1
                current_pos = -1
            else:
                signal = 0
        else:
            signal = 0

        #Code to Block invalid transitions(for example 2 after 1 doesn't make sense)
        if signal == 1 and df.iloc[i - 1]['signals'] == 1:
            signal = 0
        if signal == -1 and df.iloc[i - 1]['signals'] == -1:
            signal = 0
        if signal == 2 and current_pos == 0:
            signal = 1
        if signal == -2 and current_pos == 0:
            signal = -1
        if signal == 2 and df.iloc[i - 1]['signals'] in [1, 2]:
            signal = 0
        if signal == -2 and df.iloc[i - 1]['signals'] in [-1, -2]:
            signal = 0

        df.at[df.index[i], 'signals'] = signal

        if signal == 1:
            df.at[df.index[i], 'trade_type'] = "OPEN_LONG"
        elif signal == -1:
            df.at[df.index[i], 'trade_type'] = "OPEN_SHORT"
        elif signal == 2:
            df.at[df.index[i], 'trade_type'] = "CLOSE_SHORT_OPEN_LONG"
        elif signal == -2:
            df.at[df.index[i], 'trade_type'] = "CLOSE_LONG_OPEN_SHORT"

    return df


def main():
    df = pd.read_csv("BTC_2019_2023_1d.csv", parse_dates=['datetime'])
    df = df.set_index("datetime")
    df = process_data(df)
    df = strat(df)

    df.to_csv("results.csv")

    bt = BackTester("BTC", signal_data_path="results.csv", master_file_path="BTC_2019_2023_1d.csv", compound_flag=1)
    bt.get_trades(1000)
    stats = bt.get_statistics()

    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()