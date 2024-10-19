import pandas as pd
# !pip install numpy==1.26.4

import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# pip install catboost==1.2.5
# from catboost import CatBoostClassifier
from catboost import Pool
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint



def get_features_transformed(symbol, day_data, hour4_data, window_size = 9, alpha_type= "gamma"):

    features = [f'candle_type_T-{i}' for i in range(1, window_size + 1)] + \
               [f'rsi_T-{i}' for i in range(1, window_size + 1)] + \
               [f'Price_Range_T-{i}' for i in range(1, window_size + 1)] + \
               [f'rsi_crossover_slow_T-{i}' for i in range(1, window_size + 1)] + \
               [f'rsi_crossover_fast_T-{i}' for i in range(1, window_size + 1)] + \
               [f'harmonic_mean_low_T-{i}' for i in range(1, window_size + 1)] + \
               [f'harmonic_mean_high_T-{i}' for i in range(1, window_size + 1)] + \
               [f'slow_harmonic_mean_T-{i}' for i in range(1, window_size + 1)] + \
               [f'fast_harmonic_mean_T-{i}' for i in range(1, window_size + 1)]

    if alpha_type == "gamma":
        substrings = ['supertrend_h4']

    elif alpha_type == "beta":
        substrings = ['supertrend_h4', 'candle_type']

    elif alpha_type == "omega":
        substrings = ['supertrend_h4', 'candle_type', 'rsi_slow_crossover']

    elif alpha_type == "alpha":
        substrings = ['relative_range', 'fast_harmonic_mean']

    elif alpha_type == "delta":
        substrings = ['relative_range', 'rsi_slow_crossover', 'candle_type', 'supertrend_h4']

    elif alpha_type == "lambda":
        substrings = ['relative_range', 'candle_type', 'heikin_ashi']

    elif alpha_type == "ha":
        substrings = ['stdev_slow', 'heikin_ashi', 'slow_harmonic_mean']  # 'slow_harmonic_mean'

    elif alpha_type == "gamma_metals_fix":
        substrings = ['supertrend_h4', 'rsi_crossover_slow', 'candle_type']
        features = [f'candle_type_T-{i}' for i in range(1, window_size + 1)] + \
                   [f'rsi_T-{i}' for i in range(1, window_size + 1)] + \
                   [f'rsi_crossover_fast_T-{i}' for i in range(1, window_size + 1)] + \
                   [f'fast_harmonic_mean_T-{i}' for i in range(1, 0 + 1)]

    elif alpha_type == "gamma_indices":
        substrings = ['supertrend_h4', 'rsi_crossover_slow', 'slow_harmonic_mean']
        features = [f'candle_type_T-{i}' for i in range(1, window_size + 1)] + \
                   [f'rsi_T-{i}' for i in range(1, window_size + 1)] + \
                   [f'Price_Range_T-{i}' for i in range(1, window_size + 1)] + \
                   [f'rsi_crossover_slow_T-{i}' for i in range(1, window_size + 1)] + \
                   [f'rsi_crossover_fast_T-{i}' for i in range(1, window_size + 1)] + \
                   [f'fast_harmonic_mean_T-{i}' for i in range(1, window_size + 1)]


    day_data['pips_change'] = np.subtract(day_data['Close'].to_numpy(), day_data['Close'].shift(1).to_numpy())

    if 'JPY' in symbol:
        pips_change = day_data['pips_change'] * 10 ** 2

    else:
        pips_change = day_data['pips_change'] * 10 ** 4

    day_data['Target'] = pips_change

    day_features_X = day_data[features]

    day_features_X = day_features_X.apply(pd.to_numeric, errors='coerce')

    day_features_X = day_features_X.astype(float)

    hour4_features_X = hour4_data

    hour4_features_X = hour4_features_X.apply(pd.to_numeric, errors='coerce')

    hour4_features_X = hour4_features_X.astype(float)

    filtered_columns = [col for col in hour4_features_X.columns if any(sub in col for sub in substrings)]
    hour4_features_X = hour4_features_X[filtered_columns]

    X = day_features_X.join(hour4_features_X)

    scaler = RobustScaler()

    X_scaled = scaler.fit_transform(X)


    return X_scaled , X



# Calculate RSI
def calculate_rsi(series, period=10):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def ema(price, period):
    price = np.array(price)
    alpha = 2 / (period + 1.0)
    alpha_reverse = 1 - alpha
    data_length = len(price)

    power_factors = alpha_reverse ** (np.arange(data_length + 1))
    initial_offset = price[0] * power_factors[1:]

    scale_factors = 1 / power_factors[:-1]

    weight_factor = alpha * alpha_reverse ** (data_length - 1)

    weighted_price_data = price * weight_factor * scale_factors
    cumulative_sums = weighted_price_data.cumsum()
    ema_values = initial_offset + cumulative_sums * scale_factors[::-1]

    return ema_values


def candle_type(o, h, l, c):
    diff = abs(c - o)
    o1, c1 = np.roll(o, 1), np.roll(c, 1)  #
    min_oc = np.where(o < c, o, c)
    max_oc = np.where(o > c, o, c)

    pattern = np.where(
        np.logical_and(min_oc - l > diff, h - max_oc < diff), 6,
        np.where(np.logical_and(h - max_oc > diff, min_oc - l < diff),
                 4, np.where(np.logical_and(np.logical_and(c > o, c1 < o1), np.logical_and(c > o1, o < c1)),
                             5, np.where(min_oc - l > diff, 3,
                                         np.where(np.logical_and(h - max_oc > diff,
                                                                 min_oc - l < diff),
                                                  2, np.where(np.logical_and(np.logical_and(c > o, c1 < o1),
                                                                             np.logical_and(c > o1, o < c1)),
                                                              1, 0))))))
    return pattern


def heikin_ashi_status(ha_open, ha_close):
    candles = np.full_like(ha_close, '', dtype='U10')

    for i in range(1, len(ha_close)):

        if ha_close[i] > ha_open[i]:
            candles[i] = 2  # 'Green'

        elif ha_close[i] < ha_open[i]:
            candles[i] = 1  # 'Red'

        else:
            candles[i] = 0  # 'Neutral'

    return candles


def heikin_ashi_candles(open, high, low, close):
    ha_low, ha_close = np.empty(len(close), dtype=np.float32), np.empty(len(close), dtype=np.float32)
    ha_open, ha_high = np.empty(len(close), dtype=np.float32), np.empty(len(close), dtype=np.float32)

    ha_open[0] = (open[0] + close[0]) / 2
    ha_close[0] = (close[0] + open[0] + high[0] + low[0]) / 4

    for i in range(1, len(close)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
        ha_close[i] = (open[i] + high[i] + low[i] + close[i]) / 4
        ha_high[i] = max(high[i], ha_open[i], ha_close[i])
        ha_low[i] = min(low[i], ha_open[i], ha_close[i])

    return ha_open, ha_close, ha_high, ha_low


def true_range(high, low, close):
    close_shift = shift(close, 1)
    high_low, high_close, low_close = np.array(high - low, dtype=np.float32), \
                                      np.array(abs(high - close_shift), dtype=np.float32), \
                                      np.array(abs(low - close_shift), dtype=np.float32)

    true_range = np.max(np.hstack((high_low, high_close, low_close)).reshape(-1, 3), axis=1)

    return true_range


def shift(array, place):
    array = np.array(array, dtype=np.float32)
    shifted = np.roll(array, place)
    shifted[0:place] = np.nan
    shifted[np.isnan(shifted)] = np.nanmean(shifted)

    return shifted


def ma_based_supertrend_indicator(high, low, close, atr_length=10, atr_multiplier=3, ma_length=10):
    # Calculate True Range and Smoothed ATR
    tr = true_range(high, low, close)
    atr = ema(tr, atr_length)

    upper_band = (high + low) / 2 + (atr_multiplier * atr)
    lower_band = (high + low) / 2 - (atr_multiplier * atr)

    trend = np.zeros(len(atr))

    # Calculate Moving Average
    ema_values = ema(close, ma_length)

    if ema_values[0] > lower_band[0]:
        trend[0] = lower_band[0]
    elif ema_values[0] < upper_band[0]:
        trend[0] = upper_band[0]
    else:
        trend[0] = upper_band[0]

    # Compute final upper and lower bands
    for i in range(1, len(close)):
        if ema_values[i] > trend[i - 1]:
            trend[i] = max(trend[i - 1], lower_band[i])


        elif ema_values[i] < trend[i - 1]:
            trend[i] = min(trend[i - 1], upper_band[i])

        else:
            trend[i] = trend[i - 1]

    status_value = np.where(ema_values > trend, 1.0, -1.0)

    return trend, status_value


def supertrend_status_crossover(status_value):
    prev_status = np.roll(status_value, 1)
    supertrend_status_crossover = np.where((prev_status < 0) & (status_value > 0), 1.0,
                                           np.where((prev_status > 0) & (status_value < 0), -1.0, 0))

    return supertrend_status_crossover


def supertrend_indicator(high, low, close, period, multiplier=1.0):
    true_range_value = true_range(high, low, close)

    smoothed_atr = ema(true_range_value, period)

    upper_band = (high + low) / 2 + (multiplier * smoothed_atr)
    lower_band = (high + low) / 2 - (multiplier * smoothed_atr)

    supertrend = np.zeros(len(true_range_value))
    trend = np.zeros(len(true_range_value))

    if close[0] > upper_band[0]:
        trend[0] = upper_band[0]
    elif close[0] < lower_band[0]:
        trend[0] = lower_band[0]
    else:
        trend[0] = upper_band[0]

    for i in range(1, len(close)):

        if close[i] > upper_band[i]:
            trend[i] = upper_band[i]
        elif close[i] < lower_band[i]:
            trend[i] = lower_band[i]
        else:
            trend[i] = trend[i - 1]

    # Calculate Buy/Sell Signals using numpy where  # np.where( close > trend, '1 Buy', '-1 Sell')
    status_value = np.where(close > trend, 1.0, -1.0)

    return trend, status_value


def supertrend_status_crossover(status_value):
    prev_status = np.roll(status_value, 1)
    supertrend_status_crossover = np.where((prev_status < 0) & (status_value > 0), 1.0,
                                           np.where((prev_status > 0) & (status_value < 0), -1.0, 0))

    return supertrend_status_crossover


def rsi_crossover(smoothed_rsi):
    prev_signal = np.roll(smoothed_rsi, 1)
    direction, crossover = direction_crossover_signal_line(smoothed_rsi, prev_signal)

    return crossover
    pass


def rsi_crossover_with_sma(rsi, slow_smoothed_rsi):  # 10 period sma of actual rsi

    direction = np.where(rsi - slow_smoothed_rsi > 0, 1, -1)

    prev_direction = np.roll(direction, 1)
    crossover = np.where((prev_direction == -1) & (direction == 1), 1,
                         np.where((prev_direction == 1) & (direction == -1), -1, 0))

    return crossover


def calculate_harmonic_mean(series, period=3):
    def harmonic_mean(s):
        n = len(s)
        return n / sum(1 / x for x in s if x != 0) if n > 0 else 0

    return series.rolling(window=period).apply(harmonic_mean, raw=True)


def calculate_hm_high(series, period=45):
    def harmonic_mean_high(high):
        n = len(high)
        prev_high = np.roll(high, 1)
        diff = high - prev_high

        hm = n / sum(1 / x for x in diff if x != 0)
        y = np.sin(hm)

        y_pi = y * np.pi
        return y_pi

    return series.rolling(window=period).apply(harmonic_mean_high, raw=True)


def calculate_hm_low(series, period=45):
    def harmonic_mean_low(low):
        n = len(low)
        prev_low = np.roll(low, 1)
        diff = prev_low - low

        hm = n / sum(1 / x for x in diff if x != 0)
        y = np.sin(hm)

        y_pi = y * np.pi
        return y_pi

    return series.rolling(window=period).apply(harmonic_mean_low, raw=True)


def direction_crossover_signal_line(signal, prev_signal):
    direction = np.where(signal - prev_signal > 0, 1,
                         -1)  # current bigger then upward direction , if current small then downward direction
    prev_direction = np.roll(direction, 1)
    crossover = np.where((prev_direction == -1) & (direction == 1), 1,
                         np.where((prev_direction == 1) & (direction == -1), -1, 0))

    return direction, crossover




def add_features_day(df):

    open_value, high, low, close = df['Open'], df['High'], df['Low'], df['Close']
    # elastic_supertrend, es_status_value = ma_based_supertrend_indicator( high, low, close, atr_length=10, atr_multiplier=2.5, ma_length=10)
    # elastic_supertrend_crossover = supertrend_status_crossover(es_status_value)
    # supertrend, supertrend_status_value = supertrend_indicator(high, low, close, period= 10, multiplier=0.66)
    # supertrend_crossover = supertrend_status_crossover(supertrend_status_value)

    candle_type_value = candle_type(open_value, high, low, close)

    # ha_open, ha_close, ha_high, ha_low = heikin_ashi_candles(open_value, high, low, close)
    # heikin_ashi_candle = heikin_ashi_status(ha_open, ha_close)
    hl2 = (df['High'] + df['Low']) / 2

    rsi = calculate_rsi(hl2, period=10)
    smoothed_rsi = rsi.rolling(window=3).mean()
    slow_smoothed_rsi = rsi.rolling(window=10).mean()

    df['rsi_sma_fast'] = smoothed_rsi
    df['rsi_sma_slow'] = slow_smoothed_rsi

    df['rsi'] = rsi
    df['rsi_crossover_fast'] = rsi_crossover(smoothed_rsi)
    df['rsi_crossover_slow'] = rsi_crossover_with_sma(rsi, slow_smoothed_rsi)

    df['STDEV'] = df['Close'].rolling(window=5).std()
    df['Upper_Band'] = df['Close'].rolling(window=5).mean() + (df['Close'].rolling(window=5).std() * 2)
    df['Lower_Band'] = df['Close'].rolling(window=5).mean() - (df['Close'].rolling(window=5).std() * 2)

    df['candle_type'] = candle_type_value

    df['slow_harmonic_mean'] = calculate_harmonic_mean(df['Close'], period=9)
    df['fast_harmonic_mean'] = calculate_harmonic_mean(df['Close'], period=3)

    df['harmonic_mean_high'] = calculate_hm_high(high)

    df['harmonic_mean_low'] = calculate_hm_low(low)

    df['Price_Range'] = df['High'] - df['Low']
    df['Median_Price'] = (df['High'] + df['Low']) / 2

    df['daily_returns'] = df['Close'] - df['Close'].shift(1)

    df['Day_of_Week'] = df.index.dayofweek + 1  # Monday=1, ..., Friday=5
    df['Week_of_Month'] = (df.index.day - 1) // 7 + 1
    df['Month'] = df.index.month

    return df



def make_features_lagged_day(df, lag_by=14):
    # Create window-based features
    window_size = lag_by

    for i in range(1, window_size + 1):
        df[f'Day_of_Week_T-{i}'] = df['Day_of_Week'].shift(i)
        df[f'Week_of_Month_T-{i}'] = df['Week_of_Month'].shift(i)
        df[f'Month_T-{i}'] = df['Month'].shift(i)
        df[f'Close_T-{i}'] = df['Close'].shift(i)
        df[f'Open_T-{i}'] = df['Open'].shift(i)
        df[f'High_T-{i}'] = df['High'].shift(i)
        df[f'Low_T-{i}'] = df['Low'].shift(i)
        df[f'STDEV_T-{i}'] = df['STDEV'].shift(i)
        df[f'rsi_T-{i}'] = df['rsi'].shift(i)
        df[f'rsi_crossover_fast_T-{i}'] = df['rsi_crossover_fast'].shift(i)
        df[f'rsi_crossover_slow_T-{i}'] = df['rsi_crossover_slow'].shift(i)

        df[f'Price_Range_T-{i}'] = df['Price_Range'].shift(i)
        df[f'Median_Price_T-{i}'] = df['Median_Price'].shift(i)
        df[f'Upper_Band_T-{i}'] = df['Upper_Band'].shift(i)
        df[f'Lower_Band_T-{i}'] = df['Lower_Band'].shift(i)
        df[f'candle_type_T-{i}'] = df['candle_type'].shift(i)
        df[f'slow_harmonic_mean_T-{i}'] = df['slow_harmonic_mean'].shift(i)
        df[f'fast_harmonic_mean_T-{i}'] = df['fast_harmonic_mean'].shift(i)
        df[f'harmonic_mean_high_T-{i}'] = df['harmonic_mean_high'].shift(i)
        df[f'harmonic_mean_low_T-{i}'] = df['harmonic_mean_low'].shift(i)
        df[f'daily_returns_T-{i}'] = df['daily_returns'].shift(i)

    return df


def return_df_day(daily_path):

    # data = pd.read_csv(daily_path, sep='\t')

    df = pd.read_csv(daily_path)
    # Rename columns as requested
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
    }, inplace=True)

    datetime_series = pd.to_datetime(df ['datetime']) # , format='%Y.%m.%d'
    df['datetime'] = datetime_series
    #
    # # Set Date column as the index
    df.set_index('datetime', inplace=True)


    df = df[['Open', 'High', 'Low', 'Close']]

    return df


def combine_ohlc_into_single_day_hour_4(df):
    grouped = df.groupby(df.index.date)

    # Create a new dataframe to store the result
    reshaped_h4 = pd.DataFrame()

    # Extract Open, High, Low, Close for each 4-hour period and reshape
    for date, group in grouped:
        group = group.reset_index(drop=True)
        for i in range(0, len(group)):
            if i == 0:
                reshaped_h4.at[date, f'Open_h4_{i}'] = group.loc[i, 'Open_h4']
                reshaped_h4.at[date, f'High_h4_{i}'] = group.loc[i, 'High_h4']
                reshaped_h4.at[date, f'Low_h4_{i}'] = group.loc[i, 'Low_h4']
                reshaped_h4.at[date, f'Close_h4_{i}'] = group.loc[i, 'Close_h4']
            else:
                reshaped_h4.at[date, f'Open_h4_{i}'] = group.loc[i, 'Open_h4']
                reshaped_h4.at[date, f'High_h4_{i}'] = group.loc[i, 'High_h4']
                reshaped_h4.at[date, f'Low_h4_{i}'] = group.loc[i, 'Low_h4']
                reshaped_h4.at[date, f'Close_h4_{i}'] = group.loc[i, 'Close_h4']

    return reshaped_h4


def add_ohlc_in_lagged_hour_4(reshaped_h4, lag_by=3):
    features_h4 = pd.DataFrame()
    for candles in range(0, 6):  # 0 --> 5 all 6 candles
        for day in range(1, lag_by + 1):  # last 3 days = 6 * 3 = Last H4 18 candles

            # new name will be candle number and day shifted from
            features_h4[f'Close_h4_{candles}_T-{day}'] = reshaped_h4[f'Close_h4_{candles}'].shift(day)
            features_h4[f'High_h4_{candles}_T-{day}'] = reshaped_h4[f'High_h4_{candles}'].shift(day)
            features_h4[f'Open_h4_{candles}_T-{day}'] = reshaped_h4[f'Open_h4_{candles}'].shift(day)
            features_h4[f'Low_h4_{candles}_T-{day}'] = reshaped_h4[f'Low_h4_{candles}'].shift(day)

    return features_h4


def return_h4_df(hour4_path):
    # data_h4 = pd.read_csv(hour4_path, sep='\t')
    df = pd.read_csv(hour4_path)
    # Rename columns as requested
    df.rename( columns={
        'open': 'Open_h4',
        'high': 'High_h4',
        'low': 'Low_h4',
        'close': 'Close_h4' }, inplace=True )


    datetime_series = pd.to_datetime(df['datetime'])
    df['datetime']  = datetime_series
    #
    # # Set Date column as the index
    df.set_index('datetime', inplace=True)

    df = df[['Open_h4', 'High_h4', 'Low_h4', 'Close_h4']]

    return df

def add_features_hour_4(features_h4, lag_by=3):

    # features_h4.fillna(0.0, inplace= True)
    features_h4 = features_h4.apply(lambda x: x.fillna(x.mean()), axis=0)

    for candles in range(0, 6):  # 0 --> 5 all 6 candles
        for day in range(1, lag_by + 1):  # last 3 days = 6 * 3 = Last H4 18 candles
            pass
            # open_value  = features_h4[f'Open_h4_{candles}_T-{day}']
            close_value = features_h4[f'Close_h4_{candles}_T-{day}']
            high_value = features_h4[f'High_h4_{candles}_T-{day}']
            low_value = features_h4[f'Low_h4_{candles}_T-{day}']
            hl2 = (high_value + low_value) / 2

            features_h4[f'slow_harmonic_mean_{candles}_T-{day}'] = calculate_harmonic_mean(close_value, period=27)
            features_h4[f'fast_harmonic_mean_{candles}_T-{day}'] = calculate_harmonic_mean(close_value, period=9)

            features_h4[f'harmonic_mean_high_{candles}_T-{day}'] = calculate_hm_high(high_value)
            features_h4[f'harmonic_mean_low_{candles}_T-{day}'] = calculate_hm_low(low_value)

            rsi = calculate_rsi(hl2, period=27)
            smoothed_rsi = rsi.rolling(window=9).mean()
            slow_smoothed_rsi = rsi.rolling(window=10).mean()

            features_h4[f'rsi_sma_fast_{candles}_T-{day}'] = smoothed_rsi
            features_h4[f'rsi_sma_slow_{candles}_T-{day}'] = slow_smoothed_rsi
            features_h4[f'rsi_{candles}_T-{day}'] = rsi
            features_h4[f'rsi_crossover_fast_{candles}_T-{day}'] = rsi_crossover(smoothed_rsi)
            features_h4[f'rsi_crossover_slow_{candles}_T-{day}'] = rsi_crossover_with_sma(rsi, slow_smoothed_rsi)

            open_value = features_h4[f'Open_h4_{candles}_T-{day}'].values
            close_value = features_h4[f'Close_h4_{candles}_T-{day}'].values
            high_value = features_h4[f'High_h4_{candles}_T-{day}'].values
            low_value = features_h4[f'Low_h4_{candles}_T-{day}'].values

            open_val = features_h4[f'Open_h4_{candles}_T-{day}']
            close_val = features_h4[f'Close_h4_{candles}_T-{day}']
            high_val = features_h4[f'High_h4_{candles}_T-{day}']
            low_val = features_h4[f'Low_h4_{candles}_T-{day}']

            hlc = (features_h4[f'High_h4_{candles}_T-{day}'] + features_h4[f'Low_h4_{candles}_T-{day}'] + features_h4[
                f'Close_h4_{candles}_T-{day}']) / 3

            features_h4[f'true_range_h4_{candles}_T-{day}'] = pd.Series(high_value - low_value)

            features_h4[f'stdev_slow_{candles}_T-{day}'] = close_val.rolling(window=18).std()
            features_h4[f'stdev_fast_{candles}_T-{day}'] = close_val.rolling(window=9).std()

            highest_high = features_h4[f'High_h4_{candles}_T-{day}'].rolling(window=9).max()
            lowest_low = features_h4[f'Low_h4_{candles}_T-{day}'].rolling(window=9).min()

            # Calculate the relative range
            features_h4[f'relative_range_h4_{candles}_T-{day}'] = close_value - ((highest_high + lowest_low) / 2)

            candle_type_value = candle_type(open_value, high_value, low_value, close_value)

            # elastic_supertrend, es_status_value = ma_based_supertrend_indicator( high_value, low_value, close_value, atr_length=9, atr_multiplier=2.5, ma_length=9)
            # elastic_supertrend_crossover = supertrend_status_crossover(es_status_value)

            supertrend, supertrend_status_value = supertrend_indicator(high_value, low_value, close_value, period=27,
                                                                       multiplier=5.5)
            supertrend_crossover = supertrend_status_crossover(supertrend_status_value)

            features_h4[f'supertrend_h4_{candles}_T-{day}'] = supertrend

            features_h4[f'supertrend_status_h4_{candles}_T-{day}'] = supertrend_status_value

            features_h4[f'supertrend_crossover_h4_{candles}_T-{day}'] = supertrend_crossover

            # features_h4[f'es_supertrend_h4_{candles}_T-{day}'] = elastic_supertrend
            # features_h4[f'es_supertrend_crossover_h4_{candles}_T-{day}'] = elastic_supertrend_crossover
            # features_h4[f'es_supertrend_status_h4_{candles}_T-{day}'] = es_status_value

            features_h4[f'candle_type_h4_{candles}_T-{day}'] = candle_type_value

            ha_open, ha_close, ha_high, ha_low = heikin_ashi_candles(open_value, high_value, low_value, close_value)
            heikin_ashi_candle = heikin_ashi_status(ha_open, ha_close)
            features_h4[f'heikin_ashi_{candles}_T-{day}'] = heikin_ashi_candle

    return features_h4

#
# def make_hour4_binary_file():
#     forex_pairs = ['AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
#                    'CADCHF', 'CADJPY',
#                    'CHFJPY',
#                    'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP',
#                    'EURJPY', 'EURNZD', 'EURUSD',
#                    'GBPAUD', 'GBPCAD', 'GBPCHF',
#                    'GBPJPY', 'GBPUSD', 'GBPNZD',
#                    'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',
#                    'USDCHF', 'USDCAD', 'USDJPY']
#
#     currency_ohlc = []
#     count = 0
#     for pair in forex_pairs:
#         # daily_path = f'currency_data/{pair[:6]}_Daily.csv'
#         hour4_path = f'currency_data/{pair}_H4.csv'
#
#         data_h4 = rename_h4_df(hour4_path)
#
#         reshaped_h4 = combine_ohlc_into_single_day(data_h4)
#
#         features_h4 = add_ohlc_in_lagged(reshaped_h4, lag_by=3)
#
#         features_h4 = add_features_hour_4(features_h4, lag_by=3)
#
#         currency_df = {"hour4_features": features_h4, "symbol": f"{pair}"}
#
#         currency_ohlc.append(currency_df)
#
#     import pickle
#
#     with open('hour4_features_data_lag_by_3.bin', 'wb') as file:
#         pickle.dump(currency_ohlc, file)
#
#
# def make_daily_binary_file():
#     forex_pairs = [
#         'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
#         'CADCHF', 'CADJPY',
#         'CHFJPY',
#         'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP',
#         'EURJPY', 'EURNZD', 'EURUSD',
#         'GBPAUD', 'GBPCAD', 'GBPCHF',
#         'GBPJPY', 'GBPUSD', 'GBPNZD',
#         'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',
#         'USDCHF', 'USDCAD', 'USDJPY'
#     ]
#
#     currency_ohlc = []
#     count = 0
#     for pair in forex_pairs:
#         daily_path = f'currency_data/{pair}_Daily.csv'
#         data = return_df_day(daily_path)
#         data = add_features_day(data)
#         data = make_features_lagged_day(lag_by=14)
#
#         currency_df = {"day_features": data, "symbol": f"{pair}"}
#         currency_ohlc.append(currency_df)
#         count += 1
#         print(count)
#
#
#     import pickle
#
#     with open('day_features_data_lagby_14.bin', 'wb') as file:
#         pickle.dump(currency_ohlc, file)
