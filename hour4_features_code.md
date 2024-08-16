
        
        # WITH LESS FEATURES FOR HOUR 4 
        
        
        def rename_h4_df(hour4_path):
        
            data_h4 = pd.read_csv(hour4_path, sep='\t')
        
            # Rename columns as requested
            data_h4.rename(columns={
                '<DATE>': 'Date',
                '<OPEN>': 'Open_h4',
                '<HIGH>': 'High_h4',
                '<LOW>': 'Low_h4',
                '<CLOSE>': 'Close_h4',
                '<TICKVOL>': 'TickVol',
                '<VOL>': 'Vol',
                '<SPREAD>': 'Spread'
            }, inplace=True)
        
            data_h4['Date'] = pd.to_datetime(data_h4['Date'], format='%Y.%m.%d')
        
            # Set Date column as the index
            data_h4.set_index('Date', inplace=True)
        
            data_h4 = data_h4[['Open_h4', 'High_h4', 'Low_h4', 'Close_h4']]
        
            return data_h4
        
        
        def combine_ohlc_into_single_day(data_h4):
            grouped = data_h4.groupby(data_h4.index.date)
        
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
            
        
        def add_ohlc_in_lagged(reshaped_h4, lag_by= 7):
            
            features_h4 = pd.DataFrame()
            for candles in range(0, 6): # 0 --> 5 all 6 candles
                for day in range(1, lag_by + 1): # last 3 days = 6 * 3 = Last H4 18 candles
         
                    # new name will be candle number and day shifted from
                    features_h4[f'Close_h4_{candles}_T-{day}'] = reshaped_h4[f'Close_h4_{candles}'].shift(day)
                    features_h4[f'High_h4_{candles}_T-{day}']  = reshaped_h4[f'High_h4_{candles}'].shift(day)
                    features_h4[f'Open_h4_{candles}_T-{day}']  = reshaped_h4[f'Open_h4_{candles}'].shift(day)
                    features_h4[f'Low_h4_{candles}_T-{day}']  = reshaped_h4[f'Low_h4_{candles}'].shift(day)
            
            return  features_h4
        
        
        def add_features(features_h4, lag_by= 7):
            
            # features_h4.fillna(0.0, inplace= True)
            features_h4 = features_h4.apply(lambda x: x.fillna(x.mean()), axis=0)
            
            for candles in range(0, 6): # 0 --> 5 all 6 candles
                for day in range(1, lag_by + 1): # last 3 days = 6 * 3 = Last H4 18 candles
                    pass 
                    # open_value  = features_h4[f'Open_h4_{candles}_T-{day}']
                    close_value = features_h4[f'Close_h4_{candles}_T-{day}']
                    high_value = features_h4[f'High_h4_{candles}_T-{day}']
                    low_value  = features_h4[f'Low_h4_{candles}_T-{day}']
                    hl2 = ( high_value + low_value ) / 2
        
                    # features_h4[f'Low_h4_{candles}_T-{day}'].fillna(features_h4[f'Low_h4_{candles}_T-{day}'].mean(), inplace=True)
                    # features_h4[f'High_h4_{candles}_T-{day}'].fillna(features_h4[f'High_h4_{candles}_T-{day}'].mean(), inplace=True)
                    # features_h4[f'Close_h4_{candles}_T-{day}'].fillna(features_h4[f'Close_h4_{candles}_T-{day}'].mean(), inplace=True)
                    features_h4[f'slow_harmonic_mean_{candles}_T-{day}'] = calculate_harmonic_mean(close_value , period=27)
                    features_h4[f'fast_harmonic_mean_{candles}_T-{day}'] = calculate_harmonic_mean(close_value, period=9)
            
                    features_h4[f'harmonic_mean_high_{candles}_T-{day}'] = calculate_hm_high(high_value)
                    features_h4[f'harmonic_mean_low_{candles}_T-{day}']  = calculate_hm_low(low_value)
                    
                    rsi  = calculate_rsi(hl2, period=27)
                    smoothed_rsi =  rsi.rolling(window=9).mean()
                    slow_smoothed_rsi =  rsi.rolling(window=10).mean()
        
                    features_h4[f'rsi_sma_fast_{candles}_T-{day}'] = smoothed_rsi
                    features_h4[f'rsi_sma_slow_{candles}_T-{day}'] = slow_smoothed_rsi
                    features_h4[f'rsi_{candles}_T-{day}'] =  rsi
                    features_h4[f'rsi_crossover_fast_{candles}_T-{day}'] =  rsi_crossover(smoothed_rsi)
                    features_h4[f'rsi_crossover_slow_{candles}_T-{day}'] =  rsi_crossover_with_sma(rsi , slow_smoothed_rsi)
                    
                    
                    open_value  = features_h4[f'Open_h4_{candles}_T-{day}'].values
                    close_value = features_h4[f'Close_h4_{candles}_T-{day}'].values
                    high_value = features_h4[f'High_h4_{candles}_T-{day}'].values
                    low_value  = features_h4[f'Low_h4_{candles}_T-{day}'].values
        
                    open_val  = features_h4[f'Open_h4_{candles}_T-{day}']
                    close_val = features_h4[f'Close_h4_{candles}_T-{day}']
                    high_val = features_h4[f'High_h4_{candles}_T-{day}']
                    low_val  = features_h4[f'Low_h4_{candles}_T-{day}']
        
                    hlc = ( features_h4[f'High_h4_{candles}_T-{day}'] + features_h4[f'Low_h4_{candles}_T-{day}'] + features_h4[f'Close_h4_{candles}_T-{day}']) / 3 
        
                    features_h4[f'true_range_h4_{candles}_T-{day}'] = pd.Series(high_value- low_value)
        
                    # features_h4[f'median_h4_{candles}_T-{day}'] = pd.Series( (high_value + low_value) / 2 )
                    
                    
        #             features_h4[f'RSI_slow_{candles}_T-{day}']   = calculate_rsi(close_val, period=21)
        #             features_h4[f'RSI_fast_{candles}_T-{day}']   = calculate_rsi(close_val, period=9)
                    
                    features_h4[f'stdev_slow_{candles}_T-{day}'] = close_val.rolling(window=18).std()
                    features_h4[f'stdev_fast_{candles}_T-{day}'] = close_val.rolling(window=9).std()
            
        #             features_h4[f'Upper_Band_slow_{candles}_T-{day}'] = close_val.rolling(window=18).mean() + (close_val.rolling(window=5).std() * 2)
        #             features_h4[f'Lower_Band_slow_{candles}_T-{day}'] = close_val.rolling(window=18).mean() - (close_val.rolling(window=5).std() * 2)
        
        #             features_h4[f'Upper_Band_fast_{candles}_T-{day}'] = close_val.rolling(window=9).mean() + (close_val.rolling(window=5).std() * 2)
        #             features_h4[f'Lower_Band_fast_{candles}_T-{day}'] = close_val.rolling(window=9).mean() - (close_val.rolling(window=5).std() * 2)
        
        #             ema3_value  = hlc.ewm(span=3, adjust=False).mean()
        #             ema5_value  = hlc.ewm(span=5, adjust=False).mean()
        #             ema7_value  = hlc.ewm(span=7, adjust=False).mean()
        #             ema14_value = hlc.ewm(span=14, adjust=False).mean()
        
                    highest_high = features_h4[f'High_h4_{candles}_T-{day}'].rolling(window=9).max()
                    lowest_low   = features_h4[f'Low_h4_{candles}_T-{day}'].rolling(window=9).min()
        
                    # Calculate the relative range
                    features_h4[f'relative_range_h4_{candles}_T-{day}'] = close_value - ( ( highest_high + lowest_low ) / 2)
        
        
                    candle_type_value  = candle_type(open_value, high_value, low_value, close_value)
        
        
        #             elastic_supertrend, es_status_value = ma_based_supertrend_indicator( high_value, low_value, close_value, atr_length=9, atr_multiplier=2.5, ma_length=9)
        
        #             elastic_supertrend_crossover = supertrend_status_crossover(es_status_value)
        
                    supertrend, supertrend_status_value = supertrend_indicator(high_value, low_value, close_value, period= 27, multiplier=5.5)
                    supertrend_crossover = supertrend_status_crossover(supertrend_status_value)
        
                    features_h4[f'supertrend_h4_{candles}_T-{day}'] = supertrend
        
                    features_h4[f'supertrend_status_h4_{candles}_T-{day}'] = supertrend_status_value
        
                    features_h4[f'supertrend_crossover_h4_{candles}_T-{day}'] = supertrend_crossover
        
        
        #             features_h4[f'es_supertrend_h4_{candles}_T-{day}'] = elastic_supertrend
        
        #             features_h4[f'es_supertrend_crossover_h4_{candles}_T-{day}'] = elastic_supertrend_crossover
        
        #             features_h4[f'es_supertrend_status_h4_{candles}_T-{day}'] = es_status_value
        
                    
                    features_h4[f'candle_type_h4_{candles}_T-{day}'] = candle_type_value
        
                    # features_h4[f'smi_direct_h4_{candles}_T-{day}']  = smi_fast_direction
        
                    # features_h4[f'smi_h4_{candles}_T-{day}'] = smi_fast
        
        #             features_h4[f'ema3_h4_{candles}_T-{day}']  = pd.Series(ema3_value)
        
        #             features_h4[f'ema5_h4_{candles}_T-{day}']  = pd.Series(ema5_value)
        
        #             features_h4[f'ema7_h4_{candles}_T-{day}']  = pd.Series(ema7_value)
        
        #             features_h4[f'ema14_h4_{candles}_T-{day}'] = pd.Series(ema14_value)
        
                    ha_open, ha_close, ha_high, ha_low = heikin_ashi_candles(open_value, high_value, low_value, close_value)
                    heikin_ashi_candle = heikin_ashi_status(ha_open, ha_close)
                    features_h4[f'heikin_ashi_{candles}_T-{day}'] = heikin_ashi_candle
        
        #             selected_columns = features_h4[[ema3_column, ema5_column, ema7_column ]]
        
        #             ema_mean = np.mean(selected_columns, axis=1)
        #             features_h4[f'ema_difference_{candles}_T-{day}'] = pd.Series(np.subtract( ema_mean, ema14_value ))
        
                
            return  features_h4
        
        
        
        def ema( price, period):
        
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
            
        
        
        # Calculate RSI
        def calculate_rsi(series, period=5):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
            
        def moving_max(array, window_size):
           
            rolling_max = np.full(array.shape, 0.0)
            
            for i in range(len(array) - window_size + 1):
                window_values = array[i:i + window_size]
                rolling_max[i + window_size - 1] = np.max(window_values)
                
            rolling_max[np.isnan(rolling_max)] = np.nanmean(rolling_max)
            return rolling_max    
            
        
        def moving_min(array, window_size):
        
            rolling_min = np.full(array.shape, 0.0)
            for i in range(len(array) - window_size + 1):
                window_values = array[i:i + window_size]
                rolling_min[i + window_size - 1] = np.min(window_values)
                
            rolling_min[np.isnan(rolling_min)] = np.nanmean(rolling_min)
            return rolling_min
        
        
        
        def true_range( high, low, close):
        
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
        
        
        def ma_based_supertrend_indicator( high, low, close, atr_length=10, atr_multiplier=3, ma_length=10):
        
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
        
        
        
        
        def supertrend_status_crossover( status_value):
        
        
            prev_status = np.roll(status_value, 1)
            supertrend_status_crossover = np.where((prev_status < 0) & (status_value > 0), 1.0, np.where((prev_status > 0) & (status_value < 0), -1.0, 0))
        
            return supertrend_status_crossover
        
        
        
        
        def supertrend_indicator(high, low, close, period, multiplier=1.0):
        
            true_range_value = true_range(high, low, close)
        
            smoothed_atr = ema(true_range_value, period)
        
            upper_band = (high + low) / 2 + (multiplier * smoothed_atr)
            lower_band = (high + low) / 2 - (multiplier * smoothed_atr)
        
            supertrend = np.zeros(len(true_range_value))
            trend = np.zeros(len(true_range_value))
        
            if close[0] > upper_band[0]: trend[0] = upper_band[0]
            elif close[0] < lower_band[0]: trend[0] = lower_band[0]
            else:  trend[0] = upper_band[0]
        
            for i in range(1, len(close)):
        
                if close[i] > upper_band[i]: trend[i] = upper_band[i]
                elif close[i] < lower_band[i]: trend[i] = lower_band[i]
                else: trend[i] = trend[i - 1]
        
            # Calculate Buy/Sell Signals using numpy where  # np.where( close > trend, '1 Buy', '-1 Sell')
            status_value = np.where(close > trend, 1.0, -1.0)
        
            return trend, status_value
        
        def supertrend_status_crossover(status_value):
        
        
            prev_status = np.roll(status_value, 1)
            supertrend_status_crossover = np.where((prev_status < 0) & (status_value > 0), 1.0, np.where((prev_status > 0) & (status_value < 0), -1.0, 0))
        
            return supertrend_status_crossover
        
        
        
        
        def direction_crossover_signal_line(signal, signal_ema):
        
            direction = np.where(signal - signal_ema > 0, 1, -1)
            prev_direction = np.roll(direction, 1)
            crossover = np.where((prev_direction < 0) & (direction > 0), 1,
                                  np.where((prev_direction > 0) & (direction < 0), -1, 0))
        
            return direction, crossover
        
        
        def stochastic_momentum_index(high, low, close, period=20, ema_period=5):
            # Compute highest high and lowest low over the period
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
        
            # Compute relative range
            relative_range = close - ((highest_high + lowest_low) / 2)
        
            # Compute highest-lowest range
            highest_lowest_range = highest_high - lowest_low
        
            # Smooth relative range and highest-lowest range
            relative_range_smoothed = relative_range.ewm(span=ema_period, adjust=False).mean().ewm(span=ema_period, adjust=False).mean()
            highest_lowest_range_smoothed = highest_lowest_range.ewm(span=ema_period, adjust=False).mean().ewm(span=ema_period, adjust=False).mean()
        
            # Calculate SMI
            smi = (relative_range_smoothed / (highest_lowest_range_smoothed / 2)) * 100
            smi[smi == np.inf] = 0  # Replace infinite values with 0
            smi_ema = smi.ewm(span=ema_period, adjust=False).mean()
        
            return smi, smi_ema
        
        # def stochastic_momentum_index(high, low, close, period=20, ema_period=5):
        
        #     lengthD = ema_period
        #     lowest_low   = moving_min(low, period)
        #     highest_high = moving_max(high, period)
        #     relative_range = close - ((highest_high + lowest_low) / 2)
        #     highest_lowest_range = highest_high - lowest_low
        
        #     relative_range_smoothed = ema(ema(relative_range, ema_period), ema_period)
        #     highest_lowest_range_smoothed = ema(ema(highest_lowest_range, ema_period), ema_period)
        
        #     smi = [(relative_range_smoothed[i] / (highest_lowest_range_smoothed[i] / 2)) * 100 if
        #             highest_lowest_range_smoothed[i] != 0 else 0.0
        #             for i in range(len(relative_range_smoothed))]
        
        #     smi_ema = ema(smi, ema_period)
        
        #     return smi, smi_ema
        
        
        def candle_type(o, h, l, c):
        
            diff = abs(c - o)
            o1, c1 = np.roll(o, 1), np.roll(c, 1)  #
            min_oc = np.where(o < c, o, c)
            max_oc = np.where(o > c, o, c)
        
            pattern = np.where(
              np.logical_and( min_oc - l > diff, h - max_oc < diff), 6,
              np.where(np.logical_and( h - max_oc > diff, min_oc - l < diff),
              4, np.where(np.logical_and(np.logical_and(c > o, c1 < o1), np.logical_and(c > o1, o < c1)),
                5, np.where( min_oc - l > diff, 3,
                              np.where(np.logical_and( h - max_oc > diff,
                          min_oc - l < diff),
                              2, np.where(np.logical_and(np.logical_and(c > o, c1 < o1), np.logical_and(c > o1, o < c1)),
                              1, 0))))))
            return pattern
        
        
        
        
        
        def heikin_ashi_status( ha_open, ha_close):
        
            candles = np.full_like(ha_close, '', dtype='U10')
        
            for i in range(1, len(ha_close)):
        
                if ha_close[i] > ha_open[i]: candles[i] = 2 #'Green'
        
                elif ha_close[i] < ha_open[i]: candles[i] = 1 # 'Red'
        
                else: candles[i] = 0 #'Neutral'
        
            return candles
        
        def heikin_ashi_candles( open, high, low, close):
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
