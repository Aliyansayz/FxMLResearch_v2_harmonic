Used in version 2 : -

                    # Calculate RSI
                    def calculate_rsi(series, period=10):
                        delta = series.diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                        rs = gain / loss
                        return 100 - (100 / (1 + rs))
                    
                    
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
                    
                    
                    def rsi_crossover(smoothed_rsi):
                        
                        prev_signal = np.roll(smoothed_rsi, 1)
                        direction, crossover = direction_crossover_signal_line(smoothed_rsi, prev_signal)
                        
                        return crossover
                        pass
                    
                    def rsi_crossover_with_sma(rsi , slow_smoothed_rsi): # 10 period sma of actual rsi
                        
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
                        return  series.rolling(window=period).apply(harmonic_mean_high, raw=True)
                    
                    def harmonic_mean_high(high):
                            n = len(high)
                            prev_high = np.roll(high, 1)    
                            diff = high - prev_high
                    
                            hm = n / sum(1 / x for x in diff if x != 0)
                            y = np.sin(hm)
                    
                            y_pi = y * np.pi
                            return y_pi
                    
                    
                    def calculate_hm_low(series, period=45):
                    
                        def harmonic_mean_low(low):
                            n = len(low)
                            prev_low = np.roll(low, 1)    
                            diff = prev_low - low
                    
                            hm = n / sum(1 / x for x in diff if x != 0)
                            y = np.sin(hm)
                    
                            y_pi = y * np.pi
                            return y_pi
                        return  series.rolling(window=period).apply(harmonic_mean_low, raw=True)
                        
                    
                    def direction_crossover_signal_line(signal, prev_signal):
                    
                            direction = np.where(signal - prev_signal > 0, 1, -1) # current bigger then upward direction , if current small then downward direction
                            prev_direction = np.roll(direction, 1)
                            crossover = np.where((prev_direction == -1) & (direction == 1), 1,
                                                 np.where((prev_direction == 1) & (direction == -1), -1, 0))
                            return direction, crossover
                    
                    def supertrend_status_crossover( status_value):
                    
                    
                        prev_status = np.roll(status_value, 1)
                        supertrend_status_crossover = np.where((prev_status < 0) & (status_value > 0), 1.0, np.where((prev_status > 0) & (status_value < 0), -1.0, 0))
                    
                        return supertrend_status_crossover

Not used in model v2 :- 
                
                def ema( price, period)
                
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











        
