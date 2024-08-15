    v2 Model Using period 45 for harmonic high and harmonic low. 
    
    Below are the features that are used in v2 : 
    features =  [f'candle_type_T-{i}' for i in range(1, window_size + 1) ] +\
               [f'rsi_T-{i}' for i in range(1, window_size + 1) ] +\
               [f'Price_Range_T-{i}' for i in range(1, window_size + 1)  ] +\
               [f'rsi_crossover_slow_T-{i}' for i in range(1, window_size + 1) ] +\
               [f'rsi_crossover_fast_T-{i}' for i in range(1, window_size + 1) ] +\
               [f'harmonic_mean_low_T-{i}' for i in range(1, window_size + 1)  ] +\
               [f'harmonic_mean_high_T-{i}' for i in range(1, window_size + 1) ] +\
               [f'slow_harmonic_mean_T-{i}' for i in range(1, window_size + 1) ] +\
               [f'fast_harmonic_mean_T-{i}' for i in range(1, window_size + 1) ] 
    
    And Following Hour-4 Features:-
    if alpha_type == "gamma":
        substrings = [ 'supertrend_h4' ]
    
    elif alpha_type == "beta":
        substrings = [ 'supertrend_h4', 'candle_type']
        
        
    elif alpha_type == "omega":
        substrings = [ 'supertrend_h4', 'candle_type', 'rsi_slow_crossover']
        
    elif alpha_type == "alpha":
        substrings = [ 'relative_range', 'fast_harmonic_mean']
    
    elif alpha_type == "delta":
        substrings = [ 'relative_range', 'rsi_slow_crossover', 'candle_type', 'supertrend_h4']
        
    elif alpha_type == "lambda":
        substrings = ['relative_range', 'candle_type', 'heikin_ashi' ]
    
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
