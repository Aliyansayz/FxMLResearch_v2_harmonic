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
