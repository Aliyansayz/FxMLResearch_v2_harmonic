# V2 models
# forex_pairs = [
#     'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
#     'CADCHF', 'CADJPY',
#     'CHFJPY',
#     'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 
#     'EURJPY', 'EURNZD', 'EURUSD',
#     'GBPAUD', 'GBPCAD', 'GBPCHF', 
#     'GBPJPY', 'GBPUSD', 'GBPNZD',
#     'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',  
#     'USDCHF', 'USDCAD', 'USDJPY'
#         ]
# works when period = 45 in harmonic mean low and harmonic mean high in day candles
# audcad audusd 

test_model(symbol="USDJPY", iteration=3, learning_rate=0.2, depth=12, window_size=7, save_model=True) # version A passed
# test_model(symbol="USDJPY", iteration=3, learning_rate=0.23, depth=12, window_size=7) # version A


test_model(symbol="EURAUD", iteration=33, learning_rate=0.15, depth=10, window_size=8, save_model=True) # version -> A passed
# test_model(symbol="EURAUD", iteration=7, learning_rate=0.4, depth=7, window_size=8) # version passed

# test_model(symbol="EURAUD", iteration=3, learning_rate=0.15, depth=11, window_size=11) # version 

# xxtest_model(symbol="EURAUD", iteration=27, learning_rate=0.19, depth=10, window_size=8) # another version -> A
# test_model(symbol="EURAUD", iteration=27, learning_rate=0.15, depth=10, window_size=8) # another version -> A
# test_model(symbol="EURAUD", iteration=77, learning_rate=0.5, depth=9, window_size=10) # version 
# test_model(symbol="EURAUD", iteration=100, learning_rate=0.05, depth=6, window_size=9) # version 



test_model(symbol="EURNZD", iteration=11, learning_rate=0.15, depth=9, window_size=9, save_model=True) # version -> A passed
# test_model(symbol="EURNZD", iteration=9, learning_rate=0.25, depth=10, window_size=9) # version -> A

# Pair EURNZD__________________________experimental
# test_model(symbol="EURNZD", iteration=5, learning_rate=0.15, depth=10, window_size=6)   # another version
# test_model(symbol="EURNZD", iteration=150, learning_rate=0.02, depth=6, window_size=7) # another version
# test_model(symbol="EURNZD", iteration=66, learning_rate=0.51, depth=10, window_size=9) # another version
# test_model(symbol="EURNZD", iteration=300, learning_rate=0.01, depth=6, window_size=7) # another version


test_model(symbol="EURCAD", iteration=28, learning_rate=0.17, depth=7, window_size=6, alpha_type="beta", save_model=True ) # version A passed
# test_model(symbol="EURCAD", iteration=21, learning_rate=0.25, depth=8, window_size=9) # version A passed
# test_model(symbol="EURCAD", iteration=27, learning_rate=0.17, depth=7, window_size=6, alpha_type="beta" ) # version seems passed
#  Pair EURCAD___________________experimental
# test_model(symbol="EURCAD", iteration=1, learning_rate=0.10, depth=10, window_size=9) # version
# xxtest_model(symbol="EURCAD", iteration=50, learning_rate=0.05, depth=6, window_size=6, alpha_type="gamma" ) # version A gamma 
# test_model(symbol="EURCAD", iteration=13, learning_rate=0.05, depth=9, window_size=7, alpha_type="alpha" ) # version A alpha
# xx test_model(symbol="EURCAD", iteration=5, learning_rate=0.10, depth=10, window_size=9, alpha_type="gamma") # another version A

# test_model(symbol="EURCAD", iteration=22, learning_rate=0.25, depth=8, window_size=9) # version seems passed







test_model(symbol="EURUSD", iteration=210, learning_rate=0.05, depth=6, window_size=7, alpha_type="gamma", save_model=True ) #  version A passed

# test_model(symbol="EURUSD", iteration=5, learning_rate=0.05, depth=6, window_size=7, alpha_type="gamma" ) #  another version 
# test_model(symbol="EURUSD", iteration=1, learning_rate=0.25, depth=10, window_size=9, alpha_type="alpha"  ) # another version



test_model(symbol="USDCAD", iteration=8, learning_rate=0.11, depth=11, window_size=6, alpha_type="delta", save_model=True) # version A passed
# ______
# experimental
# test_model(symbol="USDCAD", iteration=3, learning_rate=0.11, depth=9, window_size=9, alpha_type="omega") # another version 
# test_model(symbol="USDCAD", iteration=1, learning_rate=0.20, depth=10, window_size=9, alpha_type="omega") # another version
# xxtest_model(symbol="USDCAD", iteration=5, learning_rate=0.5, depth=10, window_size=11, alpha_type="gamma") # another version 
# xxtest_model(symbol="USDCAD", iteration=5, learning_rate=0.15, depth=10, window_size=11, alpha_type="gamma") # version A 60 accuracy
# test_model(symbol="USDCAD", iteration=7, learning_rate=0.11, depth=11, window_size=6, alpha_type="delta") # version passed




test_model(symbol="NZDUSD", iteration=11, learning_rate=0.25, depth=10, window_size=7, alpha_type="gamma", save_model=True) # version A passed

# test_model(symbol="NZDUSD", iteration=77, learning_rate=0.21, depth=6, window_size=6, alpha_type="gamma") # version A
# test_model(symbol="NZDUSD", iteration=9, learning_rate=0.25, depth=10, window_size=7, alpha_type="gamma") # version A
# _____________
# experimental 
# test_model(symbol="NZDUSD", iteration=1, learning_rate=0.3, depth=7, window_size=7, alpha_type="alpha") # version 
# test_model(symbol="NZDUSD", iteration=3, learning_rate=0.7, depth=11, window_size=7, alpha_type="gamma") # version 
# test_model(symbol="NZDUSD", iteration=11, learning_rate=0.2, depth=10, window_size=7, alpha_type="gamma") # version  


# test_model(symbol="NZDUSD", iteration=17, learning_rate=0.29, depth=10, window_size=7, alpha_type="gamma") # another version
# test_model(symbol="NZDUSD", iteration=21, learning_rate=0.21, depth=6, window_size=7, alpha_type="gamma") # another version
# test_model(symbol="NZDUSD", iteration=3, learning_rate=0.15, depth=9, window_size=7, alpha_type="gamma") # another version



test_model(symbol="GBPUSD", iteration=19, learning_rate=0.19, depth=9, window_size=11, alpha_type="gamma", save_model=True) # another version A passed
# _________
# experimental
# test_model(symbol="GBPUSD", iteration=3,  learning_rate=0.25, depth=11, window_size=6,  alpha_type="delta") # version 
# test_model(symbol="GBPUSD", iteration=17,  learning_rate=0.25, depth=11, window_size=9,  alpha_type="delta") # another version 
# test_model(symbol="GBPUSD", iteration=5,  learning_rate=0.15, depth=10, window_size=11, alpha_type="gamma") # version A

# test_model(symbol="GBPUSD", iteration=7, learning_rate=0.19, depth=9, window_size=11, alpha_type="gamma") # another version A 
# test_model(symbol="GBPUSD", iteration=9, learning_rate=0.21, depth=9, window_size=11, alpha_type="omega") # another version A



test_model( symbol="GBPJPY", iteration=10, learning_rate=0.1, depth=10, window_size=5, alpha_type="lambda", save_model=True) # version A passed
# test_model( symbol="GBPJPY", iteration=27, learning_rate=0.27, depth=7, window_size=8, alpha_type="beta") # version A passed


# x test_model(symbol="GBPJPY", iteration=8, learning_rate=0.17, depth=11, window_size=13, alpha_type="delta") # version seems passed
# x test_model(symbol="GBPJPY", iteration=3, learning_rate=0.35, depth=11, window_size=6, alpha_type="alpha", save_model=True) # version passed close to A
# x test_model(symbol="GBPJPY", iteration=7, learning_rate=0.35, depth=11, window_size=6, alpha_type="alpha") # version A passed

# _______________
# experimental
# test_model(symbol="GBPJPY", iteration=7, learning_rate=0.15, depth=9, window_size=13, alpha_type="gamma") # another version accuracy by net gains 54-57 
# test_model(symbol="GBPJPY", iteration=2, learning_rate=0.7, depth=13, window_size=6, alpha_type="alpha") # another version 
# test_model(symbol="GBPJPY", iteration=5, learning_rate=0.3, depth=10, window_size=7, alpha_type="alpha") # another version 
# test_model(symbol="GBPJPY", iteration=5, learning_rate=0.3, depth=10, window_size=7, alpha_type="alpha") # another version 
# test_model(symbol="GBPJPY", iteration=7, learning_rate=0.6, depth=12, window_size=11, alpha_type="gamma") # another version 
# test_model(symbol="GBPJPY", iteration=3, learning_rate=0.7, depth=12, window_size=6, alpha_type="alpha") # another version close to A
# test_model(symbol="GBPJPY", iteration=3, learning_rate=0.19, depth=10, window_size=3, alpha_type="delta") # another version 




# test_model(symbol="GBPAUD", iteration=100, learning_rate=0.05, depth=6, window_size=9) # version seems passed
test_model(symbol="GBPAUD", iteration=12, learning_rate=0.07, depth=5, window_size=6,alpha_type="gamma", save_model=True) # version A passed





test_model(symbol="AUDJPY", iteration=1, learning_rate=0.2, depth=11, window_size=7, alpha_type="delta", save_model=True) # version A delta passed
# _________
# experimental
# test_model(symbol="AUDJPY", iteration=1, learning_rate=0.25, depth=11, window_size=7, alpha_type="delta") # version A delta passed
# test_model(symbol="AUDJPY", iteration=27, learning_rate=0.2, depth=10, window_size=11, alpha_type="omega") # version omega
# test_model(symbol="AUDJPY", iteration=1, learning_rate=0.2, depth=11, window_size=9, alpha_type="alpha") # version A
# test_model(symbol="AUDJPY", iteration=3, learning_rate=0.7, depth=12, window_size=6, alpha_type="gamma") # another version
# test_model(symbol="AUDJPY", iteration=1, learning_rate=0.01, depth=9, window_size=9, alpha_type="alpha") # another version
# test_model(symbol="AUDJPY", iteration=5, learning_rate=0.15, depth=13, window_size=7, alpha_type="alpha") # another version



test_model(symbol="EURJPY", iteration=5, learning_rate=0.2, depth=11, window_size=5, alpha_type="gamma", save_model=True) # version A passed
# test_model(symbol="EURJPY", iteration=5, learning_rate=0.2, depth=9, window_size=7, alpha_type="delta") # another version
# test_model(symbol="EURJPY", iteration=7, learning_rate=0.5, depth=5, window_size=11, alpha_type="gamma") # another version



# test_model(symbol="AUDNZD", iteration=13, learning_rate=0.5, depth=9, window_size=9, alpha_type="gamma") # version seems passed
# test_model(symbol="AUDNZD", iteration=5, learning_rate=0.15, depth=11, window_size=7, alpha_type="beta") # version A seems passed
test_model(symbol="AUDNZD", iteration=9, learning_rate=0.03, depth=10, window_size=6, alpha_type="beta", save_model=True) # version A passed


test_model(symbol="GBPNZD", iteration=12, learning_rate=0.25, depth=10, window_size=7, alpha_type="beta", save_model=True) # version A  passed

# test_model(symbol="GBPNZD", iteration=15, learning_rate=0.25, depth=10, window_size=7, alpha_type="beta") # version A seems passed



test_model(symbol="AUDUSD", iteration=14, learning_rate=0.21, depth=7, window_size=9, alpha_type="beta", save_model=True) # version A seems passed
# ______
# test_model(symbol="AUDUSD", iteration=33, learning_rate=0.15, depth=12, window_size=5, alpha_type="beta") # version A seems passed
# test_model(symbol="AUDUSD", iteration=3, learning_rate=0.1, depth=10, window_size=11, alpha_type="delta") # version seems passed
# test_model(symbol="AUDUSD", iteration=23, learning_rate=0.25, depth=8, window_size=7, alpha_type="gamma", save_model=True) # version A passed 


test_model(symbol="NZDJPY", iteration=11, learning_rate=0.35, depth=11, window_size=7, alpha_type="beta", save_model=True) # version  A passed




# test_model(symbol="USDCHF", iteration=3, learning_rate=0.25, depth=12, window_size=6, alpha_type="gamma") # version passed close to A
# test_model(symbol="USDCHF", iteration=17, learning_rate=0.15, depth=7, window_size=11, alpha_type="beta") # version passed
# test_model(symbol="USDCHF", iteration=7, learning_rate=0.22, depth=8, save_model=False, window_size=6, alpha_type="ha" ) # version passed

test_model(symbol="USDCHF", iteration=6, learning_rate=0.22, depth=8, window_size=6, alpha_type="ha", save_model=True) # version passed close to A

test_model(symbol="AUDCAD", iteration=23, learning_rate=0.25, depth=8, window_size=5, alpha_type="gamma", save_model=True) # # version passed A
# test_model(symbol="CHFJPY", iteration=11, learning_rate=0.75, depth=8, window_size=5, alpha_type="alpha" ) # version close to A

test_model(symbol="CHFJPY", iteration=21, learning_rate=0.75, depth=8, window_size=5, alpha_type="alpha" , save_model=True) # version A

test_model(symbol="GBPCAD", iteration=13, learning_rate=0.25, depth=8, window_size=7, alpha_type="alpha", save_model=True) # version passed A
# test_model(symbol="GBPCAD", iteration=15, learning_rate=0.25, depth=8, window_size=7, alpha_type="alpha") # version passed close to A


test_model(symbol="NZDCHF", iteration=22, learning_rate=0.77, depth=7, window_size=5, alpha_type="beta", save_model=True) # version A passed 


test_model(symbol="EURGBP", iteration=14, learning_rate=0.18, depth=9, window_size=6, alpha_type="omega", save_model=True) # version A passed
# test_model(symbol="EURGBP", iteration=15, learning_rate=0.18, depth=9, window_size=6, alpha_type="omega") # version A passed
# test_model(symbol="EURGBP", iteration=7, learning_rate=0.77, depth=8, window_size=7, alpha_type="gamma") # version close to A

# ______________________________________

test_model(symbol="CADCHF", iteration=14, learning_rate=0.20, depth=7, window_size=7, alpha_type="omega") # version A passed
# test_model(symbol="CADCHF", iteration=15, learning_rate=0.18, depth=8, window_size=6, alpha_type="beta") # version close to A 
# test_model(symbol="CADCHF", iteration=14, learning_rate=0.20, depth=7, window_size=7, alpha_type="beta") # version close to A

test_model(symbol="AUDCHF", iteration=25, learning_rate=0.25, depth=7, window_size=11, alpha_type="delta") # version A passed 
# test_model(symbol="AUDCHF", iteration=22, learning_rate=0.25, depth=7, window_size=11, alpha_type="delta") # version close to A
# test_model(symbol="AUDCHF", iteration=27, learning_rate=0.66, depth=8, window_size=6, alpha_type="alpha") # version close to A 

test_model(symbol="GBPCHF", iteration=13, learning_rate=0.42, depth=7, window_size=5, alpha_type="beta") # version A passed
# test_model(symbol="GBPCHF", iteration=13, learning_rate=0.49, depth=7, window_size=5, alpha_type="beta") # version close to A
# test_model(symbol="GBPCHF", iteration=12, learning_rate=0.42, depth=7, window_size=5, alpha_type="beta") # version 

test_model(symbol="EURCHF", iteration=20, learning_rate=0.18, depth=7, window_size=5, alpha_type="omega") # version A passed
# test_model(symbol="EURCHF", iteration=19, learning_rate=0.18, depth=7, window_size=5, alpha_type="omega") # version close to A
# test_model(symbol="EURCHF", iteration=12, learning_rate=0.44, depth=7, window_size=5, alpha_type="beta") # version close to A


test_model( symbol="CADJPY", iteration=7, learning_rate=0.38, depth=9, window_size=11, alpha_type="delta" ) # version A passed
# test_model( symbol="CADJPY", iteration=7, learning_rate=0.71, depth=12, window_size=6, alpha_type="beta" ) version close to A
# test_model( symbol="CADJPY", iteration=47, learning_rate=0.47, depth=9, window_size=5, alpha_type="beta" ) # version A passed


test_model( symbol="NZDCAD", iteration=7, learning_rate=0.68, depth=11, window_size=6, alpha_type="gamma" ) # version A passed
# test_model( symbol="NZDCAD", iteration=7, learning_rate=0.2, depth=12, window_size=9, alpha_type="omega" )
# test_model( symbol="NZDCAD", iteration=7, learning_rate=0.2, depth=12, window_size=9, alpha_type="gamma" ) # version close to A
# test_model( symbol="NZDCAD", iteration=6, learning_rate=0.35, depth=11, window_size=6, alpha_type="gamma" ) # version close to A
# test_model( symbol="NZDCAD", iteration=7, learning_rate=0.36, depth=11, window_size=6, alpha_type="gamma" ) # version close to A



# __________
# experimental 
# test_model(symbol="USDCHF", iteration=4, learning_rate=0.18, depth=8, window_size=5, alpha_type="beta") # version passed
# test_model(symbol="USDCHF", iteration=3, learning_rate=0.78, depth=13, window_size=7, alpha_type="omega") # version passed close to A
# test_model(symbol="USDCHF", iteration=3, learning_rate=0.78, depth=11, window_size=5, alpha_type="omega") # version passed close to A
# test_model(symbol="USDCHF", iteration=7, learning_rate=0.15, depth=8, window_size=7, alpha_type="ha") # version passed close to A
# test_model(symbol="USDCHF", iteration=3, learning_rate=0.58, depth=11, window_size=5, alpha_type="omega") # version passed 




# Day Features Settings Adjusted 

hl2 = ( data['High'] + data['Low'] ) / 2
    
rsi  = calculate_rsi(hl2, period=10)
smoothed_rsi =  rsi.rolling(window=3).mean()
slow_smoothed_rsi =  rsi.rolling(window=10).mean()

data['rsi_sma_fast'] = smoothed_rsi
data['rsi_sma_slow'] = slow_smoothed_rsi

data['rsi'] =  rsi
data['rsi_crossover_fast'] =  rsi_crossover(smoothed_rsi)
data['rsi_crossover_slow'] =  rsi_crossover_with_sma(rsi , slow_smoothed_rsi)


data['STDEV'] = data['Close'].rolling(window=5).std()
data['Upper_Band'] = data['Close'].rolling(window=5).mean() + (data['Close'].rolling(window=5).std() * 2)
data['Lower_Band'] = data['Close'].rolling(window=5).mean() - (data['Close'].rolling(window=5).std() * 2)

data['candle_type'] = candle_type_value

data['slow_harmonic_mean'] = calculate_harmonic_mean(data['Close'], period=9)
data['fast_harmonic_mean'] = calculate_harmonic_mean(data['Close'], period=3)

data['harmonic_mean_high'] = calculate_hm_high(high)
# data['harmonic_mean_high'] = harmonic_mean_high(high)
# print("New Harmonic Mean",calculate_hm_high(high)[:-20])
# print("Old Harmonic Mean",data['harmonic_mean_high'][:-20])

data['harmonic_mean_low']  = calculate_hm_low(low)





# h4 lag by 3 fixed
features_h4[f'stdev_slow_{candles}_T-{day}'] = close_val.rolling(window=18).std()
features_h4[f'stdev_fast_{candles}_T-{day}'] = close_val.rolling(window=9).std()
features_h4[f'slow_harmonic_mean_{candles}_T-{day}'] = calculate_harmonic_mean(close_value , period=27)
features_h4[f'fast_harmonic_mean_{candles}_T-{day}'] = calculate_harmonic_mean(close_value, period=9)

highest_high = features_h4[f'High_h4_{candles}_T-{day}'].rolling(window=9).max()
lowest_low   = features_h4[f'Low_h4_{candles}_T-{day}'].rolling(window=9).min()

            # Calculate the relative range
features_h4[f'relative_range_h4_{candles}_T-{day}'] = close_value - ( ( highest_high + lowest_low ) / 2)
supertrend, supertrend_status_value = supertrend_indicator(high_value, low_value, close_value, period= 27, multiplier=5.5)
supertrend_crossover = supertrend_status_crossover(supertrend_status_value)

features_h4[f'supertrend_h4_{candles}_T-{day}'] = supertrend

features_h4[f'supertrend_status_h4_{candles}_T-{day}'] = supertrend_status_value

features_h4[f'supertrend_crossover_h4_{candles}_T-{day}'] = supertrend_crossover
