# FxMLResearch_v2_harmonic


Addition Of Harmonic Mean Of the Low (the Candle difference of previous day low price and present day low price) and also Harmonic Mean Difference Of High (the Candle difference of present day high price and previous day high price)

      # in this way:- 
      harmonic mean low : 
      difference = prev_low - current low  # rolling over last 45 days 
      hm_low  = last_45_days_diff --> mean harmonic value 
      
      
      harmonic mean high:
      difference= current high - prev_high # rolling over last 45 days
      hm_high  = last_45_days_diff --> mean harmonic value 
      


Such that when there is an uptrend :-
📊 present day low becomes less lower and more higher usually
📊 present day high becomes higher and more higher usually

*  ✔️ harmonic mean low experiences a little drop due to negative values inclusion due to prev_low - current_low but `prev_low < current low`
*  ✔️ harmonic mean high experiences a rise due to positive values inclusion due to current high - prev_low as `current high > prev_high `


Addition Of Relative Strength Index Crossover

Addition Of Hour 4 Supertrend and it's crossover With large factor to only diverge when large price swing is confirmed that can be pivotal in predicting next day candle or next 6 to 18 candles of hour 4.  


Elimination of seasonal features : month, week of month, day of week

