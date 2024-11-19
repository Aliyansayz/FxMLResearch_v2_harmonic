# FxMLResearch_v2_harmonic

## Indicators, ML Library and Scaling Method
* Catboost Regressor Model
* Sklearn Robust Scaler
* Target Variable Price Change In Pips (forth decimal or second decimal place in JPY)
* Rather than price prediction we predict how much price change will be caused by next day price, if negative we sell positive then buy 
* Following features used (last 5 to 12 Days) (last 3 days Hour 4 features) :
* Relative Strength Index, Simple Moving Average of RSI,
* Candle Type 
* Harmonic Mean on `Close Price` with small and large period values,
* `High` difference of two days harmonic mean also for `Low`, difference for last 45 days
* Hour 4 RSI,
* Hour 4 Supertrend 5.5x Factor To Point High Swing In Price Visible in Daily Timeframe too
* Hour 4 Candle Type
* Hour 4 Relative Range
* Hour 4 Harmonic Mean on `Close Price` with small and large period values,



Significance Of Harmonic Mean is data drift tackling within 45 days by `Highs` and `Lows` peaks made by prices change. 

Addition Of Harmonic Mean Of the Low (the Candle difference of previous day low price and present day low price) and also Harmonic Mean Difference Of High (the Candle difference of present day high price and previous day high price)

      # in this way:- 
      harmonic mean low : 
      difference = prev_low - current low  # rolling over last 45 days 
      hm_low  = last_45_days_diff --> mean harmonic value 
      
      
      harmonic mean high:
      difference= current high - prev_high # rolling over last 45 days
      hm_high  = last_45_days_diff --> mean harmonic value 
      



Addition Of Relative Strength Index Crossover:-

Addition Of Hour 4 Supertrend and it's crossover With large factor to only diverge when large price swing is confirmed that can be pivotal in predicting next day candle or next 6 to 18 candles of hour 4.  


Elimination of seasonal features : month, week of month, day of week from day features

