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
      


Such that when there is an uptrend :-
*  📊 present day low becomes higher and more higher usually
*  📊 present day high becomes higher and more higher usually

*  ✔️ harmonic mean low experiences a little drop due to negative values inclusion as prev_low - current_low `prev_low < current low` as current low value greater than previous low value causing negative difference we subtract `prev_low - current_low` small number from a larger number .
  
*  ✔️ harmonic mean high experiences a rise due to positive values inclusion as current high - prev_high as `current high > prev_high ` as current high value greater than previous high value causing positive difference we subtract `current high - prev_high` large number from a smaller number .


Such that when there is an downtrend :-
*  📊 present day low becomes lower and more lower usually
*  📊 present day high becomes lower and more lower usually

*  ✔️ harmonic mean low experiences a little rise due to positive values inclusion as prev_low - current_low `prev_low > current low` as current low value smaller than previous low value causing positive difference we subtract large number `prev_low - current_low` from a smaller number .
  
*  ✔️ harmonic mean high experiences a little drop due to negative values inclusion as current high - prev_high `current high < prev_high ` as current high value smaller than previous high value causing negative difference we subtract `current high - prev_high` small number from a larger number .

Addition Of Relative Strength Index Crossover:-

Addition Of Hour 4 Supertrend and it's crossover With large factor to only diverge when large price swing is confirmed that can be pivotal in predicting next day candle or next 6 to 18 candles of hour 4.  


Elimination of seasonal features : month, week of month, day of week from day features

