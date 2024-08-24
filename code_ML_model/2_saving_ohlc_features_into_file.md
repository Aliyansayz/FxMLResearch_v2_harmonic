## Following code for saving Daily OHLC Features of 28 Fx Datasets
      
      import pandas as pd
      import numpy as np
      
      
      forex_pairs = [
          'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
          'CADCHF', 'CADJPY',
          'CHFJPY', 
          'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 
          'EURJPY', 'EURNZD', 'EURUSD',
          'GBPAUD', 'GBPCAD', 'GBPCHF', 
          'GBPJPY', 'GBPUSD', 'GBPNZD',
          'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',  
          'USDCHF', 'USDCAD', 'USDJPY'
              ]
      
      
      currency_ohlc = []
      count = 0
      for pair in forex_pairs:
          daily_path = f'currency_data/{pair}_Daily.csv'
          data = return_df_day(daily_path)
          data = add_features(data)
          data = make_features_lagged( lag_by= 14)
      
          currency_df = { "day_features": data, "symbol": f"{pair}"  }
          currency_ohlc.append(currency_df)
          count += 1
          print(count)
          
      
      
      import pickle
      
      with open('day_features_data_lagby_14_v2.bin', 'wb') as file:
          pickle.dump(currency_ohlc, file)
          
      
      
      
      
