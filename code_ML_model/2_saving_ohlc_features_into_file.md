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
          
      
      

## Following code for saving Hour-4 OHLC Features of 28 Fx Datasets


      forex_pairs = [ 'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
    'CADCHF', 'CADJPY',
    'CHFJPY', 
    'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 
    'EURJPY', 'EURNZD', 'EURUSD',
    'GBPAUD', 'GBPCAD', 'GBPCHF', 
    'GBPJPY', 'GBPUSD', 'GBPNZD',
    'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',  
    'USDCHF', 'USDCAD', 'USDJPY' ]
      
      
      currency_ohlc = []
      count = 0
      for pair in forex_pairs:
          # daily_path = f'currency_data/{pair[:6]}_Daily.csv'
          hour4_path = f'currency_data/{pair}_H4.csv'
      
          data_h4     = rename_h4_df(hour4_path)
      
          reshaped_h4 = combine_ohlc_into_single_day(data_h4)
      
          features_h4 = add_ohlc_in_lagged(reshaped_h4,  lag_by= 3 )
      
          features_h4 = add_features(features_h4,  lag_by= 3)
          
          currency_df = { "hour4_features": features_h4, "symbol": f"{pair}"  }
          
          currency_ohlc.append(currency_df)
          count += 1
          print(count)
      
      print(count)
      
      
      
      
      import pickle
      
      with open('hour4_features_data_lag_by_3_v2.bin', 'wb') as file:
          pickle.dump(currency_ohlc, file)
          
      
      # import pickle
      
      # with open('hour4_less_features_data_lag_by_3.bin', 'wb') as file:
      #     pickle.dump(currency_ohlc, file)
          
      
          
          
          
