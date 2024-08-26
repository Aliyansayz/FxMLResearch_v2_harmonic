      
      day_features, hour4_features = load_features_files()
      
      
      # month May
      month_key = {'8': 'oct', '7': 'nov', '6':'dec', '5': 'jan', '4':'feb', '3':'mar', '2':'april', '1':'may' }
      
      step = 1 # ____May 
      # step = 2 # April
      # step = 3 # March
      # step = 4 # February
      # step = 5 # January
      
      # step = 6 # December
      # step = 7 # November
      # step = 8 # October
      cluster = 22
      
      info = each_day_gain_loss(day_features, hour4_features, custom_sample= [step, cluster] , model_path = None)
      
      
      
      
      
      
      
      import pandas as pd
      import numpy as np
      import pickle
      from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
      from sklearn.model_selection import train_test_split
      # from sklearn.tree import DecisionTreeClassifier
      from sklearn.metrics import classification_report
      from catboost import CatBoostClassifier
      from catboost import Pool
      from catboost import CatBoostRegressor
      from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
      from sklearn.model_selection import GridSearchCV
      from sklearn.model_selection import train_test_split, RandomizedSearchCV
      from sklearn.ensemble import RandomForestRegressor
      from sklearn.metrics import mean_squared_error
      from scipy.stats import randint
      
      
      
      def test_model(symbol, iteration, learning_rate, depth, window_size=9, alpha_type = "gamma", save_model=False ):
      
          day_data, hour4_data = get_day_hour4_features(symbol, day_features, hour4_features)
      
          X_train_scaled, X_test_scaled, y_train, y_test, X_test = get_features_transformed(symbol, day_data, hour4_data,  X_test_re=True,  window_size=window_size, alpha_type=alpha_type ) 
      
          print(symbol)
          # model = load_model(symbol)
      
      
          # symbol_parameters = load_parameters(symbol)
          # iteration, learning_rate, depth = symbol_parameters['Iterations'], symbol_parameters['Lr'], symbol_parameters['depth']
      
      
          # iteration, learning_rate, depth = 15, 0.68, 6 # AUDNZD
      
          # iteration, learning_rate, depth = 210, 0.19, 7 # AUDCAD
      
          # iteration, learning_rate, depth = 17, 0.9, 7 # AUDCAD
      
          parameters = [ iteration, learning_rate, depth ]
          
          # model = finetune_model(X_train_scaled, X_test_scaled, y_train, y_test, parameters=parameters)
          model = train_model(X_train_scaled, X_test_scaled, y_train, y_test, parameters=parameters)
          if save_model == True : save_model_v2(symbol, model, parameters, alpha_type, window_size)
          # if save_model == True :     
          #     save_model(symbol, model)
      
              
      
          print("MAY")
      
          step = 1 # ____          May 
          # step = 2 # ____       April
          cluster = 22 # possible days of trading in a year data ends at 31 May
          gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  
          net_gains = gains['net_gains']
      
          accuracy_by_net_gains = gains['accuracy_by_net_gains']
          accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']
      
          print('net profit ', net_gains)
          print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)
          print("accuracy on all test points excluding last 22 points ", accuracy )
      
          print("APRIL")
          step = 2 # ____       April
          cluster = 22
          # cluster = 22 # possible days of trading in a year data ends at 31 May
          gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  
          net_gains = gains['net_gains']
      
          accuracy_by_net_gains = gains['accuracy_by_net_gains']
      
          accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']
      
          print('net profit ', net_gains)
          print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)
          print("accuracy on all test points excluding last 22 points ", accuracy )
      
      
      
          print("MARCH")
          step = 3 # ____      March
          cluster = 22 # possible days of trading in a year data ends at 31 May
          gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  
          net_gains = gains['net_gains']
      
          accuracy_by_net_gains = gains['accuracy_by_net_gains']
      
          accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']
      
          print('net profit ', net_gains)
          print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)
          print("accuracy on all test points excluding last 22 points ", accuracy )
          
          print("Iterations: ",iteration)
          print("Learning rate: ",learning_rate) 
          print("Depth: ",depth)
          
          
          print("FEB")
          step = 4 # ____      Feb
          cluster = 22 # possible days of trading in a year data ends at 31 May
          gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  
          net_gains = gains['net_gains']
      
          accuracy_by_net_gains = gains['accuracy_by_net_gains']
          accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']
      
          print('net profit ', net_gains)
          print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)
          print("accuracy on all test points excluding last 22 points ", accuracy )
          
          print("JAN")
          step = 5 # ____      Jan
          cluster = 22 # possible days of trading in a year data ends at 31 May
          gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  
          net_gains = gains['net_gains']
      
          accuracy_by_net_gains = gains['accuracy_by_net_gains']
          accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']
      
          print('net profit ', net_gains)
          print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)
          print("accuracy on all test points excluding last 22 points ", accuracy )
      
          print("DEC")
          step = 6 # ____      Jan
          cluster = 22 # possible days of trading in a year data ends at 31 May
          gains, accuracy = evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= [step, cluster], X_test=X_test ) # custom_sample= None, last_cluster = None  
          net_gains = gains['net_gains']
      
          accuracy_by_net_gains = gains['accuracy_by_net_gains']
          accuracy_by_net_gains_before = gains['accuracy_by_net_gains_before']
      
          print('net profit ', net_gains)
          print('\naccuracy by net gains on previous test data ', accuracy_by_net_gains_before)
          print("accuracy on all test points excluding last 22 points ", accuracy )
          
      
      
      
      def get_features_transformed(symbol, day_data, hour4_data, date_resume = '2020-06-01', X_test_re = False, temporal=False, window_size = 9, alpha_type= "gamma"):
          
          
          # date_resume
          # split_date = pd.to_datetime('2020-06-01') 
          
          split_date = pd.to_datetime(date_resume)
          
          day_data  = day_data.loc[split_date:]
          
          hour4_data.index =  pd.to_datetime(hour4_data.index)
      
          hour4_data = hour4_data.loc[split_date:]
      
      
          day_data['pips_change'] = np.subtract( day_data['Close'].to_numpy(), day_data['Close'].shift(1).to_numpy() )
          
          if 'JPY' in symbol: pips_change = day_data['pips_change'] * 10 ** 2 
          
          
          else :  pips_change = day_data['pips_change'] * 10 ** 4 
          
          day_data['Target'] = pips_change
          
          
          # features =  [f'candle_type_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'rsi_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'Price_Range_T-{i}' for i in range(1, window_size + 1)  ] +\
          #            [f'rsi_crossover_slow_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'rsi_crossover_fast_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'harmonic_mean_low_T-{i}' for i in range(1, window_size + 1)  ] +\
          #            [f'harmonic_mean_high_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'slow_harmonic_mean_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'fast_harmonic_mean_T-{i}' for i in range(1, window_size + 1) ]
                   
          # features = [f'Day_of_Week_T-{i}' for i in range(1, window_size + 1)] + [f'Week_of_Month_T-{i}' for i in range(1, window_size + 1)] +\
          #            [f'Month_T-{i}' for i in range(1, window_size + 1)]  +\
          #            [f'High_T-{i}' for i in range(1, window_size + 1) ] +  [f'Low_T-{i}'  for i in range(1, window_size + 1) ] + \
          #            [f'STDEV_T-{i}' for i in range(1, window_size + 1) ] + \
          #            [f'Upper_Band_T-{i}' for i in range(1, window_size + 1) ] + \
          #            [f'Lower_Band_T-{i}' for i in range(1, window_size + 1) ]+ \
          #            [f'candle_type_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'rsi_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'Price_Range_T-{i}' for i in range(1, window_size + 1)  ] +\
          #            [f'Median_Price_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'rsi_crossover_slow_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'rsi_crossover_fast_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'harmonic_mean_low_T-{i}' for i in range(1, window_size + 1)  ] +\
          #            [f'harmonic_mean_high_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'slow_harmonic_mean_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'fast_harmonic_mean_T-{i}' for i in range(1, window_size + 1) ] 
                   
                   
          # features = [f'Day_of_Week_T-{i}' for i in range(1, window_size + 1)] + [f'Week_of_Month_T-{i}' for i in range(1, window_size + 1)] +\
          #            [f'Month_T-{i}' for i in range(1, window_size + 1)]  +\
          #            [f'High_T-{i}' for i in range(1, window_size + 1) ] +  [f'Low_T-{i}'  for i in range(1, window_size + 1) ] + \
          #            [f'STDEV_T-{i}' for i in range(1, window_size + 1) ] + \
          #            [f'RSI_T-{i}' for i in range(1, window_size + 1)   ] + \
          #            [f'Upper_Band_T-{i}' for i in range(1, window_size + 1) ] + \
          #            [f'Lower_Band_T-{i}' for i in range(1, window_size + 1) ]+ \
          #            [f'candle_type_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'heikin_ashi_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'Price_Range_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'Median_Price_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'ema_3_T-{i}' for i in range(1, window_size + 1) ]  +\
          #            [f'ema_5_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'ema_7_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'ema_14_T-{i}'for i in range(1, window_size + 1) ] +\
          #            [f'ema_difference_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'supertrend_status_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'supertrend_crossover_T-{i}' for i in range(1, window_size + 1) ] + \
          #            [f'supertrend_T-{i}' for i in range(1, window_size + 1) ] +\
          #            [f'elastic_supertrend_T-{i}' for i in range(1, window_size + 1) ] + \
          #            [f'elastic_supertrend_cross_T-{i}' for i in range(1, window_size + 1) ] + \
          #            [f'elastic_supertrend_status_T-{i}' for i in range(1, window_size + 1) ]
      
          if temporal == False : 
              # [f'Day_of_Week_T-{i}' for i in range(1, window_size + 1)]  +
              features =  [f'candle_type_T-{i}' for i in range(1, window_size + 1) ] +\
                     [f'rsi_T-{i}' for i in range(1, window_size + 1) ] +\
                     [f'Price_Range_T-{i}' for i in range(1, window_size + 1)  ] +\
                     [f'rsi_crossover_slow_T-{i}' for i in range(1, window_size + 1) ] +\
                     [f'rsi_crossover_fast_T-{i}' for i in range(1, window_size + 1) ] +\
                     [f'harmonic_mean_low_T-{i}' for i in range(1, window_size + 1)  ] +\
                     [f'harmonic_mean_high_T-{i}' for i in range(1, window_size + 1) ] +\
                     [f'slow_harmonic_mean_T-{i}' for i in range(1, window_size + 1) ] +\
                     [f'fast_harmonic_mean_T-{i}' for i in range(1, window_size + 1) ] 
              
          day_features_X   =  day_data[features]
      
          day_features_X = day_features_X.apply(pd.to_numeric, errors='coerce')
      
          day_features_X = day_features_X.astype(float)
      
          hour4_features_X = hour4_data
      
          hour4_features_X = hour4_features_X.apply(pd.to_numeric, errors='coerce')
      
          hour4_features_X = hour4_features_X.astype(float)
          
          # substrings = ['relative_range', 'candle_type', 'heikin_ashi', 'stdev', 'true' ] #, 'heikin_ashi',  'supertrend_h4']
      
          # substrings = ['relative_range', 'candle_type', 'supertrend_h4' ] #, 'heikin_ashi',  'supertrend_h4']
          # substrings = [ 'supertrend_h4', 'candle_type', 'crossover', 'slow_harmonic', 'fast_harmonic' ]
          # substrings = [ 'supertrend_h4' ]
          
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
          
          elif alpha_type == "ha":
              substrings = [ 'stdev_slow', 'heikin_ashi', 'slow_harmonic_mean'  ] # 'slow_harmonic_mean'
          
          elif alpha_type == "rsi":
              substrings = [ 'rsi_sma_slow', 'candle_type', 'slow_harmonic_mean', 'supertrend_h4' ] # 'slow_harmonic_mean' ]
              # features += [f'High_T-{i}' for i in range(1, window_size + 1) ] +
              # [f'High_T-{i}' for i in range(1, window_size + 1) ] +  [f'Low_T-{i}'  for i in range(1, window_size + 1) ]
          elif alpha_type == "lambda":
              substrings = ['relative_range', 'candle_type', 'heikin_ashi' ]
      
          # substrings = [ 'supertrend_h4', 'candle_type' ]
          # substrings = [ 'supertrend_h4' ] 
          # # Filter columns that contain any of the specified substrings
          filtered_columns = [col for col in hour4_features_X.columns if any(sub in col for sub in substrings)]
          hour4_features_X = hour4_features_X[filtered_columns]
          # hour4_features_X = hour4_features_X.filter(like='relative_range')
      
          X = day_features_X.join(hour4_features_X)
          # X = day_features_X
          
          # if temporal == False :  X = day_features_X
          
          # X = day_features_X.join(hour4_features_X)
      
          y = day_data['Target']
      
          X.fillna(0.0, inplace=True)
      
          y.fillna(0.0, inplace=True)
      
          # print(len(data[features]))
          X = X.astype(float)
      
          div = int(len(X) * 0.8)
          # train = :div
          # test  = div:
          X_train = X[:div]
          X_test  = X[div:]
      
          y_train = y[:div]
          y_test  = y[div:]
      
          # # Apply RobustScaler
          scaler = RobustScaler()
          
          # scaler = StandardScaler()
      
          X_train_scaled = scaler.fit_transform(X_train)
          # X_test_scaled = scaler.transform(X_test)
          X_test_scaled = scaler.fit_transform(X_test)
          
          if not X_test_re:
              return X_train_scaled, X_test_scaled, y_train, y_test
          else :
              return X_train_scaled, X_test_scaled, y_train, y_test, X_test
      
      def train_model_2(X_train_scaled, X_test_scaled, y_train, y_test ):
          
          # # Convert to Pool (CatBoost-specific data structure)
          train_pool = Pool(X_train_scaled, y_train)
          test_pool = Pool(X_test_scaled, y_test)
          
          
          np.random.seed(42)
      
      
          # Initialize CatBoost regressor
          model = CatBoostRegressor(iterations=300, early_stopping_rounds=75, 
                                    learning_rate=0.1 , depth=5, verbose=100)
      
          # Train the model using train pool
          model.fit(train_pool)
      
      
          # Make predictions using test pool
          y_pred = model.predict(X_test_scaled)
      
          # Evaluate the model
          mse = mean_squared_error(y_test, y_pred)
          
          print(f'Mean Squared Error: {mse}')
          
          return model
      
      def finetune_model(X_train_scaled, X_test_scaled, y_train, y_test, parameters = None ):
      
      #     model = CatBoostRegressor()
      #     parameters = {'depth' : [6,8,10],'learning_rate' : [0.01, 0.05, 0.1],
      #                   'iterations'    : [30, 50, 100]}
      
      #     model = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)
      #     model.fit(X_train_scaled, y_train)
          
      #     # Make predictions using test pool
      #     y_pred = model.predict(X_test_scaled)
      
      #     # Evaluate the model
      #     mse = mean_squared_error(y_test, y_pred)
          
      #     print(f'Mean Squared Error: {mse}')
      
              # Create the model
          rf_regressor = RandomForestRegressor(random_state=42)
      
          # Define the parameter distribution
          param_dist = {
              'n_estimators': randint(50, 200),
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': [None, 10, 20, 30],
              'min_samples_split': randint(2, 11),
              'min_samples_leaf': randint(1, 5),
              'bootstrap': [True, False]
          }
      
          # Create the RandomizedSearchCV object
          model = RandomizedSearchCV(estimator=rf_regressor, param_distributions=param_dist,
                                             n_iter=100, cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error',
                                             random_state=42)
      
          # Fit the random search to the data
          model.fit(X_train_scaled, y_train)
      
          # Get the best parameters and best score
          best_params = random_search.best_params_
          best_score  = random_search.best_score_
      
          # Make predictions using test pool
          y_pred = model.predict(X_test_scaled)
      
          # Evaluate the model
          mse = mean_squared_error(y_test, y_pred)
          
          print(f'Mean Squared Error: {mse}')
          
          print("Best Parameters: ",best_params)
          print("Best Score: ",best_score)
          
          return model
          
          
      def train_model(X_train_scaled, X_test_scaled, y_train, y_test, parameters = None ):
          
          # train_model_1
          # # Convert to Pool (CatBoost-specific data structure)
          train_pool = Pool(X_train_scaled, y_train)
          test_pool = Pool(X_test_scaled, y_test)
          
          
          # np.random.seed(42)
      
      
          # Initialize CatBoost regressor
          if not parameters:
              model = CatBoostRegressor(iterations=300, early_stopping_rounds=75, 
                                    learning_rate=0.01 , depth=6, verbose=100)
          
          else : 
              iteration, learning_rate, depth = parameters[0], parameters[1], parameters[2]
              model = CatBoostRegressor(iterations=iteration, early_stopping_rounds=75, 
                                    learning_rate=learning_rate, depth=depth, verbose=100)
              
          # Train the model using train pool
          model.fit(train_pool)
      
      
          # Make predictions using test pool
          y_pred = model.predict(X_test_scaled)
      
          # Evaluate the model
          mse = mean_squared_error(y_test, y_pred)
          
          print(f'Mean Squared Error: {mse}')
          
          return model
      
          
          
      def train_model_1(X_train_scaled, X_test_scaled, y_train, y_test ):
          
          # train_model_1
          # # Convert to Pool (CatBoost-specific data structure)
          train_pool = Pool(X_train_scaled, y_train)
          test_pool = Pool(X_test_scaled, y_test)
          
          
          np.random.seed(42)
      
      
          # Initialize CatBoost regressor
          model = CatBoostRegressor(iterations=300, early_stopping_rounds=75, 
                                    learning_rate=0.01 , depth=6, verbose=100)
      
          # Train the model using train pool
          model.fit(train_pool)
      
      
          # Make predictions using test pool
          y_pred = model.predict(X_test_scaled)
      
          # Evaluate the model
          mse = mean_squared_error(y_test, y_pred)
          
          print(f'Mean Squared Error: {mse}')
          
          return model
      
      
      def load_features_files(hour4_lag_3=None):
      
          day_features_path = 'day_features_data_lagby_14_v2.bin'
          
          with open(day_features_path, 'rb') as file :
      
              day_features = pickle.load(file)
      
      
          # hour4_features_path = 'hour4_features_data.bin'  
          # hour4_features_path = 'hour4_features_data_lag_by_7.bin'
          hour4_features_path = 'hour4_features_data_lag_by_3_v2.bin'
          
          if hour4_lag_3 == True : hour4_features_path = 'hour4_features_data_lag_by_3_v2.bin'
      
          with open(hour4_features_path, 'rb') as file :
      
              hour4_features = pickle.load(file)
          
          return  day_features, hour4_features
          
      
      def evaluate_model_old(symbol, model, X_test_scaled,  y_test):
          
          sample = 0
          right  = 0
          wrong  = 0
          status = []
          pips_profit = 0 
          pips_loss = 0
      
          for point in range(1, 22 + 1 ):
      
              sample += 1
              y_pred = model.predict(X_test_scaled[-point])
              y_true = y_test[-point]
      
              if y_pred < 0.0 and y_true < 0.0 :
                  pips_profit += abs(y_true)
                  right += 1
                  status.append(1)
      
              elif y_pred > 0.0 and y_true > 0.0 :
                  pips_profit += abs(y_true)
                  right += 1
                  status.append(1)
      
              else:
                  pips_loss += abs(y_true)
                  wrong += 1
                  status.append(0)
      
          # y_pred = model.predict(test_pool )
      
              # print("actual : ",y_test[-3],"\npredicted : ",y_pred )
          print(f"{symbol}\n")
          print("accuracy on last 22 test points Month May 2024 with low learning rate 2020 dataset 80% training dataset is ",right/sample * 100 )
      
          # print("accuracy on last 100 test points is ",right) #wrong/sample * 100 )
          print(status)
          print(f"Pips On Profit Side was : ",pips_profit)
          print("Pips On Loss Side was : ",pips_loss)
      
          print("net pips profit : ", pips_profit-pips_loss )
          
          
          status = []
          for point in range(1, len(X_test_scaled) + 1 ):
      
              sample += 1
              y_pred = model.predict(X_test_scaled[-point])
              y_true = y_test[-point]
      
              if y_pred < 0.0 and y_true < 0.0 :
                  # pips_profit += abs(y_true)
                  right += 1
                  status.append(1)
      
              elif y_pred > 0.0 and y_true > 0.0 :
                  # pips_profit += abs(y_true)
                  right += 1
                  status.append(1)
      
              else:
                  # pips_loss += abs(y_true)
                  wrong += 1
                  status.append(0)
      
          print("accuracy on all test points with low learning rate 2020 dataset 80% training dataset is ",right/sample * 100 )
      
          # print("accuracy on last 100 test points is ",right) #wrong/sample * 100 )
          print(status)
          
          
          print(f"{symbol}\n==============================")
      
      
      
      
      
      def save_model_v2(symbol, model, parameters, alpha_type, window_size):
          
          import pickle
          
          # parameters = [ iteration, learning_rate, depth ]
          iterations = parameters[0]
          learning_rate = parameters[1] 
          depth = parameters[2]
          model_file = {"model": model ,  "model_info": { "learning_rate": learning_rate, "depth": depth, 
                        "iterations": iterations, "alpha_type": alpha_type, "window_size": window_size,
                        "scaling": "robust_scaling", "version": "v2" }, "symbol": symbol+"_pips_change" }
          with open(f"v2_forex_models/{symbol}_model_pips_change", 'wb') as file:
              
              pickle.dump(model_file, file)
          print("saved successfully")
              
          # model.save_model(f"v2_forex_models/{symbol}_model_pips_change")
      
          
          
      def save_model(symbol, model):
          
          model.save_model(f"v2_forex_models/{symbol}_model_pips_change")
      
          
          
      def load_features_files():
      
          day_features_path = 'day_features_data_lagby_14_v2.bin' 
      
      
          with open(day_features_path, 'rb') as file :
      
              day_features = pickle.load(file)
      
      
          # hour4_features_path = 'hour4_features_data.bin'  
          hour4_features_path = 'hour4_features_data_lag_by_3_v2.bin'
      
          with open(hour4_features_path, 'rb') as file :
      
              hour4_features = pickle.load(file)
              
          return  day_features, hour4_features
          
          
      def get_day_hour4_features(symbol, day_features, hour4_features):
          
          
          
          for day_pair in day_features: 
              if  day_pair['symbol'] == symbol: 
      
                  day_data = day_pair['day_features'] 
                  break
      
      
          for hour4_pair in hour4_features: 
              if  hour4_pair['symbol'] == symbol: 
      
                  hour4_data = hour4_pair['hour4_features'] 
                  break
          
          return  day_data, hour4_data
      
      def load_model(symbol, path = None):
          
          
          if path: 
              pass
              path = f"{path}/{symbol}_model_pips_change"
              
          else: 
              path = f"v2_forex_models/{symbol}_model_pips_change"
          
          model = CatBoostRegressor()
          
          model.load_model(path)
          return model
      
      import pickle
      
      
      def load_model_v2(symbol, path= None):
      
          if path:
              pass
              path = f"{path}/{symbol}_model_pips_change"
      
          else:
              path = f"v2_forex_models/{symbol}_model_pips_change"
      
          
          with open(path, 'rb') as file :
      
              model_file = pickle.load(file)
      
          # model = CatBoostRegressor()
          
          model = model_file["model"]
          model_info = model_file["model_info"]
      
          return model, model_info
      
      # {"model": model ,  "model_info": { "learning_rate": learning_rate, "depth": depth, 
      #   "iterations": iterations, "alpha_type": alpha_type, "window_size": window_size,
      #   "scaling": "robust_scaling", "version": "v2" }, "symbol": symbol+"_pips_change" }
       
              
      # def load_model_old(symbol):
          
      #     pass
      #     model = CatBoostRegressor()
      #     path = f'forex_models_low_lr_few_h4features_300 iteration/{symbol}_model_pips_change'
      #     model.load_model(path)
      #     return model
      
      
      def evaluate_model(symbol, model, X_test_scaled, y_test):
          
          sample = 0
          right  = 0
          wrong  = 0
          status = []
          pips_profit = 0 
          pips_loss = 0
      
          for point in range(23, 45 + 1 ): # may 1 - 22 # april 23 - 55
      
              sample += 1
              y_pred = model.predict(X_test_scaled[-point])
              y_true = y_test[-point]
      
              if y_pred < 0.0 and y_true < 0.0 :
                  pips_profit += abs(y_true)
                  right += 1
                  status.append(1)
      
              elif y_pred > 0.0 and y_true > 0.0 :
                  pips_profit += abs(y_true)
                  right += 1
                  status.append(1)
      
              else:
                  pips_loss += abs(y_true)
                  wrong += 1
                  status.append(0)
      
          # y_pred = model.predict(test_pool )
      
              # print("actual : ",y_test[-3],"\npredicted : ",y_pred )
          print(f"{symbol}\n")
          print("accuracy on last 22 test points Month May 2024 with low learning rate 2020 dataset 80% training dataset is ",right/sample * 100 )
      
          # print("accuracy on last 100 test points is ",right) #wrong/sample * 100 )
          print(status)
          print(f"Pips On Profit Side was : ",pips_profit)
          print("Pips On Loss Side was : ",pips_loss)
      
          print("net pips profit : ", pips_profit-pips_loss )
          
          status = []
          for point in range(1, len(X_test_scaled) + 1 ):
      
              sample += 1
              y_pred = model.predict(X_test_scaled[-point])
              y_true = y_test[-point]
      
              if y_pred < 0.0 and y_true < 0.0 :
                  # pips_profit += abs(y_true)
                  right += 1
                  status.append(1)
      
              elif y_pred > 0.0 and y_true > 0.0 :
                  # pips_profit += abs(y_true)
                  right += 1
                  status.append(1)
      
              else:
                  # pips_loss += abs(y_true)
                  wrong += 1
                  status.append(0)
      
          print("accuracy on all test points excluding last 22 points with low learning rate 2020 dataset 80% training dataset is ",right/sample * 100 )
      
          # print("accuracy on last 100 test points is ",right) #wrong/sample * 100 )
          print(status)
          
          
          print(f"{symbol}\n==============================")
          # try:
          #     return  pips_profit-pips_loss, right/sample * 100
          # except : pass
      
          # net_gains , accuracy
          
      def evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, custom_sample= None, X_test = None ):
          
          sample = 0
          right  = 0
          wrong  = 0
          status = []
          pips_profit = 0
          pips_loss = 0
          
          # custom_sample=[1,22]
          if custom_sample:
              step, cluster = custom_sample[0] , custom_sample[1] 
              start , end = 0 , 0
              # for i in range(step):
              #     start += 1 + end # 23
              #     end   += cluster      # 
              
              for i in range(step): # 1-> 22 + 1 # 2
                  end   += cluster
              end += 1
              start = end - cluster
              
          else : start, end = 1, 22
          
      #     # Convert net gains to numpy array
      # returns = np.array(net_gains)
      
      # # Calculate mean return
      # mean_return = np.mean(returns)
      
      # # Calculate standard deviation of returns
      # std_return = np.std(returns)
      
      # # Assuming risk-free rate is 0 for simplicity
      # risk_free_rate = 0
      
      # # Calculate Sharpe ratio
      # sharpe_ratio = (mean_return - risk_free_rate) / std_return
      
      # print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
      
      
          net_gains_list = []
          for point in range(start, end  ): # may 1 - 22 # april 23 - 45
      
              sample += 1
              y_pred = model.predict(X_test_scaled[-point])
              y_true = y_test[-point]
      
              if y_pred < 0.0 and y_true < 0.0 :
                  pips_profit += abs(y_true)
                  right += 1
                  net_gains_list.append(abs(y_true))
                  status.append(-1)
      
              elif y_pred > 0.0 and y_true > 0.0 :
                  pips_profit += abs(y_true)
                  right += 1
                  net_gains_list.append(y_true)
                  status.append(1)
      
              else:
                  pips_loss += abs(y_true)
                  wrong += 1
                  # status.append(0)
                  net_gains_list.append(-1*abs(y_true))
      
                  if y_pred > 0.0 and y_true < 0.0 : status.append(str('+1-1'))
      
                  elif y_pred < 0.0 and y_true > 0.0 : status.append(str('-1+1'))
              
      
          # y_pred = model.predict(test_pool )
          
          # print("actual : ",y_test[-3],"\npredicted : ",y_pred )
          # print(f"{symbol}\n")
          # print("accuracy on last 22 test points Month May 2024 with low learning rate 2020 dataset 80% training dataset is ",right/sample * 100 )
          
          returns = np.array(net_gains_list)
      
          # Calculate mean return
          mean_return = np.mean(returns)
      
          # Calculate standard deviation of returns
          std_return = np.std(returns)
      
          # Assuming risk-free rate is 0 for simplicity
          risk_free_rate = 0
      
          # Calculate Sharpe ratio
          sharpe_ratio = (mean_return - risk_free_rate) / std_return
          if len(X_test) != None :
              date   = X_test.index.values[-end]
              date = str(date.astype('datetime64[D]'))
              print(symbol , f" Starting date : {date} ")
              
          status = status[::-1] # starting from end because end is the first most day of the chosen month
          
          print(status)
          # print("accuracy on last 100 test points is ",right) #wrong/sample * 100 )
          accuracy_by_days = right/sample * 100
          # print(f"Accuracy by days :",accuracy_by_days)
          print(f'Sharpe Ratio: {sharpe_ratio:.2f}')
          
          print(f"Pips On Profit Side was : ",pips_profit)
          print("Pips On Loss Side was : ",pips_loss)
          accuracy_by_net_gains = pips_profit / (pips_profit+pips_loss)*100
          print('accuracy by days', accuracy_by_days )
          
          print('accuracy_by_net_gains ',accuracy_by_net_gains )
          net_gains = pips_profit-pips_loss
      #     print("net pips profit : ", pips_profit-pips_loss )
          
          status = []
          right, wrong = 0, 0
          sample = 0
          pips_profit, pips_loss = 0, 0
          for point in range(end, len(X_test_scaled) + 1 ): # excluding May April 55  # excluding May 23 -> till end of length
      
              sample += 1
              y_pred = model.predict(X_test_scaled[-point])
              y_true = y_test[-point]
      
              if y_pred < 0.0 and y_true < 0.0 :
                  pips_profit += abs(y_true)
                  right += 1
                  status.append(-1)
      
              elif y_pred > 0.0 and y_true > 0.0 :
                  pips_profit += abs(y_true)
                  right += 1
                  status.append(1)
      
              else:
                  pips_loss += abs(y_true)
                  wrong += 1
                  # status.append(0)
                  if y_pred > 0.0 and y_true < 0.0 : status.append(str('+1-1'))
      
                  elif y_pred < 0.0 and y_true > 0.0 : status.append(str('-1+1'))
      
          # print("accuracy on all test points excluding last 22 points with low learning rate 2020 dataset 80% training dataset is ",right/sample * 100 )
      
          # print("accuracy on last 100 test points is ",right) #wrong/sample * 100 )
          # print(status)
          accuracy_by_days_before = right/sample * 100
          accuracy_by_net_gains_before = pips_profit / (pips_profit+pips_loss)*100
          gains = {}
          gains['net_gains'] = net_gains
          gains['accuracy_by_days']  = accuracy_by_days
          gains['accuracy_by_days_before']  = accuracy_by_days_before
          gains['accuracy_by_net_gains_before'] = accuracy_by_net_gains_before
          gains['accuracy_by_net_gains'] = accuracy_by_net_gains
          gains['sharpe_ratio'] = sharpe_ratio
          
          # print(f"{symbol}\n==============================")
          try:
              return  gains, accuracy_by_days_before
          except : pass
      
      
      
      
      def update_profit_and_loss(pl_info, date, pips_profit, pips_loss, net_gain):
          if date in pl_info:
              pl_info[date]['pips_profit'] += pips_profit
              pl_info[date]['pips_loss'] += pips_loss
              pl_info[date]['net_gain'] += net_gain
          else:
              pl_info[date] = {'pips_profit': pips_profit, 'pips_loss': pips_loss, 'net_gain': net_gain}
      
          return pl_info
              
      
      
      def each_day_gain_loss(day_features, hour4_features, custom_sample= None , model_path = None):
      
          pass
          # custom_sample = [ 1, 22 ]  
          forex_pairs = [
          'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
          'CADCHF', 'CADJPY',
          'CHFJPY', 
          'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 
          'EURJPY', 'EURNZD', 'EURUSD',
          'GBPAUD', 'GBPCAD', 'GBPCHF', 
          'GBPJPY', 'GBPUSD', 'GBPNZD',
          'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',  
          'USDCHF', 'USDCAD', 'USDJPY']
          symbols_store = {}
          
          sample = 0
          right  = 0
          wrong  = 0
          status = []
          pips_profit = 0
          pips_loss = 0
          pl_info = {}
          # custom_sample=[1,22]
          if custom_sample :
              step, cluster = custom_sample[0] , custom_sample[1] 
              start , end = 0 , 0
              for i in range(step): # 1-> 22 + 1 # 2
                  end   += cluster
              end += 1
              start = end - cluster
          
          else : start, end = 1, 22
          
          for point in range(start, end): # may 1 - 22 # april 23 - 45
              
              pips_profit, pips_loss = 0, 0
              net_gains = 0
              
              sample += 1
              for symbol in forex_pairs:
          
                  
                  if symbol not in symbols_store:
      
      
                      if model_path: model, model_info = load_model_v2(symbol, path = model_path)
                      
                      else:  model, model_info = load_model_v2(symbol, path = "forex_models" )
      
                      
      # {"model": model ,  "model_info": { "learning_rate": learning_rate, "depth": depth, 
      #   "iterations": iterations, "alpha_type": alpha_type, "window_size": window_size,
      #   "scaling": "robust_scaling", "version": "v2" }, "symbol": symbol+"_pips_change" }
                      
                      # learning_rate, depth, iterations = model_info["learning_rate"] ,  model_info["depth"],  model_info["iterations"]
      
                      alpha_type, window_size = model_info["alpha_type"],  model_info["window_size"]
                      
                      day_data, hour4_data = get_day_hour4_features(symbol, day_features, hour4_features)
                      # X_train_scaled, X_test_scaled, y_train, y_test, X_test = get_features_transformed(symbol, day_data, hour4_data, date_resume='2020-06-01',  X_test_re=True)
      
                      X_train_scaled, X_test_scaled, y_train, y_test, X_test = get_features_transformed(symbol, day_data, hour4_data,  X_test_re=True,  window_size=window_size, alpha_type=alpha_type )
      
                      
                      symbols_store[symbol] = { 'X_test_scaled': X_test_scaled,  'y_test': y_test, 'model': model, 'X_test': X_test  }
                      # symbols_store[symbol] = { 'y_test': y_test }
                      # symbols_store[symbol] = { 'model': model }
                      # symbols_store[symbol] = { 'X_test': X_test }
                  
                  else:
                      X_test_scaled = symbols_store[symbol]['X_test_scaled']
                      y_test = symbols_store[symbol]['y_test']
                      model  = symbols_store[symbol]['model']
                      X_test = symbols_store[symbol]['X_test']
                  
                  y_pred = model.predict(X_test_scaled[-point])
                  y_true = y_test[-point]
                  date   = X_test.index.values[-point]
                  # date = np.datetime64(date)
                  date = str(date.astype('datetime64[D]'))
                  
                  if y_pred < 0.0 and y_true < 0.0 :
                      pips_profit += abs(y_true)
                      right += 1
                      status.append(-1)
      
                  elif y_pred > 0.0 and y_true > 0.0 :
                      pips_profit += abs(y_true)
                      right += 1
                      status.append(1)
      
                  else:
                      pips_loss += abs(y_true)
                      wrong += 1
                      # status.append(0)
                      if y_pred > 0.0 and y_true < 0.0 : status.append(str('+1-1'))
      
                      elif y_pred < 0.0 and y_true > 0.0 : status.append(str('-1+1'))
                  
              net_gain = int(pips_profit - pips_loss)
              pl_info  =  update_profit_and_loss(pl_info, date, int(pips_profit), int(pips_loss), net_gain)
      
      
          return pl_info
      
      
      def update_symbol_profit_and_loss(pl_info, date, pips_profit, pips_loss, net_gain, symbol):
            
      #     symbol
          # if 'date_info' not in pl_info:  pl_info['date_info'] = date
      #     if symbol in pl_info:
      #         pl_info[symbol]['pips_profit'] += pips_profit
      #         pl_info[symbol]['pips_loss'] += pips_loss
      #         pl_info[symbol]['net_gain'] += net_gain
              
      #     else:
      #         # pl_info[date] = {'pips_profit': pips_profit, 'pips_loss': pips_loss, 'net_gain': net_gain}
          pl_info[symbol] = {'pips_profit': pips_profit, 'pips_loss': pips_loss, 'net_gain': net_gain, 'date': date}
      
          return pl_info
      
      # '2024-01-29': {'pips_profit': 137, 'pips_loss': 1127, 'net_gain': -989} 
      
      
      def each_day_gain_loss_with_lookback_filter(day_features, hour4_features, custom_sample= None , model_path = None):
      
          pass
          # custom_sample = [ 1, 22 ]  
          forex_pairs = [
          'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
          'CADCHF', 'CADJPY',
          'CHFJPY', 
          'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 
          'EURJPY', 'EURNZD', 'EURUSD',
          'GBPAUD', 'GBPCAD', 'GBPCHF', 
          'GBPJPY', 'GBPUSD', 'GBPNZD',
          'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',  
          'USDCHF', 'USDCAD', 'USDJPY']
          symbols_store = {}
          
          sample = 0
          right  = 0
          wrong  = 0
      
          pips_profit = 0
          pips_loss = 0
          pl_info = {}
          # custom_sample=[1,22]
          if custom_sample :
              step, cluster = custom_sample[0] , custom_sample[1] 
              start , end = 0 , 0
              for i in range(step): # 1-> 22 + 1 # 2
                  end   += cluster
              end += 1
              start = end - cluster
          
          else : start, end = 1, 22
          
          for point in range(start, end): # may 1 - 22 # april 23 - 45
              
              pips_profit, pips_loss = 0, 0
              net_gains = 0
              
              sample += 1
              for symbol in forex_pairs:
          
                  
                  if symbol not in symbols_store:
                      
                      status = []
                      if model_path: model, model_info = load_model_v2(symbol, path = model_path)
                      
                      else:  model, model_info = load_model_v2(symbol, path = "forex_models" )
      
                      
      
      
                      alpha_type, window_size = model_info["alpha_type"],  model_info["window_size"]
                      
                      day_data, hour4_data = get_day_hour4_features(symbol, day_features, hour4_features)
                      # X_train_scaled, X_test_scaled, y_train, y_test, X_test = get_features_transformed(symbol, day_data, hour4_data, date_resume='2020-06-01',  X_test_re=True)
      
                      X_train_scaled, X_test_scaled, y_train, y_test, X_test = get_features_transformed(symbol, day_data, hour4_data,  X_test_re=True,  window_size=window_size, alpha_type=alpha_type )
      
                      
                      symbols_store[symbol] = { 'X_test_scaled': X_test_scaled,  'y_test': y_test, 'model': model, 'X_test': X_test  
                                              ,'status': status }
                      
                      # symbols_store[symbol] = { 'y_test': y_test }
                      # symbols_store[symbol] = { 'model': model }
                      # symbols_store[symbol] = { 'X_test': X_test }
                  
                  else:
                      X_test_scaled = symbols_store[symbol]['X_test_scaled']
                      y_test = symbols_store[symbol]['y_test']
                      model  = symbols_store[symbol]['model']
                      X_test = symbols_store[symbol]['X_test']
                      status = symbols_store[symbol]['status']
      
                  
                  y_pred = model.predict(X_test_scaled[-point])
                  y_true = y_test[-point]
                  date   = X_test.index.values[-point]
                  # date = np.datetime64(date)
                  date = str(date.astype('datetime64[D]'))
      
                  try : last_status = status[-1]
                  except : last_status = status
                  
                  if last_status ==  -1 or last_status == 1 or len(status) == 0 :  
                      
                      if y_pred < 0.0 and y_true < 0.0 :
                          pips_profit += abs(y_true)
                          status.append(-1)
                  
                      elif y_pred > 0.0 and y_true > 0.0 :
                          pips_profit += abs(y_true)
                          status.append(1)
      
                      else:
                          pips_loss += abs(y_true)
                          if y_pred > 0.0 and y_true < 0.0 : status.append(str('+1-1'))
                          elif y_pred < 0.0 and y_true > 0.0 :  status.append(str('-1+1'))
      
                  else:
                      if y_pred < 0.0 and y_true < 0.0 :    status.append(-1)
                      elif y_pred > 0.0 and y_true > 0.0 :  status.append(1)
                      
                      else:
                          if y_pred > 0.0 and y_true < 0.0 : status.append(str('+1-1'))
                      
                          elif y_pred < 0.0 and y_true > 0.0 : status.append(str('-1+1'))
              
      
                  symbols_store[symbol]['status'] = status
                                                  
              
              net_gain = int(pips_profit - pips_loss)
              pl_info  =  update_profit_and_loss(pl_info, date, int(pips_profit), int(pips_loss), net_gain)
      
      
          return pl_info
      
      
      
      def each_symbol_gain_loss_by_date(day_features, hour4_features, custom_sample= None , model_path = None, date_val='2024-01-29'):
      
          pass
          # custom_sample = [ 1, 22 ]  
          forex_pairs = [
          'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
          'CADCHF', 'CADJPY',
          'CHFJPY', 
          'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 
          'EURJPY', 'EURNZD', 'EURUSD',
          'GBPAUD', 'GBPCAD', 'GBPCHF', 
          'GBPJPY', 'GBPUSD', 'GBPNZD',
          'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',  
          'USDCHF', 'USDCAD', 'USDJPY']
          symbols_store = {}
          
          sample = 0
          right  = 0
          wrong  = 0
          status = []
          pips_profit = 0
          pips_loss = 0
          pl_info = {}
          # custom_sample=[1,22]
          if custom_sample :
              step, cluster = custom_sample[0] , custom_sample[1] 
              start , end = 0 , 0
              for i in range(step): # 1-> 22 + 1 # 2
                  end   += cluster
              end += 1
              start = end - cluster
          
          else : start, end = 1, 22
          
          
          for point in range(start, end ): # may 1 - 22 # april 23 - 45
              
              pips_profit, pips_loss = 0, 0
              net_gains = 0
              
              sample += 1
              for symbol in forex_pairs:
                  
                  pips_profit, pips_loss = 0, 0
                  if symbol not in symbols_store:
      
                      if model_path: model, model_info = load_model_v2(symbol, path = model_path)
                      
                      else:  model, model_info = load_model_v2(symbol, path = "forex_models" )
                      
      # {"model": model ,  "model_info": { "learning_rate": learning_rate, "depth": depth, 
      #   "iterations": iterations, "alpha_type": alpha_type, "window_size": window_size,
      #   "scaling": "robust_scaling", "version": "v2" }, "symbol": symbol+"_pips_change" }
                      
                      # learning_rate, depth, iterations = model_info["learning_rate"] ,  model_info["depth"],  model_info["iterations"]
      
                      alpha_type, window_size = model_info["alpha_type"],  model_info["window_size"]
                      
                      day_data, hour4_data = get_day_hour4_features(symbol, day_features, hour4_features)
                      # X_train_scaled, X_test_scaled, y_train, y_test, X_test = get_features_transformed(symbol, day_data, hour4_data, date_resume='2020-06-01',  X_test_re=True)
      
                      X_train_scaled, X_test_scaled, y_train, y_test, X_test = get_features_transformed(symbol, day_data, hour4_data,  X_test_re=True,  window_size=window_size, alpha_type=alpha_type )
                      
                      symbols_store[symbol] = { 'X_test_scaled': X_test_scaled,  'y_test': y_test, 'model': model, 'X_test': X_test  }
                      # symbols_store[symbol] = { 'y_test': y_test }
                      # symbols_store[symbol] = { 'model': model }
                      # symbols_store[symbol] = { 'X_test': X_test }
                  
                  else:
                      X_test_scaled = symbols_store[symbol]['X_test_scaled']
                      y_test = symbols_store[symbol]['y_test']
                      model  = symbols_store[symbol]['model']
                      X_test = symbols_store[symbol]['X_test']
                  
                  y_pred = model.predict(X_test_scaled[-point])
                  y_true = y_test[-point]
                  date   = X_test.index.values[-point]
                  # date = np.datetime64(date)
                  date = str(date.astype('datetime64[D]'))
                  
                  if date != date_val :
                      continue
                  
                  if date != date_val : 
                      point += 1
                      break
                  
                  y_pred = model.predict(X_test_scaled[-point])
                  y_true = y_test[-point]
                  
                  if  y_pred < 0.0 and y_true < 0.0 :
                      pips_profit = abs(y_true)
                      right += 1
                      status.append(-1)
      
                  elif y_pred > 0.0 and y_true > 0.0 :
                      pips_profit = abs(y_true)
                      right += 1
                      status.append(1)
      
                  else:
                      pips_loss = abs(y_true)
                      wrong += 1
                      # status.append(0)
                      if y_pred > 0.0 and y_true < 0.0 : status.append(str('+1-1'))
      
                      elif y_pred < 0.0 and y_true > 0.0 : status.append(str('-1+1'))
                  
                  net_gain = int(pips_profit - pips_loss) # net gain on symbol basis for a particular day
                  pl_info  =  update_symbol_profit_and_loss(pl_info, date, int(pips_profit), int(pips_loss), net_gain, symbol)
      
      
          return pl_info
      
      
      
      def each_day_gain_loss_with_threshold(day_features, hour4_features, custom_sample= None , model_path = None, threshold = 5 ):
      
          pass
          # custom_sample = [ 1, 22 ]  
          forex_pairs = [
          'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
          'CADCHF', 'CADJPY',
          'CHFJPY', 
          'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 
          'EURJPY', 'EURNZD', 'EURUSD',
          'GBPAUD', 'GBPCAD', 'GBPCHF', 
          'GBPJPY', 'GBPUSD', 'GBPNZD',
          'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',  
          'USDCHF', 'USDCAD', 'USDJPY']
          symbols_store = {}
          
          sample = 0
          right  = 0
          wrong  = 0
          status = []
          pips_profit = 0
          pips_loss = 0
          pl_info = {}
          # custom_sample=[1,22]
          if custom_sample :
              step, cluster = custom_sample[0] , custom_sample[1] 
              start , end = 0 , 0
              for i in range(step):
                  end   += cluster + 1
      
              start = end - cluster
          
          
          else : start, end = 1, 22
          
          era = start + 10
          
          for point in range(start, end + 1 ): # may 1 - 22 # april 23 - 45
              
              pips_profit, pips_loss = 0, 0
              net_gains = 0
              
              sample += 1
              for symbol in forex_pairs:
          
                  
                  if symbol not in symbols_store:
                  
                      day_data, hour4_data = get_day_hour4_features(symbol, day_features, hour4_features)
                      X_train_scaled, X_test_scaled, y_train, y_test, X_test = get_features_transformed(symbol, day_data, hour4_data, date_resume='2020-06-01',  X_test_re=True)
                      
                      if model_path: model = load_model(symbol, path = model_path)
                      else : model = load_model(symbol)
                      
                      
                      symbols_store[symbol] = { 'X_test_scaled': X_test_scaled,  
                                               'y_test': y_test, 'model': model, 'X_test': X_test  }
                      # symbols_store[symbol] = { 'y_test': y_test }
                      # symbols_store[symbol] = { 'model': model }
                      # symbols_store[symbol] = { 'X_test': X_test }
                  
                  else:
                      X_test_scaled = symbols_store[symbol]['X_test_scaled']
                      y_test = symbols_store[symbol]['y_test']
                      model  = symbols_store[symbol]['model']
                      X_test = symbols_store[symbol]['X_test']
                  
                  y_pred = model.predict(X_test_scaled[-point])
                  y_true = y_test[-point]
                  date   = X_test.index.values[-point]
                  # date = np.datetime64(date)
                  date = str(date.astype('datetime64[D]'))
                  if point < era : 
                      if abs(y_pred) < threshold : continue
                  
                  if y_pred < 0.0 and y_true < 0.0 :
                      pips_profit += abs(y_true)
                      right += 1
                      status.append(-1)
      
                  elif y_pred > 0.0 and y_true > 0.0 :
                      pips_profit += abs(y_true)
                      right += 1
                      status.append(1)
      
                  else:
                      pips_loss += abs(y_true)
                      wrong += 1
                      # status.append(0)
                      if y_pred > 0.0 and y_true < 0.0 : status.append(str('+1-1'))
      
                      elif y_pred < 0.0 and y_true > 0.0 : status.append(str('-1+1'))
                  
              net_gain = int(pips_profit - pips_loss)
              pl_info  =  update_profit_and_loss(pl_info, date, int(pips_profit), int(pips_loss), net_gain)
      
      
          return pl_info
      
      #     accuracy_by_days = right/sample * 100
      
      #     accuracy_by_net_gains = pips_profit / (pips_profit+pips_loss)*100
      
      #     net_gains = pips_profit-pips_loss
      
          
      import pickle
      
      parameters = {
          'NZDJPY': {
              'Iterations': 15,
              'Lr': 0.001,
              'scaler': 'RobustScaler()',
              'depth': 6
          },
          'AUDJPY': {
              'Iterations': 5,
              'Lr': 0.01,
              'scaler': 'RobustScaler()',
              'depth': 6
          },
          'USDJPY': {
              'Iterations': 5,
              'Lr': 0.01,
              'depth': 6
          },
          'CADJPY': {
              'Iterations': 5,
              'Lr': 0.01,
              'depth': 6
          },
          'NZDCAD': {
              'Iterations': 7,
              'Lr': 0.06,
              'scaler': 'RobustScaler()',
              'depth': 8,
              'window_size': 9,
              'accuracy': 52,
              'net pips profit': 114
          },
          'CADCHF': {
              'Iterations': 7,
              'Lr': 0.06,
              'scaler': 'RobustScaler()',
              'depth': 8,
              'window_size': 9,
              'accuracy': 52,
              'net pips profit': 158
          },
          'USDCHF2': {
              'Iterations': 7,
              'Lr': 0.06,
              'scaler': 'RobustScaler()',
              'depth': 8,
              'window_size': 9,
              'accuracy': 50,
              'net pips profit': 179
          },
          'EURAUD': {
              'Iterations': 7,
              'Lr': 0.06,
              'scaler': 'RobustScaler()',
              'depth': 8,
              'window_size': 9,
              'accuracy': 56,
              'net pips profit': 306
          },
          'EURNZD3': {
              'Iterations': 15,
              'Lr': 0.1,
              'scaler': 'RobustScaler()',
              'depth': 7,
              'window_size': 9,
              'accuracy': 52,
              'net pips profit': 278
          },
          'GBPNZD': {
              'Iterations': 15,
              'Lr': 0.1,
              'scaler': 'RobustScaler()',
              'depth': 7,
              'window_size': 9,
              'accuracy': 50,
              'net pips profit': 224
          },
          'NZDCHF2': {
              'Iterations': 15,
              'Lr': 0.1,
              'scaler': 'RobustScaler()',
              'depth': 7,
              'window_size': 9,
              'accuracy': 50,
              'net pips profit': 140
          },
          'EURUSD3': {
              'Iterations': 15,
              'Lr': 0.1,
              'scaler': 'RobustScaler()',
              'depth': 7,
              'window_size': 9,
              'accuracy': 56,
              'net pips profit': 306
          },
          'GBPCAD': {
              'Iterations': 5,
              'Lr': 0.001,
              'scaler': 'RobustScaler()',
              'depth': 7,
              'window_size': 9,
              'accuracy': 54,
              'net pips profit': 151
          },
          'GBPUSD2': {
              'Iterations': 5,
              'Lr': 0.001,
              'scaler': 'RobustScaler()',
              'depth': 7,
              'window_size': 9,
              'accuracy': 52,
              'net pips profit': 217
          },
          'EURGBP': {
              'Lr': 0.001 ,
              'Iterations': 188,
              'depth': 7,
              'accuracy' :52,
              'pips net profit' : 97
          },
          'EURCAD2': {
              'Iterations': 15,
              'Lr': 0.1,
              'scaler': 'RobustScaler()',
              'depth': 7,
              'window_size': 9,
              'accuracy': 51,
              'pips net profit':  228
          }, 
          'GBPCHF': {
              'Iterations': 15,
              'Lr': 0.1,
              'scaler': 'RobustScaler()',
              'depth': 8,
              'window_size': 9,
              'accuracy': 53,
              'pips net profit':  241
          }, 
          'NZDUSD': {
              'Iterations': 5, 
              'Lr': 0.77, 
              'depth': 7,
              'accuracy': 54
          }, 
          'EURUSD2': {
              'Lr':0.025,
              'Iterations': 450, 
              'depth': 5,
              'accuracy': 53
          },
          'EURNZD2':{
           'Iterations': 29, 
           'Lr': 0.63, 
           'depth': 7,
           'accuracy': 50
          }, 
          'GBPUSD': {
           'Iterations': 13, 
           'Lr': 0.77, 
           'depth': 7,
           'accuracy': 53
          },
          'GBPAUD': {
          'Iterations': 13,
          'Lr': 0.77,
          'depth': 8,
          'accuracy': 55       
          },
          'AUDUSD2': {
          'Iterations': 5,
          'Lr': 0.7,
          'depth': 7,
          'accuracy': 50       
          },
          'AUDCHF': {
          'Iterations': 7,
          'Lr': 0.71,
          'depth' : 7
          }, 
          'AUDCAD':{
          'Iterations' : 9,
          'Lr': 0.67,
          'depth': 7
          },
          'AUDNZD':{
          'Iterations': 15,
          'Lr': 0.68,
          'depth': 7 
          }, 
          'EURCAD':{ 
          'Iterations': 20,
          'Lr': 0.039,
          'depth': 7
          },
          'NZDCHF' : {
          'Iterations': 27,
          'Lr' : 0.35,
          'depth' : 7
          },
          'USDCAD': {
          'Iterations': 20,
          'Lr': 0.039,
          'depth': 7
          },
          'USDCHF':{
          'Iterations': 188,
          'Lr': 0.67,
          'depth': 7
          },
          'NZDCAD':{
          'Iterations': 17,
          'Lr': 0.38,
          'depth': 7
          }, 
          'AUDUSD' :{
          'Iterations': 11 ,
          'Lr': 0.211,
          'depth': 7
          },
          'EURUSD':{
          'Iterations': 23,
          'Lr': 0.51,
          'depth': 7
          },
          'EURNZD':{
          'Iterations': 188,
          'Lr' :0.55,
          'depth': 7
          }
      }
      
      # with open('parameters', 'wb') as file:
      #     pickle.dump(parameters, file )
          
      
      def load_parameters(symbol):
          
          with open('parameters', 'rb') as file:
              parameters = pickle.load(file )
              
          return parameters[symbol]
              
