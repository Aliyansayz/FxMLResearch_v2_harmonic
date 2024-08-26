
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
            

        day_features_X   =  day_data[features]

            day_features_X = day_features_X.apply(pd.to_numeric, errors='coerce')
        
            day_features_X = day_features_X.astype(float)
        
            hour4_features_X = hour4_data
        
            hour4_features_X = hour4_features_X.apply(pd.to_numeric, errors='coerce')
        
            hour4_features_X = hour4_features_X.astype(float)

        
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

        elif alpha_type == "ha":
        substrings = [ 'stdev_slow', 'heikin_ashi', 'slow_harmonic_mean'  ] # 'slow_harmonic_mean'




    filtered_columns = [col for col in hour4_features_X.columns if any(sub in col for sub in substrings)]
    hour4_features_X = hour4_features_X[filtered_columns]

    X = day_features_X.join(hour4_features_X)


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

    # Apply RobustScaler
    scaler = RobustScaler()
    

    X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    X_test_scaled = scaler.fit_transform(X_test)
    
    if not X_test_re:
        return X_train_scaled, X_test_scaled, y_train, y_test
    else :
        return X_train_scaled, X_test_scaled, y_train, y_test, X_test
