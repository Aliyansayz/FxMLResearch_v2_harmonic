
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
