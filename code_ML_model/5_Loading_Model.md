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
