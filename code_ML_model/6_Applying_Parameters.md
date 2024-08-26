    if model_path: model, model_info = load_model_v2(symbol, path = model_path)
                    
                else:  model, model_info = load_model_v2(symbol, path = "forex_models" )

                alpha_type, window_size = model_info["alpha_type"],  model_info["window_size"]
                
                day_data, hour4_data = get_day_hour4_features(symbol, day_features, hour4_features)
          
          
          X_train_scaled, X_test_scaled, y_train, y_test, X_test = get_features_transformed(symbol, day_data, hour4_data,  X_test_re=True,  window_size=window_size, alpha_type=alpha_type )
          symbols_store[symbol] = { 'X_test_scaled': X_test_scaled,  'y_test': y_test, 'model': model, 'X_test': X_test  }


            y_pred = model.predict(X_test_scaled[-point])
            y_true = y_test[-point]
            date   = X_test.index.values[-point]
            # date = np.datetime64(date)
            date = str(date.astype('datetime64[D]'))


            if y_pred < 0.0  :
                pips_profit += abs(y_true)
                right += 1
                status.append(-1)

            elif y_pred > 0.0  :
                pips_profit += abs(y_true)
                right += 1
                status.append(1)


            pl_info  =  update_symbol_profit_and_loss(pl_info, date, int(pips_profit), int(pips_loss), net_gain, symbol)




      def update_symbol_profit_and_loss(pl_info, date, pips_profit, pips_loss, net_gain, symbol):
            
      
          pl_info[symbol] = {'pips_profit': pips_profit, 'pips_loss': pips_loss, 'net_gain': net_gain, 'date': date}
      
          return pl_info
