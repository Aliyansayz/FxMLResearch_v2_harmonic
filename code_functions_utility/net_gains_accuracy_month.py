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
      





# use of last month features if last_month = 3 this means latest_month



def evaluate_model_return_gains_accuracy(symbol, model, X_test_scaled, y_test, month_keys= None, last_months = None, X_test = None ):
      
      """
      When Using Month Keys
      Assuming that our symbol-pair dataset is not more than last 11 months or
      No two months with different year like, May 2024 in dataset but we cannot allow May 2023 
      
      """
      
  
      month_store = { 1: 'January', 2:'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
                     7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December' }
      month_indexes = {}
      
      last_month = 1 # 2 3 4 
      month_keys  = [1,4,5]

      dataset_info = {}
      date   = X_test.index.values[-1] # latest date to get latest month
      date = str(date.astype('datetime64[D]'))
      latest_month = int(date.split('-')[1])

      if month_keys != None :  #  then checks for 
        pass
        date_string = '2024-05-27'
        month = date_string.split('-')[1]
        latest_month = max(month_key)
        if max(month_keys)
      else :  last_month_mode = True
        
      if latest_month in month_key
  
      date_string = '2024-05-27'
      month = date_string.split('-')[1]
      latest_month = max(month_key)
      last_months = 3
      break_month = last_months + 1
  
  
      sample, right, wrong = 0, 0, 0
      status = []
      pips_profit = 0
      pips_loss = 0
      
      # custom_sample=[1,22]
      if custom_sample:
          step, cluster = custom_sample[0], custom_sample[1]
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
      for point in range(1, len(X_test_scaled) + 1 ): #
  
          sample += 1
          # y_pred = model.predict(X_test_scaled[-point])
          # y_true = y_test[-point]
          date   = str(X_test.index.values[-end].astype('datetime64[D]'))
          month_num   = int(date.split('-')[1])
          if break_month == month_num and last_month_mode : break
          else :
            if month_num not in month_indexes:  month_indexes[month_num] = { "index": [point] }
            else : month_indexes[month_num]["index"].append(point)
            
      # to be continued...
        
          month_indexes[]
  
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


  


