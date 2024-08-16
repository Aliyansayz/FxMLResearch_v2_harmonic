              from datetime import datetime, timedelta
              
              # Get today's date
              today = datetime.today()
              
              # Get the date 75 days ago
              previous_date = today - timedelta(days=95)
              
              # Get the date 1 day ago
              yesterday = today - timedelta(days=1)
              # Format the date
              yesterday_str = yesterday.strftime('%Y-%m-%d')
              
              # Format the dates
              today_str = today.strftime('%Y-%m-%d')
              previous_date_str = previous_date.strftime('%Y-%m-%d')
              
              print("Today's Date:", today_str)
              print("75 Days Ago Date:", previous_date_str)
              
              
              from twelvedata import TDClient
              api_key = "_"
              # Initialize client - apikey parameter is requiered
              td = TDClient(apikey=api_key)
              
              # Construct the necessary time series
              ts = td.time_series(
              symbol="EUR/USD",
              interval="4h",
              start_date=f"{previous_date_str}",
              end_date= f"{today_str}",
              outputsize=450
              )
              
              # Returns pandas.DataFrame
              print(ts.as_pandas())
