Whole Code : -




              from datetime import datetime, timedelta

              def get_start_end_date(daylength = 95):
              # Get today's date
                  today = datetime.today()
              
              # Get the date 75 days ago
                  previous_date = today - timedelta(days=daylength) # 95
              
              # Get the date 1 day ago
              # yesterday = today - timedelta(days=1)
              # # Format the date
              # yesterday_str = yesterday.strftime('%Y-%m-%d')
              
                  # Format the dates
                  today_str = today.strftime('%Y-%m-%d')
                  previous_date_str = previous_date.strftime('%Y-%m-%d')
                  start_date =  previous_date_str
                  end_date = today_str
              
                  return  start_date, end_date
              
              
              
              
              import threading
              import time
              from twelvedata import TDClient
              
              # Function to simulate processing of a forex pair
              def process_forex_pair(symbol, tf, start_date, end_date, output_size, td ):
              
                  print(f"Processing {symbol}...")
                  if tf == None : tf = "4h"
              

                  symbol_ = symbol[:3]+"/"+symbol[:-3]
                  ts = td.time_series(
                      symbol= symbol_, # "EUR/USD", # symbol_
                      interval=tf , #"4h", # tf
                      start_date=f"{start_date}",
                      end_date=f"{end_date}",
                      outputsize=output_size ) # 450 ) # approximate length for
              
                  # Returns pandas.DataFrame
                  df = ts.as_pandas()
                  folder_path = "currency_data"
              
                  if not os.path.exists(folder_path):
                      os.makedirs(folder_path)
              
                  filename = f"{symbol}.csv"
              
                  if tf == "1day":
                      filename = f"{symbol}_Daily.csv"
              
                  elif tf == "4h":
                      filename = f"{symbol}_H4.csv"
              
                  file_path = os.path.join(folder_path, filename)
              
                  df.to_csv(f"{file_path}")
              
                  # Simulate some work with a sleep (replace with actual processing logic)
                  time.sleep(0.5)
                  print(f"Completed processing {symbol}")
              
              
                  # Main function to manage threading
                  def main():
                      td = TDClient(apikey="")
                
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
                    start_date, end_date = get_start_end_date()
                
                    daylength = 95
                    tf = "1day"
                    if tf == "1day":
                        outputsize = daylength  # last 95 days
                    elif tf == "4h":
                        outputsize = daylength * 6  # 95 days length
                
                
                    batch_size = 8 # limit of requests api can make  # Number of threads to run simultaneously
                
                    output_size = daylength
                    for i in range(0, len(forex_pairs), batch_size):
                        # Create a batch of threads
                        # threads = []
                        for symbol in forex_pairs[i:i + batch_size]:
                            process_forex_pair(symbol, tf, start_date, end_date, output_size, td)
                
                        # Wait for 60 seconds before moving on to the next batch
                        print("Waiting for 60 seconds before processing the next batch...")
                        time.sleep(60)
                
                    print("All forex pairs have been processed.")
    
    
            # Run the main function
            if __name__ == "__main__":
                main()

# Initial Working: - 

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




