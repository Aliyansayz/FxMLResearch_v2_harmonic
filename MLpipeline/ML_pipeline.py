import threading
import time
from twelvedata import TDClient
from features_engineering import return_df_day, return_h4_df
from features_engineering import add_features_day, make_features_lagged_day
from features_engineering import combine_ohlc_into_single_day_hour_4, add_ohlc_in_lagged_hour_4, add_features_hour_4
import os
import pandas as pd



class ohlc_transformations:
    forex_pairs = [
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
        'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD',
        'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'EURUSD',
        'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPUSD', 'GBPNZD',
        'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',
        'USDCHF', 'USDCAD', 'USDJPY'
    ]
    lag_info = { "model" : { "v2": {"hour_4_lag": 3, "max_day_lag": 14  } }  }


    pass

    def make_day_ohlc(self):

        self.lag_info["model"]["v2"]["max_day_lag"]
        max_day_lag_by = 14
        currency_ohlc = []
        count = 0
        for pair in forex_pairs:
            daily_path = f'currency_data/{pair}_Daily.csv'
            df = return_df_day(daily_path)
            df = combine_ohlc_into_single_day_hour_4(df)

            df = add_features_day(df)
            df = make_features_lagged_day(lag_by=14)

            currency_df = {"day_features": df, "symbol": f"{pair}"}
            currency_ohlc.append(currency_df)
            count += 1
        print(count)

        filename = "day_features_data_lagby_14_v2.bin"

        folder_path = "currency_ohlc"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = os.path.join(folder_path, filename)

        self.save_ohlc_with_features(currency_ohlc, path)



    def make_hour_4_ohlc(self):

        hour_4_lag_by = self.lag_info["model"]["v2"]["hour_4_lag"]
        currency_ohlc = []
        count = 0

        for pair in forex_pairs:
            # daily_path = f'currency_data/{pair[:6]}_Daily.csv'
            hour4_path = f'currency_data/{pair}_H4.csv'

            data_h4 = return_h4_df(hour4_path)

            reshaped_h4 = combine_ohlc_into_single_day_hour_4(data_h4)
            reshaped_h4 =  self.add_latest_index_into_ohcl(reshaped_h4)
            features_h4 = add_ohlc_in_lagged_hour_4(reshaped_h4, lag_by=3)

            features_h4 = add_features_hour_4(features_h4, lag_by=3)
            currency_df = {"hour4_features": features_h4, "symbol": f"{pair}"}
            currency_ohlc.append(currency_df)
            count += 1
        print(count)

        filename = "hour4_features_data_lag_by_3_v2.bin"

        folder_path = "currency_ohlc"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = os.path.join(folder_path, filename)

        self.save_ohlc_with_features(currency_ohlc, path)



    def save_ohlc_with_features(self, currency_ohlc, path ):

        import pickle

        with open(path, 'wb') as file:
            pickle.dump(currency_ohlc, file)

        # with open('hour4_features_data_lag_by_3_v2.bin', 'wb') as file:
        #     pickle.dump(currency_ohlc, file)


    def add_latest_index_into_ohcl(self, df):

        formatted_date = self.get_current_date()
        new_date   = pd.to_datetime(f'{formatted_date}')  # adding tomorrow date if it's sat, sun then automatically monday selected.
        df.loc[new_date] = np.nan
        return df


    def get_current_date(self, tomorrow=None):
        from datetime import date, timedelta

        # Get today's date
        if tomorrow:
            today_date = date.today() + timedelta(days=1)
        else:
            today_date = date.today()

        if today_date.isoweekday() == 6:
            today_date += timedelta(days=2)  # Saturday
        elif today_date.isoweekday() == 7:
            today_date += timedelta(days=1)  # Sunday

        formatted_date = today_date.strftime('%Y-%m-%d')

        return formatted_date


from features_engineering import get_features_transformed

class ohlc_models (ohlc_transformations) :



    def apply_predictions(self, symbol, day_data, hour4_data):

        model, model_info =  self.load_model(symbol)
        alpha_type, window_size = model_info["alpha_type"], model_info["window_size"]
        X_scaled, X = get_features_transformed(symbol, day_data, hour4_data, window_size = window_size, alpha_type= alpha_type)

        y_pred = model.predict(X_scaled[-1])
        if y_pred < 0.0: status = "-1"
        elif y_pred >= 0.0 : status = "1"

        return  status

    def get_date(self):

        from datetime import datetime
        today = datetime.today()
        today_str = today.strftime('%Y-%m-%d')
        return  today_str


    def fx_status_function(self, fx_status_info, date, status, symbol):

        if status == 1    : action = "buy"
        elif status == -1 : action = "sell"

        fx_status_info[symbol] = { 'date': date, "action": action, 'prediction_status': status }

        return fx_status_info


    def load_model(self, symbol, path=None):

        if path: path = f"{path}/{symbol}_model_pips_change"
        path = f"{path}/{symbol}_model_pips_change"
        # else:  path = f"v2_forex_models/{symbol}_model_pips_change"

        with open(path, 'rb') as file:
            model_file = pickle.load(file)

        model = model_file["model"]
        model_info = model_file["model_info"]

        return model, model_info


    def load_features_files(self):


        day_features_path = 'day_features_data_lagby_14_v2.bin'

        with open(day_features_path, 'rb') as file:
            day_features = pickle.load(file)

        # hour4_features_path = 'hour4_features_data.bin'
        hour4_features_path = 'hour4_features_data_lag_by_3_v2.bin'

        with open(hour4_features_path, 'rb') as file:
            hour4_features = pickle.load(file)

        return day_features, hour4_features

    def get_day_hour4_features(self, symbol, day_features, hour4_features):

        for day_pair in day_features:
            if day_pair['symbol'] == symbol:
                day_data = day_pair['day_features']
                break

        for hour4_pair in hour4_features:
            if hour4_pair['symbol'] == symbol:
                hour4_data = hour4_pair['hour4_features']
                break

        return day_data, hour4_data


    def load_models(self):
        pass


    def apply_parameters(self):
        pass




class twelve_data_ohlc( ohlc_models ) :
    forex_pairs = [
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
        'CADCHF', 'CADJPY','CHFJPY', # wait


        'EURAUD', 'EURCAD',
        'EURCHF', 'EURGBP','EURJPY', 'EURNZD', 'EURUSD',
        'GBPAUD', # wait
        
        'GBPCAD', 'GBPCHF','GBPJPY', 'GBPUSD', 'GBPNZD',
        'NZDCAD', 'NZDCHF', 'NZDJPY', # wait
        
        
        'NZDUSD',
        'USDCHF', 'USDCAD', 'USDJPY'
    ]
    batch_size = 8


    def get_ohlc_1d_4h(self):

        self.save_ohlc(tf="1day")

        self.save_ohlc(tf="4h")


    def get_start_end_date(self, daylength=95):
        from datetime import datetime, timedelta

        # Get today's date
        today = datetime.today()

        # Get the date 75 days ago
        previous_date = today - timedelta(days=daylength)  # 95

       
        # Format the dates
        today_str = today.strftime('%Y-%m-%d')
        previous_date_str = previous_date.strftime('%Y-%m-%d')
        start_date = previous_date_str
        end_date = today_str

        return start_date, end_date


    # Function to simulate processing of a forex pair
    def process_forex_pair(self, symbol, tf, start_date, end_date, output_size, td):

        print(f"Processing {symbol}...")
        if tf == None: tf = "4h"

        symbol_ = symbol[:3] + "/" + symbol[3:]
        print(symbol_)
        ts = td.time_series(
            symbol=symbol_,  # "EUR/USD", # symbol_
            interval=tf,  # "4h", # tf
            start_date=f"{start_date}",
            end_date=f"{end_date}",
            outputsize=output_size)  # 450 ) # approximate length for

        # Returns pandas.DataFrame
        df = ts.as_pandas()
        folder_path = "currency_data"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = f"{symbol}.csv"

        if tf == "1day": filename = f"{symbol}_Daily.csv"

        elif tf == "4h": filename = f"{symbol}_H4.csv"


        file_path = os.path.join(folder_path, filename)

        self.append_new_data( csv_file_path= file_path, new_data_df= df)

        # df.to_csv(f"{file_path}")

        # Simulate some work with a sleep (replace with actual processing logic)
        time.sleep(0.5)
        print(f"Completed processing {symbol}")


    def append_new_data(self, csv_file_path, new_data_df):

        # new_data_df['datetime'] = pd.to_datetime(new_data_df['datetime'])

        # Step 1: Read the existing data from the CSV file
        try:
            existing_df = pd.read_csv(csv_file_path, parse_dates=["datetime"])
            if not existing_df.empty:
                filtered_new_data = new_data_df[~new_data_df['datetime'].isin(existing_df['datetime'])]

        except FileNotFoundError:
            existing_df = pd.DataFrame()  # Create an empty DataFrame if the file doesn't exist
            filtered_new_data = new_data_df
            filtered_new_data.to_csv(csv_file_path)

            print(f"Data successfully appended to {csv_file_path}")
            # Step 2: Ensure the date column in the new data is in datetime format


         # Step 3: Filter out data from `new_data_df` that is already in `existing_df`

        # Step 4: Append the new data to the existing data
        combined_df = pd.concat([existing_df, filtered_new_data])

        # Step 5: Save the combined data back to the CSV file
        combined_df.to_csv(csv_file_path)

        print(f"Data successfully appended to {csv_file_path}")


    # Main function to manage threading
    def save_ohlc(self, tf="1day", daylength=95):

        td = TDClient(apikey="")
        start_date, end_date = self.get_start_end_date()

        # daylength = 95
        if tf == "1day":
            outputsize = daylength  # last 95 days
        elif tf == "4h":
            outputsize = daylength * 6  # 95 days length

        batch_size = self.batch_size  # limit of requests api can make  # Number of threads to run simultaneously

        output_size = daylength
        for i in range(0, len(self.forex_pairs), batch_size):

            for symbol in self.forex_pairs[i:i + batch_size]:
                self.process_forex_pair(symbol, tf, start_date, end_date, output_size, td)

            # Wait for 60 seconds before moving on to the next batch
            print("Waiting for 60 seconds before processing the next batch...")
            time.sleep(60)


    
class access_resources(twelve_data_ohlc):
    pass


class mlpipeline(access_resources):

    @classmethod
    def return_fx_prediction_status(cls):


        resource = cls()

        resource.save_ohlc(tf="1day")
        resource.save_ohlc(tf="4h")
        
        resource.make_day_ohlc()
        resource.make_hour_4_ohlc()
        
        date = resource.get_date()

        day_features, hour4_features = resource.load_features_files()
        fx_status_info = {}
        for symbol in resource.forex_pairs:
            pass
            day_data, hour4_data = resource.get_day_hour4_features(symbol, day_features, hour4_features)

            status = resource.apply_predictions(symbol, day_data, hour4_data)
            fx_status_info = resource.fx_status_function(fx_status_info, date, status, symbol)

        pass


mlpipeline.return_fx_prediction_status()
