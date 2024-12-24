import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import json

import threading
import time, pytz
from features_engineering import return_df_day, return_h4_df
from features_engineering import add_features_day, make_features_lagged_day
from features_engineering import combine_ohlc_into_single_day_hour_4, add_ohlc_in_lagged_hour_4, add_features_hour_4
import os, pickle
import pandas as pd
import numpy as np
from os_functions import resolve_currency_data_path, resolve_ml_models_path, resolve_records_path, resolve_currency_ohlc_bin_path, resolve_updated_ml_record
from mt5_ohlc_download import get_ohlc_data_h4, get_ohlc_data_day


class twelve_data_ohlc():
    forex_pairs = [
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
        'CADCHF', 'CADJPY', 'CHFJPY',  # wait

        'EURAUD', 'EURCAD',
        'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'EURUSD',
        'GBPAUD',  # wait

        'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPUSD', 'GBPNZD',
        'NZDCAD', 'NZDCHF', 'NZDJPY',  # wait

        'NZDUSD',
        'USDCHF', 'USDCAD', 'USDJPY'
    ]
    batch_size = 8

    def get_ohlc_1d_4h(self):

        self.save_ohlc(tf="1day")

        self.save_ohlc(tf="4h")

    def user_request_end_date(self, to_use= False):
        pass


    def get_start_end_date(self, daylength=95):
        from datetime import datetime, timedelta

        # Get today's date
        today = datetime.today()

        # Get the date 95 days ago
        previous_date = today - timedelta(days=daylength)  # 95

        # Format the dates
        today_str = today.strftime('%Y-%m-%d')
        previous_date_str = previous_date.strftime('%Y-%m-%d')
        start_date = previous_date_str
        end_date = today_str
        return start_date, end_date

    def check_df_exists(self, path, filename):
        csv_file_path = os.path.abspath(os.path.join(path, filename))
        # csv_file_path = os.path.join(path, filename)
        try:
            df = pd.read_csv(csv_file_path, parse_dates=["datetime"])
            return True, df
        except:
            return False, None


    def t_day_minus_1(self, df=None):
        from datetime import datetime, timedelta
        today = datetime.today()
        previous_date = today - timedelta(days=1)
        start_date =  today - timedelta(days=95)
        previous_date_str = previous_date.strftime('%Y-%m-%d')

        start_date_str = start_date.strftime('%Y-%m-%d')
        if previous_date_str and start_date_str in df['datetime'].values:  return True
        else: return False


    # Function to simulate processing of a forex pair
    def process_forex_pair(self, symbol, tf, start_date, end_date, output_size):

        # print(f"Processing {symbol}...")
        if tf == None: tf = "4h"
        folder_path = resolve_currency_data_path()
        filename = f"{symbol}.csv"

        if tf == "1day": filename, df = f"{symbol}_Daily.csv" , get_ohlc_data_day(symbol)

        elif tf == "4h": filename, df = f"{symbol}_H4.csv" ,  get_ohlc_data_h4(symbol)

        # symbol_ = symbol[:3] + "/" + symbol[3:]
        #
        # print(symbol_)
        """
        # check if dataframe exist if exists then create
        check_df, df = self.check_df_exists(folder_path, filename)
        if check_df:
           date_status =  self.t_day_minus_1(df)
           if date_status == True : return
        """

        # Returns pandas.DataFrame

        # folder_path = "currency_data"
        folder_path = resolve_currency_data_path()
        self.append_new_data(csv_file_path=folder_path, new_data_df=df)
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
                self.process_forex_pair(symbol, tf, start_date, end_date, output_size)

            # Wait for 60 seconds before moving on to the next batch
            print("Waiting for 60 seconds before processing the next batch...")
            time.sleep(10)


class ohlc_transformations(twelve_data_ohlc):
    forex_pairs = [
        'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD',
        'CADCHF', 'CADJPY', 'CHFJPY', 'EURAUD', 'EURCAD',
        'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'EURUSD',
        'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPUSD', 'GBPNZD',
        'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',
        'USDCHF', 'USDCAD', 'USDJPY'
    ]
    lag_info = {"model": {"v2": {"hour_4_lag": 3, "max_day_lag": 14}}}

    pass

    def make_day_ohlc(self):

        self.lag_info["model"]["v2"]["max_day_lag"]
        currency_data_path = resolve_currency_data_path()
        max_day_lag_by = 14
        currency_ohlc = []
        count = 0
        for pair in self.forex_pairs:
            daily_path = f'{currency_data_path}/{pair}_Daily.csv'
            df = return_df_day(daily_path)
            df = df.sort_index(ascending=True)  ##### experimental

            df = self.add_latest_index_into_ohlc(df)

            df = add_features_day(df)
            df = make_features_lagged_day(df, lag_by=14)


            currency_df = {"day_features": df, "symbol": f"{pair}"}
            currency_ohlc.append(currency_df)
            count += 1
        print(count)

        filename = "day_features_data_lagby_14.bin"
        folder_path = resolve_currency_ohlc_bin_path()
        # folder_path = "currency_ohlc"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = os.path.join(folder_path, filename)

        self.save_ohlc_with_features(currency_ohlc, path)

    def make_hour_4_ohlc(self):

        hour_4_lag_by = self.lag_info["model"]["v2"]["hour_4_lag"]
        currency_data_path = resolve_currency_data_path()
        metals = ['XAUUSD', 'XAGUSD']
        currency_ohlc = []
        count = 0

        for pair in self.forex_pairs:
            # daily_path = f'currency_data/{pair[:6]}_Daily.csv'
            hour4_path = f'{currency_data_path}/{pair}_H4.csv'
            data_h4 = return_h4_df(hour4_path)
            data_h4 = data_h4.sort_index(ascending=True)  ##### experimental

            # reindexed_h4 = self.add_latest_index_into_ohlc(data_h4)
            reindexed_h4_day = combine_ohlc_into_single_day_hour_4(data_h4)

            if pair in metals:
                lag_by = 5
            else:
                lag_by = 3
            reshaped_lagged_h4 = add_ohlc_in_lagged_hour_4(reindexed_h4_day, lag_by=lag_by)


            features_h4 = add_features_hour_4(reshaped_lagged_h4, lag_by=lag_by)


            currency_df = {"hour4_features": features_h4, "symbol": f"{pair}"}
            currency_ohlc.append(currency_df)
            count += 1

        print(count)

        filename = "hour4_features_data_lag_by_3.bin"
        folder_path = resolve_currency_ohlc_bin_path()
        # folder_path = "currency_ohlc"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        path = os.path.join(folder_path, filename)

        self.save_ohlc_with_features(currency_ohlc, path)

    def save_ohlc_with_features(self, currency_ohlc, path):

        import pickle

        with open(path, 'wb') as file:
            pickle.dump(currency_ohlc, file)

        # with open('hour4_features_data_lag_by_3_v2.bin', 'wb') as file:
        #     pickle.dump(currency_ohlc, file)

    def add_latest_index_into_ohlc(self, df):
        """
        Use : Optional when Metatrader 5 is used present day date already present in dataset
        """
        # Get the latest index datetime
        latest_date = df.index.max()

        # Calculate the next date (assuming daily frequency)
        next_date = latest_date + pd.Timedelta(days=1)

        # Check if the next date is Saturday (5) or Sunday (6)
        if next_date.weekday() == 5:  # Saturday
            next_date += pd.Timedelta(days=2)
        elif next_date.weekday() == 6:  # Sunday
            next_date += pd.Timedelta(days=1)

        # Insert the next date with a NaN value
        df.loc[next_date] = np.nan
        return df

    """
    def add_latest_index_into_ohcl(self, df):

        formatted_date = self.get_current_date()

        new_date = pd.to_datetime(
            f'{formatted_date}')  # adding tomorrow date if it's sat, sun then automatically monday selected.
        df.loc[new_date] = np.nan
        return df

    """
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


class ohlc_models(ohlc_transformations):

    def apply_predictions(self, symbol, day_data, hour4_data, day=1):
        models_path = resolve_ml_models_path()
        model, model_info = self.load_model(symbol, models_path) # "forex_models"
        alpha_type, window_size = model_info["alpha_type"], model_info["window_size"]
        X_scaled, X = get_features_transformed(symbol, day_data, hour4_data, window_size=window_size,
                                               alpha_type=alpha_type)

        date_val = X.index.values[-day]

        y_pred = model.predict(X_scaled[-day])
        if y_pred < 0.0:
            status = -1
        elif y_pred >= 0.0:
            status = 1

        return status, y_pred, date_val

    def get_date(self):

        from datetime import datetime, timedelta
        today = datetime.today()
        today_str = today.strftime('%Y-%m-%d')
        gmt_zone = pytz.timezone('UTC')
        current_gmt_time = datetime.now(gmt_zone)

        if current_gmt_time.isoweekday() == 6:
            current_gmt_time += timedelta(days=2)  # Saturday
        elif current_gmt_time.isoweekday() == 7:
            current_gmt_time += timedelta(days=1)  # Sunday

        formatted_date = current_gmt_time.strftime('%Y-%m-%d')

        return formatted_date

    def fx_status_function(self, fx_status_info, lot_size, status, symbol, net_change):

        action = ""
        if status == 1:
            action = "buy"
        elif status == -1:
            action = "sell"
        if net_change:
            fx_status_info[symbol] = {"action": action, "lot_size": lot_size, "pips_change": int(net_change)}

        else:
            fx_status_info[symbol] = {"action": action, "lot_size": lot_size}  # , 'prediction_status': status }

        return fx_status_info

    def load_model(self, symbol, path=None):

        if path: model_path = f"{path}/{symbol}_model_pips_change"
        model_path = f"{path}/{symbol}_model_pips_change"
        # else:  path = f"v2_forex_models/{symbol}_model_pips_change"

        with open(model_path, 'rb') as file:
            model_file = pickle.load(file)

        model = model_file["model"]
        model_info = model_file["model_info"]

        return model, model_info

    def load_features_files(self):

        # folder_path = "currency_ohlc"
        folder_path =  resolve_currency_ohlc_bin_path()
        filename_day = "day_features_data_lagby_14.bin"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        day_features_path = os.path.join(folder_path, filename_day)

        with open(day_features_path, 'rb') as file:
            day_features = pickle.load(file)

        # hour4_features_path = 'hour4_features_data.bin'
        filename_hour4 = 'hour4_features_data_lag_by_3.bin'
        if not os.path.exists(folder_path): os.makedirs(folder_path)

        hour4_features_path = os.path.join(folder_path, filename_hour4)
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

    def previous_day_prediction(self, symbol, day_data, hour4_data, day=2):

        pass
        models_path = resolve_ml_models_path()
        model, model_info = self.load_model(symbol,  models_path) # "forex_models"
        alpha_type, window_size = model_info["alpha_type"], model_info["window_size"]
        X_scaled, X, pips_change = get_features_transformed(symbol, day_data, hour4_data, window_size=window_size,
                                                            alpha_type=alpha_type, lookback_analysis=True)

        date_val = X.index.values[-day]

        y_pred = model.predict(X_scaled[-day])
        change_val = pips_change[-day]
        y_actual = change_val

        if y_pred < 0.0:
            status = -1
        elif y_pred >= 0.0:
            status = 1

        prediction_outcome = True if y_actual < 0.0 and y_pred < 0.0 or y_actual >= 0.0 and y_pred >= 0.0 else False
        # change_val    # return status, y_pred, date_val
        status, y_pred, date_val = self.apply_predictions(symbol, day_data, hour4_data, day=2)

        return status, y_pred, date_val, prediction_outcome

    def load_models(self):
        pass

    def apply_parameters(self):
        pass

    def make_record_schema_file(self, record_info):
        col = [ 'date',
            'AUDCAD', 'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADCHF', 'CADJPY', 'CHFJPY',
            'EURAUD', 'EURCAD', 'EURCHF', 'EURGBP', 'EURJPY', 'EURNZD', 'EURUSD',
            'GBPAUD', 'GBPCAD', 'GBPCHF', 'GBPJPY', 'GBPUSD', 'GBPNZD', 'NZDCAD', 'NZDCHF', 'NZDJPY', 'NZDUSD',
            'USDCHF',
            'USDCAD', 'USDJPY'
        ]
        df = pd.DataFrame(columns=col)
        df = df.astype('object')

        # df['date'] = record_info["date"]
        # df.set_index('date', inplace=True)
        df.index.name = 'date'

        date = record_info["date"]
        symbols = record_info["symbol"]

        # Convert the symbols dictionary into a DataFrame with date as index
        df = pd.DataFrame([symbols], index=[date])


        return df


    def append_next_date_record(self, df, record_info):
        next_date = record_info['date']
        next_symbol_record = record_info['symbol']
        # Check if next date already exists in the DataFrame
        if next_date in df.index:
            # If it exists, update the record for that date
            df.loc[next_date] = next_symbol_record
        else:
            # If it doesn't exist, add a new row for the next date
            df_next = pd.DataFrame([next_symbol_record], index=[next_date])
            df = pd.concat([df, df_next])

        return df


    # def store_fx_records(self, fx_status_info):
    #     # folder = "../records"
    #     folder = resolve_records_path()
    #     file_name = "ml_record_info.csv"
    #     file_path = os.path.join(folder, file_name)
    #     # if not os.path.exists(folder): os.makedirs(folder)
    #
    #     try:
    #         df = pd.read_csv(file_path)
    #         df = self.append_next_date_record(df, fx_status_info)
    #         df.to_csv(file_path, index = False)
    #     except:
    #         df = self.make_record_schema_file(fx_status_info)
    #         df = self.append_next_date_record(df, fx_status_info)
    #         df.to_csv(file_path)
    #
    #     pass

    def store_fx_records(self, fx_status_info):
        # folder = "../records"
        folder = resolve_records_path()
        file_name = "ml_record_info.json"
        file_path = os.path.join(folder, file_name)
        # if not os.path.exists(folder): os.makedirs(folder)

        try:  # path
            with open(file_path, 'r') as file:
                record_info = json.load(file)  # Load the JSON data into a Python dictionary
        except FileNotFoundError:
            record_info = {}  # If the file doesn't exist, start with an empty dictionary
        if len(record_info) == 0:
            record_info.update(fx_status_info)
        elif len(record_info) < 3:
            record_info = [record_info]
            record_info.append(fx_status_info)

        else:
            record_info.append(fx_status_info)


        # Step 3: Write the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(record_info, file, indent=4)


    def save_updated_in_services(self, fx_status_info):
        try:
            with open('fx_status_info.txt', 'r') as file:
                record_info = json.load(file)  # Load the JSON data into a Python dictionary

            record_info.update(fx_status_info)

            with open('fx_status_info.txt', 'w') as file:
                json.dump(fx_status_info, file)
        except FileNotFoundError:
            # record_info = {}
            with open('fx_status_info.txt', 'w') as file:
                json.dump(fx_status_info, file)


    def save_fxml_updated_ml_record(self, fx_status_info):
        path = resolve_updated_ml_record()
        filename = "trade_invoke.json"
        _path = os.path.join(path, filename)

        with open(f'{_path}', 'w') as file:
            json.dump(fx_status_info, file)
        pass
   

class access_resources(ohlc_models):
    pass


class mlpipeline(access_resources):

    @classmethod
    def return_fx_prediction_status(cls, symbols=None):
        resource = cls()
        # resource.get_ohlc_1d_4h()
        if symbols:
            resource.forex_pairs = symbols

        resource.save_ohlc(tf="1day") # now function added to made temp orary
        resource.save_ohlc(tf="4h")  # now function added to automate

        # decision whether date index is in csv file temp orary

        resource.make_day_ohlc()
        resource.make_hour_4_ohlc()

        date = resource.get_date()
        # exit()
        # function to get lot size
        lot_size = 0.01

        day_features, hour4_features = resource.load_features_files()
        fx_status_info = {}
        fx_status_info['date'] = date
        fx_status_info['symbol'] = {}
        for symbol in resource.forex_pairs:
            pass
            day_data, hour4_data = resource.get_day_hour4_features(symbol, day_features, hour4_features)
            status, net_change, _ = resource.apply_predictions(symbol, day_data, hour4_data, day=1)
            prev_status, _, prev_date, prev_prediction_outcome = resource.previous_day_prediction(symbol, day_data,
                                                                                                  hour4_data, day=2)

            symbol_status = resource.fx_status_function(fx_status_info['symbol'], lot_size, status, symbol, net_change)
            symbol_status[symbol]["prev_status"] = prev_status
            symbol_status[symbol]["prev_outcome"] = prev_prediction_outcome
            fx_status_info['symbol'] = symbol_status

        # resource.store_fx_records(fx_status_info)


        # symbol_status["prev_date"] = prev_date

        # resource.store_fx_records(fx_status_info)

        resource.save_fxml_updated_ml_record(fx_status_info)

        # resource.save_updated_in_services(fx_status_info)


        # with open('fx_status_info.txt', 'w') as file:
        #     json.dump(fx_status_info, file)




# mlpipeline.return_fx_prediction_status()

# folder_path = "currency_data"
# folder_path = resolve_currency_data_path()
# print(folder_path)
