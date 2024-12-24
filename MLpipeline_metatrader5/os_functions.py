import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def resolve_data_transformation_for_model_evaluation_path(source_type="metatrader5", destination= "build-ready"):

    # folder_name = "fxml_app/data_sources/metatrader5/build-ready"
    parent_folder = "data_sources"
    folder_name = source_type
    # Get the current working directory
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)  # Get the last part of the path (directory name)

    # print(current_dir_name)
    # Determine where to place the `records` directory
    if current_dir_name in ["views", "controllers", "services", "resources"]:
        # Place `records` outside the current directory
        folder_path = os.path.abspath(os.path.join(current_dir, "..", parent_folder, source_type, destination))
        location = "outside"

    elif current_dir_name == "data_sources" or current_dir_name != "fxml_app":
        # Place `records` inside the `app` directory
        folder_path = os.path.join(current_dir, folder_name, destination)
        location = "inside app"


    elif current_dir_name == "fxml_app" or current_dir_name != "fxml_app":
        # Place `records` inside the `app` directory
        folder_path = os.path.join(current_dir, parent_folder, source_type, destination)
        location = "inside app"

    # Ensure the `records` directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        # print(f"'records' directory not found. Created {location}: {folder_path}")
    # else:
    #     print(f"'records' directory exists {location}: {folder_path}")

    # Create the full file path
    _path = folder_path
    # print(location)
    # print(f"Full path: {_path}")

    return _path

# _path = resolve_data_transformation_for_model_evaluation_path(destination= "build-ready")
# print(_path)

def resolve_updated_ml_record():

    pass
    folder_name = "updated_Ml_record"
    # file_name = "ml_record_info.csv"

    # Get the current working directory
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)  # Get the last part of the path (directory name)

    # Determine the base directory for placing the `records` folder
    if current_dir_name in ["views", "controllers", "services"]:
        # Place `records` outside the current directory
        base_dir = os.path.abspath(os.path.join(current_dir, ".."))
        location = "outside current directory"
    elif current_dir_name == "fxml_app" or current_dir_name != "fxml_app":
        # Place `records` inside the `fxml_app` directory
        base_dir = current_dir
        location = "inside fxml_app"

    # Ensure the `records` directory exists
    _path = os.path.join(base_dir, folder_name)

    if not os.path.exists(_path):
        os.makedirs(_path, exist_ok=True)
    #     print(f"'records' directory not found. Created {location}: {_path}")
    # else:
    #     print(f"'records' directory exists {location}: {_path}")

    # Create the absolute path for the file
    # _path = os.path.join(base_dir, folder_name) # os.path.join(folder_path, file_name)
    # print(f"Full file path: {_path}")

    return _path




def resolve_records_path():
    folder_name = "records"
    file_name = "ml_record_info.csv"

    # Get the current working directory
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)  # Get the last part of the path (directory name)

    # Determine the base directory for placing the `records` folder
    if current_dir_name in ["views", "controllers", "services"]:
        # Place `records` outside the current directory
        base_dir = os.path.abspath(os.path.join(current_dir, ".."))
        location = "outside current directory"
    elif current_dir_name == "fxml_app" or current_dir_name != "fxml_app" :
        # Place `records` inside the `fxml_app` directory
        base_dir = current_dir
        location = "inside fxml_app"
    # else:
    #     # Default behavior: place `records` inside the current directory
    #     base_dir = current_dir
    #     location = "inside current directory"

    # Create the absolute path for the `records` folder
    # folder_path = os.path.join(base_dir, folder_name)

    # Ensure the `records` directory exists
    _path = os.path.join(base_dir, folder_name)

    if not os.path.exists(_path):
        os.makedirs(_path, exist_ok=True)
    #     print(f"'records' directory not found. Created {location}: {_path}")
    # else:
    #     print(f"'records' directory exists {location}: {_path}")

    # Create the absolute path for the file
    # _path = os.path.join(base_dir, folder_name) # os.path.join(folder_path, file_name)
    # print(f"Full file path: {_path}")

    return _path

def resolve_currency_ohlc_bin_path():

    folder_name = "currency_ohlc_bin"

    # Get the current working directory
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)  # Get the last part of the path (directory name)

    # print(current_dir_name)
    # Determine where to place the `records` directory
    if current_dir_name in ["views", "controllers", "resources"]:
        # Place `records` outside the current directory
        folder_path = os.path.abspath(os.path.join(current_dir, "..", "services", folder_name))
        location = "outside but inside services"

    elif current_dir_name == "services" :
        folder_path = os.path.join(current_dir, folder_name)
        location = "inside"
        pass

    elif current_dir_name == "fxml_app" or current_dir_name != "fxml_app":
        # Place `records` inside the `app` directory
        folder_path = os.path.join(current_dir, "services", folder_name)
        location = "inside services"

    # Ensure the `records` directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    #     print(f"'records' directory not found. Created {location}: {folder_path}")
    # else:
    #     print(f"'records' directory exists {location}: {folder_path}")

    # Create the full file path
    _path = folder_path
    # print(location)
    # print(f"Full path: {_path}")

    return _path


def resolve_currency_data_path():

    folder_name = "currency_data"


    # Get the current working directory
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)  # Get the last part of the path (directory name)

    # print(current_dir_name)
    # Determine where to place the `records` directory
    if current_dir_name in ["views", "controllers", "services", "resources"]:
        # Place `records` outside the current directory
        folder_path = os.path.abspath(os.path.join(current_dir, "..", folder_name))
        location = "outside"
    elif current_dir_name == "fxml_app" or current_dir_name != "fxml_app":
        # Place `records` inside the `app` directory
        folder_path = os.path.join(current_dir, folder_name)
        location = "inside app"

    # Ensure the `records` directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        # print(f"'records' directory not found. Created {location}: {folder_path}")
    # else:
    #     print(f"'records' directory exists {location}: {folder_path}")

    # Create the full file path
    _path = folder_path
    # print(location)
    # print(f"Full path: {_path}")

    return _path


def resolve_ml_models_path():

    folder_name = "ml_models/baseline_ml_models/forex_models"


    # Get the current working directory
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)  # Get the last part of the path (directory name)

    print(current_dir_name)
    # Determine where to place the `records` directory
    if current_dir_name in ["views", "controllers", "services", "resources"]:
        # Place `records` outside the current directory
        folder_path = os.path.abspath(os.path.join(current_dir, "..", folder_name))
        location = "outside"
    elif current_dir_name == "fxml_app" or current_dir_name != "fxml_app":
        # Place `records` inside the `app` directory
        folder_path = os.path.join(current_dir, folder_name)
        location = "inside app"

    # Ensure the `records` directory exists
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path, exist_ok=True)
    #     print(f"'records' directory not found. Created {location}: {folder_path}")
    # else:
    #     print(f"'records' directory exists {location}: {folder_path}")

    # Create the full file path
    _path = folder_path
    # print(location)
    print(f"Full path: {_path}")

    return _path


def resolve_profile_path(profile_name, models_type ):

    folder_name = "ml_models/baseline_ml_models/forex_models"
    folder_name = f"ml_models/{profile_name}"

    # Get the current working directory
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)  # Get the last part of the path (directory name)

    print(current_dir_name)
    # Determine where to place the `records` directory
    if current_dir_name in ["views", "controllers", "services", "resources"]:
        # Place `records` outside the current directory
        folder_path = os.path.abspath(os.path.join(current_dir, "..", folder_name))
        location = "outside"
    elif current_dir_name == "fxml_app" or current_dir_name != "fxml_app":
        # Place `records` inside the `app` directory
        folder_path = os.path.join(current_dir, folder_name)
        location = "inside app"


    # Create the full file path
    _path = folder_path
    # print(location)
    print(f"Full path: {_path}")

    return _path


def resolve_license_bin_path():

    folder_name = "config"
    file_name = "license_key.bin"

    # Get the current working directory
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)  # Get the last part of the path (directory name)

    print(current_dir_name)
    # Determine where to place the `records` directory
    if current_dir_name in ["views", "controllers", "services", "resources"]:
        # Place `records` outside the current directory
        folder_path = os.path.abspath(os.path.join(current_dir, "..", folder_name))
        location = "outside"
    elif current_dir_name == "fxml_app" or current_dir_name != "fxml_app":
        # Place `records` inside the `app` directory
        folder_path = os.path.join(current_dir, folder_name)
        location = "inside app"

    # Ensure the `records` directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"'records' directory not found. Created {location}: {folder_path}")
    else:
        print(f"'records' directory exists {location}: {folder_path}")

    # Create the full file path
    file_path = os.path.join(folder_path, file_name)
    print(f"Full file path: {file_path}")

    return file_path


def resolve_fx_evaluation_tables():
    folder_name = "fx_evaluation_table"

    # Get the current working directory
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)  # Get the last part of the path (directory name)

    print(current_dir_name)
    # Determine where to place the `records` directory
    if current_dir_name in ["views", "controllers", "resources", "services"]:
        # Place `records` outside the current directory
        folder_path = os.path.abspath(os.path.join(current_dir, "..", folder_name))
        location = "outside "

    # elif current_dir_name == "services":
    #     folder_path = os.path.join(current_dir, folder_name)
    #     location = "inside"
    #     pass

    elif current_dir_name == "fxml_app" or current_dir_name != "fxml_app":
        # Place `records` inside the `app` directory
        folder_path = os.path.join(current_dir, folder_name)
        location = "inside"

    # Ensure the `records` directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    #     print(f"'records' directory not found. Created {location}: {folder_path}")
    # else:
    #     print(f"'records' directory exists {location}: {folder_path}")

    # Create the full file path
    _path = folder_path
    # print(location)
    # print(f"Full path: {_path}")

    return _path

def resolve_api_path():

    folder_name = "config"
    file_name = "api_key.bin"

    # Get the current working directory
    current_dir = os.getcwd()
    current_dir_name = os.path.basename(current_dir)  # Get the last part of the path (directory name)

    print(current_dir_name)
    # Determine where to place the `records` directory
    if current_dir_name in ["views", "controllers", "services", "resources"]:
        # Place `records` outside the current directory
        folder_path = os.path.abspath(os.path.join(current_dir, "..", folder_name))
        location = "outside"
    elif current_dir_name == "fxml_app" or current_dir_name != "fxml_app":
        # Place `records` inside the `app` directory
        folder_path = os.path.join(current_dir, folder_name)
        location = "inside app"

    # Ensure the `records` directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"'records' directory not found. Created {location}: {folder_path}")
    else:
        print(f"'records' directory exists {location}: {folder_path}")

    # Create the full file path
    file_path = os.path.join(folder_path, file_name)
    print(f"Full file path: {file_path}")

    return file_path

# print(resolve_updated_ml_record())
# resolve_fx_evaluation_tables()

# resolve_ml_models_path()
#
# resolve_currency_data_path()
