
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from glob import glob
from os.path import dirname, abspath, join, exists
from sklearn import linear_model
from typing import Tuple

# Pysolar
from pysolar.solar import get_altitude, radiation

# ClearML
from clearml import Task
from clearml import Dataset

# Constants 
BUILDING_LAT = 39.9649
BUILDING_LON = -75.1396
PROJECT_DIR = dirname((abspath('')))
DATA_DIR = abspath(join(PROJECT_DIR, "data"))
LOC_BUILDING_DATA_PATH = abspath(join(DATA_DIR, "2021-10-19_2022-01-09_CF2.csv"))
LOC_WEATHER_DATA_PATH = abspath(join(DATA_DIR, "CF2_Weather_2020-2022.csv"))

#TODO: replace prints with logging
class DataGenerator():
    """
    Generates data for training using weather and power data.
    """
    clearml_task: Task = None
    building_data_dir: str = ""
    weather_data_dir: str = ""

    # Data
    start_date = None
    end_date = None
    building_data_keep_columns = []
    weather_data_keep_columns = []
    building_data: pd.DataFrame = None        # Building data
    weather_data: pd.DataFrame = None         # Weather data

    def __init__(self, use_local_data: bool = True, weather_features: list = None, building_features: list = None):
        """
        Initialize data generator.

        Arguments:
            use_local_data (bool)   : whether or not to use local data or ClearML server
        """
        self.use_local_data = use_local_data    # track for later use

        if use_local_data:
            self.building_data_dir = LOC_BUILDING_DATA_PATH
            self.weather_data_dir = LOC_WEATHER_DATA_PATH
        else:
            # ClearML Stuff
            self.clearml_task = Task.init(project_name="amplify", task_name="power-ss-notebook")

            self.building_data_dir = glob(
                Dataset.get(
                    dataset_project="amplify",
                    dataset_name="building_data"
                ).get_local_copy()
                + "/**"
            )[0]

            self.weather_data_dir = glob(
                Dataset.get(
                    dataset_project="amplify",
                    dataset_name="weather_data"
                ).get_local_copy()
                + "/**"
            )[0]

        self.weather_data_keep_columns = weather_features if weather_features else\
            [
                'temp',
                'pressure',
                'humidity',
                'clouds_all'
            ]

        self.building_data_keep_columns = building_features if building_features else\
            [
                "True Power (kW)",
                "Total Energy (kWh)",
                "Reactive Energy (kVARh)",
                "Reactive Power (kVAR)",
                "Apparent Power (kVA)",
                "Apparent Energy (kVAh)",
                "dPF",
                "aPF",
                "Current (A)"
            ]

    def LoadData(self, building_data_path: str = None, weather_data_path: str = None) -> Tuple[bool, pd.DataFrame, pd.DataFrame]:
        """
        Loads data for a specified building/weather data paths. If no paths are specified, uses
        default paths.

        *NOTE: if providing a path, will only use local data for paths not provided

        Arguments:
            building_data_path (str)    : building data file path (optional)
            weather_data_path (str)     : weather data file path (optional)

        Return:
            (load_success, building_data, weather_data)

            load_success (bool)         : whether or not loading data was successful
            building_data (dataframe)   : building data
            weather_data (dataframe)    : weather data
        """
        load_success = False
        if building_data_path:
            self.building_data_dir = building_data_path
            self.use_local_data = True
        if weather_data_path:
            self.weather_data_dir = weather_data_path
            self.use_local_data = True

        if self._LoadBuildingData() and self._LoadWeatherData():
            load_success = True

        return (load_success, self.building_data, self.weather_data)


    def _LoadBuildingData(self) -> bool:
        """
        Load and format building data from file

        Return:
            whether or not loading building data was successful (bool)
        """
        load_success = True
        #TODO: make try-catch exception specific
        try:
            self.building_data = pd.read_csv(self.building_data_dir, header=None, low_memory=False)

            # Forward fill the header name for each PowerScout
            self.building_data.iloc[0] = self.building_data.T[0].fillna(method="ffill")

            # Rename the 'nan' block
            self.building_data.loc[0, 0] = "Timestamp"

            # Create the multi-index
            self.building_data.columns = [list(self.building_data.iloc[0]), list(self.building_data.iloc[1])]

            # Drop the first two rows because they're just the column names, and any column with only nulls
            self.building_data = self.building_data[2:]

            # Convert timestamp column to datetime format 
            self.building_data.Timestamp = pd.to_datetime(
                self.building_data.Timestamp.Timestamp, 
                infer_datetime_format=True,
            )

            # Set Timestamp column as index, set columns to type 'float', rename index
            self.building_data = (
                self.building_data.set_index([("Timestamp", "Timestamp")])
                .replace("-", np.nan)
                .astype(float)
            )
            self.building_data.index.rename("Timestamp", inplace=True)

            # Set building_data to Eastern timezone and then convert to UTC
            self.building_data = self.building_data.tz_localize(
                'America/New_York',
                ambiguous=True).tz_convert('UTC')

            # deduplicate index
            self.building_data = self.building_data.drop_duplicates(keep='last')

            # Drop any column or row with all nulls
            self.building_data = self.building_data.dropna(how="all", axis=1).dropna(how="all", axis=0)

            # remove noise (zeros) from the building_data
            self.building_data = self.building_data.replace(0, np.nan).fillna(method="ffill")

            # Slice to the two power systems we're monitoring and rename columns
            #TODO: make the power systems vairable at some point
            self.building_data = self.building_data[["PowerScout DPS126", "PowerScout DPS121"]].rename(
                columns={"PowerScout DPS126": "solar", "PowerScout DPS121": "usage"}
            )

            # Create our separate y-columns: solar pwr generated, buildling pwr used
            idx = pd.IndexSlice

            # Create DF with only Energy - keep just the last value (meter readings)
            self.building_data = self.building_data.loc[idx[:], idx[:, self.building_data_keep_columns]]

            # Set some relevant datetimes based on building power data timeframe
            self.end_date = dt.datetime.today()
            self.start_date = self.building_data.index[-1]

            print("Info: Successfully loaded building data!")

        except:
            print("Error: Cannot load building data!")
            load_success = False

        return load_success

    def _LoadWeatherData(self) -> bool:
        """
        Load and format weather data from file

        *NOTE: presumes that building data has already been loaded

        Return:
            whether or not loading data was successful (bool)
        """
        if self.building_data is None:
            return False

        load_success = True
        #TODO: make try-catch exception specific
        try:
            if self.use_local_data:
                self.weather_data = pd.read_csv(self.weather_data_dir)
            else:
                self.weather_data = pd.read_csv(self.weather_data_dir, header=0, low_memory=False)

            #TODO clip weather data to start and end on same dates as building data

            ## Clean up datetime, drop 2nd datetime column
            # Convert from POSIX to ISO UTC time
            self.weather_data.dt = pd.to_datetime(
                self.weather_data.dt,
                unit='s',
                utc=True,
                infer_datetime_format=True
            )

            # Set date as index
            self.weather_data = self.weather_data.set_index('dt')

            # Drop 2nd datetime column that's not needed
            self.weather_data = self.weather_data[self.weather_data_keep_columns]

            # Add Solar Irradiance
            self.weather_data['irradiance'] = np.nan
            date_list = list(self.weather_data.index)
            for date in date_list:
                altitude_deg = get_altitude(BUILDING_LAT, BUILDING_LON, date.to_pydatetime())
                self.weather_data.loc[date, 'irridance'] =\
                    radiation.get_radiation_direct(date.to_pydatetime(), altitude_deg)

            # Trim weather data to match building data timeframe:
            self.weather_data =\
                self.weather_data[(self.weather_data.index >= self.building_data.index[-1])
                                & (self.weather_data.index <= self.building_data.index[0])]

            # Fill nulls in irradiance with 0
            self.weather_data = self.weather_data.replace(np.nan, 0)

            # deduplicate index
            self.weather_data = self.weather_data.drop_duplicates()

            ## Match weather_data index to building_data index timestamps, and ffill missing weather data
            # Add new index locations for the missing timestamps
            self.weather_data =\
                self.weather_data.append(
                    pd.DataFrame(set(self.building_data.index) - set(self.weather_data.index)).set_index(0))

            # Embed the added index timestamps into the correct time
            self.weather_data = self.weather_data.reset_index().sort_values('index').set_index('index')

            # Forward fill the weather_data
            self.weather_data = self.weather_data.fillna(method='ffill')

            print("Info: Successfully loaded weather data!")

        except:
            print("Error: Cannot load weather data")
            load_success = False

        return load_success
