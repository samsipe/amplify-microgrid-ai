import json
import os

import numpy as np
import pandas as pd
import requests
from pysolar.radiation import get_radiation_direct
from pysolar.solar import get_altitude, get_azimuth
from tensorflow.keras.layers import Normalization


class PredictData:
    def __init__(
        self,
        lat: float = 39.9649,
        lon: float = -75.1396,
        features: list = ["dt", "temp", "clouds"],
        cyclical_features: list = ["azimuth", "day_of_week"],
    ):
        self.lat = lat
        self.lon = lon
        self.features = features
        self.cyclical_features = cyclical_features
        
        """
        ##1) Takes in weather prediction data (API connection)
        ##2) Runs that data through weather_cleaner method to clean the data
        ##3) Runs additional_features method to add irradiance and azimuth
        ##4) Outputs clean features of weather prediction + pysolar for 48hrs
        ##-> 5) Split datetime index to a separate df
        -> 6) Run model.predict(weather_prediction)
        -> 7) Combine datetime df with predict output
        -> 8) ???
        -> 9) Profit
        """

    def weather_forecast(self):
        """
        This is the main callable function. It runs the other functions
        in order, passing the results of one to the next, to return properly
        prepared weather forecast data.
        """

        # Run method for API call to retrive data and convert JSON -> dataframe
        self.raw_weather = self._get_forecast(
            lat=self.lat, lon=self.lon, features=self.features
        )

        # Run method to clean weather data retrieved from API call
        self.clean_weather = self._forecast_clean(
            raw_forecast=self.raw_weather, cyclical_features=self.cyclical_features
        )

        # Run method to add azimuth, irradiance, and day of week
        self.all_features = self._additional_features(
            clean_weather=self.clean_weather,
            cyclical_features=self.cyclical_features,
            lat=self.lat,
            lon=self.lon,
        )

        # Run method to split datetime column into separate DF
        self.dt_split_features = self._split_dt_index(
            prepped_features=self.all_features
        )

        return self.dt_split_features

    def _get_forecast(self, lat, lon, features):
        self.lat, self.lon = lat, lon
        self.features = features

        # Set URL for API
        self.url = (
            "https://api.openweathermap.org/data/2.5/onecall?lat="
            + str(self.lat)
            + "&lon="
            + str(self.lon)
            + "&units=metric&exclude=current,minutely,daily,alerts&appid="
            + OW_API_KEY
        )

        # Pull data into JSON format
        self.data = requests.get(self.url).json()

        # Convert data into dataframe, pulling just the 3 columns we need
        self.weather_data = pd.json_normalize(self.data["hourly"])[self.features]

        print("Info: Successfully retrieved forecast data")
        return self.weather_data

    def _forecast_clean(self, raw_forecast, cyclical_features):
        self.weather_data = raw_forecast.copy()
        self.cyclical_features = cyclical_features

        # Convert from POSIX to ISO UTC time
        self.weather_data.dt = pd.to_datetime(
            self.weather_data.dt, unit="s", utc=True, infer_datetime_format=True
        )

        # Set date as index
        self.weather_data = self.weather_data.set_index("dt")

        # Create columns to capture data to be added after merge
        self.weather_data[["azimuth", "irradiance", "day_of_week"]] = np.nan

        # Create columns for capturing cyclical feature conversions
        for feature in self.cyclical_features:
            self.feat_sin, self.feat_cos = feature + "_sin", feature + "_cos"
            self.weather_data[[self.feat_sin, self.feat_cos]] = np.nan

        # Fill nulls in irradiance and azimuth with 0
        self.weather_data = self.weather_data.replace(np.nan, 0)

        print("Info: Successfully cleaned Weather data!")
        return self.weather_data

    def _additional_features(self, clean_weather, cyclical_features, lat, lon):
        self.output_df = clean_weather.copy()
        self.lat = lat
        self.lon = lon
        self.cyclical_features = cyclical_features

        # Create date list from index for calculating pysolar data
        self.date_list = list(self.output_df.index)

        for date in self.date_list:
            self.pydate = date.to_pydatetime()
            self.date = date

            # Calculate Solar Azimuth
            self.output_df.loc[self.date, "azimuth"] = get_azimuth(
                self.lat, self.lon, self.pydate
            )

            # Calculate Solar Altitude
            self.altitude_deg = get_altitude(self.lat, self.lon, self.pydate)

            # Calculate Solar Irradiance
            self.output_df.loc[self.date, "irradiance"] = get_radiation_direct(
                self.pydate, self.altitude_deg
            )

        # Replace NaNs with 0
        self.output_df = self.output_df.replace(np.nan, 0)

        print("Info: Successfully added Azimuth and Irradiance data!")
        print("Info: Converting Cyclical Features.")

        # Add day of week to Weather Data (+1 is for correct spread in next step)
        self.output_df.day_of_week = self.output_df.index.strftime("%w").astype(int) + 1

        # Add cyclical functions for columns identified in 'cyclical_features' argument
        for feature in self.cyclical_features:
            # Set column names to capture new values
            self.feat_sin, self.feat_cos = feature + "_sin", feature + "_cos"

            # Calculate spread of values in column
            self.feat_spread = (
                int(self.output_df[feature].max()) - int(self.output_df[feature].min())
            ) + 1

            # Convert to sinusoidal and cosinal values
            self.output_df[self.feat_sin] = np.sin(
                2 * np.pi * self.output_df[feature].astype(float) / self.feat_spread
            )
            self.output_df[self.feat_cos] = np.cos(
                2 * np.pi * self.output_df[feature].astype(float) / self.feat_spread
            )

            # Clean up dataframe by dropping original feature column
            self.output_df.drop([feature], axis=1, inplace=True)

        print("Info: Successfully converted cyclical features! Data is ready!")

        return self.output_df.round(2)

    def _split_dt_index(self, prepped_features):
        self.prepped_features = prepped_features.copy()

        self.prepped_features.reset_index(inplace=True)

        return pd.DataFrame(self.prepped_features.dt), self.prepped_features.drop(
            "dt", axis=1
        )