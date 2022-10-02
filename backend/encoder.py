import geocoder
import numpy as np
import pandas as pd
import ee  # earth engine api


class Encoder:
    def __init__(self):
        service_account = 'service_account'
        credentials = ee.ServiceAccountCredentials(service_account, 'key.json')
        ee.Initialize(credentials)
        ee.Initialize()
        self._image_collection = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY")

    def encode_features(self, year, location):
        latitude, longitude = self._geoencode(location)
        if latitude is np.nan or longitude is np.nan:
            return None
        data_frame = self._get_image_info(latitude, longitude, year)
        return data_frame

    def _geoencode(self, location):
        try:
            lat, long = geocoder.arcgis(location, maxRows=1).latlng
            return lat, long
        except:
            return np.nan, np.nan

    def _get_image_info(self, lon, lat, year):
        end_date, start_date = self._dates_by_hemisphere(lat, year)
        func_collection = self._annual_collection(end_date, start_date)

        # Point of interest in lon/lat
        location = ee.Geometry.Point(lon, lat)

        data, header = self._formulate_data(func_collection, location)
        data, header = self._construct_features(data, header)
        data_frame = self._convert_to_data_frame(data, header)

        return data_frame

    def _convert_to_data_frame(self, data, header):
        data_frame = pd.DataFrame(data=data, index=header).T
        return data_frame

    def _construct_features(self, data, header):
        # Store temperature data in a separate array and calculate maximum variation from mean
        temp_data = data[:, 3]
        max_variation = np.max(np.abs(temp_data - np.mean(temp_data)))
        # Transpose the data and take the mean of each row
        data = np.mean(data.T, axis=1)
        # Combine the wind speeds into one column and add temperature variation
        data = np.insert(data, 10, np.sqrt(data[8] ** 2 + data[9] ** 2), axis=0)  # wind speed
        header = np.insert(header, 10, 'wind_speed', axis=0)
        data = np.insert(data, 11, max_variation, axis=0)  # temperature variation
        header = np.insert(header, 11, 'max_temp_variation', axis=0)
        # Remove lat/lon, time, and wind components
        data = np.delete(data, [0, 1, 2, 8, 9], axis=0)
        header = np.delete(header, [0, 1, 2, 8, 9], axis=0)
        return data, header

    def _formulate_data(self, func_collection, location):
        # Load data into numpy array
        data = np.asarray(func_collection.getRegion(location, 1000).getInfo())
        # Convert all rows past the first to floats
        header = data[0][1:]
        data = data[1:, 1:].astype(float)
        return data, header

    def _annual_collection(self, end_date, start_date):
        func_collection = self._image_collection.select('skin_temperature', 'dewpoint_temperature_2m',
                                                        'volumetric_soil_water_layer_1', 'surface_pressure',
                                                        'total_precipitation',
                                                        'u_component_of_wind_10m',
                                                        'v_component_of_wind_10m').filterDate(start_date, end_date)
        return func_collection

    def _dates_by_hemisphere(self, lat, year):
        if lat > 0:  # northern hemisphere
            start_date = f'{year}-04-01'
            end_date = f'{year}-10-31'
        else:  # southern hemisphere
            start_date = f'{year - 1}-10-31'
            end_date = f'{year}-04-01'
        return end_date, start_date
