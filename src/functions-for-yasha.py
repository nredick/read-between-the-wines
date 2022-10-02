# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.9.13 (''env'': venv)'
#     language: python
#     name: python3
# ---

# %%
import geocoder
import numpy as np
import ee # import the earth engine api 
import datetime
import folium
import matplotlib.pyplot as plt
import pandas as pd
import re


# %% [markdown]
# ### THIS FUNCTION IS FOR TAKING A WINERY NAME (STRING) AND CONVERTING IT TO LAT/LON

# %%
def get_coords(location):
    try:
        lat, long = geocoder.arcgis(location, maxRows=1).latlng
        return lat, long
    except:
        return np.nan, np.nan


# %% [markdown]
# The outputs of this function (lat, long) are then passed to the function named `get_image_info` below, along with the year (also user input)

# %%
# authenticate & init google earth engine
ee.Authenticate()
ee.Initialize()

# %% [markdown]
# To authorize access needed by Earth Engine, open the following URL in a web browser and follow the instructions:
#
# https://code.earthengine.google.com/client-auth?scopes=https%3A//www.googleapis.com/auth/earthengine%20https%3A//www.googleapis.com/auth/devstorage.full_control&request_id=HqtVhos9abJBMjeBJSUu7Q6afHLElBFxgAaH49TmASM&tc=ufA9FqcFzej14_TXxnPHko0eUFB1NHQ_9hVaY68yr3E&cc=rAwpTTsKKJxpmWwVQsZCIIM_3DxJ5aRQI7AAMyCHfiE
#
# The authorization workflow will generate a code, which you should paste in the box below.

# %%
import geemap

img_collection = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY")

def get_image_info(lon, lat, year):

    # Filtering dates by northern and southern hemispheres
    if lat > 0: # northern hemisphere
        start_date = f'{year}-04-01'
        end_date = f'{year}-10-31'
    else: # southern hemisphere
        start_date = f'{year - 1}-10-31'
        end_date = f'{year}-04-01'

    # Collection unique to the year
    func_collection = img_collection.select('skin_temperature', 'dewpoint_temperature_2m', 
                                            'volumetric_soil_water_layer_1', 'surface_pressure', 'total_precipitation', 
                                            'u_component_of_wind_10m', 'v_component_of_wind_10m').filterDate(start_date, end_date)

    # Point of interest in lon/lat
    location = ee.Geometry.Point(lon, lat)

    # Load data into numpy array
    data = np.asarray(func_collection.getRegion(location, 1000).getInfo())

    # Convert all rows past the first to floats
    header = data[0][1:]
    data = data[1:, 1:].astype(float)

    # Store temperature data in a separate array and calculate maximum variation from mean
    temp_data = data[:, 3]
    max_variation = np.max( np.abs(temp_data - np.mean(temp_data)) )

    # Transpose the data and take the mean of each row
    data = np.mean(data.T, axis=1)

    # Combine the wind speeds into one column and add temperature variation
    data = np.insert(data, 10, np.sqrt(data[8]**2 + data[9]**2), axis=0) # wind speed
    header = np.insert(header, 10, 'wind_speed', axis=0)

    data = np.insert(data, 11, max_variation, axis=0) # temperature variation
    header = np.insert(header, 11, 'max_temp_variation', axis=0)

    # Remove lat/lon, time, and wind components
    data = np.delete(data, [0, 1, 2, 8, 9], axis=0)
    header = np.delete(header, [0, 1, 2, 8, 9], axis=0)

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data=data, index=header).T

    return df

# %% [markdown]
# The data frame this function outputs has one row and multiple columns. Each column is one input into the model.
