# ---
# jupyter:
#   jupytext:
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

# %% [markdown]
# # Read Between the Wines 
#
# ## Imports

# %%
import ee # import the earth engine api 
import datetime
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% [markdown]
# ## Retrieve the data
#
# ### Download the Google Earth Engine data

# %%
# authenticate & init google earth engine
ee.Authenticate()
ee.Initialize()

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

    # Combine the wind speeds into one column
    data = np.insert(data, 10, np.sqrt(data[8]**2 + data[9]**2), axis=0)
    header = np.insert(header, 10, 'wind_speed', axis=0)

    # Add the maximum variation to the end of the data
    data = np.insert(data, 11, max_variation, axis=0)
    header = np.insert(header, 11, 'max_temp_variation', axis=0)
    
    # Remove lat/lon, time, and wind components
    data = np.delete(data, [0, 1, 2, 8, 9], axis=0)
    header = np.delete(header, [0, 1, 2, 8, 9], axis=0)

    # Convert the data to a pandas dataframe
    df = pd.DataFrame(data=data, index=header).T

    return display(df)


# get image
get_image_info(135, 35, 1994)
