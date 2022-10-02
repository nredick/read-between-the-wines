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
import re

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


# %%
# Import csv of wine data
wine_data = pd.read_csv('../data/wine.csv')

# Sample 33% of the data
wine_data = wine_data.sample(frac=0.33)

def extract_year(x):
    try:
        y = int( (re.findall("[1-3][0-9]{3}", x))[0]) 
    except:
        y = np.nan
    return y

# Extract years from wine data and convert to int
wine_data['year'] = wine_data['title'].apply(extract_year)

# Drop rows with no year
wine_data = wine_data.dropna(subset=['year'])

# Drop rows with years outside of 1980-2017 and no lat/lon
wine_data = wine_data[(wine_data['year'] >= 1981) & (wine_data['year'] <= 2021) & (wine_data['lat'].notna()) & (wine_data['lon'].notna())]

# Convert years to int
wine_data['year'] = wine_data['year'].astype(int)

# Extract EE data for each row and add as new columns in the dataframe
for index, row in wine_data.iterrows():
    df = get_image_info(row['lon'], row['lat'], row['year'])
    for col in df.columns:
        wine_data.loc[index, col] = df[col][0]
        
    print(f'Finished row {index}')

# Save dataframe to csv
wine_data.to_csv('../data/wine_processed.csv', index=False)


