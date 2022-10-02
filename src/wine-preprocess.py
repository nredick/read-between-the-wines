#!/usr/bin/env python
# coding: utf-8
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
#     display_name: 'Python 3.10.7 (''venv_maishacks2022'': venv)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocess Wine Data 

# %%
import numpy as np
import pandas as pd
import geocoder


# %%
path = "../data/winemag-data-130k-v2.csv"
wine = pd.read_csv(path)

# define a function to create search string for winery location
def get_location(x):
    return " ".join(map(str, [x['winery'], "Winery", x['province'], x['country']]))

# remove rows with nan values
wine.dropna(subset=['winery', 'province', 'country', 'points', 'title', 'price'], inplace=True, axis='rows')

wine['location'] = wine.apply(lambda x: get_location(x), axis=1)

wine = wine.drop(wine.columns.difference(['points', 'price', 'title', 'location']), axis=1)


# %%


unique_locs = wine['location'].unique()


# %%
# define a function to get coordinates from location
def get_coords(location):
    try:
        lat, long = geocoder.arcgis(location, maxRows=1).latlng
        return lat, long
    except:
        return np.nan, np.nan
    

wine['lat'], wine['lon'] = wine.apply(lambda x: get_coords(x['location']), axis=1)

latlon = pd.Series(unique_locs).map(get_coords)
latlon = pd.DataFrame(list(latlon), columns=['lat', 'lon'])


# %%
unique_locs = pd.Series(unique_locs)
latlon_list = latlon.values.tolist()


# %%
def get_latlon(row, get='lat'):
    loc = row.T.location
    index = unique_locs.index[unique_locs == loc].tolist()[0]
    lat, lon = latlon_list[index]
    
    if get == 'lat':
        return lat
    else:
        return lon

wine['lat'] = wine.apply(lambda x: get_latlon(x), axis=1)
wine['lon'] = wine.apply(lambda x: get_latlon(x, get='lon'), axis=1)

wine.head(5)    


# %%
wine.to_csv("wine.csv", header=0, index=False)

