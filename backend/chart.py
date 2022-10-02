from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
# Imports
import pandas as pd
import numpy as np
from bokeh.plotting import ColumnDataSource, figure, output_file, show, gmap
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper, Slider, Select
from bokeh.tile_providers import CARTODBPOSITRON, get_provider
import os
import haversine as hs
from pyproj import Proj, transform
import flask


def plot_html(plot):
    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # render template
    script, div = components(plot)
    html = flask.render_template(
        'chart.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    )
    return html


def wine_price():
    # ## Plot of quality vs price

    # +
    # Read in data into numpy array
    df = pd.read_csv('../data/wine_processed.csv', header=0)

    # Convert points and prices to numpy arrays
    points = df['points'].to_numpy()
    prices = df['price'].to_numpy()

    # Calculate mean of prices at each quality value
    mean_price = []
    for quality in np.unique(points):
        # All indices at some price "quality"
        indices = np.asarray(points == quality).nonzero()[0]

        # Take mean of prices (at values that correspond to quality)
        mean_price.append(np.mean(prices[indices]))

    output_file("toolbar.html")

    # Set dataframe as source for bokeh
    source = ColumnDataSource(data=dict(
        Points=df['points'],
        Price=df['price'],
        Variety=df['title'],
        Winery=df['location'],
        Year=df['year']
    ))

    TOOLTIPS = [
        ('Points', '@Points'),
        ('Price', '@Price'),
        ('Variety', '@Variety'),
        ('Winery', '@Winery'),
        ('Year', '@Year')
    ]

    p = figure(plot_width=500, plot_height=500, title='Wine Data', tooltips=TOOLTIPS, x_range=(0, 1000), y_range=(80, 103))

    p.circle(x='Price', y='Points', source=source, size=4, color='red', alpha=0.5)
    p.circle(x=mean_price, y=np.unique(points), size=7, color='blue', alpha=1)
    p.line(x=mean_price, y=np.unique(points), line_width=3, color='blue', alpha=1)

    p.xaxis.axis_label = 'Price in USD'
    p.yaxis.axis_label = 'Quality (out of 100)'
    return p

# -

# ## Bang for you buck?

# Mean price list starts at 80, ends at 100
def good_price(user_price, user_quality):

    # Normalize
    user_quality = user_quality - 80

    ideal_price = np.round(mean_price[user_quality])

    if user_price > ideal_price: # Overpaying
        return f"You're paying too much for this bottle! An average wine of this quality only costs ${ideal_price}"
    elif user_price == ideal_price:
        return f"The average wine of this quality costs ${ideal_price}--good job!"
    else:
        return f"The average bottle of this quality costs ${ideal_price}. Steal!"


def full_data_frame():
    # Read in data into numpy array
    df = pd.read_csv('../data/wine.csv', header=0)
    df = df.dropna()
    return df


def nearby_wineries(lat, lon, df):
    # Convert lats and lons to numpy arrays
    lats = df['lat'].to_numpy()
    lons = df['lon'].to_numpy()

    # Find distance to each winery
    distances = []
    for j in range(len(lats)):
        distances.append(hs.haversine((lat, lon), (lats[j], lons[j])))

    # Append distances to dataframe
    df['distance'] = np.round(distances)

    # Sort by distance and return top 10
    df = df.sort_values(by=['distance'])

    # Count number of unique wineries from top
    unique_wineries = set()
    for i in range(len(df)):
        unique_wineries.add((df.iloc[i]['location'], df.iloc[i]['distance'], df.iloc[i]['lat'], df.iloc[i]['lon'], df.iloc[i]['points'], df.iloc[i]['price']))
        if len(unique_wineries) == 60:
            break

    return unique_wineries


# -

# ## Nearby wineries on map

# +
def map_wineries(user_lat, user_lon, df):
    # Define lists
    lats, lons, distances, wineries, qualities, prices = [[] for i in range(6)]

    for vinyard in nearby_wineries(user_lat, user_lon, df):
        # Unpack set
        name, distance, lat, lon, quality, price = vinyard[0], vinyard[1], vinyard[2], vinyard[3], vinyard[4], vinyard[5]

        # Append to lists
        if not name in wineries : wineries.append(name)
        if not distance in distances : distances.append(distance)
        if not lat in lats : lats.append(lat), prices.append(price)
        if not lon in lons : lons.append(lon), qualities.append(quality)

    # Projection in/out
    inproj = Proj(init='epsg:4326')
    outproj = Proj(init='epsg:3857')

    # Transforming lat/lons to web mercator
    x, y = transform(inproj, outproj, lons, lats)
    user_x, user_y = transform(inproj, outproj, user_lon, user_lat)

    output_file("tile.html")

    tile_provider = get_provider(CARTODBPOSITRON)

    source = ColumnDataSource(
        data=dict(lat=y,
                  lon=x,
                  winery=wineries,
                  distance=distances,
                  quality=qualities,
                  price=prices)
    )

    TOOLTIPS_2 = [
        ('Winery', '@winery'),
        ('Distance', '@distance km'),
        ('Quality', '@quality'),
        ('Price', '@price')
    ]

    # range bounds supplied in web mercator coordinates
    p = figure(x_range=(user_x - 1000000, user_x + 1000000), y_range=(user_y - 1000000, user_y + 1000000),
               x_axis_type="mercator", y_axis_type="mercator", tooltips=TOOLTIPS_2)
    p.add_tile(tile_provider)

    # Surrounding wineries in blue and user location in red
    p.circle(x="lon", y="lat", size=15, fill_color="blue", fill_alpha=0.8, source=source)
    p.circle(x=user_x, y=user_y, size=15, fill_color="red", fill_alpha=1)

    return p
