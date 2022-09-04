#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'> Data Project from Gett - Analysis of Cancelled Orders <h1>

# ![download.png](attachment:download.png)

# Gett, previously known as GetTaxi, is an Israeli-developed technology platform solely focused on corporate Ground Transportation Management (GTM). They have an application where clients can order taxis, and drivers can accept their rides (offers). At the moment, when the client clicks the Order button in the application, the matching system searches for the most relevant drivers and offers them the order. In this task, we would like to investigate some matching metrics for orders that did not completed successfully, i.e., the customer didn't end up getting a car.

# ## Packages 

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt


# ## Data

# In[3]:


df_orders = pd.read_csv("C:/Users/visha/OneDrive/Documents/GETT Taxi Failed Orders Analysis/datasets/data_orders.csv")
df_offers = pd.read_csv("C:/Users/visha/OneDrive/Documents/GETT Taxi Failed Orders Analysis/datasets/data_offers.csv")
df = df_orders.merge(df_offers, how='inner', on='order_gk')


# ## Data Dictionary
# 
# We have two data sets: data_orders and data_offers, both being stored in a CSV format. 
# 
# The data_orders data set contains the following columns:
# 
# - order_datetime - time of the order
# - origin_longitude - longitude of the order
# - origin_latitude - latitude of the order
# - m_order_eta - time before order arrival
# - order_gk - order number
# - order_status_key - status, an enumeration consisting of the following mapping:
#     - 4 - cancelled by client,
#     - 9 - cancelled by system, i.e., a reject
# - is_driver_assigned_key - whether a driver has been assigned
# - cancellation_time_in_seconds - how many seconds passed before cancellation
# 
# The data_offers data set is a simple map with 2 columns:
# 
# - order_gk - order number, associated with the same column from the orders data set
# - offer_id - ID of an offer

# ## Understanding Data

# In[4]:


print("Size of orders datasest: {}".format(df_orders.shape))
print("Size of offers datasest: {}".format(df_offers.shape))


# In[5]:


df_orders.head()


# In[6]:


df_offers.head()


# In[7]:


df.head()


# ## Cancellations before and after driver assignment

# In[8]:


sns.countplot(data=df, x='is_driver_assigned_key')


# Above countplot shows that orders with no driver assigned has higher chances (3:1) of being cancelled than orders with drivers assigned.
# It makes sense, because of the following situations in the case of order with no drivers assigned:
# 1. There are may be some locations, which is least preferred by drivers.
# 2. There can be some locations which have low volume of drivers.
# 3. There can be a possibility, majority of orders gets cancelled during peak time when there are less availability of drivers.
# 4. There can be specific type of orders, which didn't preferred by drivers.

# In[9]:


df.groupby('is_driver_assigned_key').size()


# ## Distribution of failed orders by hours

# In[10]:


df['hr_of_day'] = pd.to_datetime(df.order_datetime).dt.hour


# In[11]:


df.head()


# In[12]:


df['hr_of_day'].value_counts().sort_index()
df['hr_of_day'].nunique()


# In[13]:


fig = plt.figure(figsize=(10,6))
ax = sns.countplot(data=df, x="hr_of_day", order=df['hr_of_day'].value_counts().index)


# In[14]:


fig = plt.figure(figsize=(10,6))
tmp_df = df.hr_of_day.value_counts()
ax = sns.lineplot(x=tmp_df.index, y=tmp_df, legend='full')
g = ax.set_xticks(range(0, 24))
g = ax.set_xticklabels(range(0, 24))


# Based on order hour, we can divide order failure into two groups.
# 1. Order cancellation due to peak hours(time during which taxi demand is higher than usual) i.e. from 7AM to 9AM and 8PM to 10PM.
# 2. Order cancellation due to odd hours i.e. 11PM to 3AM.

# ##  Average time to cancellation with and without driver, by the hour

# In[15]:


df.cancellations_time_in_seconds.describe()
tmp_df = df.cancellations_time_in_seconds
min_outlier_value = tmp_df.mean() - tmp_df.std() * 3
max_outlier_value = tmp_df.mean() + tmp_df.std() * 3
df = df[(df.cancellations_time_in_seconds > min_outlier_value) & (df.cancellations_time_in_seconds < max_outlier_value)]



fig = plt.figure(figsize=(10,6))
sns.histplot(df.cancellations_time_in_seconds, kde=True)


# In[16]:


df.cancellations_time_in_seconds.describe()


# In[17]:


temp_df = df.groupby(['is_driver_assigned_key', 'hr_of_day'])['cancellations_time_in_seconds'].mean()


# In[18]:


fig = plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=temp_df.reset_index(), x='hr_of_day', y='cancellations_time_in_seconds', hue='is_driver_assigned_key')
g = ax.set_xticks(range(0, 24))
g = ax.set_xticklabels(range(0, 24))


# Above lineplot shows that mean cancellations time is more for orders if drivers has been assigned. It makes some sense as driver matching and driver acceptance for orders takes few seconds. So, if client has cancelled order after driver assignment then cancellation time include driver matching and driver acceptance time, hence it has more average cancellations time across each hour of the day.

# ## Distribution of average ETA by hours

# In[19]:


temp_df = df.groupby('hr_of_day')['m_order_eta'].mean()
temp_df = temp_df.reset_index()


# In[20]:


fig = plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=temp_df, x='hr_of_day', y='m_order_eta')
g = ax.set_xticks(range(0, 24))
g = ax.set_xticklabels(range(0, 24))


# ## calculate how many sizes 8 hexes contain 80% of all orders from the original data sets and visualise the hexes, colouring them by the number of fails on the map

# In[26]:


import h3
import folium

df["hex_id"] = df.apply(
    func=lambda row: h3.geo_to_h3(lat=row["origin_latitude"], lng=row["origin_longitude"], resolution=8), axis=1)

grouped_q5 = df.groupby(by="hex_id")["order_gk"].count()
grouped_q5.shape


# In[27]:


grouped_q5 = grouped_q5.reset_index()
grouped_q5.sample(n=5, random_state=42)


# To find the number of hexes that contain 80% of the orders, we will apply a cumulative percentage operation over the order_gk count column in the grouped_q5 DataFrame. This consists of the following steps:
# 
# - Sort the DataFrame by the count.
# - Find the total number (sum) of failed orders.
# - Apply the method cumsum to find the cumulative sum of the order-count column.
# - Divide by the total sum to generate percentages.
# - Filter to find the row that is closest to 80%.

# In[28]:


grouped_q5 = grouped_q5.sort_values(by="order_gk")  # 1
total_orders = grouped_q5["order_gk"].sum()  # 2
grouped_q5["cum_sum"] = grouped_q5["order_gk"].cumsum()  # 3
grouped_q5["cum_perc"] = 100 * grouped_q5["cum_sum"] / total_orders  # 4
grouped_q5[grouped_q5["cum_perc"] <= 80]  # 5


# There are 131 rows in the final output, and 133 rows in the original grouped DataFrame, meaning that 131 hexagons contain around 80% of data, and only 2 hexagons contain the other 20%!

# In[29]:


map = folium.Map(location=[df["origin_latitude"].mean(), df["origin_longitude"].mean()],
                 zoom_start=8.5,  # after a bit of experimentation, we thought this presents the map best
                 tiles="cartodbpositron")


# In[31]:


import json
import geojson


def to_geojson(row):
    """Transform hex_id into a geojson object."""
    geometry = {
        "type": "Polygon",
        "coordinates": [h3.h3_to_geo_boundary(h=row["hex_id"], geo_json=True)]
    }
    return geojson.Feature(id=row["hex_id"], geometry=geometry, properties={"order_gk": row["order_gk"]})


geojsons = grouped_q5.apply(func=to_geojson, axis=1).values.tolist()
geojson_str: str = json.dumps(geojson.FeatureCollection(geojsons))


# In[33]:


import matplotlib

# instantiate a colormap object for better visualisation
colormap = matplotlib.cm.get_cmap(name="plasma")
max_order_gk = grouped_q5["order_gk"].max()
min_order_gk = grouped_q5["order_gk"].min()


# In[34]:


_ = folium.GeoJson(data=geojson_str, style_function=lambda f: {
    "fillColor": matplotlib.colors.to_hex(
        colormap((f["properties"]["order_gk"] - min_order_gk) / (max_order_gk - min_order_gk))),
    "color": "black",
    "weight": 1,
    "fillOpacity": 0.7
}).add_to(map)


# In[35]:


map

