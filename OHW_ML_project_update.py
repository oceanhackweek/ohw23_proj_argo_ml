#!/usr/bin/env python
# coding: utf-8

# In[1]:


from argopy import DataFetcher as ArgoDataFetcher
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import xarray
from netCDF4 import Dataset
import scipy as sp


# In[2]:


final_table = pd.DataFrame()
for i in range(11):
    print(i)
    month_str_end = f'{str(i+2).zfill(2)}'  # Ensure month is formatted with leading zero
    month_str_start = f'{str(i+1).zfill(2)}'  # Ensure month is formatted with leading zero

    start_date = f'2021-{month_str_start}-01'
    end_date = f'2021-{month_str_end}-01'  # You need to specify the end date accordingly
    
    region = [-50, -20, 30, 50, 0, 2000, start_date, end_date]
    argo_loader = ArgoDataFetcher()
    ds = argo_loader.region(region).to_xarray()

    lon = np.array(ds.variables['LONGITUDE'])
    lat = np.array(ds.variables['LATITUDE'])
    temp = np.array(ds.variables['TEMP'])
    time = np.array(ds.variables['TIME'])
    depth = np.array(ds.variables['PRES'])
    salinity = np.array(ds.variables['PSAL'])
   # df = pd.DataFrame({'Longitude': lon, 'Latitude': lat, 'Depth': depth, 'Temperature': temp, 'Time': time})
    df = pd.DataFrame({'Longitude': lon, 'Latitude': lat, 'Depth': depth, 'salinity': salinity, 'Time': time})
    df_new = df.copy()
    df_new.Depth = df_new.Depth.round()
    table = pd.pivot_table(df_new, values='salinity', index=['Longitude', 'Latitude', 'Time'], columns='Depth')
    
    if final_table.shape[0] == 0:
        final_table = table
    else:
        final_table = pd.concat([final_table, table])
   
    


# In[3]:


final_table_2 = pd.DataFrame()


region = [-50, -20, 30, 50, 0, 2000,  '2022-12-1', '2022-12-31']
argo_loader = ArgoDataFetcher()
ds = argo_loader.region(region).to_xarray()

lon = np.array(ds.variables['LONGITUDE'])
lat = np.array(ds.variables['LATITUDE'])
#temp = np.array(ds.variables['TEMP'])
time = np.array(ds.variables['TIME'])
depth = np.array(ds.variables['PRES'])
salinity = np.array(ds.variables['PSAL'])

#df = pd.DataFrame({'Longitude': lon, 'Latitude': lat, 'Depth': depth, 'Temperature': temp, 'Time': time})
df = pd.DataFrame({'Longitude': lon, 'Latitude': lat, 'Depth': depth, 'salinity': salinity, 'Time': time})
df_new = df.copy()
df_new.Depth = df_new.Depth.round()
final_table_2 = pd.pivot_table(df_new, values='salinity', index=['Longitude', 'Latitude', 'Time'], columns='Depth')
    
final_table = pd.concat([final_table, final_table_2])


# In[4]:


table_intrp = final_table.interpolate(axis=1)


# In[5]:


table_intrp=table_intrp.iloc[:,10:].dropna()


# In[6]:


table_intrp.to_csv('salin.csv')


# In[8]:


final_table


# In[ ]:




