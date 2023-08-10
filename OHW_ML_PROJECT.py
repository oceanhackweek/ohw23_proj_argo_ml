#!/usr/bin/env python
# coding: utf-8

# In[2]:


from argopy import DataFetcher as ArgoDataFetcher
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import xarray
from netCDF4 import Dataset
import scipy as sp


# # Read the Argo data

# In[16]:


region = [-50, -20, 30, 50, 0, 500, '2021-01', '2022-01']
argo_loader = ArgoDataFetcher()
ds = argo_loader.region(region).to_xarray()


# In[9]:


ds.keys()


# In[10]:


lon = np.array(ds.variables['LONGITUDE'])
lat = np.array(ds.variables['LATITUDE'])
temp = np.array(ds.variables['TEMP'])
time = np.array(ds.variables['TIME'])
depth = np.array(ds.variables['PRES'])


# In[11]:


ds.variables['POSITION_QC']


# In[ ]:





# # creat data frame

# In[12]:


df = pd.DataFrame({'Longitude': lon, 'Latitude': lat, 'Depth': depth, 'Temperature': temp})
table = pd.pivot_table(df, values='Temperature', index=['Longitude', 'Latitude'], columns='Depth')


# # Round the depth values (could refine!)

# In[13]:


df_new = df.copy()
df_new.Depth = df_new.Depth.round()
df_new
table = pd.pivot_table(df_new, values='Temperature', index=['Longitude', 'Latitude'], columns='Depth')
table


# In[27]:


df


# In[15]:


table.to_csv('data.csv')  


# In[47]:


# for i in range(len(depth) - 1):
#     if np.where(isinstance(depth[i], float)==True):
#         j = sp.interpolate(i,i+1, type = int, kind = 'linear')
        
#     else:
#         continue


# In[49]:


print(depth)


# In[ ]:




