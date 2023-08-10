from ioos_qc import qartod
import pandas as pd
import numpy as np


argo_df = pd.read_csv('data_for_spikes.csv')

argo_df = argo_df.drop(argo_df.columns[0],axis=1)

noise = np.random.normal(0,2,[len(argo_df.index),len(argo_df.columns)]) # creating a dataframe of normally distributed values around 0, with a stndev of 2. Stndev can be changed to make more or less noisy data

argo_noise = argo_df+noise

argo_noise.to_csv('argo_noise.csv',index=False)

spike_flags = pd.DataFrame(index=argo_noise.index,columns=argo_noise.columns)
range_flags = pd.DataFrame(index=argo_noise.index,columns=argo_noise.columns)

for index in argo_noise.index:
   flags = qartod.spike_test(inp=argo_noise.iloc[index],suspect_threshold=3.0,fail_threshold=8.0) # using 3°C as suspect and 8°C as fail, per IOOS recommendations
   spike_flags.iloc[index] = flags
   flags = qartod.gross_range_test(inp=argo_noise.iloc[index],fail_span=[0,30]) #flags any value temp that is less than 0°C or higher than 30°C
   range_flags.iloc[index] = flags 

spike_flags.to_csv('spike_flags.csv',index=False)
range_flags.to_csv('range_flags.csv',index=False)
