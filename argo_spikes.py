from ioos_qc import qartod
import pandas as pd
import numpy as np


argo_df = pd.read_csv('data_for_spikes.csv')

argo_df = argo_df.drop(argo_df.columns[0],axis=1)

noise = np.random.normal(0,2,[len(argo_df.index),len(argo_df.columns)])

argo_noise = argo_df+noise

all_flags = pd.DataFrame(index=argo_noise.index,columns=argo_noise.columns)

for index in argo_noise.index:
   flags = qartod.spike_test(inp=argo_noise.iloc[index],suspect_threshold=3.0,fail_threshold=8.0)
   all_flags.iloc[index] = flags

all_flags.to_csv('spike_flags.csv',index=False)