from ioos_qc import qartod
import pandas as pd
import numpy as np

class ArgoSpikes():
   
   def __init__(self, filename):
      self.filename = filename
      self.argo_df = pd.read_csv(f'../raw_data/{self.filename}')

   def generate_spikes(self, noise_effect=2):
      # note: this will generate random noise on every run - so the results will not be the same between runs
      argo_df_new = self.argo_df.iloc[:, 3:]

      noise = np.random.normal(0,noise_effect,[len(argo_df_new.index),len(argo_df_new.columns)]) # creating a dataframe of normally distributed values around 0, with a stndev of 2. Stndev can be changed to make more or less noisy data
      self.argo_noise = argo_df_new+noise

      self.argo_df[self.argo_noise.columns] == self.argo_noise
      self.argo_df.to_csv(f'../raw_data/{self.filename}_noise.csv',index=False)

   def qc_spikes(self):
      # note: this will generate random noise on every run - so the results will not be the same between runs   
      self.spike_flags = pd.DataFrame(index=self.argo_noise.index,columns=self.argo_noise.columns)
      self.range_flags = pd.DataFrame(index=self.argo_noise.index,columns=self.argo_noise.columns)

      for index in self.argo_noise.index:
         flags = qartod.spike_test(inp=self.argo_noise.iloc[index],suspect_threshold=3.0,fail_threshold=8.0) # using 3°C as suspect and 8°C as fail, per IOOS recommendations
         self.spike_flags.iloc[index] = flags
         flags = qartod.gross_range_test(inp=self.argo_noise.iloc[index],fail_span=[0,30]) #flags any value temp that is less than 0°C or higher than 30°C
         self.range_flags.iloc[index] = flags 

      self.spike_flags.to_csv(f'../raw_data/{self.filename}_spike_flags.csv',index=False)
      self.range_flags.to_csv(f'../raw_data/{self.filename}_range_flags.csv',index=False)

      self.all_flags = self.spike_flags.copy()
      self.all_flags[self.all_flags == 2] = 1
      self.all_flags[self.all_flags == 3] = 1
      self.all_flags[self.all_flags == 4] = 0

      self.all_flags.to_csv(f'../raw_data/{self.filename}_flags.csv',index=False)
