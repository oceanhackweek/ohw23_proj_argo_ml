# ohw23_proj_argo_ml

## 1. Get argo data
Argopy: https://argopy.readthedocs.io/en/latest/

  - Rob: I'll help work on this

- Define which data we are going to use
Temperature and salinity

- Define limits for Argo data: locations, time

Rob: ArgoPy can select for a time period and in a rectangular area using lat/long bounds and pressure bounds. Alternatively can select specific floats or profiles.
    
North Atlantic

Last year of data

## 2. Prepare data for the Machine Learning Model
Prepare the data vertically to be equal gridded. Use of numpy,interp?
![image](https://github.com/oceanhackweek/ohw23_proj_argo_ml/assets/47478764/3db8a0b9-2238-491d-8312-8a3e7cd39fd2)

The input data needs to have a format similar to each:
- a dataframe or a csv file
- each row represents a profile
- each column represents a data value in a specific depth
- you can add two more columns on the data related to the position of the profiles

Need to turn full Argo data DF to two dataframes, one for temp and one for salinity, with the following columns: LATITUDE, LONGITUDE, and each unique value of PRES
- Each row is a unique lat/long pair, and has the TEMP or PSAL at the depth corresponding to each PRES column

## 3. Generate spikes on the data
Add some random noise to the GDAC data

## 4. Apply IOOS QC on the "fake" data

If we have time, can use the ioos_qc module's qartod.spike_test function. 

## 5. Define some configurations on the ML model
09AUG - talk about that

## 6. Apply the ML model
09AUG - talk about that

## 7. Prepare the final result
Jupyter notebook with all the steps

## (Optional) Prepare the data for the Dense Neural Network (DNN) model

## (Optional) Apply the DNN model
