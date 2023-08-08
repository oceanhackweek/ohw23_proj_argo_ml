# ohw23_proj_argo_ml

## Get argo data
Argopy: https://argopy.readthedocs.io/en/latest/

## Define which data we are going to use
Temperature and salinity

## Define limits for Argo data: locations, time
North Atlantic
Last year of data

## Prepare data for the Machine Learning Model
Prepare the data vertically to be equal gridded. Use of numpy,interp?
![image](https://github.com/oceanhackweek/ohw23_proj_argo_ml/assets/47478764/3db8a0b9-2238-491d-8312-8a3e7cd39fd2)

The input data needs to have a format similar to each:
- a dataframe or a csv file
- each row represents a profile
- each column represents a data value in a specific depth
- you can add two more columns on the data related to the position of the profiles

## Define some configurations on the ML model
09AUG - talk about that

## Apply the ML model
09AUG - talk about that

## Generate spikes on the data
Add some random noise to the GDAC data

## Prepare the final result
Jupyter notebook with all the steps

## (Optional) Apply IOOS QC on the "fake" data

## (Optional) Prepare the data for the Dense Neural Network (DNN) model

## (Optional) Apply the DNN model
