#%%
import pandas as pd
import numpy as np
from compute_ET import *

#%%define the data directory
data_dir = "data/"
#altitude of Boechout in meters
altitude = 13.42
#latitude of Boechout in degrees
latitude = 51.13

#%%read the data
climate_data = pd.read_csv(data_dir + "boechout_climate_data.csv", index_col=0, parse_dates=True)
climate_data.index = pd.to_datetime(climate_data.index, format="%d/%m/%Y")
relative_humidity = pd.read_csv(data_dir + "Boechout_mswx_relative_humidity.csv", index_col=0, parse_dates=True)/100

#%%compute components of evaporation: 
#gamma(psychrometric constant), delta(slope of vapor pressure curve), vapor pressure deficit, wind speed, net radiation
#The soil heat flux G is ignored in this model
delta=compute_slope_of_vapor_pressure_curve(climate_data["mean_daily_temperature"])
gamma=compute_psychrometric_constant(altitude)
vapor_p_deficit=vapor_pressure_deficit(climate_data["max_daily_temperature"], climate_data["min_daily_temperature"], relative_humidity["relative_humidity"][0:len(climate_data)])
net_radiation=climate_data["global_radiation"]*3.6 #convert from kwh/m2/day to MJ/m2/day
G=0
soil_evap_numerator=0.408*delta*(net_radiation-G)+gamma*(900/(273+climate_data["mean_daily_temperature"]))*climate_data["wind_speed"]*vapor_p_deficit

if __name__ == "__main__":
    print("The components of evaporation have been computed successfully")


# %%
