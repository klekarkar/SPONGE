# This script initializes the model parameters and pre-allocates arrays for the simulation.
# It sets up the initial conditions for soil moisture, evaporation stress, and other variables.

#%%
import numpy as np
import pandas as pd
from parameters import *         # brings in s_fc, s_wp, s_sat, etc.
from archived.data_processing import *     # brings in delta, gamma, etc.
from compute_daily_LAI import process_modisLAI, get_daily_lai

#=======================================================
# Load your CSV input DAILY data

df = pd.read_csv("./data/Boechout_precip_ETo.csv", index_col=0)
pET_k = 0.408 * (delta/(delta + gamma)) * net_radiation #delta and gamma from data_processing.py

# add this:
precip = df["precipitation"]   # a pd.Series of rainfall
n_steps    = len(precip)
precip.index=pd.to_datetime(precip.index, format='%d/%m/%Y')

#LAI data
annual_lai_cycle = process_modisLAI('./data/modisLAI_2002_2020.csv')
dailyLAI = get_daily_lai(precip.index.min(), precip.index.max(), annual_lai_cycle)

# =======================================================
# Pre‐allocate all arrays where the model will store its results 
#soil moisture balance variables
interception     = np.zeros(n_steps)
infil            = np.zeros(n_steps)
perco            = np.zeros(n_steps)
sm               = np.zeros(n_steps)
E_stress_tc      = np.zeros(n_steps)
E_stress_sc      = np.zeros(n_steps)
E_stress_bs      = np.zeros(n_steps)
evap_actual_tc   = np.zeros(n_steps)
evap_actual_sc   = np.zeros(n_steps)
evap_actual_bs   = np.zeros(n_steps)
total_evap       = np.zeros(n_steps)
storage_capacity = np.zeros(n_steps)
run_off          = np.zeros(n_steps)
throughfall       = np.zeros(n_steps)
canopy_evap     = np.zeros(n_steps)

#groundwater level
y                = np.zeros(n_steps)

#wetland
E_veg_wl         = np.zeros(n_steps)
E_wat_wl         = np.zeros(n_steps)
y_wl            = np.zeros(n_steps)
sm_wl          = np.zeros(n_steps)
infil_wl        = np.zeros(n_steps)
perco_wl        = np.zeros(n_steps)
vol_wl        = np.zeros(n_steps)
run_off_wl      = np.zeros(n_steps)

#=======================================================
# rectangular wetland geometry


#======================================================
# Initial conditions at t=0 
dt = 1.0 # time step in days
soil_depth        = 300.0 #mm
sm[0]             = s_fc
E_stress_tc[0]    = 1.0
E_stress_sc[0]    = 1.0
E_stress_bs[0]    = 1.0
infil[0]          = 0.5 * s_fc
perco[0]          = 0.0
total_evap[0]     = 0.0
evap_actual_tc[0] = 0.5
evap_actual_sc[0] = 0.5
evap_actual_bs[0] = 0.5
interception[0]   = 0.0
storage_capacity[0] = max(0, (s_sat - sm[0]) * soil_depth)
run_off[0]        = 0.0
y[0]              = 0.0
y_wl[0]           = 50
sm_wl[0]         = 0.0
infil_wl[0]      = 0.5 * s_fc
vol_wl[0]        = top_width * top_length * y_wl[0] # initial volume of water in wetland
run_off_wl[0]  = 0.0
canopy_evap[0] = 0.0
throughfall[0] = 0.0

#wetland volume
wetland_area= top_width * top_length
max_vol_wl  = wetland_area * h_max  # m³
