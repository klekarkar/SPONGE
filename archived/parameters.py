"""Model parameters and initial conditions for the water balance model"""
import numpy as np

# ====== GROUNDWATER PARAMETERS ======
A1 = 38755
Kh_GWlocal = 2000
Kh_GWout = 1800
Kh_GWreg = 1
L_GWlocal = 15000
L_GWout = 3000
L_GWreg = 3e5
M = 0.7
MeanRootDepth = 2000
kevap_BS = 0.8
kv_WL = 1.5
kveg = 1.1
kvegsat = 0.9
n = 0.20
phi_GW = n
phi_GWlocal = n
phi_GWout = n
ref_elev_upstream = 13.42 * 1000 #elevation of the catchment in mm
ref_elev_downstream = 10.9 * 1000 #elevation of the wetland in mm
r_P = 0.2
s_molecular_suction = 0.20
s_sce = 0.20
s_scs = 0.50

#====== SOIL WATER BALANCE PARAMETERS ======
#Evapotranspiration parameters
alpha_tall_canopy = 0.8
alpha_short_canopy = 1.26 #short veg. and bare soil
alpha_bare_soil = 1.26 #similar to alpha for short canopy
beta = 0.07 

#landuse composition
frac_tall_canopy = 0.5
frac_short_canopy = 0.3
frac_bare_soil = 0.2

#Soil parameters
s_fc = 0.321
s_sat = 0.482 #Saturation in m3/m3 (Silt Loam) -->>> from SPAW model
s_wp = 0.137 #Wilting point in m3/m3 (Silt Loam) -->>> from SPAW model
Ks=12.19 #Saturated hydraulic conductivity in mm/h (Silt Loam) -->>> from SPAW model
S=34.3 #Sorptivity in mm/h^0.5

#Vegetation parameters
LAI = 2.5
tau = 0.8 #or 0.2 a parameter accounting for the development of vegetation over the year (vegetation optical depth)--->>> from Miralles

#======== INTERCEPTION PARAMETERS ======
# Interception parameters
k1 = 0.2 #mm


#======== WETLAND PARAMETERS ======
frac_wl_veg = 0.5
frac_wl_water = 1-frac_wl_veg
top_width = 500 #m
top_length = 700 #m
h_max = 1.50 #m


# Initialize water balance variables and variable catchment properties empty arrays.
arrays = {
    "A_ratio": np.array([ ]),
    "A2": np.array([ ]),
    "CALCdeltaH_GW": np.array([ ]),
    "CALCdeltaH_WL": np.array([ ]),
    "day": np.array([ ]),
    "deltaH_groundwater": np.array([ ]),
    "deltaH_wetland": np.array([ ]),
    "evap_BS": np.array([ ]),
    "evap_baresoil": np.array([ ]),
    "evap_baresoil_cont_to_s": np.array([ ]),
    "evap_BS_mm": np.array([ ]),
    "ep_WL": np.array([ ]),
    "eveg_sat": np.array([ ]),
    "eveg_sat_cont_to_y": np.array([ ]),
    "eveg_us": np.array([ ]),
    "eveg_us_cont_to_s": np.array([ ]),
    "gradient_local": np.array([ ]),
    "gradient_outflow": np.array([ ]),
    "Inf_cont_to_s_cm": np.array([ ]),
    "Inf_cont_to_s_moist": np.array([ ]),
    "interception_threshold": np.array([ ]),
    "net_precipitation": np.array([ ]),
    "moisture_dep_factor": np.array([ ]),
    "plant_stress_factor": np.array([ ]),
    "Q_GWreg": np.array([ ]),
    "Qss_GW": np.array([ ]),
    "qGW_Local": np.array([ ]),
    "qGW_out": np.array([ ]),
    "qLOCAL_GW": np.array([ ]),
    "qLOCAL_WL": np.array([ ]),
    "Recharge": np.array([ ]),
    "recharge_cont_to_y": np.array([ ]),
    "Rlinear": np.array([ ]),
    "R_us": np.array([ ]),
    "s": np.array([ ]),
    "water_table_elevation": np.array([ ]),
    "downstrean_wl_elev": np.array([ ]),
    "tR_local": np.array([ ]),
    "tR_out": np.array([ ]),
    "water_level_wetland": np.array([ ]),
    "water_table_depth": np.array([ ]),
    "AWC": np.array([ ]),
    "y": np.array([ ]),
    "y_rech": np.array([ ]),
    "y_WL": np.array([ ]),
    "net_p_input": np.array([ ]),
    "run_off": np.array([ ]),
    "interception": np.array([ ]),
    "s_max": np.array([ ]),
    "delta_s": np.array([ ]),
    "cumulative_infiltration": np.array([ ]),
    "total_evap": np.array([ ]),
    "E_stress_tc": np.array([ ]),
    "E_stress_sc": np.array([ ]),
    "E_stress_bs": np.array([ ]),
    "evap_actual_tc": np.array([ ]),
    "evap_actual_sc": np.array([ ]),
    "evap_actual_bs": np.array([ ]),
    "total_evap": np.array([ ]),
    "infil": np.array([ ]),
    "sm": np.array([ ]),
    "perco": np.array([ ]),
    'upstream_wl_elev': np.array([ ]),
    'downstream_wl_elev': np.array([ ]),
    'y_ds': np.array([ ]),
    }

# model/parameters.py

# Initialize water balance variables and variable catchment properties as np.array([ ])
A_ratio = np.array([])
A2 = np.array([ ])
CALCdeltaH_GW = np.array([ ])
CALCdeltaH_WL = np.array([ ])
day = np.array([ ])
deltaH_groundwater = np.array([ ])
deltaH_wetland = np.array([ ])
evap_BS = np.array([ ])
eb_US = np.array([ ])
eb_US_cont_to_s = np.array([ ])
evap_BS_mm = np.array([ ])
ep_WL = np.array([ ])
eveg_sat = np.array([ ])
eveg_sat_cont_to_y = np.array([ ])
eveg_us = np.array([ ])
eveg_us_cont_to_s = np.array([ ])
gradient_local = np.array([ ])
gradient_outflow = np.array([ ])
Inf_cont_to_s_cm = np.array([ ])
Inf_cont_to_s_moist = np.array([ ])
net_precipitation = np.array([ ])
moisture_dep_factor = np.array([ ])
plant_stress_factor = np.array([ ])
Q_GWreg = np.array([ ])
Qss_GW = np.array([ ])
qGW_Local = np.array([ ])
qGW_out = np.array([ ])
qLOCAL_GW = np.array([ ])
qLOCAL_WL = np.array([ ])
Recharge = np.array([ ])
recharge_cont_to_y = np.array([ ])
Rlinear = np.array([ ])
R_us = np.array([ ])
s = np.array([ ])
upstream_wl_elev = np.array([ ])
downstream_wl_elev = np.array([ ])
tR_local = np.array([ ])
tR_out = np.array([ ])
water_level_wetland = np.array([ ])
water_table_depth = np.array([ ])
AWC = np.array([ ])
y = np.array([ ])
y_rech = np.array([ ])
y_ds = np.array([ ])
net_p_input = np.array([ ])
run_off = np.array([ ])
interception = np.array([ ])
s_max = np.array([ ])
delta_s = np.array([ ])
interception_threshold = np.array([ ])
evap_baresoil = np.array([ ])
evap_baresoil_cont_to_s = np.array([ ])
cumulative_infiltration = np.array([ ])
total_evap = np.array([ ])
E_stress_tc = np.array([ ]) # Evaporation stress tall canopy
E_stress_sc = np.array([ ]) # Evaporation stress short canopy
E_stress_bs = np.array([ ]) # Evaporation stress
evap_actual_tc = np.array([ ])
evap_actual_sc = np.array([ ])
evap_actual_bs = np.array([ ])
total_evap = np.array([ ])
infil = np.array([ ])
sm = np.array([ ])
perco = np.array([ ])
