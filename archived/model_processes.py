
# -*- coding: utf-8 -*-
"""
This script simulates the soil water balance for a given time step.
It calculates interception, evapotranspiration, infiltration, runoff, and percolation.
"""

import numpy as np
from archived.compute_interception import daily_interception_threshold
from compute_ET import *
from compute_daily_LAI import *
from archived.infiltration_models import InfiltrationModel
import archived.model_init as mi   # ← pulls in all the arrays & constants
from parameters import *  # ← pulls in all the parameters
#==============================================================================
"""Soil Moisture Balance Model

Initial arrays of soil moisture balance variables
"""
#==============================================================================

    
def simulate_wb(time):

    """
    Simulate the water balance for a single time step.
    """

    #ensure fraction of vegetation types sum to 1
    sum_f = frac_tall_canopy + frac_short_canopy + frac_bare_soil
    if abs(sum_f - 1.0) > 1e-6:
        raise ValueError(f"Vegetation fractions must sum to 1.0 (got {sum_f:.2f})")

    #Vegetation interception
    P   = mi.precip.iloc[time]
    LAI = mi.dailyLAI.iloc[time]
    I_max = mi.k1 * LAI #maximum interception capacity (mm)

    # Get canopy interception from previous time step
    I_prev = mi.interception[time-1]

    #Compute canopy evaporation
    if I_prev > 0 and I_max > 0:
        Ec = (I_prev / I_max) ** (2/3) * mi.pET_k.iloc[time] 
        Ec = min(Ec, I_prev) #cannot evaporate more than what is stored in the canopy
    else:
        Ec = 0.0

    #Update canopy storage
    I_temp = I_prev - Ec + P

    if I_temp > I_max:
        throughfall = I_temp - I_max
        I_new = I_max
    else:
        throughfall = 0.0
        I_new = I_temp
    mi.interception[time] = I_new
    mi.throughfall[time] = throughfall
    mi.canopy_evap[time] = Ec

    ######

    mi.interception[time] = min(mi.interception[time-1] + P, I_max)
    throughfall = P - mi.interception[time]

    #==========EVAPOTRANSPIRATION===================
    if mi.canopy_evap[time] >= mi.pET_k.iloc[time]:
        mi.canopy_evap[time] = mi.pET_k.iloc[time]
        mi.evap_actual_tc[time] = 0
        mi.evap_actual_sc[time] = 0
        mi.evap_actual_bs[time] = 0
        soil_ET = 0
        mi.E_stress_tc[time] = 0
        mi.E_stress_sc[time] = 0
        mi.E_stress_bs[time] = 0
    else:
        remaining_PET = mi.pET_k.iloc[time] - mi.canopy_evap[time]
        ratio = np.clip((s_fc - mi.sm[time-1]) / (s_fc - s_wp), 0, 1)
        E_tc = 1 - ratio**2
        E_sc = max(0, 0.5*(1 - np.sqrt(ratio) + tau/0.8))
        E_bs = 1 - np.sqrt(ratio)
        mi.E_stress_tc[time] = E_tc
        mi.E_stress_sc[time] = E_sc
        mi.E_stress_bs[time] = E_bs

        pot_tc = remaining_PET*frac_tall_canopy*alpha_tall_canopy
        pot_sc = remaining_PET*frac_short_canopy*alpha_short_canopy
        pot_bs = remaining_PET*frac_bare_soil*alpha_bare_soil

        mi.evap_actual_tc[time] = pot_tc*E_tc
        mi.evap_actual_sc[time] = pot_sc*E_sc
        mi.evap_actual_bs[time] = pot_bs*E_bs

        soil_ET = (mi.evap_actual_tc[time]
                  +mi.evap_actual_sc[time]
                  +mi.evap_actual_bs[time])
        
        mi.total_evap[time] = mi.interception[time] + soil_ET

    #==========INFILTRATION===================
    # Calculate the storage capacity based on the soil moisture at the previous time step

    sc = max(0, (s_sat - mi.sm[time-1]) * mi.soil_depth)
    mi.storage_capacity[time] = sc

    infil_model = InfiltrationModel(Ks, S)
    daily_inf   = infil_model.valiantzas_model(mi.dt) * 24.0

    mi.infil[time] = min(daily_inf, throughfall, sc)
    infil_excess   = max(0, throughfall - mi.infil[time])

    # update soil moisture with infiltration ---
    mi.sm[time] = mi.sm[time-1] + mi.infil[time]/mi.soil_depth

    # runoff (infiltration‐excess + saturation‐excess) ---
    if mi.sm[time] > s_sat:
        sat_ex = (mi.sm[time] - s_sat) * mi.soil_depth
        mi.sm[time] = s_sat
    else:
        sat_ex = 0.0

    mi.run_off[time] = infil_excess + sat_ex

    # subtract ET from soil moisture ---
    mi.sm[time] -= soil_ET / mi.soil_depth

    # percolation as excess over field capacity ---
    excess_fc      = max(0, mi.sm[time] - s_fc)
    mi.perco[time] = 0.7 * excess_fc * mi.soil_depth
    mi.sm[time]   -= mi.perco[time] / mi.soil_depth

    # enforce bounds of soil moisture---
    mi.sm[time] = np.clip(mi.sm[time], mi.s_wp, mi.s_sat)

    # === Groundwater dynamics ===
    #update groundwater level based on percolation and evaporation
    mi.y[time] = min(0, mi.y[time - 1] + (mi.perco[time] - 0.5*soil_ET) / mi.n*(1 -mi.s_fc))




















