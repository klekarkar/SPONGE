
import numpy as np
from archived.compute_interception import daily_interception_threshold
from compute_ET import *
from archived.infiltration_models import InfiltrationModel
import archived.model_init as mi   # ← pulls in all the arrays & constants
from parameters import *  # ← pulls in all the parameters

def simulate_wb(
    time, sm, pET_k, precip, LAI, s_fc, s_wp, tau, soil_depth, Ks, S, dt, 
    frac_tall_canopy, frac_short_canopy, frac_bare_soil, alpha_tall_canopy, 
    alpha_short_canopy, alpha_bare_soil, interception, total_evap, evap_actual_tc, 
    evap_actual_sc, evap_actual_bs, infil, perco, run_off,
    E_stress_tc_arr, E_stress_sc_arr, E_stress_bs_arr  #track E_stress over time
):

    for time in range(1, n_steps):
        assert abs(frac_tall_canopy + frac_short_canopy + frac_bare_soil - 1.0) < 1e-6, \
            "Vegetation fractions must sum to 1"

        # === INTERCEPTION ===
        interception[time] = min(precip['precipitation'].iloc[time], 0.05 * LAI)
        throughfall = precip['precipitation'].iloc[time] - interception[time]

        # === EVAPOTRANSPIRATION ===
        if interception[time] >= pET_k.iloc[time]:
            total_evap[time] = pET_k.iloc[time]
            evap_actual_tc[time] = 0
            evap_actual_sc[time] = 0
            evap_actual_bs[time] = 0
            soil_ET = 0
            E_stress_tc_arr[time] = 0
            E_stress_sc_arr[time] = 0
            E_stress_bs_arr[time] = 0
        else:
            remaining_PET = pET_k.iloc[time] - interception[time]

            ratio = np.clip((s_fc - sm[time - 1]) / (s_fc - s_wp), 0, 1)
            E_stress_tc = 1 - ratio**2
            E_stress_sc = max(0, 0.5 * (1 - np.sqrt(ratio) + tau / 0.8))
            E_stress_bs = 1 - np.sqrt(ratio)

            # Save to array
            E_stress_tc_arr[time] = E_stress_tc
            E_stress_sc_arr[time] = E_stress_sc
            E_stress_bs_arr[time] = E_stress_bs

            pot_evap_tc = remaining_PET * frac_tall_canopy * alpha_tall_canopy
            pot_evap_sc = remaining_PET * frac_short_canopy * alpha_short_canopy
            pot_evap_bs = remaining_PET * frac_bare_soil * alpha_bare_soil

            evap_actual_tc[time] = pot_evap_tc * E_stress_tc
            evap_actual_sc[time] = pot_evap_sc * E_stress_sc
            evap_actual_bs[time] = pot_evap_bs * E_stress_bs

            soil_ET = evap_actual_tc[time] + evap_actual_sc[time] + evap_actual_bs[time]

            #calculate total evaporation
            if interception[time] > pET_k.iloc[time]:
                total_evap[time] = pET_k.iloc[time]
            else:
                total_evap[time] = interception[time] + soil_ET

        # === INFILTRATION ===

        # Calculate the storage capacity based on the soil moisture at the previous time step
        sc =  max(0, (s_sat - sm[time - 1]) * soil_depth)
        storage_capacity[time] = sc

        infiltration_model = InfiltrationModel(Ks, S)
        daily_infiltration = infiltration_model.valiantzas_model(dt) * 24  # mm/day

        infil[time] = min(daily_infiltration, throughfall, sc)

        # Infiltration excess runoff occurs if throughfall > infiltration capacity
        infil_excess = max(0, throughfall - infil[time])

        # === UPDATE SOIL MOISTURE due to INFILTRATION ===
        sm[time] = sm[time - 1] + infil[time] / soil_depth

        # === RUNOFF ===
        if sm[time] > s_sat:
            sat_excess = (sm[time] - s_sat) * soil_depth
            sm[time] = s_sat
        else:
            sat_excess = 0

        # Total runoff = infiltration-excess + saturation-excess
        run_off[time] = infil_excess + sat_excess

        # === UPDATE SOIL MOISTURE due to EVAPOTRANSPIRATION ===
        sm[time] = sm[time] - soil_ET / soil_depth

        # === PERCOLATION ===
        excess_over_fc = max(0, sm[time] - s_fc)
        perco[time] = 0.7 * excess_over_fc * soil_depth
        sm[time] -= perco[time] / soil_depth

        # === BOUNDS ===
        sm[time] = min(sm[time], s_sat)
        sm[time] = max(sm[time], s_wp)


        # === Groundwater dynamics ===
        #update groundwater level based on percolation and evaporation
        y[time] = min(0, y[time - 1] + (perco[time] - 0.5*soil_ET) / n*(1 -s_fc))


        #return dictionary with results

        return {
            "interception": interception,
            "total_evap": total_evap,
            "evap_actual_tc": evap_actual_tc,
            "evap_actual_sc": evap_actual_sc,
            "evap_actual_bs": evap_actual_bs,
            "infil": infil,
            "percolation": perco,
            "sm": sm,
            "run_off": run_off,
            "E_stress_tc_arr": E_stress_tc_arr,
            "E_stress_sc_arr": E_stress_sc_arr,
            "E_stress_bs_arr": E_stress_bs_arr,
            "storage_capacity": storage_capacity,
            'y': y
        }
