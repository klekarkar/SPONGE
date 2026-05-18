"""
Model process orchestration functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm.auto import trange

from soil_processes import (
    pack_root_frac_to_2d,
    step_infiltration,
    step_et_partition,
    step_drainage_unit_gradient,
)

from wetland_processes import wetland_water_balance_step


#----------------------------------------------------------------------------------
""" 
Initialize model Fluxes and States arrays
----------------------------------------------------------------------------------
"""
def initialize_model_fluxes(daily_index, freq, n_layers, sm0=None):
    """
    Initialize model fluxes and states arrays to store results.

    Parameters
    ----------
    daily_index : pd.DatetimeIndex
        Daily date index for the simulation period.
    freq : str
        Output frequency, taken from the config (e.g., inputs.sim_freq).
        Use "D" for daily outputs or "H" for hourly outputs.
    n_layers : int
        Number of soil layers.
    sm0 : array-like or None, optional
        Initial soil moisture per layer (volumetric). If None, initialized to zeros.
        If scalar, applied to all layers.

    Returns
    -------
    dict
        Dictionary of initialized fluxes and states arrays.
    """
    daily_index = pd.DatetimeIndex(daily_index).normalize()

    freq = str(freq).strip().upper()
    if freq == "D":
        out_index = daily_index
    elif freq == "H":
        start = daily_index.min()
        end = daily_index.max() + pd.Timedelta(days=1)  # end-exclusive
        out_index = pd.date_range(start, end, freq="h", inclusive="left")
    else:
        raise ValueError("Invalid frequency. Use 'D' or 'H' (from config sim_freq).")

    n = len(out_index)

    fluxes = {
        "index": out_index,

        "interception":      np.zeros(n, dtype=float),
        "throughfall":       np.zeros(n, dtype=float),
        "canopy_evap":       np.zeros(n, dtype=float),
        "canopy_evap_grid": np.zeros(n, dtype=float), #grid-cell area weighted for vegetated fraction


        "infiltration":      np.zeros(n, dtype=float),
        "surface_run_off":   np.zeros(n, dtype=float),

        "evap_actual_bs":    np.zeros(n, dtype=float),
        "total_evap":        np.zeros(n, dtype=float),
        "unmet_T":           np.zeros(n, dtype=float),
        "wetland_storage":   np.zeros(n, dtype=float),
        "wetland_depth":     np.zeros(n, dtype=float),
        "wetland_active_area_fraction": np.zeros(n, dtype=float),
    }
    
    # initial soil moisture (volumetric)
    if sm0 is None:
        sm0 = np.zeros(n_layers, dtype=float)
    sm0 = np.asarray(sm0, dtype=float)
    if sm0.size == 1:
        sm0 = np.full(n_layers, float(sm0), dtype=float)
    if len(sm0) != n_layers:
        raise ValueError("sm0 must be scalar or length n_layers.")

    # soil moisture + ET per layer
    for k in range(1, n_layers + 1):
        fluxes[f"soil_moisture_L{k}"] = np.zeros(n, dtype=float)
        fluxes[f"aET_L{k}"] = np.zeros(n, dtype=float)
        fluxes[f"soil_moisture_L{k}"][0] = sm0[k - 1]

    # percolation between layers
    for k in range(1, n_layers):
        fluxes[f"percolation_L{k}L{k+1}"] = np.zeros(n, dtype=float)

    # deep recharge from bottom layer
    fluxes[f"recharge_L{n_layers}"] = np.zeros(n, dtype=float)

    # initial canopy storage
    fluxes["interception"][0] = 0.0

    return fluxes

"""
----------------------------------------------------------------------------------
 Soil Water Balance for n layers
----------------------------------------------------------------------------------
Soil Water Balance Module (multi-LU, n-layer)
- Canopy model already produced TF_arr and Ec_arr as (n_lu, n_t) arrays (mm/timestep).
- step_infiltration updates sm[0] in-place and returns (infil_mm, runoff_mm).
- step_et_partition updates sm in-place and returns (T_layers_out, Esoil, ET_total, unmet_T, Ec_grid).
- step_drainage_unit_gradient updates sm in-place, fills perc_out in-place, and returns recharge_mm.
- All soil moisture states are volumetric (m3/m3). z_layers is mm.
"""

def run_land_surface_water_balance(
    model_fluxes: dict,
    forcing: pd.DataFrame,                        # columns: P, PET at model timestep
    z_layers: np.ndarray,                         # (N,) mm
    theta_sat: np.ndarray,
    theta_fc: np.ndarray,
    theta_wp: np.ndarray,
    theta_r: np.ndarray,
    Ks_cm_d: np.ndarray,
    vG_m: np.ndarray,
    b_infilt: float,
    tau_vG: float,
    f_cr: float,
    q_sm: float,
    q_soil: float,
    lu_types: list[str],
    lu_fracs: np.ndarray,
    root_frac_by_lu: dict[str, np.ndarray],       # {lu: (N,) root fractions}
    TF_arr: np.ndarray,                           # (n_lu, n_t) mm/timestep
    Ec_arr: np.ndarray,                           # (n_lu, n_t) mm/timestep
    beta_arr: np.ndarray,                         # (n_lu, n_t) suppression factor (0-1)
    sim_freq: str,
    stress_method: str,
    drain_above_fc: bool = False,
    wetland_params: dict | None = None,
    show_progress: bool = True,
    logger=None,
):
    """
    Multi-LU soil water balance driven by canopy outputs packed as 2D arrays:
      TF_arr[i,t], Ec_arr[i,t] for LU i at timestep t.

    Stores ONLY catchment-average outputs in model_fluxes.
    """

    # -----------------------------
    # Validate inputs
    # -----------------------------
    sim_freq = str(sim_freq).upper().strip()
    if sim_freq not in ("H", "D"):
        raise ValueError("sim_freq must be 'H' or 'D'.")

    if not {"P", "PET"}.issubset(forcing.columns):
        raise ValueError("forcing must have columns ['P','PET'].")

    idx = pd.DatetimeIndex(model_fluxes["index"])
    if not idx.equals(pd.DatetimeIndex(forcing.index)):
        raise ValueError("model_fluxes['index'] must match forcing.index exactly.")

    n_t = len(idx)

    # ensure arrays
    z_layers  = np.asarray(z_layers, dtype=np.float64)
    theta_sat = np.asarray(theta_sat, dtype=np.float64)
    theta_fc  = np.asarray(theta_fc, dtype=np.float64)
    theta_wp  = np.asarray(theta_wp, dtype=np.float64)
    theta_r   = np.asarray(theta_r, dtype=np.float64)
    Ks_cm_d   = np.asarray(Ks_cm_d, dtype=np.float64)
    vG_m      = np.asarray(vG_m, dtype=np.float64)
    tau_vG    = float(tau_vG)
    b_infilt   = float(b_infilt)
    f_cr      = float(f_cr)
    q_sm      = float(q_sm)
    q_soil    = float(q_soil)
    N = z_layers.size
    n_lu = len(lu_types)

    # after: theta_wp, theta_fc, f_cr, stress_method are defined
    smethod = str(stress_method).strip().lower()
    theta_cr_arr = None
    if smethod in ("canopy_resistance", "vic_gsm", "araki_2025"):
        theta_cr_arr = theta_wp + f_cr * (theta_fc - theta_wp)


    lu_fracs = np.asarray(lu_fracs, dtype=np.float64)
    if lu_fracs.size != n_lu:
        raise ValueError("lu_types and lu_fracs must have same length.")
    if abs(float(lu_fracs.sum()) - 1.0) > 1e-3:
        raise ValueError(f"lu_fracs must sum to 1 (±1e-3). Got {lu_fracs.sum()}")

    TF_arr = np.asarray(TF_arr, dtype=np.float64)
    Ec_arr = np.asarray(Ec_arr, dtype=np.float64)
    if TF_arr.shape != (n_lu, n_t):
        raise ValueError(f"TF_arr must be shape {(n_lu, n_t)}, got {TF_arr.shape}")
    if Ec_arr.shape != (n_lu, n_t):
        raise ValueError(f"Ec_arr must be shape {(n_lu, n_t)}, got {Ec_arr.shape}")

    P   = forcing["P"].to_numpy(dtype=np.float64)
    PET = forcing["PET"].to_numpy(dtype=np.float64)

    dt_hours = 1.0 if sim_freq == "H" else 24.0

    # -----------------------------
    # Ensure output arrays exist
    # -----------------------------
    model_fluxes.setdefault("infiltration",    np.zeros(n_t, dtype=np.float64))
    model_fluxes.setdefault("surface_run_off", np.zeros(n_t, dtype=np.float64))
    model_fluxes.setdefault("evap_actual_bs",  np.zeros(n_t, dtype=np.float64))
    model_fluxes.setdefault("total_evap",      np.zeros(n_t, dtype=np.float64))
    model_fluxes.setdefault("unmet_T",         np.zeros(n_t, dtype=np.float64))
    model_fluxes.setdefault("canopy_evap_grid",np.zeros(n_t, dtype=np.float64))

    for k in range(N):
        model_fluxes.setdefault(f"aET_L{k+1}",          np.zeros(n_t, dtype=np.float64))
        model_fluxes.setdefault(f"soil_moisture_L{k+1}",np.zeros(n_t, dtype=np.float64))

    for k in range(N - 1):
        model_fluxes.setdefault(f"percolation_L{k+1}L{k+2}", np.zeros(n_t, dtype=np.float64))
    model_fluxes.setdefault(f"recharge_L{N}", np.zeros(n_t, dtype=np.float64))

    model_fluxes.setdefault("pre", np.zeros(n_t, dtype=np.float64))
    model_fluxes.setdefault("pet", np.zeros(n_t, dtype=np.float64))

    wetland_output_names = [
    "wetland_storage",
    "wetland_depth",
    "wetland_active_area_fraction",
    "wetland_routed_runoff",
    "wetland_precip",
    "wetland_evap",
    "wetland_infiltration",
    "wetland_overflow",
    "wetland_recharge",
    "surface_runoff_after_wetland",
]

    for name in wetland_output_names:
        model_fluxes.setdefault(name, np.zeros(n_t, dtype=np.float64))

    # -----------------------------
    # Pack roots (constant)
    # -----------------------------
    root_frac_2d = pack_root_frac_to_2d(lu_types, root_frac_by_lu, N)

    is_bare  = np.all(np.isclose(root_frac_2d, 0.0), axis=1)   # (n_lu,)
    pveg_lu  = (~is_bare).astype(np.float64)
    pbare_lu = is_bare.astype(np.float64)

    # -----------------------------
    # Initialize per-LU soil moisture (n_lu, N)
    # -----------------------------
    sm_init = np.array([model_fluxes[f"soil_moisture_L{k+1}"][0] for k in range(N)], dtype=np.float64)
    sm_by_lu = np.repeat(sm_init[None, :], repeats=n_lu, axis=0)

    # -----------------------------
    # Scratch arrays (reused)
    # -----------------------------
    sm_c       = np.zeros(N,     dtype=np.float64)
    T_layers_c = np.zeros(N,     dtype=np.float64)
    perc_c     = np.zeros(N - 1, dtype=np.float64)

    sm_tmp     = np.zeros(N,     dtype=np.float64)
    T_tmp      = np.zeros(N,     dtype=np.float64)
    stress_tmp = np.ones(N,      dtype=np.float64)
    perc_tmp   = np.zeros(N - 1, dtype=np.float64)



    # -----------------------------
    # Wetland state
    # -----------------------------
    if wetland_params is None:
        wetland_params = {"enabled": False}

    wetland_storage = (
        float(wetland_params.get("initial_depth_m", 0.0))
        * 1000.0
        * float(wetland_params.get("area_fraction", 0.0))
    )

    loop = trange(n_t, desc="Running Land Surface WB (multi-LU)") if show_progress else range(n_t)

    # =============================================================================
    # TIME LOOP
    # =============================================================================
    for t in loop:
        P_t = float(P[t])
        PET_t = float(PET[t])

        infil_c = 0.0
        runoff_c = 0.0
        Esoil_c = 0.0
        ET_c = 0.0
        unmetT_c = 0.0
        Ec_grid_c = 0.0
        rech_c = 0.0

        sm_c.fill(0.0)
        T_layers_c.fill(0.0)
        perc_c.fill(0.0)

        # -----------------------------
        # LU LOOP
        # -----------------------------
        for i in range(n_lu):
            frac = float(lu_fracs[i])

            # copy LU state to scratch
            sm_tmp[:] = sm_by_lu[i, :]

            TF_lu = float(TF_arr[i, t])
            Ec_lu = float(Ec_arr[i, t])

            bare = bool(is_bare[i])
            beta_t = float(beta_arr[i, t]) if not bare else 0.0

            pveg = float(pveg_lu[i])
            pbare = float(pbare_lu[i])

            # Rain reaching soil:
            # - bare tile: all precipitation to soil
            # - vegetated tile: only throughfall to soil
            P_to_soil = P_t if bare else TF_lu

            # Step 1: infiltration/runoff
            infil, runoff = step_infiltration(
                sm=sm_tmp,
                throughfall=P_to_soil,
                z_layers=z_layers,
                theta_sat=theta_sat,
                b_infilt=b_infilt,
                theta_r=theta_r,
            )

            # Step 2: ET partitioning
            T_layers, Esoil, ET_total, unmet_T, Ec_grid = step_et_partition(
                sm=sm_tmp,
                z_layers=z_layers,
                pet_t=PET_t,
                canopy_evap_t=0.0 if bare else Ec_lu,
                beta_t=beta_t,
                pveg=pveg,
                pbare=pbare,
                root_frac=root_frac_2d[i, :],
                theta_fc=theta_fc,
                theta_wp=theta_wp,
                theta_r=theta_r,
                T_layers_out=T_tmp,
                stress_layers_scratch=stress_tmp,
                stress_method=stress_method,
                f_cr=f_cr,
                q_sm=q_sm,
                q_soil=q_soil,
                theta_cr=theta_cr_arr,
            )

            # Step 3: drainage/percolation
            recharge = step_drainage_unit_gradient(
                sm=sm_tmp,
                z_layers=z_layers,
                theta_sat=theta_sat,
                theta_fc=theta_fc,
                theta_r=theta_r,
                Ks_cm_d=Ks_cm_d,
                vG_m=vG_m,
                tau_vG=tau_vG,
                dt_hours=dt_hours,
                perc_out=perc_tmp,
                drain_above_fc=drain_above_fc,
            )

            # write back LU state
            sm_by_lu[i, :] = sm_tmp

            # aggregate catchment means
            infil_c += frac * float(infil)
            runoff_c += frac * float(runoff)
            Esoil_c += frac * float(Esoil)
            ET_c += frac * float(ET_total)
            unmetT_c += frac * float(unmet_T)
            Ec_grid_c += frac * float(Ec_grid)
            rech_c += frac * float(recharge)

            sm_c += frac * sm_tmp
            T_layers_c += frac * T_layers
            perc_c += frac * perc_tmp

        # -----------------------------
        # Wetland module
        # -----------------------------
        # The wetland receives aggregated catchment surface runoff.
        wetland_state = wetland_water_balance_step(
            storage_mm_grid=wetland_storage,
            precipitation_mm=P_t,
            pet_mm=PET_t,
            surface_runoff_mm=runoff_c,
            wetland_params=wetland_params,
            dt_hours=dt_hours,
        )

        wetland_storage = wetland_state["storage_mm_grid"]

        # -----------------------------
        # Store catchment averages
        # -----------------------------
        model_fluxes["infiltration"][t] = infil_c

        # Runoff generated before wetland routing
        model_fluxes["surface_run_off"][t] = runoff_c

        # Runoff exported after wetland routing/storage/overflow
        model_fluxes["surface_runoff_after_wetland"][t] = wetland_state[
            "surface_runoff_after_wetland_mm"
        ]

        model_fluxes["evap_actual_bs"][t] = Esoil_c

        # Soil/canopy evap only. Wetland evap stored separately.
        model_fluxes["total_evap"][t] = ET_c

        model_fluxes["unmet_T"][t] = unmetT_c
        model_fluxes["canopy_evap_grid"][t] = Ec_grid_c

        model_fluxes["pre"][t] = P_t
        model_fluxes["pet"][t] = PET_t

        for k in range(N):
            model_fluxes[f"soil_moisture_L{k+1}"][t] = sm_c[k]
            model_fluxes[f"aET_L{k+1}"][t] = T_layers_c[k]

        for k in range(N - 1):
            model_fluxes[f"percolation_L{k+1}L{k+2}"][t] = perc_c[k]

        # Soil-profile recharge only. Wetland recharge stored separately.
        model_fluxes[f"recharge_L{N}"][t] = rech_c

        # -----------------------------
        # Store wetland outputs
        # -----------------------------
        model_fluxes["wetland_storage"][t] = wetland_state["storage_mm_grid"]
        model_fluxes["wetland_depth"][t] = wetland_state["depth_m"]
        model_fluxes["wetland_active_area_fraction"][t] = wetland_state[
            "active_area_fraction"
        ]

        model_fluxes["wetland_routed_runoff"][t] = wetland_state["routed_runoff_mm"]
        model_fluxes["wetland_precip"][t] = wetland_state["wetland_precip_mm"]
        model_fluxes["wetland_evap"][t] = wetland_state["wetland_evap_mm"]
        model_fluxes["wetland_infiltration"][t] = wetland_state[
            "wetland_infiltration_mm"
        ]
        model_fluxes["wetland_overflow"][t] = wetland_state["wetland_overflow_mm"]
        model_fluxes["wetland_recharge"][t] = wetland_state["wetland_recharge_mm"]

    return model_fluxes


