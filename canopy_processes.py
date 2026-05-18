from __future__ import annotations
import logging
import logging
import numpy as np
import pandas as pd
from tqdm.auto import trange
logger = logging.getLogger(__name__)

#Canopy water balance functions

""" HELPER FUNCTIONS """
def _pack_daily_lai_to_2d(lu_types, forcing_index, lai_daily_by_lu) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack daily LAI into (n_lu, n_days) float64 and return (lai_2d, unique_days).
    - unique_days is a normalized daily DatetimeIndex spanning forcing days.
    - lai_2d[i, d] corresponds to LU i LAI at unique_days[d].
    """
    forcing_index = pd.DatetimeIndex(forcing_index)
    days = forcing_index.normalize()
    unique_days = pd.DatetimeIndex(days.unique()).sort_values()

    n_lu = len(lu_types)
    n_days = len(unique_days)
    lai_2d = np.zeros((n_lu, n_days), dtype=np.float64)

    # map day->col index
    day_to_j = {d: j for j, d in enumerate(unique_days)}

    for i, lu in enumerate(lu_types):
        if lu not in lai_daily_by_lu:
            raise ValueError(f"Missing LAI series for lu_type='{lu}' in lai_daily_by_lu.")

        s = lai_daily_by_lu[lu].copy()
        s.index = pd.DatetimeIndex(s.index).normalize()

        # Ensure every needed day exists (fast check)
        missing = unique_days.difference(s.index)
        if len(missing) > 0:
            raise ValueError(f"LAI series for '{lu}' missing {len(missing)} day(s), first: {missing[0].date()}")

        # Fill lai_2d for required days
        # Use reindex to guarantee aligned order, then numpy
        lai_2d[i, :] = s.reindex(unique_days).to_numpy(dtype=np.float64)

    return lai_2d, unique_days


def canopy_step_liang_1994(P_t, PET_t, I_prev, LAI, c_int):
    """
    Liang et al. (1994):  https://doi.org/10.1029/94JD00483
    Returns: (I_new, TF, Ec, beta)
    beta here is wet fraction proxy used for suppression (0-1).
    """
    LAI = float(np.asarray(LAI).squeeze())
    c_int = float(np.asarray(c_int).squeeze())
    
    I_max = max(0.0, c_int * LAI)

    if I_prev > 0.0 and I_max > 0.0:
        beta = float(np.clip((I_prev / I_max) ** (2.0 / 3.0), 0.0, 1.0))
    else:
        beta = 0.0

    Ec_max = beta * PET_t if PET_t > 0.0 else 0.0

    # IMPORTANT: use ONLY stored water if you want to avoid instant “rain-powered” Ec spikes
    Ec = min(Ec_max, I_prev)

    I_after = I_prev + P_t - Ec
    I_new = min(I_max, max(0.0, I_after))
    TF = max(0.0, I_after - I_max)

    return I_new, TF, Ec, beta

#----------------------------------------------------------------------------------
def canopy_step_swat(P_t, PET_t, I_prev, LAI, LAI_max_lu, canmx_mm=2.0):
    """
    SWAT-style:
      can_day = canmx * LAI/LAImax
      fill storage first, spill remainder as TF
      evaporate intercepted water from storage with PET limit
    Returns: (I_new, TF, Ec, beta)
    beta returned as wet fraction proxy = I_mid / can_day (0-1) after rain.
    """

    if isinstance(canmx_mm, (list, tuple, np.ndarray)):
        canmx_mm = canmx_mm[0]
    canmx_mm = float(canmx_mm)


    LAImax = max(float(LAI_max_lu), 1e-6)
    can_day = max(0.0, float(canmx_mm) * float(LAI) / LAImax)

    # 1) fill storage first
    space = max(0.0, can_day - I_prev)
    store_add = min(P_t, space)
    TF = max(0.0, P_t - store_add) #throughfall is excess after filling storage
    I_mid = I_prev + store_add

    # wetness proxy after rain filling
    beta = 0.0 if can_day <= 0 else float(np.clip(I_mid / can_day, 0.0, 1.0))

    # 2) evaporate from storage
    Ec = min(float(PET_t), float(I_mid))
    I_new = I_mid - Ec

    return I_new, TF, Ec, beta

#----------------------------------------------------------------------------------
"""Canopy Water Balance Simulation
 Liang et al. (1994):  https://doi.org/10.1029/94JD00483"""
#----------------------------------------------------------------------------------
def run_canopy_wb(
    forcing,
    flux_arrays,
    sim_freq,
    lu_types,
    lu_fracs,
    lai_daily_by_lu,
    canopy_method,
    I0=0.0,
    c_int=0.1, #interception capacity per LAI (mm), used in Liang 1994, but not in SWAT
    canmx_mm=2.0, #default canmx in SWAT is 2.0 mm, but we can override with config
    lai_max_mode="climatology",
    lai_max_fixed=5.0,
    show_progress=True,
    return_per_lu_series=True,
    logger=None,
):
    """
    Fast multi-LU canopy interception / canopy evaporation,
    Multi-LU canopy interception and canopy evaporation using Liang et al. (1994) logic.

    What this function does:
      1) Reads forcing P and PET at the model timestep (hourly or daily).
      2) For each timestep, runs canopy interception *separately per LU* because LAI differs.
      3) Aggregates per-LU results into catchment-average fluxes using LU area fractions.

    What it returns:
      - flux_arrays updated in-place (catchment averages only).
      - Per-LU internal canopy storages (I_lu) are kept internally (state), not saved as outputs.

    with low overhead:
      - LAI packed to 2D
      - per-LU state stored as arrays
      - per-LU TF/Ec/I stored as 2D arrays
      - no pandas .loc/.iloc in the hot loop
    """

    sim_freq = str(sim_freq).upper().strip()
    if sim_freq not in ("H", "D"):
        raise ValueError("sim_freq must be 'H' or 'D'.")

    if not {"P", "PET"}.issubset(forcing.columns):
        raise ValueError("forcing must have columns ['P','PET'].")

    out_index = pd.DatetimeIndex(forcing.index)
    n_t = len(out_index)

    if "index" not in flux_arrays:
        raise ValueError("flux_arrays must contain 'index'.")
    if not pd.DatetimeIndex(flux_arrays["index"]).equals(out_index):
        raise ValueError("flux_arrays['index'] does not match forcing index.")

    lu_fracs = np.asarray(lu_fracs, dtype=np.float64)
    if len(lu_types) != lu_fracs.size:
        raise ValueError("lu_types and lu_fracs must have same length.")
    if abs(float(lu_fracs.sum()) - 1.0) > 1e-3:
        raise ValueError(f"lu_fracs must sum to 1 (±1e-3). Got {lu_fracs.sum()}")

    n_lu = len(lu_types)

    # forcing arrays
    P = forcing["P"].to_numpy(dtype=np.float64)
    PET = forcing["PET"].to_numpy(dtype=np.float64)

    # Pack LAI daily to 2D and create day-index mapping for each timestep
    lai_2d, unique_days = _pack_daily_lai_to_2d(lu_types, out_index, lai_daily_by_lu)
    day_to_j = {d: j for j, d in enumerate(unique_days)}
    day_idx = out_index.normalize()
    day_j = np.fromiter((day_to_j[d] for d in day_idx), dtype=np.int64, count=n_t)

    # Catchment output arrays (ensure present)
    flux_arrays["interception"][:n_t] = 0.0
    flux_arrays["throughfall"][:n_t]  = 0.0
    flux_arrays["canopy_evap"][:n_t]  = 0.0
    flux_arrays["interception"][0]    = float(I0)

    # Per-LU state I (mm)
    I_lu = np.full(n_lu, float(I0), dtype=np.float64)

    # Per-LU outputs (2D) (mm/timestep or mm state)
    TF_lu = np.zeros((n_lu, n_t), dtype=np.float64)
    Ec_lu = np.zeros((n_lu, n_t), dtype=np.float64)
    I_store_lu = np.zeros((n_lu, n_t), dtype=np.float64)
    beta_lu = np.zeros((n_lu, n_t), dtype=np.float64) #wet fraction 

    #Canopy method selection
    # Canopy method selection (robust to accidental tuple/list)
    if isinstance(canopy_method, (list, tuple, np.ndarray)):
        if len(canopy_method) != 1:
            raise ValueError(f"canopy_method must be a single string, got {canopy_method}")
        canopy_method = canopy_method[0]

    canopy_method = str(canopy_method).strip().lower()
    # LU-specific LAImax (needed for SWAT)
    if canopy_method == "swat":
        if str(lai_max_mode).strip().lower() == "fixed":
            LAImax_lu = np.full(n_lu, float(lai_max_fixed), dtype=np.float64)
        else:
            # climatology max per LU across the whole sim period
            LAImax_lu = np.maximum(np.max(lai_2d, axis=1), 1e-6)
    else:
        LAImax_lu = None


    loop = trange(n_t, dynamic_ncols=False, desc="Running Canopy WB (multi-LU)") if show_progress else range(n_t)

    for t in loop:
        P_t = float(P[t])
        PET_t = float(PET[t])

        # catchment accumulators
        I_c = 0.0
        TF_c = 0.0
        Ec_c = 0.0

        jday = day_j[t]  # day column index for lai_2d

        for i in range(n_lu):
            LAI = float(lai_2d[i, jday])
            I_prev = float(I_lu[i])

            if canopy_method == "liang_1994":
                I_new, TF, Ec, beta = canopy_step_liang_1994(
                    P_t=P_t, PET_t=PET_t, I_prev=I_prev, LAI=LAI, c_int=c_int
                )

            elif canopy_method == "swat":
                I_new, TF, Ec, beta = canopy_step_swat(
                    P_t=P_t, PET_t=PET_t, I_prev=I_prev, LAI=LAI,
                    LAI_max_lu=LAImax_lu[i], canmx_mm=canmx_mm
                )
            else:
                raise ValueError(f"Unknown canopy_method: {canopy_method}")

            # update LU state
            I_lu[i] = I_new

            # store per-LU series
            I_store_lu[i, t] = I_new
            TF_lu[i, t] = TF
            Ec_lu[i, t] = Ec
            beta_lu[i, t] = beta

            frac = float(lu_fracs[i])
            I_c  += frac * I_new
            TF_c += frac * TF
            Ec_c += frac * Ec

        flux_arrays["interception"][t] = I_c
        flux_arrays["throughfall"][t]  = TF_c
        flux_arrays["canopy_evap"][t]  = Ec_c

    if not return_per_lu_series:
        # return arrays for max speed downstream (recommended)
        return flux_arrays, TF_lu, Ec_lu, I_store_lu, beta_lu

    # compatibility mode: return dict-of-Series like before
    TF_by_lu = {lu: pd.Series(TF_lu[i, :], index=out_index, name=f"TF_{lu}") for i, lu in enumerate(lu_types)}
    Ec_by_lu = {lu: pd.Series(Ec_lu[i, :], index=out_index, name=f"Ec_{lu}") for i, lu in enumerate(lu_types)}
    I_by_lu  = {lu: pd.Series(I_store_lu[i, :], index=out_index, name=f"I_{lu}")  for i, lu in enumerate(lu_types)}
    beta_by_lu = {lu: pd.Series(beta_lu[i, :], index=out_index, name=f"beta_{lu}")  for i, lu in enumerate(lu_types)}

    return flux_arrays, TF_by_lu, Ec_by_lu, I_by_lu, beta_by_lu



