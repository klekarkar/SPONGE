""" 
Model Flux Calculation Functions
"""
import pandas as pd
import numpy as np
from tqdm.auto import trange
from typing import Optional
from wetland_processes import wetland_water_balance_step


""" HELPER FUNCTIONS """
def pack_lu_timeseries_to_2d(lu_types, idx, series_by_lu, name="TS") -> np.ndarray:
    """
    series_by_lu: dict {lu: pd.Series indexed exactly by idx}
    returns: (n_lu, n_t) float64 array
    """
    idx = pd.DatetimeIndex(idx)
    n_lu = len(lu_types)
    n_t = len(idx)
    arr = np.zeros((n_lu, n_t), dtype=np.float64)

    for i, lu in enumerate(lu_types):
        if lu not in series_by_lu:
            raise ValueError(f"Missing {name}_by_lu for lu='{lu}'")
        s = series_by_lu[lu]
        if not pd.DatetimeIndex(s.index).equals(idx):
            raise ValueError(f"{name}_by_lu['{lu}'] index does not match model index.")
        arr[i, :] = s.to_numpy(dtype=np.float64)

    return arr


def pack_root_frac_to_2d(lu_types, root_frac_by_lu, n_layers: int) -> np.ndarray:
    """
    root_frac_by_lu: dict {lu: (n_layers,) array}
    returns: (n_lu, n_layers) float64 array
    """
    n_lu = len(lu_types)
    out = np.zeros((n_lu, n_layers), dtype=np.float64)

    for i, lu in enumerate(lu_types):
        if lu not in root_frac_by_lu:
            raise ValueError(f"Missing root_frac_by_lu for lu='{lu}'")
        rf = np.asarray(root_frac_by_lu[lu], dtype=np.float64)
        if rf.shape != (n_layers,):
            raise ValueError(f"root_frac_by_lu['{lu}'] must have shape {(n_layers,)}, got {rf.shape}")
        out[i, :] = rf

    return out


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
        "wetland_active_area_fraction": np.zeros(n, dtype=float)
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

#----------------------------------------------------------------------------------
"""Canopy Water Balance Simulation
"""
#----------------------------------------------------------------------------------
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


"""----------------------------------------------------------------------------------
111
Soil Water Balance Simulation
-------------------------------------------------------------------------------------
"""
"""Stress functions for transpiration reduction due to soil moisture stress."""

""" Linear soil moisture stress function """
def sm_stress_linear_vec(sm, s_fc, s_wp):
    sm = np.asarray(sm, np.float64)
    s_fc = np.asarray(s_fc, np.float64)
    s_wp = np.asarray(s_wp, np.float64)
    denom = np.maximum(s_fc - s_wp, 1e-12)
    return np.clip((sm - s_wp) / denom, 0.0, 1.0)


""" VIC-style soil moisture stress function (root-zone integrated) """
def vic_gsm_inverse(sm, z, theta_wp, theta_cr):
    W  = sm * z
    Ww = theta_wp * z
    Wc = theta_cr * z
    if W >= Wc:
        return 1.0
    elif W <= Ww:
        return 0.0
    else:
        return (W - Ww) / (Wc - Ww + 1e-12)


def vic_rootzone_gsm(sm, z_layers, root_frac, theta_wp, theta_cr):
    gsm_inv_layers = np.empty(sm.shape[0], dtype=np.float64)
    for k in range(sm.shape[0]):
        gsm_inv_layers[k] = vic_gsm_inverse(sm[k], z_layers[k], theta_wp[k], theta_cr[k])

    gsm_inv = float(np.sum(root_frac * gsm_inv_layers))
    gsm_inv = float(np.clip(gsm_inv, 0.0, 1.0))
    if gsm_inv <= 1e-6:
        return np.inf
    return 1.0 / gsm_inv

def araki_2025_rootzone_stress(sm, theta_wp, theta_cr, root_frac, q_sm):
    """
    Root-zone stress based on Araki_2025 nonlinear Stage-II loss:
     https://doi.org/10.1029/2024GL111403

      f_layer = ((sm - wp)/(cr - wp))^q_sm, clipped [0,1]
      f_rootzone = sum(root_frac * f_layer)
    """
    sm = np.asarray(sm, np.float64)
    theta_wp = np.asarray(theta_wp, np.float64)
    theta_cr = np.asarray(theta_cr, np.float64)
    root_frac = np.asarray(root_frac, np.float64)

    denom = np.maximum(theta_cr - theta_wp, 1e-12)
    x = (sm - theta_wp) / denom
    x = np.clip(x, 0.0, 1.0)

    q_sm = float(q_sm)
    f_layer = x ** q_sm
    f_rootzone = float(np.clip(np.sum(root_frac * f_layer), 0.0, 1.0))
    return f_rootzone

#----------------------------------------------------------------------------------

""" Infiltration into the top soil layer (VIC variable infiltration curve) """
def step_infiltration(
    sm: np.ndarray,          # (N,) updated in-place
    throughfall: float,      # mm/timestep
    z_layers: np.ndarray,    # (N,) mm
    theta_sat: np.ndarray,   # (N,)
    b_infilt: float,
    theta_r: Optional[np.ndarray] = None            # (N,) or None
) -> tuple[float, float]:
    """
    Updates sm[0] in-place only.
    Returns (infil, runoff) in mm/timestep.
    """
    P = float(throughfall)
    z1 = float(z_layers[0])
    b = float(b_infilt)

    if theta_r is None:
        W  = max(0.0, sm[0] * z1)
        Wc = max(1e-12, theta_sat[0] * z1)
        W_r = 0.0
    else:
        W_r = float(theta_r[0]) * z1
        W  = max(0.0, (sm[0] * z1) - W_r)
        Wc = max(1e-12, (theta_sat[0] * z1) - W_r)

    im = (1.0 + b) * Wc
    frac = float(np.clip(1.0 - (W / Wc), 0.0, 1.0))
    i0 = im * (1.0 - frac ** (1.0 / (1.0 + b)))

    if i0 + P >= im:
        runoff = max(0.0, P - (Wc - W))
    else:
        runoff = P - Wc + W + Wc * (1.0 - (i0 + P) / im) ** (1.0 + b)
        runoff = max(0.0, runoff)

    infil = P - runoff
    W_new = min(Wc, W + infil)

    if theta_r is None:
        sm[0] = W_new / z1
    else:
        sm[0] = (W_r + W_new) / z1

    sm[0] = float(np.clip(sm[0], 0.0, theta_sat[0]))
    return float(infil), float(runoff)


def allocate_transpiration_root_water_stress(
    Tpot: float,
    f_root: np.ndarray,
    stress: np.ndarray,
    sm: np.ndarray,
    z_layers: np.ndarray,
    theta_min: np.ndarray,
    T_layer_out: np.ndarray,   # (N,) preallocated, written in-place
    max_passes: Optional[int] = None,
    eps: float = 1e-12
) -> float:
    """
    Writes T_layer_out (mm per layer). Returns unmet transpiration (mm).
    """
    f_root = np.asarray(f_root, dtype=np.float64)
    stress = np.asarray(stress, dtype=np.float64)

    N = f_root.shape[0]
    if max_passes is None:
        max_passes = N

    Tpot = float(max(0.0, Tpot))
    T_layer_out.fill(0.0)
    if Tpot <= eps:
        return 0.0

    rooted = f_root > 0.0

    # extractable water per layer (mm)
    w_avail = (sm - theta_min) * z_layers
    w_avail = np.maximum(w_avail, 0.0)
    w_avail[~rooted] = 0.0

    demand = Tpot

    # scratch arrays (local small allocations are OK; but we can still avoid heavy ones)
    # N is small (<= ~10), so this is not the dominant cost.
    for _ in range(max_passes):
        if demand <= eps:
            break

        remain = w_avail - T_layer_out
        remain = np.maximum(remain, 0.0)
        remain[~rooted] = 0.0

        w = f_root * stress * remain
        wsum = float(w.sum())
        if wsum <= eps:
            break

        add = demand * (w / wsum)
        add = np.minimum(add, remain)

        T_layer_out += add
        demand = Tpot - float(T_layer_out.sum())

    unmet = max(0.0, Tpot - float(T_layer_out.sum()))
    return float(unmet)


def step_et_partition(
    sm: np.ndarray,                 # (N,) updated in-place
    z_layers: np.ndarray,
    pet_t: float,
    canopy_evap_t: float,
    beta_t: float,
    pveg: float,
    pbare: float,
    root_frac: np.ndarray,          # (N,)
    theta_fc: np.ndarray,
    theta_wp: np.ndarray,
    theta_r: np.ndarray,
    T_layers_out: np.ndarray,       # (N,) scratch output filled in-place
    stress_layers_scratch: np.ndarray,  # (N,) scratch
    stress_method: str,
    f_cr: float,
    q_sm: float,
    q_soil: float,
    theta_cr: Optional[np.ndarray] = None
) -> tuple[np.ndarray, float, float, float, float]:
    """
    Updates sm in-place.
    Returns (T_layers_out, Esoil, ET_total, unmet_T, Ec_grid).
    """
    PET = float(pet_t)
    Ec_veg = min(float(canopy_evap_t), PET)
    PET_veg_rem = max(0.0, PET - Ec_veg)
    Tpot = float(pveg) * PET_veg_rem

    #wet-canopy suppression (WEP-style) DOI: 10.1002/hyp.275
    if beta_t is not None:
        Tpot *= (1.0 - float(np.clip(beta_t, 0.0, 1.0)))

    stress_method = str(stress_method).strip().lower()

    if stress_method == "linear":
        stress_layers_scratch[:] = sm_stress_linear_vec(sm, theta_fc, theta_wp)
        Tpot_eff = Tpot

    elif stress_method in ("canopy_resistance", "vic_gsm"):
        # build theta_cr_use
        if theta_cr is None:
            theta_cr_use = theta_wp + float(f_cr) * (theta_fc - theta_wp)
        else:
            theta_cr_use = np.asarray(theta_cr, dtype=np.float64)
            if theta_cr_use.size == 1:
                theta_cr_use = np.full_like(theta_wp, float(theta_cr_use))

        g_sm = vic_rootzone_gsm(sm, z_layers, root_frac, theta_wp, theta_cr_use)
        s = 0.0 if not np.isfinite(g_sm) else 1.0 / g_sm
        s = float(np.clip(s, 0.0, 1.0))

        Tpot_eff = Tpot * s
        stress_layers_scratch.fill(1.0)

    elif stress_method == "araki_2025":
        # need theta_cr_use here too (same definition)
        if theta_cr is None:
            theta_cr_use = theta_wp + float(f_cr) * (theta_fc - theta_wp)
        else:
            theta_cr_use = np.asarray(theta_cr, dtype=np.float64)

        s = araki_2025_rootzone_stress(
            sm=sm,
            theta_wp=theta_wp,
            theta_cr=theta_cr_use,
            root_frac=root_frac,
            q_sm=q_sm,
        )

        Tpot_eff = Tpot * s
        stress_layers_scratch.fill(1.0)

    else:
        raise ValueError(f"Unknown stress_method: {stress_method}")


    unmet_T = allocate_transpiration_root_water_stress(
        Tpot=Tpot_eff,
        f_root=root_frac,
        stress=stress_layers_scratch,
        sm=sm,
        z_layers=z_layers,
        theta_min=theta_wp,
        T_layer_out=T_layers_out,
        max_passes=sm.shape[0]
    )

    # remove transpiration
    sm -= T_layers_out / z_layers

    
    Wtop = max(0.0, (sm[0] - theta_wp[0]) * z_layers[0])

    # ===========bare soil evaporation===============
    beta = float(np.clip((sm[0] - theta_wp[0]) / (theta_fc[0] - theta_wp[0] + 1e-12), 0.0, 1.0))
    q_soil = float(q_soil)
    if q_soil != 1.0:
        beta = beta ** q_soil

    Esoil_pot = float(pbare) * PET * beta

    Esoil = min(Esoil_pot, Wtop)
    sm[0] -= Esoil / z_layers[0]

    # enforce residual
    sm[:] = np.maximum(sm, theta_r)

    Ec_grid = Ec_veg
    ET_total = Ec_grid + float(T_layers_out.sum()) + float(Esoil)

    return T_layers_out, float(Esoil), float(ET_total), float(unmet_T), float(Ec_grid)


def step_drainage_unit_gradient(
    sm: np.ndarray,               # (N,) updated in-place
    z_layers: np.ndarray,
    theta_sat: np.ndarray,
    theta_fc: np.ndarray,
    theta_r: np.ndarray,
    Ks_cm_d: np.ndarray,
    vG_m: np.ndarray,
    tau_vG: float,
    dt_hours: float,
    perc_out: np.ndarray,         # (N-1,) filled in-place
    drain_above_fc: bool = False
) -> float:
    """
    Updates sm in-place, fills perc_out, returns recharge (mm).
    """
    n_layers = sm.shape[0]
    perc_out.fill(0.0)

    # percolation between layers
    for k in range(n_layers - 1):
        denom = (theta_sat[k] - theta_r[k])
        Se = (sm[k] - theta_r[k]) / denom if denom > 0 else 0.0
        Se = float(np.clip(Se, 0.0, 1.0))

        K_theta = Ks_cm_d[k] * (Se**tau_vG) * (1.0 - (1.0 - Se**(1.0 / vG_m[k]))**vG_m[k])**2
        Kk = K_theta * 10.0 / 24.0  # cm/d -> mm/h
        Fk_pot = Kk * dt_hours

        if drain_above_fc:
            Wavail = max(0.0, (sm[k] - theta_fc[k]) * z_layers[k])
        else:
            Wavail = max(0.0, (sm[k] - theta_r[k]) * z_layers[k])

        Wspace = max(0.0, (theta_sat[k+1] - sm[k+1]) * z_layers[k+1])
        Fk = min(Fk_pot, Wavail, Wspace)

        sm[k]   -= Fk / z_layers[k]
        sm[k+1] += Fk / z_layers[k+1]
        perc_out[k] = Fk

    # bottom recharge
    denom = (theta_sat[-1] - theta_r[-1])
    Se = (sm[-1] - theta_r[-1]) / denom if denom > 0 else 0.0
    Se = float(np.clip(Se, 0.0, 1.0))

    K_theta = Ks_cm_d[-1] * (Se**tau_vG) * (1.0 - (1.0 - Se**(1.0 / vG_m[-1]))**vG_m[-1])**2
    Kb = K_theta * 10.0 / 24.0
    Fb_pot = Kb * dt_hours

    if drain_above_fc:
        Wavail_b = max(0.0, (sm[-1] - theta_fc[-1]) * z_layers[-1])
    else:
        Wavail_b = max(0.0, (sm[-1] - theta_r[-1]) * z_layers[-1])

    recharge = min(Fb_pot, Wavail_b)
    sm[-1] -= recharge / z_layers[-1]

    sm[:] = np.maximum(sm, theta_r)
    return float(recharge)


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

def soil_water_balance_nlayers_multi_lu(
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


    loop = trange(n_t, desc="Running Soil WB (multi-LU)") if show_progress else range(n_t)

    # =============================================================================
    # TIME LOOP
    # =============================================================================
    for t in loop:
        P_t   = float(P[t])
        PET_t = float(PET[t])

        infil_c   = 0.0
        runoff_c  = 0.0
        Esoil_c   = 0.0
        ET_c      = 0.0
        unmetT_c  = 0.0
        Ec_grid_c = 0.0
        rech_c    = 0.0

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
            beta_t = float(beta_arr[i, t]) if (not bare) else 0.0

            pveg = float(pveg_lu[i])
            pbare = float(pbare_lu[i])
            

            # Rain reaching soil:
            # - bare tile: all precipitation to soil
            # - vegetated tile: only throughfall to soil
            P_to_soil = P_t if bare else TF_lu

            # ---- Step 1: infiltration/runoff (updates sm_tmp[0] in-place)
            infil, runoff = step_infiltration(
                sm=sm_tmp,
                throughfall=P_to_soil,
                z_layers=z_layers,
                theta_sat=theta_sat,
                b_infilt=b_infilt,
                theta_r=theta_r
            )

            # ---- Step 2: ET partitioning (updates sm_tmp in-place)
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
                theta_cr=theta_cr_arr
            )
            # sm_tmp already updated in-place by step_et_partition

            # ---- Step 3: drainage/percolation (updates sm_tmp in-place; fills perc_tmp)
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
                drain_above_fc=drain_above_fc
            )
            # sm_tmp updated in-place; perc_tmp filled; recharge returned

            wetland_state = wetland_water_balance_step(
                        storage_mm_grid=wetland_storage,
                        precipitation_mm=pre_t,
                        pet_mm=pet_t,
                        surface_runoff_mm=surface_runoff_t,
                        wetland_params=wetland_params,
                        dt_hours=dt_hours,
                    )

            wetland_storage = wetland_state["storage_mm_grid"]

            surface_runoff_t = wetland_state["surface_runoff_after_wetland_mm"]
            recharge_t = recharge_t + wetland_state["wetland_recharge_mm"]
            total_evap_t = total_evap_t + wetland_state["wetland_evap_mm"]


            # write back LU state
            sm_by_lu[i, :] = sm_tmp

            # aggregate (catchment mean)
            infil_c   += frac * float(infil)
            runoff_c  += frac * float(runoff)
            Esoil_c   += frac * float(Esoil)
            ET_c      += frac * float(ET_total)
            unmetT_c  += frac * float(unmet_T)
            Ec_grid_c += frac * float(Ec_grid)
            rech_c    += frac * float(recharge)

            sm_c       += frac * sm_tmp
            T_layers_c += frac * T_layers
            perc_c     += frac * perc_tmp

        # -----------------------------
        # Store catchment averages
        # -----------------------------
        model_fluxes["infiltration"][t]     = infil_c
        model_fluxes["surface_run_off"][t]  = runoff_c
        model_fluxes["evap_actual_bs"][t]   = Esoil_c
        model_fluxes["total_evap"][t]       = ET_c
        model_fluxes["unmet_T"][t]          = unmetT_c
        model_fluxes["canopy_evap_grid"][t] = Ec_grid_c

        model_fluxes["pre"][t] = P_t
        model_fluxes["pet"][t] = PET_t

        for k in range(N):
            model_fluxes[f"soil_moisture_L{k+1}"][t] = sm_c[k]
            model_fluxes[f"aET_L{k+1}"][t]           = T_layers_c[k]

        for k in range(N - 1):
            model_fluxes[f"percolation_L{k+1}L{k+2}"][t] = perc_c[k]
        model_fluxes[f"recharge_L{N}"][t] = rech_c

    return model_fluxes



