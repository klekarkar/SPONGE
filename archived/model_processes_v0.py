""" 
Model Flux Calculation Functions
"""
import pandas as pd
import numpy as np
import prepare_inputs as prep
from tqdm.auto import trange


""" HELPER FUNCTIONS """
def pack_lu_timeseries_to_2d(lu_types, idx, series_by_lu, name="TF"):
    """
    Helper function to pack per-LU time series into a 2D array for easier processing.

    series_by_lu: dict {lu: pd.Series indexed exactly by idx}
    returns: (n_lu, n_t) float array
    """
    idx = pd.DatetimeIndex(idx)
    n_lu = len(lu_types)
    n_t = len(idx)
    arr = np.zeros((n_lu, n_t), dtype=np.float64)

    for i, lu in enumerate(lu_types):
        s = series_by_lu[lu]
        if not pd.DatetimeIndex(s.index).equals(idx):
            raise ValueError(f"{name}_by_lu['{lu}'] index does not match model index.")
        arr[i, :] = s.to_numpy(dtype=np.float64)

    return arr


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

#----------------------------------------------------------------------------------
"""Canopy Water Balance Simulation
 Liang et al. (1994):  https://doi.org/10.1029/94JD00483"""
#----------------------------------------------------------------------------------
def run_canopy_wb(
    forcing: pd.DataFrame,                  # model-timestep forcing with columns: P, PET (mm/timestep)
    flux_arrays: dict,                      # your existing catchment-averaged output container
    sim_freq: str,                          # "H" or "D" (used mainly for validation / logging)
    lu_types: list[str],                    # land-use / vegetation class names (e.g., IGBP types + "Bare")
    lu_fracs: np.ndarray,                   # same length as lu_types, must sum to 1
    lai_daily_by_lu: dict[str, pd.Series],  # {lu: daily LAI Series indexed by date (normalized)}
    I0: float = 0.0,                        # initial canopy water storage per LU (mm)
    show_progress: bool = True,
    logger=None
):
    """
    Multi-LU canopy interception and canopy evaporation using Liang et al. (1994) logic.

    What this function does:
      1) Reads forcing P and PET at the model timestep (hourly or daily).
      2) For each timestep, runs canopy interception *separately per LU* because LAI differs.
      3) Aggregates per-LU results into catchment-average fluxes using LU area fractions.

    What it returns:
      - flux_arrays updated in-place (catchment averages only).
      - Per-LU internal canopy storages (I_lu) are kept internally (state), not saved as outputs.
    """

    # -------------------------------------------------------------------------
    # 0) Basic checks and normalization
    # -------------------------------------------------------------------------
    sim_freq = str(sim_freq).upper().strip()
    if sim_freq not in ("H", "D"):
        raise ValueError("sim_freq must be 'H' or 'D'.")

    # forcing must provide precipitation and PET at the model timestep
    if not {"P", "PET"}.issubset(forcing.columns):
        raise ValueError("forcing must have columns ['P','PET'].")

    # model timestep index (hourly or daily)
    out_index = pd.DatetimeIndex(forcing.index)
    n = len(out_index)

    # enforce that the output arrays were initialized on the same index as forcing
    if "index" not in flux_arrays:
        raise ValueError("flux_arrays must contain 'index'.")
    if not pd.DatetimeIndex(flux_arrays["index"]).equals(out_index):
        raise ValueError("flux_arrays['index'] does not match forcing index.")

    # LU definitions must align
    lu_fracs = np.asarray(lu_fracs, dtype=float)
    if len(lu_types) != len(lu_fracs):
        raise ValueError("lu_types and lu_fracs must have same length.")
    if abs(lu_fracs.sum() - 1.0) > 1e-3:
        raise ValueError(f"lu_fracs must sum to 1 (±1e-3). Got {lu_fracs.sum()}")

    # Normalize LAI series indices (we expect daily series indexed by normalized dates)
    lai_by_lu = {}
    for lu in lu_types:
        if lu not in lai_daily_by_lu:
            raise ValueError(f"Missing LAI series for lu_type='{lu}' in lai_daily_by_lu.")
        s = lai_daily_by_lu[lu].copy()
        s.index = pd.DatetimeIndex(s.index).normalize()
        lai_by_lu[lu] = s

    # Convert forcing columns to arrays for faster indexing in the loop
    P = forcing["P"].to_numpy(dtype=float)
    PET = forcing["PET"].to_numpy(dtype=float)

    # -------------------------------------------------------------------------
    # 1) Initialize catchment-averaged output arrays (your existing container)
    # -------------------------------------------------------------------------
    # Note: your variable "interception" is storing canopy storage (I),
    # not "interception loss". This matches your old implementation.
    flux_arrays["interception"][:n] = 0.0   # canopy storage (catchment avg)
    flux_arrays["throughfall"][:n]  = 0.0   # water reaching soil (catchment avg)
    flux_arrays["canopy_evap"][:n]  = 0.0   # evaporation from canopy (catchment avg)
    flux_arrays["interception"][0]  = float(I0)

    # -------------------------------------------------------------------------
    # 2) Internal per-LU canopy storage state
    # -------------------------------------------------------------------------
    # This is the key difference vs the old single-LAI model:
    # each LU has its own canopy storage because LAI differs => I_max differs.
    I_lu = {lu: float(I0) for lu in lu_types}

    # Per-LU outputs to return (mm/timestep, per LU tile area)
    throughfall_by_lu = {lu: np.zeros(n, dtype=float) for lu in lu_types}
    canopy_evap_by_lu = {lu: np.zeros(n, dtype=float) for lu in lu_types}
    interception_store_by_lu = {lu: np.zeros(n, dtype=float) for lu in lu_types}  # optional but handy

    # -------------------------------------------------------------------------
    # 3) Time loop: compute per-LU canopy WB and aggregate to catchment average
    # -------------------------------------------------------------------------
    loop = trange(n, dynamic_ncols=False, desc="Running Canopy WB (multi-LU)") if show_progress else range(n)

    for t in loop:
        # We index LAI with the "day" (normalized), because LAI is daily.
        day = out_index[t].normalize()

        # Model-timestep forcing (mm/timestep)
        P_t = float(P[t])
        PET_t = float(PET[t])

        # Catchment accumulators start at zero each timestep.
        # We will add (+=) each LU contribution weighted by its fraction.
        I_c = 0.0    # catchment-average canopy storage (state diagnostic)
        TF_c = 0.0   # catchment-average throughfall flux
        Ec_c = 0.0   # catchment-average canopy evaporation flux

        # ---- loop over land-use tiles ----
        for lu, frac in zip(lu_types, lu_fracs):

            # Daily LAI for this LU (mm? no, unitless LAI)
            if day not in lai_by_lu[lu].index:
                raise ValueError(f"LAI series for '{lu}' missing day {day.date()}")

            LAI = float(lai_by_lu[lu].loc[day])

            # Maximum canopy storage capacity (Liang-style linear with LAI)
            I_max = max(0.0, 0.2 * LAI) #Eq. 2

            # Previous canopy storage for this LU
            I_prev = I_lu[lu]

            # Max canopy evaporation depends on how full the canopy is (beta term)
            if I_prev > 0 and I_max > 0 and PET_t > 0:
                beta = (I_prev / I_max) ** (2.0 / 3.0)
                Ec_max = max(0.0, beta * PET_t)
            else:
                Ec_max = 0.0

            # Actual canopy evaporation (limited by available water in canopy + rainfall)
            # f is the fraction of timestep required for canopy evap to exhaust intercepted water (Eq. 10)
            f = min(1.0, (I_prev + P_t) / Ec_max) if Ec_max > 0 else 0.0
            Ec = min(f * Ec_max, I_prev + P_t) #Eq. 9

            # Update canopy storage after evaporation
            I_after = I_prev + P_t - Ec

            # Canopy storage is capped by I_max.
            # Excess becomes throughfall (water that reaches soil).
            I_new = min(I_max, max(0.0, I_after))
            TF = max(0.0, I_after - I_max)

            # ---- update per-LU state ----
            I_lu[lu] = I_new

            # save per-LU series (per tile area, not weighted)
            interception_store_by_lu[lu][t] = I_new
            throughfall_by_lu[lu][t] = TF
            canopy_evap_by_lu[lu][t] = Ec

            # ---- aggregate to catchment-average outputs ----
            # Catchment average = sum over tiles (frac * tile_value)
            I_c  += frac * I_new
            TF_c += frac * TF
            Ec_c += frac * Ec

        # Store catchment-average results in your existing arrays
        flux_arrays["interception"][t] = I_c
        flux_arrays["throughfall"][t]  = TF_c
        flux_arrays["canopy_evap"][t]  = Ec_c

    # convert per-LU arrays to Series with the model index
    throughfall_by_lu = {lu: pd.Series(v, index=out_index, name=f"TF_{lu}") for lu, v in throughfall_by_lu.items()}
    canopy_evap_by_lu = {lu: pd.Series(v, index=out_index, name=f"Ec_{lu}") for lu, v in canopy_evap_by_lu.items()}
    interception_store_by_lu = {lu: pd.Series(v, index=out_index, name=f"I_{lu}") for lu, v in interception_store_by_lu.items()}

    return flux_arrays, throughfall_by_lu, canopy_evap_by_lu, interception_store_by_lu

#----------------------------------------------------------------------------------

"""----------------------------------------------------------------------------------
111
Soil Water Balance Simulation
-------------------------------------------------------------------------------------
"""
""" Linear soil moisture stress function """
def sm_stress_linear_vec(sm, s_fc, s_wp):
    sm = np.asarray(sm, float)
    s_fc = np.asarray(s_fc, float)
    s_wp = np.asarray(s_wp, float)
    denom = np.maximum(s_fc - s_wp, 1e-12)
    return np.clip((sm - s_wp) / denom, 0.0, 1.0)
#----------------------------------------------------------------------------------
""" Canopy resistance function to determine transpiration reduction based on soil moisture stress (VIC-style) 
Based on Liang et al. (1994).
"""
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
    """
    Root-zone VIC soil moisture stress factor g_sm (dimensionless).

    Returns g_sm (>= 1.0); use 1/g_sm as transpiration reduction.
    """
    gsm_inv_layers = np.array([
        vic_gsm_inverse(sm[k], z_layers[k], theta_wp[k], theta_cr[k])
        for k in range(len(sm))
    ])

    # Root-weighted average (VIC Eq. 8 concept)
    gsm_inv = np.sum(root_frac * gsm_inv_layers)

    # Safety
    gsm_inv = np.clip(gsm_inv, 0.0, 1.0)

    if gsm_inv <= 1e-6:
        return np.inf  # fully stressed
    else:
        return 1.0 / gsm_inv

#----------------------------------------------------------------------------------


""" Infiltration into the top soil layer (VIC variable infiltration curve) """
def step_infiltration(
    sm, throughfall, z_layers, theta_sat,
    b_infilt=0.2,
    theta_r=None
):
    """
    Returns
    -------
    sm : updated (copy) soil moisture vector
    infil : mm/timestep
    runoff : mm/timestep
    """
    sm = sm.copy()
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
    frac = np.clip(1.0 - (W / Wc), 0.0, 1.0)
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
    return sm, float(infil), float(runoff)


#----------------------------------------------------------------------------------

""" Allocate transpiration into soil layers based on root distribution, water stress, and extractable water """

def allocate_transpiration_root_water_stress(
    Tpot,
    f_root,
    stress,
    sm,
    z_layers,
    theta_min,          # usually theta_wp (or theta_r)
    max_passes=None,
    eps=1e-12
):
    """
    Allocate potential transpiration Tpot (mm) into layers based on:
      - root fractions f_root (sum~1 over rooted layers),
      - stress factors stress (0..1),
      - extractable water (sm - theta_min) * z_layers.

    Returns
    -------
    T_layer : (N,) array of mm extracted per layer
    unmet   : scalar mm unmet transpiration
    """
    f_root   = np.asarray(f_root, dtype=float)
    stress   = np.asarray(stress, dtype=float)
    sm       = np.asarray(sm, dtype=float)
    z_layers = np.asarray(z_layers, dtype=float)
    theta_min = np.asarray(theta_min, dtype=float)

    N = len(f_root)
    if max_passes is None:
        max_passes = N

    # sanitize
    Tpot = float(max(0.0, Tpot))
    f_root = np.clip(f_root, 0.0, None)
    stress = np.clip(stress, 0.0, 1.0)

    # rooted mask
    rooted = f_root > 0.0

    # extractable water per layer (mm)
    w_avail = np.maximum(0.0, (sm - theta_min) * z_layers)
    w_avail[~rooted] = 0.0

    T_layer = np.zeros(N, dtype=float)
    demand = Tpot

    for _ in range(max_passes):
        if demand <= eps:
            break

        remain = np.maximum(0.0, w_avail - T_layer)
        remain[~rooted] = 0.0

        # weights: prefer layers with roots, water, and low stress
        w = f_root * stress * remain
        wsum = w.sum()

        if wsum <= eps:
            break

        add = demand * (w / wsum)
        add = np.minimum(add, remain)

        T_layer += add
        demand = Tpot - T_layer.sum()

    unmet = float(max(0.0, Tpot - T_layer.sum()))
    return T_layer, unmet

#----------------------------------------------------------------------------------

""" ET partitioning (transpiration by roots+stress + bare soil evaporation)"""

#assumes canopy evap is already computed and stored in model_fluxes["canopy_evap"][t]

def step_et_partition(
    sm, z_layers,
    pet_t, canopy_evap_t,
    pveg, pbare,
    root_frac, theta_fc, theta_wp, theta_r,
    allocate_transpiration_root_water_stress,
    stress_method="canopy_resistance",
    f_cr=0.65,            # user option (only used if theta_cr is not provided)
    theta_cr=None,       # user option: array (N,) or scalar
):
    sm = sm.copy()

    PET = float(pet_t)
    Ec_veg = min(float(canopy_evap_t), PET)
    PET_veg_rem = max(0.0, PET - Ec_veg)
    Tpot = pveg * PET_veg_rem

    # ---- transpiration stress handling ----
    if stress_method == "linear":
        # (your original behavior) layerwise 0..1 affects allocation weights
        stress_layers = sm_stress_linear_vec(sm, theta_fc, theta_wp)
        Tpot_eff = Tpot

    elif stress_method in ("canopy_resistance", "vic_gsm"):
        # define theta_cr (critical) properly
        if theta_cr is None:
            theta_cr = theta_wp + float(f_cr) * (theta_fc - theta_wp)
        theta_cr = np.asarray(theta_cr, float)
        if theta_cr.size == 1:
            theta_cr = np.full_like(sm, float(theta_cr), dtype=float)

        # g_sm is resistance factor >=1, so reduction is g_sm^{-1} in [0,1]
        g_sm = vic_rootzone_gsm(sm, z_layers, root_frac, theta_wp, theta_cr)
        gsm_inv = 0.0 if not np.isfinite(g_sm) else 1.0 / g_sm

        Tpot_eff = Tpot * float(np.clip(gsm_inv, 0.0, 1.0))
        stress_layers = np.ones_like(sm, dtype=float)  # allocation by roots+water only

    else:
        raise ValueError(f"Unknown stress_method: {stress_method}")

    # allocate transpiration (extractable water limit still enforced)
    T_layers, unmet_T = allocate_transpiration_root_water_stress(
        Tpot=Tpot_eff,
        f_root=root_frac,
        stress=stress_layers,
        sm=sm,
        z_layers=z_layers,
        theta_min=theta_wp,
        max_passes=len(sm)
    )

    sm -= T_layers / z_layers

    # ---- bare soil evaporation unchanged ----
    Wtop = max(0.0, (sm[0] - theta_wp[0]) * z_layers[0])
    beta = np.clip((sm[0] - theta_wp[0]) / (theta_fc[0] - theta_wp[0] + 1e-12), 0.0, 1.0)
    Esoil_pot = pbare * PET * beta
    Esoil = min(Esoil_pot, Wtop)
    sm[0] -= Esoil / z_layers[0]

    sm = np.maximum(sm, theta_r)
    Ec_grid = pveg * Ec_veg

    return sm, T_layers, float(Esoil), float(Ec_grid + float(T_layers.sum()) + Esoil), float(unmet_T), float(Ec_grid)

#----------------------------------------------------------------------------------
""" Soil Drainage between Layers """
def step_drainage_unit_gradient(
    sm, z_layers,
    theta_sat, theta_fc, theta_r,
    Ks_cm_d, vG_m,
    dt_hours,
    drain_above_fc=False
):
    sm = sm.copy()
    n_layers = len(sm)
    tau = 0.5 #used in van Genuchten hydraulic conductivity function

    perc = np.zeros(n_layers - 1, dtype=float)

    # hydraulic conductivity function based on unsaturated soil moisture (vG)
    for k in range(n_layers - 1):
        # compute effective saturation Se for layer k
        denom = (theta_sat[k] - theta_r[k])
        Se = (sm[k] - theta_r[k]) / denom if denom > 0 else 0.0
        Se = float(np.clip(Se, 0.0, 1.0))

        # unsaturated hydraulic conductivity using van Genuchten model
        K_theta = Ks_cm_d[k] * (Se**tau) * (1.0 - (1.0 - Se**(1.0 / vG_m[k]))**vG_m[k])**2
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
        perc[k] = Fk

    # bottom recharge
    denom = (theta_sat[-1] - theta_r[-1])
    Se = (sm[-1] - theta_r[-1]) / denom if denom > 0 else 0.0
    Se = float(np.clip(Se, 0.0, 1.0))

    K_theta = Ks_cm_d[-1] * (Se**tau) * (1.0 - (1.0 - Se**(1.0 / vG_m[-1]))**vG_m[-1])**2
    Kb = K_theta * 10.0 / 24.0
    Fb_pot = Kb * dt_hours

    if drain_above_fc:
        Wavail_b = max(0.0, (sm[-1] - theta_fc[-1]) * z_layers[-1])
    else:
        Wavail_b = max(0.0, (sm[-1] - theta_r[-1]) * z_layers[-1])

    recharge = min(Fb_pot, Wavail_b)
    sm[-1] -= recharge / z_layers[-1]

    sm = np.maximum(sm, theta_r)
    return sm, perc, float(recharge)


"""
----------------------------------------------------------------------------------
 Soil Water Balance for n layers
----------------------------------------------------------------------------------
"""
TF_arr = pack_lu_timeseries_to_2d(lu_types, forcing.index, TF_by_lu, name="TF")
Ec_arr = pack_lu_timeseries_to_2d(lu_types, forcing.index, Ec_by_lu, name="Ec")



def soil_water_balance_nlayers_multi_lu(
    model_fluxes: dict,
    forcing: pd.DataFrame,                        # columns: P, PET at model timestep
    z_layers: np.ndarray,                         # (N,) mm
    theta_sat: np.ndarray, theta_fc: np.ndarray, theta_wp: np.ndarray,
    Ks_cm_d: np.ndarray, vG_m: np.ndarray,    
    lu_types: list[str],
    lu_fracs: np.ndarray,
    root_frac_by_lu: dict[str, np.ndarray],       # {lu: (N,) root fractions}
    TF_by_lu: dict[str, pd.Series],               # {lu: Series indexed by forcing.index} (mm/timestep)
    Ec_by_lu: dict[str, pd.Series],               # {lu: Series indexed by forcing.index} (mm/timestep)
    sim_freq: str,
    theta_r: np.ndarray,
    drain_above_fc: bool = False,
    show_progress: bool = True,
    logger=None,
):
    """
    Multi-LU soil water balance driver.

    Each LU tile has its own soil moisture state because:
      - Root fractions differ by LU => transpiration extraction profile differs
      - Throughfall differs by LU (from canopy model)
      - Canopy evaporation differs by LU (from canopy model) => affects remaining PET for transpiration

    We still return / store ONLY catchment-average outputs in model_fluxes,
    matching your current design.

    Requirements:
      - forcing.index must match model_fluxes['index']
      - TF_by_lu and Ec_by_lu series indices must match forcing.index
      - root_frac_by_lu[lu] exists for vegetated LUs (zeros for bare LUs is fine)
    """

    # ---------------------------------------------------------------------
    # 0) Basic checks and setup
    # ---------------------------------------------------------------------
    sim_freq = str(sim_freq).upper().strip()
    if sim_freq not in ("H", "D"):
        raise ValueError("sim_freq must be 'H' or 'D'.")

    if not {"P", "PET"}.issubset(forcing.columns):
        raise ValueError("forcing must have columns ['P','PET'].")

    idx = pd.DatetimeIndex(model_fluxes["index"])
    if not idx.equals(pd.DatetimeIndex(forcing.index)):
        raise ValueError("model_fluxes['index'] must match forcing.index exactly.")

    n_timesteps = len(idx)

    z_layers   = np.asarray(z_layers, float)
    theta_sat  = np.asarray(theta_sat, float)
    theta_fc   = np.asarray(theta_fc, float)
    theta_wp   = np.asarray(theta_wp, float)
    Ks_cm_d    = np.asarray(Ks_cm_d, float)
    vG_m       = np.asarray(vG_m, float)

    # number of soil layers
    N = len(z_layers)
    if theta_r is None:
        theta_r = theta_wp.copy()
    theta_r = np.asarray(theta_r, float)

    lu_fracs = np.asarray(lu_fracs, dtype=float)
    if len(lu_types) != len(lu_fracs):
        raise ValueError("lu_types and lu_fracs must have same length.")
    if abs(lu_fracs.sum() - 1.0) > 1e-3:
        raise ValueError(f"lu_fracs must sum to 1 (±1e-3). Got {lu_fracs.sum()}")

    # forcing arrays for speed
    P = forcing["P"].to_numpy(dtype=float)
    PET = forcing["PET"].to_numpy(dtype=float)

    dt_hours = 1.0 if sim_freq == "H" else 24.0

    sm_c = np.zeros(N, dtype=np.float64)
    T_layers_c = np.zeros(N, dtype=np.float64)
    perc_c = np.zeros(N - 1, dtype=np.float64)


    # ---------------------------------------------------------------------
    # 1) Ensure outputs exist in model_fluxes (catchment average series)
    # ---------------------------------------------------------------------
    model_fluxes.setdefault("infiltration", np.zeros(n_timesteps))
    model_fluxes.setdefault("surface_run_off", np.zeros(n_timesteps))
    model_fluxes.setdefault("evap_actual_bs", np.zeros(n_timesteps))
    model_fluxes.setdefault("total_evap", np.zeros(n_timesteps))
    model_fluxes.setdefault("unmet_T", np.zeros(n_timesteps))
    model_fluxes.setdefault("canopy_evap_grid", np.zeros(n_timesteps))

    # per-layer outputs for ET and soil moisture
    for k in range(N):
        model_fluxes.setdefault(f"aET_L{k+1}", np.zeros(n_timesteps))
        model_fluxes.setdefault(f"soil_moisture_L{k+1}", np.zeros(n_timesteps))

    # percolation between layers
    for k in range(N - 1):
        model_fluxes.setdefault(f"percolation_L{k+1}L{k+2}", np.zeros(n_timesteps))
    model_fluxes.setdefault(f"recharge_L{N}", np.zeros(n_timesteps))

    # Optional: keep forcing copies in model_fluxes like your old code
    model_fluxes.setdefault("pre", np.zeros(n_timesteps))
    model_fluxes.setdefault("pet", np.zeros(n_timesteps))

    # ---------------------------------------------------------------------
    # 2) Initialize per-LU soil moisture states from the existing initial condition
    # ---------------------------------------------------------------------
    sm_init = np.array([model_fluxes[f"soil_moisture_L{k+1}"][0] for k in range(N)], dtype=float)
    sm_by_lu = {lu: sm_init.copy() for lu in lu_types}

    # ---------------------------------------------------------------------
    # 3) Validate that TF_by_lu and Ec_by_lu are aligned and available
    # ---------------------------------------------------------------------
    for lu in lu_types:
        if lu not in TF_by_lu:
            raise ValueError(f"Missing TF_by_lu for lu='{lu}'")
        if lu not in Ec_by_lu:
            raise ValueError(f"Missing Ec_by_lu for lu='{lu}'")

        if not pd.DatetimeIndex(TF_by_lu[lu].index).equals(idx):
            raise ValueError(f"TF_by_lu['{lu}'] index does not match model time index.")
        if not pd.DatetimeIndex(Ec_by_lu[lu].index).equals(idx):
            raise ValueError(f"Ec_by_lu['{lu}'] index does not match model time index.")

        if lu not in root_frac_by_lu:
            raise ValueError(f"Missing root_frac_by_lu for lu='{lu}'")

        rf = np.asarray(root_frac_by_lu[lu], float)
        if rf.shape != (N,):
            raise ValueError(f"root_frac_by_lu['{lu}'] must have shape (n_layers,) = {(N,)}, got {rf.shape}")

    # ---------------------------------------------------------------------
    # 4) Time loop
    # ---------------------------------------------------------------------
    loop = trange(n_timesteps, desc="Running Soil WB (multi-LU)") if show_progress else range(n_timesteps)

    for t in loop:

        P_t = float(P[t])
        PET_t = float(PET[t])

        # catchment accumulators
        infil_c = 0.0
        runoff_c = 0.0
        Esoil_c = 0.0
        ET_c = 0.0
        unmetT_c = 0.0
        Ec_grid_c = 0.0

        #catchment-average states
        sm_c = np.zeros(N, dtype=float)
        T_layers_c = np.zeros(N, dtype=float)
        perc_c = np.zeros(N - 1, dtype=float)
        rech_c = 0.0

        # ---- loop over LU tiles ----
        for lu, frac in zip(lu_types, lu_fracs):

            # current LU soil state
            sm = sm_by_lu[lu].copy()

            # (A) Rain input to soil for this LU = throughfall from canopy model
            TF_lu = float(TF_by_lu[lu].iloc[t])

            # (B) Canopy evaporation for this LU (interception loss)
            Ec_lu = float(Ec_by_lu[lu].iloc[t])

            # (C) Root fractions for this LU
            root_frac = np.asarray(root_frac_by_lu[lu], float)

            # Decide whether this LU is "bare" based on root profile (simple rule):
            # if roots are all zero => treat as bare tile (no transpiration, no canopy evap effect).
            is_bare = bool(np.allclose(root_frac, 0.0))

            # tile fractions for ET partitioning
            pveg = 0.0 if is_bare else 1.0
            pbare = 1.0 if is_bare else 0.0

            # -----------------------------------------------------------------
            # Step 1: Infiltration + surface runoff (Liang/VIC variable infiltration curve)
            # -----------------------------------------------------------------
            # For bare tiles, rainfall to soil should be direct precipitation, not throughfall.
            P_to_soil = P_t if is_bare else TF_lu

            sm_inf, flx_inf = step_infiltration(
                sm=sm,
                throughfall=P_to_soil,
                z_layers=z_layers,
                theta_sat=theta_sat,
                b_infilt=0.2,
                theta_r=theta_r
            )

            # we follow your original behavior: only top layer gets updated by infiltration
            sm[0] = sm_inf[0]

            # -----------------------------------------------------------------
            # Step 2: ET partitioning (canopy evap reduces PET available for transpiration)
            # -----------------------------------------------------------------
            # IMPORTANT: canopy_evap_t is per LU tile area (not area-weighted).
            # step_et_partition will compute Ec_grid = pveg * canopy_evap_t (tile scale).
            sm, flx_et = step_et_partition(
                sm=sm,
                z_layers=z_layers,
                pet_t=PET_t,
                canopy_evap_t=0.0 if is_bare else Ec_lu,   # bare has no canopy
                pveg=pveg,
                pbare=pbare,
                root_frac=root_frac,                      # LU-specific rooting
                theta_fc=theta_fc,
                theta_wp=theta_wp,
                theta_r=theta_r,
                allocate_transpiration_root_water_stress=allocate_transpiration_root_water_stress
            )

            # -----------------------------------------------------------------
            # Step 3: Drainage / percolation / recharge
            # -----------------------------------------------------------------
            sm, flx_dr = step_drainage_unit_gradient(
                sm=sm,
                z_layers=z_layers,
                theta_sat=theta_sat,
                theta_fc=theta_fc,
                theta_r=theta_r,
                Ks_cm_d=Ks_cm_d,
                vG_m=vG_m,
                dt_hours=dt_hours,
                drain_above_fc=drain_above_fc
            )


            # store updated LU state
            sm_by_lu[lu] = sm

            # -----------------------------------------------------------------
            # Step 4: Aggregate to catchment averages (area-weighted)
            # -----------------------------------------------------------------
            infil_c   += frac * float(flx_inf["infiltration"])
            runoff_c  += frac * float(flx_inf["surface_runoff"])
            Esoil_c   += frac * float(flx_et["Esoil"])
            ET_c      += frac * float(flx_et["ET_total"])
            unmetT_c  += frac * float(flx_et["unmet_T"])
            Ec_grid_c += frac * float(flx_et["Ec_grid"])

            sm_c       += frac * sm
            T_layers_c += frac * np.asarray(flx_et["T_layers"], float)
            perc_c     += frac * np.asarray(flx_dr["percolation"], float)
            rech_c     += frac * float(flx_dr["recharge"])

        # ---------------------------------------------------------------------
        # Write catchment-average results
        # ---------------------------------------------------------------------
        model_fluxes["infiltration"][t] = infil_c
        model_fluxes["surface_run_off"][t] = runoff_c
        model_fluxes["evap_actual_bs"][t] = Esoil_c
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
        model_fluxes[f"recharge_L{N}"][t] = rech_c

    return model_fluxes


