from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

# Soil water balance functions

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

