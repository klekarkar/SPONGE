
"""
---------------------------------------------------------------------------------
Data preparation functions
---------------------------------------------------------------------------------
"""
from __future__ import annotations
import warnings
from typing import Dict, Tuple
import numpy as np
import pandas as pd


"""
----------------------------------------------------------------------------------
Ensure key column for matching
----------------------------------------------------------------------------------
"""

def _ensure_key_col(df: pd.DataFrame, name_col: str = "lu_type", key_col: str = "_key") -> pd.DataFrame:
    """Ensure df has a lowercased key column for matching."""
    df = df.copy()
    if key_col not in df.columns:
        df[key_col] = df[name_col].astype(str).str.strip().str.lower()
    return df

"""
----------------------------------------------------------------------------------
Build per-landuse root fraction profiles across soil layers
----------------------------------------------------------------------------------
"""
def build_root_frac_by_lu(
    landcover_df: pd.DataFrame,
    root_params_df: pd.DataFrame,
    layer_edges_m: np.ndarray,
    wetland_proxy_type: str = "Grassland",
    strict: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """
    Build per-landuse root fraction profiles across soil layers.

    Parameters
    ----------
    landcover_df : DataFrame
        Must contain: lu_type, is_bare. (lu_frac not required here)
    root_params_df : DataFrame
        Indexed by lowercased IGBP name, with columns a,b,dr.
        (e.g., index 'cropland', 'deciduous broadleaf tree', ...)
    layer_edges_m : np.ndarray
        Layer edges (m), length n_layers+1.
    wetland_proxy_type : str
        Proxy IGBP type to use when lu_type == 'Permanent wetland' has missing a/b/dr.
    strict : bool
        If True, missing root params for non-bare LU raises ValueError.
        If False, warns and assigns zeros.

    Returns
    -------
    root_frac_by_lu : dict
        lu_type -> np.ndarray (n_layers,)
    root_params_by_lu : dict
        lu_type -> dict with keys a,b,dr,used_type
    """
    lc = _ensure_key_col(landcover_df, "lu_type", "_key")
    n_layers = len(layer_edges_m) - 1

    # safety: ensure root_params_df index is lowercased
    rp = root_params_df.copy()
    if rp.index.dtype != "object":
        rp.index = rp.index.astype(str)
    rp.index = rp.index.str.strip().str.lower()

    proxy_key = str(wetland_proxy_type).strip().lower()
    if proxy_key not in rp.index:
        raise ValueError(f"wetland_proxy_type '{wetland_proxy_type}' not found in root params table.")

    root_frac_by_lu: Dict[str, np.ndarray] = {}
    root_params_by_lu: Dict[str, dict] = {}

    missing = []

    for lu, key, is_bare in zip(lc["lu_type"], lc["_key"], lc["is_bare"]):
        if bool(is_bare):
            root_frac_by_lu[lu] = np.zeros(n_layers, dtype=float)
            root_params_by_lu[lu] = {"a": np.nan, "b": np.nan, "dr": np.nan, "used_type": "bare"}
            continue

        if key not in rp.index:
            missing.append(lu)
            root_frac_by_lu[lu] = np.zeros(n_layers, dtype=float)
            root_params_by_lu[lu] = {"a": np.nan, "b": np.nan, "dr": np.nan, "used_type": "missing"}
            continue

        row = rp.loc[key]
        a, b, dr = row["a"], row["b"], row["dr"]

        # handle missing wetland params
        used_type = key
        if pd.isna(a) or pd.isna(b) or pd.isna(dr):
            if key == "permanent wetland":
                prow = rp.loc[proxy_key]
                a, b, dr = float(prow["a"]), float(prow["b"]), float(prow["dr"])
                used_type = proxy_key
            else:
                # present but incomplete
                missing.append(lu)
                root_frac_by_lu[lu] = np.zeros(n_layers, dtype=float)
                root_params_by_lu[lu] = {"a": np.nan, "b": np.nan, "dr": np.nan, "used_type": "missing"}
                continue
        else:
            a, b, dr = float(a), float(b), float(dr)

        # compute layer fractions
        rf = np.zeros(n_layers, dtype=float)
        for i in range(n_layers):
            z1 = float(layer_edges_m[i])
            z2 = float(layer_edges_m[i + 1])
            rf[i] = compute_root_fraction_layer(a, b, dr, z1, z2)

        s = float(rf.sum())
        if s <= 0:
            raise ValueError(f"Root fractions sum to {s} for lu_type='{lu}' (used '{used_type}').")
        rf /= s

        root_frac_by_lu[lu] = rf
        root_params_by_lu[lu] = {"a": a, "b": b, "dr": dr, "used_type": used_type}

    if missing:
        msg = (
            "Missing/incomplete root parameters for these lu_type values: "
            + ", ".join(missing)
            + "."
        )
        if strict:
            raise ValueError(msg + " Fix igbp_root_params.csv or landcover lu_type names.")
        warnings.warn(msg + " Assigned zero root fractions for them.", UserWarning)

    return root_frac_by_lu, root_params_by_lu

"""
----------------------------------------------------------------------------------
Build daily LAI time series per LU from DOY climatology
----------------------------------------------------------------------------------
"""

def build_lai_daily_by_lu(
    landcover_df: pd.DataFrame,
    lai_cycles_doy: pd.DataFrame,
    start_date,
    end_date,
    strict: bool = False,
    daily_index: pd.DatetimeIndex | None = None,
) -> Dict[str, pd.Series]:
    """
    Build daily LAI time series per LU from DOY climatology.

    Parameters
    ----------
    landcover_df : DataFrame
        Must contain: lu_type, is_bare.
    lai_cycles_doy : DataFrame
        Index: doy 1..366, columns include LU types (IGBP names).
    start_date, end_date : date-like
        Daily run period.
    strict : bool
        If True, missing LAI column raises ValueError.
        If False, warns and assigns LAI=0.

    Returns
    -------
    lai_daily_by_lu : dict
        lu_type -> pd.Series indexed by daily dates [start_date, end_date]
    """
    lc = _ensure_key_col(landcover_df, "lu_type", "_key")

    if daily_index is None:
        daily_index = pd.date_range(start=start_date, end=end_date, freq="D")
    else:
        # safety
        daily_index = pd.DatetimeIndex(daily_index).normalize()

    # check DOY index
    if lai_cycles_doy.index.min() != 1 or lai_cycles_doy.index.max() != 366:
        raise ValueError("lai_cycles_doy must be indexed by doy 1..366.")
    if len(lai_cycles_doy.index) != 366:
        raise ValueError("lai_cycles_doy must have exactly 366 rows (doy 1..366).")

    lai_daily_by_lu: Dict[str, pd.Series] = {}
    missing = []

    for lu, is_bare in zip(lc["lu_type"], lc["is_bare"]):
        if bool(is_bare):
            lai_daily_by_lu[lu] = pd.Series(
                0.0, index=daily_index, name=f"LAI_{lu}"
            )
            continue

        if lu not in lai_cycles_doy.columns:
            missing.append(lu)
            lai_daily_by_lu[lu] = pd.Series(
                0.0, index=daily_index, name=f"LAI_{lu}"
            )
            continue

        col = pd.to_numeric(lai_cycles_doy[lu], errors="coerce")
        if col.isna().any():
            # fill small gaps in the climatology
            col = col.interpolate("linear").ffill().bfill()

        cycle = pd.Series(col.to_numpy(), index=lai_cycles_doy.index)
        s = get_daily_lai(start_date, end_date, cycle)  # your existing function
        s = s.reindex(daily_index)
        s.name = f"LAI_{lu}"
        lai_daily_by_lu[lu] = s

    if missing:
        msg = "Missing LAI DOY cycles for these lu_type values: " + ", ".join(missing) + "."
        if strict:
            raise ValueError(msg + " Add columns to lai_doy_cycles.csv.")
        warnings.warn(msg + " Assigned LAI=0 for them.", UserWarning)

    return lai_daily_by_lu


""" 
----------------------------------------------------------------------------------
Disaggregate annual doy LAI to model date range 
----------------------------------------------------------------------------------
""" 
#Interpolate the data to get the daily values for each year
#This will give the 366 values for the daily LAI data (with DOYs 1-366)
# Function to grab any date range
 
def get_daily_lai(start_date, end_date, LAIcycle: pd.Series) -> pd.Series:
    """
    Returns a pandas Series of daily LAI between start_date and end_date,
    by mapping each day-of-year into the 1–366 climatology cycle.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    doys = dates.dayofyear

    # ensure LAIcycle indexed by 1..366
    if not isinstance(LAIcycle.index, pd.Index) or LAIcycle.index.min() != 1 or LAIcycle.index.max() != 366:
        raise ValueError("LAIcycle must be a Series indexed by doy 1..366.")

    daily_vals = LAIcycle.loc[doys].to_numpy()
    return pd.Series(daily_vals, index=dates, name="LAI")

"""
----------------------------------------------------------------------------------
Disaggregate daily to hourly meteorology 
----------------------------------------------------------------------------------
"""
#PET disaggregation functions
def _monthly_daylength_hours(lat_deg):
    """
    Monthly daylength (hours, float) for the 15th of each month at latitude lat_deg.
    Geometric daylength (no refraction correction).
    """
    lat = np.deg2rad(lat_deg)
    doy_15 = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)

    delta = np.deg2rad(23.44) * np.sin(2*np.pi*(284 + doy_15)/365.0)
    x = -np.tan(lat) * np.tan(delta)
    x = np.clip(x, -1.0, 1.0)

    omega = np.arccos(x)
    daylen = 24.0/np.pi * omega
    return np.clip(daylen, 0.0, 24.0)

def _disagg_pet_smooth(PET_d, date, latitude, fnight_pet=0.0):
    """
    Disaggregate one daily PET total (mm/day) to 24 hourly PET (mm/hour)
    using a smooth half-sine diurnal shape during daylight.

    fnight_pet: fraction (0..1) of PET assigned uniformly to night hours.
               Typically small (0.0–0.15). If 0, night PET is exactly 0.
    """
    if PET_d <= 0:
        return np.zeros(24, dtype=float)

    month = date.month
    daylen = _monthly_daylength_hours(latitude)[month - 1]  # float hours

    # Handle polar edge cases (not relevant for Belgium but safe)
    if daylen <= 0.0:
        # all "night"
        hourly = np.full(24, PET_d / 24.0, dtype=float) if fnight_pet > 0 else np.zeros(24, dtype=float)
        return hourly
    if daylen >= 24.0:
        # all "day"
        return np.full(24, PET_d / 24.0, dtype=float)

    # Define sunrise/sunset centered around noon
    sunrise = 12.0 - daylen / 2.0
    sunset  = 12.0 + daylen / 2.0

    hours = np.arange(24, dtype=float)
    day_mask = (hours >= sunrise) & (hours < sunset)

    # Night allocation (uniform over night hours)
    hourly = np.zeros(24, dtype=float)
    PET_night = PET_d * fnight_pet
    PET_day = PET_d - PET_night

    n_night = 24 - int(day_mask.sum())
    if n_night > 0 and PET_night > 0:
        hourly[~day_mask] = PET_night / n_night

    # Day allocation with half-sine weights (smooth, peak at noon)
    if PET_day > 0 and day_mask.any():
        # map each daylight hour to [0, pi]
        t = (hours[day_mask] + 0.5 - sunrise) / daylen  # +0.5 uses hour-center
        w = np.sin(np.pi * t)                            # half-sine shape
        w = np.maximum(w, 0.0)
        w_sum = w.sum()
        if w_sum > 0:
            hourly[day_mask] = PET_day * (w / w_sum)
        else:
            # fallback: uniform over day hours
            hourly[day_mask] = PET_day / day_mask.sum()

    # Conservation check
    if not np.isclose(hourly.sum(), PET_d, atol=1e-10):
        # minor floating error fix
        hourly *= PET_d / hourly.sum()

    return hourly

def generate_hourly_pet(pet_daily: pd.Series, latitude, fnight_pet=0.05):
    daily = pet_daily.copy()
    daily.index = daily.index.normalize()

    hourly_vals = np.empty(len(daily) * 24, dtype=float)
    hourly_times = np.empty(len(daily) * 24, dtype="datetime64[ns]")

    for i, date in enumerate(daily.index):
        PET_h = _disagg_pet_smooth(daily.iloc[i], date, latitude=latitude, fnight_pet=fnight_pet)
        sl = slice(i*24, (i+1)*24)
        hourly_vals[sl] = PET_h
        hourly_times[sl] = pd.date_range(date, periods=24, freq="h").values

    return pd.DataFrame({"pet_h": hourly_vals}, index=pd.DatetimeIndex(hourly_times))


#Disaggregate daily precipitation to hourly precipitation
def disagg_P_hourly(daily_precip, dn_ratios_df, month, day_start=6, day_end=18):
    """
    Disaggregate one daily precipitation total (mm/day) to 24 hourly precipitation (mm/hour),
    using month-specific day/night ratios. Day hours are defined by day_start and day_end.

    Inputs:
    - daily_precip: float, daily precipitation total (mm/day)
    - dn_ratios_df: DataFrame, with columns 'day_ratio' and 'night_ratio' indexed by month (1-12)
    - month: int, month of the year (1-12)
    - day_start: int, hour of day when daytime starts (default: 6)
    - day_end: int, hour of day when daytime ends (default: 18)
    Outputs:
    - hourly: ndarray of shape (24,), hourly precipitation (mm/hour)

    """

    day_ratio  = float(dn_ratios_df.loc[month, 'day_ratio'])
    night_ratio = float(dn_ratios_df.loc[month, 'night_ratio'])

    # recommended: enforce conservation
    if not np.isclose(day_ratio + night_ratio, 1.0, atol=1e-6):
        raise ValueError(f"Month {month}: day_ratio + night_ratio != 1 (got {day_ratio + night_ratio})")

    hourly = np.zeros(24, dtype=float)
    day_hours = [(h >= day_start) and (h < day_end) for h in range(24)]

    hourly[day_hours] = daily_precip * (day_ratio / 12.0)
    hourly[~np.array(day_hours)] = daily_precip * (night_ratio / 12.0)
    return hourly

def generate_hourly_precipitation(precip_daily: pd.Series, dn_ratios_df: pd.DataFrame,
                                  day_start=6, day_end=18):
    """
    Generate hourly precipitation DataFrame from daily precipitation Series.
    """
    daily = precip_daily.copy()
    daily.index = daily.index.normalize()

    hourly_index = pd.date_range(daily.index.min(),
                                 daily.index.max() + pd.Timedelta(hours=23),
                                 freq="h")

    hourly_vals = np.empty(len(daily) * 24, dtype=float)

    for i, date in enumerate(daily.index):
        hourly_vals[i*24:(i+1)*24] = disagg_P_hourly(
            daily_precip=daily.iloc[i],
            dn_ratios_df=dn_ratios_df,
            month=date.month,
            day_start=day_start,
            day_end=day_end
        )

    return pd.DataFrame({"pre_h": hourly_vals}, index=hourly_index)


#----------------------------------------------------------------------------------

""" 
Per-soil layer root distrbution function according to IGBP type
Acccording to Zeng (2001): https://doi.org/10.1175/1525-7541(2001)002<0525:GVRDFL>2.0.CO;2
----------------------------------------------------------------------------------
"""
def root_cdf_Y(d_m: float, a: float, b: float) -> float:
    """Cumulative root fraction above depth d (meters). Equation 2 of the paper.
    Parameters
    ----------
    d_m : float
        Depth in meters.
    a : float
        Root distribution parameter a.
    b : float
        Root distribution parameter b.
    Returns
    -------
    float
        Cumulative root fraction above depth d (0...99).
    """
    d_m = float(max(d_m, 0.0))
    return float(1.0 - 0.5 * (np.exp(-a * d_m) + np.exp(-b * d_m)))


def compute_root_fraction_layer(a: float, b: float, dr: float, z1: float, z2: float) -> float:
    """
    Root fraction in layer [z1,z2] (meters), using Zeng-style Y(d).
    Depths deeper than dr are capped to dr.
    """
    z1 = float(max(z1, 0.0))
    z2 = float(max(z2, z1))
    dr = float(max(dr, 0.0))

    if dr > 0:
        z1 = min(z1, dr)
        z2 = min(z2, dr)

    f = root_cdf_Y(z2, a, b) - root_cdf_Y(z1, a, b)
    return float(max(f, 0.0))

def root_fracs_multilayer(layer_edges_m, a, b, dr):
    n_layers = len(layer_edges_m) - 1
    f = np.zeros(n_layers, dtype=float)

    for i in range(n_layers):
        z1 = float(layer_edges_m[i])
        z2 = float(layer_edges_m[i+1])
        f[i] = compute_root_fraction_layer(a, b, dr, z1, z2)

    s = f.sum()
    if s <= 0:
        raise ValueError("Computed root fractions sum to 0. Check a,b,dr and layer depths.")
    return f / s
#----------------------------------------------------------------------------------

"""
----------------------------------------------------------------------------------
Convenience wrapper
----------------------------------------------------------------------------------
"""
def build_derived_inputs(raw, *, strict_roots: bool = True, strict_lai: bool = False):
    """
    Convenience wrapper: from the object returned by config_loader.load_raw(...)
    compute per-LU roots and per-LU daily LAI.
    """
    root_frac_by_lu, root_params_by_lu = build_root_frac_by_lu(
        landcover_df=raw.landcover,
        root_params_df=raw.root_params,
        layer_edges_m=raw.layer_edges_m,
        wetland_proxy_type=raw.wetland_proxy_type,
        strict=strict_roots,
    )

    lai_daily_by_lu = build_lai_daily_by_lu(
        landcover_df=raw.landcover,
        lai_cycles_doy=raw.lai_cycles_doy,
        start_date=raw.start_dt,
        end_date=raw.end_dt,
        strict=strict_lai,
        daily_index=raw.meteo.index,   # aligns perfectly to meteo
    )

    return root_frac_by_lu, root_params_by_lu, lai_daily_by_lu
