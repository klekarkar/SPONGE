# model_runner.py
# End-to-end runner: config -> raw -> forcing -> canopy -> soil -> outputs

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from scenarios import load_scenario, apply_scenario

from config_loader import load_raw, get_logger, load_parameters
# from config_loader import derive_soil_params_from_table
from soil_utils import derive_soil_params_from_table
import prepare_inputs as prep
from forcings import build_forcing
from canopy_processes import run_canopy_wb


from model_processes_v1 import (
    initialize_model_fluxes,
    run_land_surface_water_balance,
)

from canopy_processes import run_canopy_wb

# =============================================================================
# Helpers
# =============================================================================

def trim_warmup(df: pd.DataFrame, warmup_days: int, sim_freq: str) -> pd.DataFrame:
    """Drop warmup period from a timeseries dataframe."""
    if warmup_days <= 0:
        return df

    sim_freq = str(sim_freq).strip().upper()
    steps = warmup_days if sim_freq == "D" else warmup_days * 24

    if steps >= len(df):
        raise ValueError(f"Warmup removes all data: warmup_steps={steps}, n={len(df)}")

    return df.iloc[steps:].copy()


def fluxes_to_df(fluxes: dict) -> pd.DataFrame:
    """Convert model_fluxes dict -> single wide DataFrame indexed by datetime."""
    idx = pd.DatetimeIndex(fluxes["index"])
    n = len(idx)

    cols = {}
    for k, v in fluxes.items():
        if k == "index":
            continue
        arr = np.asarray(v)
        if arr.ndim == 1 and len(arr) == n:
            cols[k] = arr

    df = pd.DataFrame(cols, index=idx)
    df.index.name = "time"
    return df


def _init_sm0(sm0_mode, theta_fc, theta_wp, theta_sat) -> np.ndarray:
    """Build initial soil moisture profile (volumetric) per layer."""
    if isinstance(sm0_mode, (list, tuple, np.ndarray)):
        sm0 = np.asarray(sm0_mode, dtype=float)
        return sm0

    key = str(sm0_mode).strip().lower()
    if key == "fc":
        return np.asarray(theta_fc, float).copy()
    if key == "wp":
        return np.asarray(theta_wp, float).copy()
    if key == "sat":
        return np.asarray(theta_sat, float).copy()
    if key == "zero":
        return np.zeros_like(np.asarray(theta_fc, float))
    raise ValueError("sm0_mode must be 'fc','wp','sat','zero' or an array-like of length n_layers.")


# =============================================================================
# Main runner
# =============================================================================
def run_model(
    paths_nml: str | Path,
    run_nml: str | Path,
    params_nml: str | Path,
    scenario_nml: str | Path | None = None,
    I0: float = 0.0,
    sm0_mode="fc",
    logger=None,
):

    """
    Run full model using three namelists (paths.nml + run.nml + scenario.nml).

    Returns
    -------
    raw : InputsRaw
    forcing : pd.DataFrame
    fluxes : dict
    df_all : pd.DataFrame
    df_trim : pd.DataFrame
    """
    logger = logger or get_logger()

    # -------------------------------------------------------------------------
    # 1) Load config + raw inputs
    # -------------------------------------------------------------------------
    raw = load_raw(paths_nml, run_nml, params_nml)
    logger.info("Loaded raw inputs: %s -> %s", raw.start_dt.date(), raw.end_dt.date())

    scenario_cfg = load_scenario(scenario_nml)
    raw = apply_scenario(raw, scenario_cfg)

    raw.soil_params_layer = derive_soil_params_from_table(
    soil_tbl=raw.soil_layers_table,
    soil_layer_names=raw.soil_layer_names,
    ptf_method=raw.ptf_method,
    ksat_method=raw.ksat_method,
)

    logger.info(
        "Scenario applied: %s | landcover = %s",
        getattr(raw, "scenario_name", "baseline"),
        raw.landcover[["lu_type", "lu_frac"]].to_dict(orient="records"),
)

    # -------------------------------------------------------------------------
    # 2) Build forcing at model timestep (D or H)
    # -------------------------------------------------------------------------
    forcing = build_forcing(
        meteo_daily=raw.meteo,
        dn_ratios_df=raw.dn_ratios,
        latitude=raw.latitude,
        sim_freq=raw.sim_freq,
        precip_method=raw.precip_method,
        pet_method=raw.pet_method,
        fnight_pet=raw.fnight_pet,
    )
    logger.info("Built forcing (%s): %d steps", raw.sim_freq, len(forcing))

    # -------------------------------------------------------------------------
    # 3) Build per-LU rooting and per-LU LAI (daily series)
    # -------------------------------------------------------------------------
    root_frac_by_lu, root_params_by_lu = prep.build_root_frac_by_lu(
        landcover_df=raw.landcover,
        root_params_df=raw.root_params,
        layer_edges_m=raw.layer_edges_m,
        wetland_proxy_type=raw.wetland_proxy_type,
        strict=True,
    )

    lai_daily_by_lu = prep.build_lai_daily_by_lu(
        landcover_df=raw.landcover,
        lai_cycles_doy=raw.lai_cycles_doy,
        start_date=raw.start_dt,
        end_date=raw.end_dt,
        strict=False,  # warn + LAI=0 if missing
    )
    logger.info("Prepared LU inputs: %d LU types", len(raw.landcover))

    # -------------------------------------------------------------------------
    # 4) Soil geometry + soil params (per-layer arrays already mapped in load_raw)
    # -------------------------------------------------------------------------
    edges_m = np.asarray(raw.layer_edges_m, dtype=float)
    z_layers_mm = np.diff(edges_m) * 1000.0
    n_layers = len(z_layers_mm)

    sp = raw.soil_params_layer
    theta_sat = np.asarray(sp["theta_sat"], float)
    theta_fc  = np.asarray(sp["theta_fc"], float)
    theta_wp  = np.asarray(sp["theta_wp"], float)
    theta_r   = np.asarray(sp.get("theta_r", theta_wp), float)
    Ks_cm_day   = np.asarray(sp["Ks_cm_day"], float)
    vG_m = np.asarray(sp["vG_m"], float)

    # initial soil moisture
    sm0 = _init_sm0(sm0_mode, theta_fc=theta_fc, theta_wp=theta_wp, theta_sat=theta_sat)
    if sm0.shape != (n_layers,):
        raise ValueError(f"sm0 must have shape (n_layers,)={(n_layers,)}, got {sm0.shape}")
    
    #-------------------------------------------------------------------------
    # Parameters from parameters.nml
    #-------------------------------------------------------------------------
    par = load_parameters(params_nml)

    b_infilt = par.b_infilt
    tau_vG   = par.tau_vG
    f_cr     = par.f_cr
    stress_method  = par.stress_method
    drain_above_fc = par.drain_above_fc
    lai_k          = par.lai_k
    use_lai_pveg   = par.use_lai_pveg
    q_sm   = par.q_sm
    q_soil = par.q_soil
    canopy_method=par.canopy_method
    canmx_mm=par.canmx_mm
    lai_max_mode=par.lai_max_mode
    lai_max_fixed=par.lai_max_fixed
    c_int=par.c_int

    # -------------------------------------------------------------------------
    # 5) Initialize storage for outputs ONCE
    # -------------------------------------------------------------------------
    flux_arrays = initialize_model_fluxes(
        daily_index=raw.meteo.index,
        freq=raw.sim_freq,
        n_layers=n_layers,
        sm0=sm0,
    )

    # -------------------------------------------------------------------------
    # 6) Canopy WB (multi-LU) -> returns TF_by_lu, Ec_by_lu
    # -------------------------------------------------------------------------
    lu_types = raw.landcover["lu_type"].tolist()
    lu_fracs = raw.landcover["lu_frac"].to_numpy(dtype=float)

    flux_arrays, TF_arr, Ec_arr, I_arr, beta_arr = run_canopy_wb(
        forcing=forcing,
        flux_arrays=flux_arrays,
        sim_freq=raw.sim_freq,
        lu_types=lu_types,
        lu_fracs=lu_fracs,
        lai_daily_by_lu=lai_daily_by_lu,
        canopy_method=canopy_method,
        canmx_mm=canmx_mm,
        lai_max_mode=lai_max_mode,
        lai_max_fixed=lai_max_fixed,
        I0=I0,
        c_int=c_int,
        show_progress=True,
        return_per_lu_series=False,  # <-- arrays
    )

    logger.info("Canopy WB complete.")

    # -------------------------------------------------------------------------
    # 7) Soil WB (multi-LU)
    # --------------------------------------------------------------------------

    flux_arrays = run_land_surface_water_balance(
        model_fluxes=flux_arrays,
        forcing=forcing,
        z_layers=z_layers_mm,
        theta_sat=theta_sat,
        theta_fc=theta_fc,
        theta_wp=theta_wp,
        theta_r=theta_r,
        Ks_cm_d=Ks_cm_day,
        vG_m=vG_m,
        b_infilt=b_infilt,
        tau_vG=tau_vG,
        f_cr=f_cr,
        q_sm=q_sm,
        q_soil=q_soil,
        lu_types=lu_types,
        lu_fracs=lu_fracs,
        root_frac_by_lu=root_frac_by_lu,
        TF_arr=TF_arr,          # (n_lu, n_t)
        Ec_arr=Ec_arr,          # (n_lu, n_t)
        beta_arr=beta_arr,
        sim_freq=raw.sim_freq,
        stress_method=stress_method,
        drain_above_fc=drain_above_fc,
        wetland_params=getattr(raw, "wetland_params", {"enabled": False}),
        show_progress=True,
        logger=logger,
    )

    logger.info("Soil WB complete.")

    # -------------------------------------------------------------------------
    # 8) Outputs to DataFrame + trim warmup
    # -------------------------------------------------------------------------
    df_all = fluxes_to_df(flux_arrays)
    df_trim = trim_warmup(df_all, raw.warmup_days, raw.sim_freq)

    return raw, forcing, flux_arrays, df_all, df_trim


# =============================================================================
# Model Reports
# =============================================================================


def write_simulation_report(inputs, filename: str = "inputs_report.txt") -> Path:
    """
    Write a concise inputs validation report to outputs folder as a .txt file.
    No console printing.
    """
    out_dir = Path(inputs.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / filename

    met = inputs.meteo
    start = met.index.min().date()
    end = met.index.max().date()

    # Missing days check
    expected = pd.date_range(met.index.min(), met.index.max(), freq="D")
    missing_days = expected.difference(met.index)
    missing_n = len(missing_days)

    # DN ratios
    dn = inputs.dn_ratios
    dn_shape = dn.shape
    dn_cols = list(dn.columns)

    # Landcover
    lc = inputs.landcover.copy()
    lc_sum = float(lc["lu_frac"].sum())
    lc_rows = [
        f"  - {r.lu_type}: {float(r.lu_frac):.4f}" + (" (bare)" if bool(r.is_bare) else "")
        for r in lc.itertuples(index=False)
    ]

    # Soil / layers
    n_layers = inputs.layer_edges_m.size - 1
    soil_lines = []
    soil_lines.append(f"  n_layers      : {n_layers}")
    soil_lines.append(f"  layer_edges_m : {inputs.layer_edges_m.tolist()}")
    if hasattr(inputs, "soil_layer_names"):
        soil_lines.append(f"  soil_layers   : {list(inputs.soil_layer_names)}")
    if hasattr(inputs, "ptf_method"):
        soil_lines.append(f"  ptf_method    : {inputs.ptf_method}")
    if hasattr(inputs, "ksat_method"):
        soil_lines.append(f"  ksat_method   : {inputs.ksat_method}")

    # Optional: derived soil params per layer (compact)
    if hasattr(inputs, "soil_params_layer") and hasattr(inputs, "soil_layer_names"):
        sp = inputs.soil_params_layer
        layer_ids = list(inputs.soil_layer_names)
        soil_lines.append("")
        soil_lines.append("  DERIVED SOIL PARAMS (per layer)")
        keys = ["theta_sat", "theta_fc", "theta_wp", "theta_r", "Ks_mm_h"]
        # include alpha/vG_n/vG_m if present
        for k in ["alpha", "vG_n", "vG_m"]:
            if k in sp:
                keys.append(k)

        header = "  layer_ID " + " ".join([f"{k:>10s}" for k in keys])
        soil_lines.append(header)
        for i, lid in enumerate(layer_ids):
            row = [lid]
            for k in keys:
                val = sp.get(k, np.full(n_layers, np.nan))[i]
                row.append(f"{float(val):10.4f}" if np.isfinite(val) else f"{'nan':>10s}")
            soil_lines.append("  " + f"{row[0]:7s} " + " ".join(row[1:]))

    # LAI checks (missing LU columns)
    lai_missing = []
    if hasattr(inputs, "lai_cycles_doy"):
        lai_cols = set(inputs.lai_cycles_doy.columns)
        for r in lc.itertuples(index=False):
            if (not bool(r.is_bare)) and (r.lu_type not in lai_cols):
                lai_missing.append(r.lu_type)
    else:
        lai_missing = ["(lai_cycles_doy not found on inputs object)"]

    # Root fractions check
    root_sums = []
    if hasattr(inputs, "root_frac_by_lu"):
        for r in lc.itertuples(index=False):
            lu = r.lu_type
            rf = inputs.root_frac_by_lu.get(lu, None)
            if rf is None:
                root_sums.append(f"  - {lu}: MISSING")
            else:
                root_sums.append(f"  - {lu}: rf sum={rf}")
    else:
        root_sums = ["  (root_frac_by_lu not found on inputs object)"]

    # Build text
    lines = []
    lines.append("MODEL INPUTS REPORT")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("PATHS")
    for k in ["project_root", "db_dir", "input_dir", "meteo_dir", "lai_dir",
              "landcover_dir", "validation_dir", "out_dir"]:
        if hasattr(inputs, k):
            lines.append(f"  {k:15s}: {getattr(inputs, k)}")
    lines.append("")

    lines.append("RUN SETTINGS")
    lines.append(f"  sim_freq     : {inputs.sim_freq}")
    lines.append(f"  warmup_days  : {inputs.warmup_days}")
    lines.append(f"  meteo_period : {start} -> {end}")
    lines.append(f"  meteo_cols   : {list(met.columns)}")
    lines.append(f"  missing_days : {missing_n}")
    if missing_n > 0:
        lines.append(f"  first_missing: {[d.strftime('%Y-%m-%d') for d in missing_days[:10]]}")
    lines.append("")

    lines.append("DISAGGREGATION")
    lines.append(f"  precip_method: {getattr(inputs, 'precip_method', '')}")
    lines.append(f"  pet_method   : {getattr(inputs, 'pet_method', '')}")
    lines.append(f"  fnight_pet   : {getattr(inputs, 'fnight_pet', '')}")
    lines.append(f"  dn_ratios    : shape={dn_shape}, cols={dn_cols}")
    lines.append("")

    lines.append("SOIL / LAYERS")
    lines.extend(soil_lines)
    lines.append("")

    lines.append("LANDCOVER")
    lines.append(f"  sum(lu_frac) : {lc_sum:.6f}")
    lines.extend(lc_rows)
    lines.append("")

    lines.append("LAI")
    if lai_missing:
        lines.append("  WARNING: Missing LAI cycle column(s) for:")
        for x in lai_missing:
            lines.append(f"    - {x}")
    else:
        lines.append("  All non-bare landcover types have LAI cycles.")
    lines.append("")

    lines.append("ROOT FRACTIONS")
    lines.extend(root_sums)
    lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return txt_path
#-----------------------------------------------------------------------------

# ==============================WATER BALANCE CHECK===============================================
def water_balance_report(raw, forcing: pd.DataFrame, df_all: pd.DataFrame, filename: str = "water_balance_report.txt") -> Path:
    """
    Export lightweight water-balance sanity checks to outputs folder as a .txt file.
    No console printing.

    Parameters
    ----------
    raw : InputsRaw-like
        Must have out_dir, meteo, sim_freq
    forcing : pd.DataFrame
        Forcing time index used by the model (hourly/daily)
    df_all : pd.DataFrame
        Model outputs table with columns like:
          pre, total_evap, surface_run_off, *recharge*
    filename : str
        Output text file name.

    Returns
    -------
    Path to saved report.
    """
    out_dir = Path(raw.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / filename

    lines = []
    lines.append("WATER BALANCE REPORT")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    # ----------------
    # Period info
    # ----------------
    lines.append("=== PERIOD ===")
    try:
        lines.append(f"meteo   : {raw.meteo.index.min().date()} -> {raw.meteo.index.max().date()}")
    except Exception:
        lines.append("meteo   : (not available)")

    try:
        lines.append(f"forcing : {forcing.index.min()} -> {forcing.index.max()}")
    except Exception:
        lines.append("forcing : (not available)")

    lines.append(f"sim_freq: {getattr(raw, 'sim_freq', '(unknown)')}")
    try:
        lines.append(f"steps   : {len(forcing)}")
    except Exception:
        lines.append("steps   : (not available)")
    lines.append("")

    # ----------------
    # Water balance
    # ----------------
    lines.append("=== WATER BALANCE (rough, timestep totals) ===")

    required = ["pre", "total_evap", "surface_run_off"]
    missing = [c for c in required if c not in df_all.columns]

    if missing:
        lines.append("Cannot compute full balance (missing required columns):")
        for c in missing:
            lines.append(f"  - {c}")
        lines.append("")
    else:
        P = float(df_all["pre"].sum())
        ET = float(df_all["total_evap"].sum())
        R = float(df_all["surface_run_off"].sum())

        recharge_cols = [c for c in df_all.columns if "recharge" in str(c).lower()]
        if recharge_cols:
            Q = float(df_all[recharge_cols].sum(axis=1).sum())
        else:
            Q = float("nan")

        lines.append(f"Sum P        = {P:.6f}")
        lines.append(f"Sum ET       = {ET:.6f}")
        lines.append(f"Sum runoff   = {R:.6f}")
        lines.append(f"Sum recharge = {Q:.6f}" if np.isfinite(Q) else "Sum recharge = NaN (no recharge columns found)")

        resid = P - ET - R - (Q if np.isfinite(Q) else 0.0)
        lines.append(f"Check P - ET - R - Q = {resid:.6f}")
        lines.append("")

        # Helpful: list which columns were used for recharge
        if recharge_cols:
            lines.append("Recharge columns used:")
            for c in recharge_cols:
                lines.append(f"  - {c}")
            lines.append("")

    # ----------------
    # Minimal column inventory (optional but handy)
    # ----------------
    lines.append("=== OUTPUT COLUMNS (df_all) ===")
    lines.append(", ".join(map(str, df_all.columns.tolist())))
    lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return txt_path

