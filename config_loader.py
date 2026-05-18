# config_loader.py
# Load model configuration from TWO namelists (paths.nml + run.nml)
# and prepare inputs to fit model structure.
#
# Assumptions / enforced contracts:
# - Folder layout (under project_root):
#     model_rho_b/     (lookup tables)
#     inputs/
#       meteo/
#       lai/
#       landcover/
#       validation/
#     outputs/
#
# - Fixed column names (users must NOT rename):
#     meteo_file:          index=date, columns include "P" and "PET"
#     landcover_fractions: columns "lu_type", "lu_frac"
#     LAI DOY cycles:      column "doy" and columns matching lu_type (IGBP names)
#     soil LUT:            columns m, Ksat_cm_h, theta_S, theta_FC, theta_PWP
#     root params LUT:     columns igbp_type, a, b, dr
#   
#
# - IGBP names everywhere:
#     landcover lu_type must match igbp_root_params.csv igbp_type (case-insensitive),
#     except bare-like types which are allowed (Bare/Barren/Urban/Rock...).
#
# - LAI:
#     LAI is provided as DOY climatology per LU type (lai_doy_cycles.csv),
#     expanded to daily using prep.get_daily_lai(start, end, LAIcycle).

from __future__ import annotations
from dataclasses import dataclass
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import f90nml
from datetime import datetime
from soil_utils import load_soil_table, derive_soil_params_from_table
import prepare_inputs as prep

#----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

def _resolve(base: Path, maybe_rel: str | Path) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (base / p)

def require_columns(df: pd.DataFrame, required: list[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}. Found: {list(df.columns)}")

#----------------------------------------------------------------------------------------------------------------------
def get_logger(name: str = "WaTRE model", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoids duplicate handlers in notebooks
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = False
    return logger

@dataclass
class InputsRaw:
    # dirs
    project_root: Path
    db_dir: Path
    input_dir: Path
    meteo_dir: Path
    lai_dir: Path
    validation_dir: Path
    landcover_dir: Path
    out_dir: Path

    # run settings
    sim_freq: str
    warmup_days: int

    # time window used (final)
    start_dt: pd.Timestamp
    end_dt: pd.Timestamp

    # data tables
    meteo: pd.DataFrame
    dn_ratios: pd.DataFrame
    landcover: pd.DataFrame          # lu_type, lu_frac, is_bare
    lai_cycles_doy: pd.DataFrame     # index=doy, cols=lu_type
    root_params: pd.DataFrame        # index=igbp_type lower, cols a,b,dr

     # layers / soils
    layer_edges_m: np.ndarray
    soil_layer_names: list[str]          # NEW: from run.nml soil_layers
    soil_layers_table: pd.DataFrame      # NEW: raw table loaded from soil_textures_file
    soil_params_layer: dict[str, np.ndarray]  # computed arrays

    # root fractions per LU (computed)
    root_frac_by_lu: dict[str, np.ndarray]   # key=lu_type
    root_params_by_lu: dict[str, dict[str, float]]  # key=lu_type, value=dict of root params a,b,dr
    root_frac_df: pd.DataFrame                # labeled view (LU x layer)

    # methods
    ptf_method: str
    ksat_method: str

    # process settings
    latitude: float
    precip_method: str
    pet_method: str
    fnight_pet: float
    infiltration_model: str
    infil_use_layer: int
    wetland_proxy_type: str
    
    #parameters
    params: Params


@dataclass
class Params:
    params_nml: Path
    b_infilt: float
    tau_vG: float
    drain_above_fc: bool
    stress_method: str
    f_cr: float
    lai_k: float
    use_lai_pveg: bool
    q_sm: float
    q_soil: float
    canopy_method: str
    canmx_mm: float
    lai_max_mode: str
    lai_max_fixed: float
    c_int: float
    


# Parameters
def load_parameters(params_nml: str | Path) -> Params:
    params_nml = Path(params_nml)
    cfg = f90nml.read(params_nml)

    if "parameters" not in cfg:
        raise ValueError(f"Missing &PARAMETERS block in {params_nml}")

    par = cfg["parameters"]

    required = ["b_infilt", "tau_vG", "drain_above_fc", "stress_method", "f_cr", "lai_k", "use_lai_pveg", "canopy_method", "canmx_mm", "lai_max_mode", "lai_max_fixed", "c_int"]
    missing = [k for k in required if k not in par]
    if missing:
        raise ValueError(f"Missing parameter(s) in {params_nml} under &PARAMETERS: {missing}")

    return Params(
        params_nml=params_nml,
        b_infilt=float(par["b_infilt"]),
        tau_vG=float(par["tau_vG"]),
        drain_above_fc=bool(par["drain_above_fc"]),
        stress_method=str(par["stress_method"]).strip(),
        f_cr=float(par["f_cr"]),
        lai_k=float(par["lai_k"]),
        use_lai_pveg=bool(par["use_lai_pveg"]),
        q_sm=float(par["q_sm"]),
        q_soil=float(par['q_soil']),
        canopy_method=str(par["canopy_method"]).strip(),
        canmx_mm=float(par["canmx_mm"]),
        lai_max_mode=str(par["lai_max_mode"]).strip(),
        lai_max_fixed=float(par["lai_max_fixed"]),
        c_int=float(par["c_int"]),
    )

#-------------------------------------------------------------------------------------------------

def load_raw(paths_nml: str | Path, run_nml: str | Path, params_nml: str | Path) -> InputsRaw:
    paths_cfg = f90nml.read(Path(paths_nml))
    run_cfg   = f90nml.read(Path(run_nml))
    params = load_parameters(params_nml)   # <- the ONLY thing you need from params.nml

    # --- paths ---
    p = paths_cfg.get("paths", {})
    project_root = Path(p.get("project_root", ".")).resolve()
    db_dir = _resolve(project_root, p.get("db_dir", "./model_db/"))
    input_dir = _resolve(project_root, p.get("input_dir", "./inputs/"))
    out_dir = _resolve(project_root, p.get("out_dir", "./outputs/"))

    meteo_dir = input_dir / "meteo"
    lai_dir = input_dir / "lai"
    landcover_dir = input_dir / "landcover"
    soil_dir = input_dir / "soil"
    validation_dir = input_dir / "validation"

    db = paths_cfg.get("db_files", {})
    igbp_root_params_file = _resolve(db_dir, db["igbp_root_params_file"])


    # --- run settings ---
    sim_freq = str(run_cfg.get("model", {}).get("sim_freq", "H")).strip().upper()
    warmup_days = int(run_cfg.get("model", {}).get("warmup_days", 0))

    # --- time window ---
    time_cfg = run_cfg.get("time", {})
    start_dt = pd.to_datetime(time_cfg.get("start_date"), dayfirst=True)
    end_dt = pd.to_datetime(time_cfg.get("end_date"), dayfirst=True)

    # --- meteo ---
    met = run_cfg["meteo"]
    meteo_file = _resolve(meteo_dir, met["meteo_file"])
    dn_file = _resolve(meteo_dir, met["dn_ratios_file"])
    latitude = float(met["latitude"])

    meteo = pd.read_csv(meteo_file, index_col=0)
    require_columns(meteo, ["P", "PET"], f"meteo ({meteo_file.name})")
    idx = meteo.index.astype(str).str.strip()

    # Attempt multiple date parsing strategies to handle common formats (YYYY-mm-dd, YYYY/mm/dd, dd/mm/YYYY)
    try:
        dt = pd.to_datetime(idx, errors="raise", dayfirst=False)  # handles YYYY/mm/dd, YYYY-mm-dd
    except Exception:
        dt = pd.to_datetime(idx, errors="raise", dayfirst=True)   # handles dd/mm/yyyy
    
    #If neither is provided, this will raise an error which is good to catch bad formats. \\

    meteo.index = dt.normalize()

    meteo = meteo.sort_index().loc[start_dt:end_dt]

    # continuous daily
    expected = pd.date_range(meteo.index.min(), meteo.index.max(), freq="D")
    miss = expected.difference(meteo.index)
    if len(miss) > 0:
        raise ValueError(f"meteo has missing days (first 10): {list(miss[:10].strftime('%Y-%m-%d'))}")

    dn_ratios = pd.read_csv(dn_file, index_col=0)

    # --- disagg ---
    dis = run_cfg.get("disagg", {})
    precip_method = str(dis.get("precip_method", "dn_ratio"))
    pet_method = str(dis.get("pet_method", "daylength"))
    fnight_pet = float(dis.get("fnight_pet", 0.05))


    # ------------------------------------------------------------------
    # SOIL: layer edges + per-layer textures table + PTF selection
    # ------------------------------------------------------------------
    soil_cfg = run_cfg.get("soil", {})

    layer_edges_m = np.array(soil_cfg.get("layer_edges_m", []), dtype=float)
    if layer_edges_m.size < 2:
        raise ValueError("soil.layer_edges_m must contain at least 2 values.")

    n_layers = int(layer_edges_m.size - 1)

    soil_layer_names = [str(x).strip() for x in soil_cfg.get("soil_layers", [])]
    if len(soil_layer_names) != n_layers:
        raise ValueError(
            f"soil.soil_layers length ({len(soil_layer_names)}) must equal n_layers ({n_layers}) "
            f"where n_layers = len(layer_edges_m)-1."
        )

    soil_tbl = load_soil_table(
        soil_cfg=soil_cfg,
        soil_dir=soil_dir,
        soil_layer_names=soil_layer_names,
    )

    ptf_method = soil_cfg.get("ptf_method", None)
    ksat_method = soil_cfg.get("ksat_method", None)

    if ptf_method is None:
        raise ValueError("Missing &soil ptf_method in run.nml")
    if ksat_method is None:
        raise ValueError("Missing &soil ksat_method in run.nml")

    ptf_method = str(ptf_method).strip()
    ksat_method = str(ksat_method).strip()

    soil_params_layer = derive_soil_params_from_table(
        soil_tbl=soil_tbl,
        soil_layer_names=soil_layer_names,
        ptf_method=ptf_method,
        ksat_method=ksat_method,
    )

    
    # #---------------------------------------------------------------------------------------------------------------
    # """ Export derived soil WRC parameters to CSV if requested in run.nml. """
    # #---------------------------------------------------------------------------------------------------------------
    # export_flag = bool(soil_cfg.get("export_derived_soil_params", False))
    # derived_name = str(soil_cfg.get("derived_soil_water_retention_file", "derived_soil_params.csv"))

    # if export_flag:
    #     derived_path = soil_dir / derived_name

    #     derived_df = soil_tbl[["layer_ID", "sand", "silt", "clay", "rho_b", "OM", "topsoil"]].copy()
    #     derived_df["theta_sat"] = theta_sat
    #     derived_df["theta_fc"]  = theta_fc
    #     derived_df["theta_wp"]  = theta_wp
    #     derived_df["theta_r"]   = theta_r
    #     derived_df["alpha"]     = alpha_arr
    #     derived_df["vG_n"]      = n_arr
    #     derived_df["vG_m"]      = m_arr
    #     derived_df["Ks_cm_day"]   = ks_cm_day
    #     derived_df["ptf_method"] = ptf_method
    #     derived_df["ksat_method"] = ksat_method

    #     derived_df.to_csv(derived_path, index=False)
    
    # get_logger().info("Exported derived soil hydraulic properties to %s", derived_path)

    #---------------------------------------------------------------------------------------------------------------
    # --- landcover ---
    lc_file = _resolve(landcover_dir, run_cfg["landcover"]["landcover_file"])
    landcover = pd.read_csv(lc_file)
    require_columns(landcover, ["lu_type", "lu_frac"], f"landcover ({lc_file.name})")
    landcover["lu_type"] = landcover["lu_type"].astype(str).str.strip()
    landcover["lu_frac"] = pd.to_numeric(landcover["lu_frac"], errors="raise")

    landcover["_key"] = landcover["lu_type"].str.lower()
    landcover = landcover.groupby("_key", as_index=False).agg(lu_type=("lu_type","first"), lu_frac=("lu_frac","sum"))

    bare_aliases = {"bare", "barren", "bare soil", "urban", "rock"}
    landcover["is_bare"] = landcover["_key"].isin(bare_aliases)

    if abs(float(landcover["lu_frac"].sum()) - 1.0) > 1e-3:
        raise ValueError("landcover fractions must sum to 1.")

    # --- LAI cycles (raw) ---
    lai_cycles_file = _resolve(lai_dir, run_cfg.get("lai", {}).get("lai_cycles_file", "lai_doy_cycles.csv"))
    lai_cycles = pd.read_csv(lai_cycles_file)
    require_columns(lai_cycles, ["doy"], f"LAI cycles ({Path(lai_cycles_file).name})")
    lai_cycles["doy"] = pd.to_numeric(lai_cycles["doy"], errors="raise").astype(int)
    lai_cycles = lai_cycles.set_index("doy").sort_index()

    # --- process ---
    proc = run_cfg.get("process", {})
    infiltration_model = str(proc.get("infiltration_model", "valiantzas_model")).strip()
    infil_use_layer = int(proc.get("infil_use_layer", 1))
    wetland_proxy_type = str(run_cfg.get("vegetation", {}).get("wetland_proxy_type", "Grassland")).strip()

        # --- root params (raw) ---
    root_params = pd.read_csv(igbp_root_params_file)
    require_columns(root_params, ["igbp_type", "a", "b", "dr"], f"root params ({igbp_root_params_file.name})")
    root_params["_key"] = root_params["igbp_type"].astype(str).str.strip().str.lower()
    root_params = root_params.set_index("_key")

    # ---- ROOT FRACTIONS (compute once during config loading) ----
    root_frac_by_lu, root_params_by_lu = prep.build_root_frac_by_lu(
        landcover_df=landcover,
        root_params_df=root_params,
        layer_edges_m=layer_edges_m,
        wetland_proxy_type=wetland_proxy_type,
        strict=True,
    )
    # Build labeled LU x layer table
    root_frac_df = pd.DataFrame.from_dict(root_frac_by_lu, orient="index")
    root_frac_df.columns = soil_layer_names  # e.g. ["L_01","L_02","L_03","L_04"]
    root_frac_df.index.name = "lu_type"


    return InputsRaw(
        project_root=project_root,
        db_dir=db_dir,
        input_dir=input_dir,
        meteo_dir=meteo_dir,
        lai_dir=lai_dir,
        landcover_dir=landcover_dir,
        validation_dir=validation_dir,
        out_dir=out_dir,
        sim_freq=sim_freq,
        warmup_days=warmup_days,
        start_dt=meteo.index.min(),
        end_dt=meteo.index.max(),
        meteo=meteo,
        dn_ratios=dn_ratios,
        landcover=landcover[["lu_type","lu_frac","is_bare","_key"]].copy(),
        lai_cycles_doy=lai_cycles,
        root_params=root_params,
        root_params_by_lu=root_params_by_lu,
        root_frac_by_lu=root_frac_by_lu,
        root_frac_df=root_frac_df,
        ptf_method=ptf_method,
        ksat_method=ksat_method,
        soil_params_layer=soil_params_layer,  
        layer_edges_m=layer_edges_m,
        soil_layer_names=soil_layer_names,
        soil_layers_table=soil_tbl,          
        latitude=latitude,
        precip_method=precip_method,
        pet_method=pet_method,
        fnight_pet=fnight_pet,
        infiltration_model=infiltration_model,
        infil_use_layer=infil_use_layer,
        wetland_proxy_type=wetland_proxy_type,
        params=params
    )

#---------------------------------------------------------------------------------



""" 
--------------------------------------------------------------------------------
Write concise inputs report to outputs folder
--------------------------------------------------------------------------------
"""

#Log file
def write_inputs_report(inputs, filename: str = "inputs_report.log") -> Path:
    """
    Write a concise inputs validation report to outputs folder.
    Does NOT include P/PET statistics.
    """
    out_dir = Path(inputs.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / filename

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

    # LAI per LU
    lai_keys = list(getattr(inputs, "lai_daily_by_lu", {}).keys())
    lai_missing = []
    if hasattr(inputs, "lai_cycles_doy"):
        lai_cols = set(inputs.lai_cycles_doy.columns)
        for r in lc.itertuples(index=False):
            if (not bool(r.is_bare)) and (r.lu_type not in lai_cols):
                lai_missing.append(r.lu_type)
    else:
        lai_missing = ["(lai_cycles_doy not found on inputs object)"]


    lines = []
    lines.append("MODEL INPUTS REPORT")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    lines.append("PATHS")
    for k in ["project_root", "rho_b_dir", "input_dir", "meteo_dir", "lai_dir", "landcover_dir", "validation_dir", "out_dir"]:
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
    if hasattr(inputs, "options"):
        o = inputs.options
        lines.append(f"  precip_method: {o.get('precip_method')}")
        lines.append(f"  pet_method   : {o.get('pet_method')}")
        lines.append(f"  fnight_pet   : {o.get('fnight_pet')}")
    lines.append(f"  dn_ratios    : shape={dn_shape}, cols={dn_cols}")
    lines.append("")

    lines.append("SOIL / LAYERS")
    lines.append(f"  n_layers       : {inputs.layer_edges_m.size -1}")
    lines.append(f"  layer_edges_m  : {inputs.layer_edges_m.tolist()}")
    lines.append(f"  soil_layers    : {inputs.soil_layer_names}")
    lines.append(f"  ptf_method     : {inputs.ptf_method}")
    lines.append(f"  ksat_method    : {inputs.ksat_method}")


    lines.append("LANDCOVER")
    lines.append(f"  sum(lu_frac) : {lc_sum:.6f}")
    lines.extend(lc_rows)
    lines.append("")

    lines.append("LAI")
    lines.append(f"  lai_daily_by_lu keys: {lai_keys}")
    if lai_missing:
        lines.append("  WARNING: Missing LAI cycle column(s) for:")
        for x in lai_missing:
            lines.append(f"    - {x}")
    else:
        lines.append("  All non-bare landcover types have LAI cycles.")
    lines.append("")

        # Root fractions per LU
    lines.append("ROOT FRACTIONS")
    if hasattr(inputs, "root_frac_df"):
        for lu, row in inputs.root_frac_df.iterrows():
            lines.append(f"  - {lu}: sum={float(row.sum()):.6f} | " +
                        ", ".join([f"{c}={row[c]:.3f}" for c in inputs.root_frac_df.columns]))
    else:
        lines.append("  (root_frac_df not found on inputs object)")
    lines.append("")

    log_path.write_text("\n".join(lines), encoding="utf-8")
    return log_path
