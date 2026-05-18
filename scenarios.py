# scenarios.py
from __future__ import annotations

from pathlib import Path
import copy
import pandas as pd
import f90nml

#============================SCENARIO LOADING============================

def load_scenario(scenario_nml: str | Path | None) -> dict:
    """
    Load scenario namelist.

    If scenario_nml is None, return a disabled baseline scenario.
    """
    if scenario_nml is None:
        return {
            "scenario": {
                "name": "baseline",
                "enabled": False,
            },
            "landcover_scenario": {
                "enabled": False,
                "from_lu": "",
                "to_lu": "",
                "fraction": 0.0,
            },
        }

    scenario_nml = Path(scenario_nml)

    if not scenario_nml.exists():
        raise FileNotFoundError(f"Scenario namelist not found: {scenario_nml}")

    return f90nml.read(scenario_nml)

#============================SCENARIO APPLICATION============================

def apply_scenario(raw, scenario_cfg: dict):
    raw = copy.deepcopy(raw)

    sc = scenario_cfg.get("scenario", {})
    scenario_name = str(sc.get("name", "baseline")).strip()
    scenario_enabled = bool(sc.get("enabled", False))

    raw.scenario_name = scenario_name
    raw.scenario_enabled = scenario_enabled

    if not scenario_enabled:
        raw.wetland_enabled = False
        raw.wetland_params = {"enabled": False}
        return raw

    raw = apply_landcover_scenario(raw, scenario_cfg)
    raw = apply_soil_scenario(raw, scenario_cfg)
    raw = apply_wetland_scenario(raw, scenario_cfg)

    return raw

#============================LAND-COVER CONVERSION SCENARIO FUNCTIONS============================

def apply_landcover_scenario(raw, scenario_cfg: dict):
    """
    Apply a simple land-cover conversion scenario.
    """
    cfg = scenario_cfg.get("landcover_scenario", {})

    if not bool(cfg.get("enabled", False)):
        return raw

    from_lu = str(cfg.get("from_lu", "")).strip()
    to_lu = str(cfg.get("to_lu", "")).strip()
    fraction = float(cfg.get("fraction", 0.0))
    fraction_mode = str(cfg.get("fraction_mode", "source")).strip().lower()

    if not from_lu:
        raise ValueError("landcover_scenario.from_lu is empty.")
    if not to_lu:
        raise ValueError("landcover_scenario.to_lu is empty.")
    if fraction <= 0:
        raise ValueError("landcover_scenario.fraction must be > 0.")

    raw.landcover = convert_landcover_fraction(
        landcover=raw.landcover,
        from_lu=from_lu,
        to_lu=to_lu,
        fraction=fraction,
        fraction_mode=fraction_mode,
    )

    return raw

#============================LAND COVER CHANGE ============================

def convert_landcover_fraction(
    landcover: pd.DataFrame,
    from_lu: str,
    to_lu: str,
    fraction: float,
    fraction_mode: str = "absolute",
) -> pd.DataFrame:
    """
    Convert land-cover from one class to another.

    Parameters
    ----------
    landcover : pd.DataFrame
        Must contain columns: lu_type, lu_frac.
    from_lu : str
        Existing land-cover class to reduce.
    to_lu : str
        Land-cover class to increase or create.
    fraction : float
        If fraction_mode='absolute', this is the absolute grid-cell fraction.
        If fraction_mode='source', this is the fraction of from_lu to convert.
    fraction_mode : {'absolute', 'source'}
        Determines how fraction is interpreted.

    Returns
    -------
    pd.DataFrame
        Modified land-cover table with fractions summing to 1.
    """
    lc = landcover.copy()

    required = {"lu_type", "lu_frac"}
    missing = required.difference(lc.columns)
    if missing:
        raise ValueError(f"landcover table missing required columns: {sorted(missing)}")

    from_lu = str(from_lu).strip()
    to_lu = str(to_lu).strip()
    fraction = float(fraction)
    fraction_mode = str(fraction_mode).strip().lower()

    if fraction <= 0:
        raise ValueError("fraction must be > 0.")

    if fraction_mode not in ("absolute", "source"):
        raise ValueError(
            f"Unknown fraction_mode '{fraction_mode}'. "
            "Use 'absolute' or 'source'."
        )

    # Case-insensitive matching while preserving original names
    lc["_key"] = lc["lu_type"].astype(str).str.strip().str.lower()
    from_key = from_lu.lower()
    to_key = to_lu.lower()

    if from_key not in set(lc["_key"]):
        available = lc["lu_type"].tolist()
        raise ValueError(
            f"from_lu '{from_lu}' not found in landcover. "
            f"Available classes: {available}"
        )

    available_frac = float(lc.loc[lc["_key"] == from_key, "lu_frac"].sum())

    if fraction_mode == "absolute":
        convert_frac = fraction
    else:
        convert_frac = available_frac * fraction

    if convert_frac > available_frac + 1e-12:
        raise ValueError(
            f"Cannot convert {convert_frac:.4f} from '{from_lu}'. "
            f"Only {available_frac:.4f} is available."
        )

    # Reduce source class
    lc.loc[lc["_key"] == from_key, "lu_frac"] -= convert_frac

    # Increase or create target class
    if to_key in set(lc["_key"]):
        lc.loc[lc["_key"] == to_key, "lu_frac"] += convert_frac
    else:
        bare_aliases = {"bare", "barren", "bare soil", "urban", "rock"}

        new_row = {
            "lu_type": to_lu,
            "lu_frac": convert_frac,
            "is_bare": to_key in bare_aliases,
            "_key": to_key,
        }

        for col in lc.columns:
            if col not in new_row:
                new_row[col] = None

        lc = pd.concat([lc, pd.DataFrame([new_row])], ignore_index=True)

    # Remove near-zero classes
    lc = lc.loc[lc["lu_frac"] > 1e-12].copy()

    # Normalise tiny floating-point drift
    total = float(lc["lu_frac"].sum())
    if total <= 0:
        raise ValueError("Land-cover fractions sum to zero after scenario conversion.")

    lc["lu_frac"] = lc["lu_frac"] / total

    if abs(float(lc["lu_frac"].sum()) - 1.0) > 1e-9:
        raise ValueError("Land-cover fractions do not sum to 1 after normalisation.")

    return lc

#============================SOIL SCENARIO FUNCTIONS============================
def apply_soil_scenario(raw, scenario_cfg: dict):
    """
    Apply soil-property changes for scenario runs.

    Useful for testing the impact of soil changes associated with land-cover
    change, e.g. afforestation, wetland creation or infiltration ponds.

    This modifies raw.soil_layers_table, not raw.soil_params_layer.
    Soil hydraulic properties should be recalculated after this step.

    Notes
    -----
    Guo & Gifford (2002) values are reported land-use-change effects on
    soil carbon stocks. Here, they are applied as relative multipliers to OM.

    Guo et al. (2021) values are depth-specific afforestation effects on
    SOC and bulk density. These are applied by layer.

    Mayer et al. (2020) values provide additional tree-type/management
    sensitivity scenarios, including broadleaf, conifer and N-fixing effects.
    """
    cfg = scenario_cfg.get("soil_scenario", {})

    if not bool(cfg.get("enabled", False)):
        return raw

    method = str(cfg.get("method", "none")).strip().lower()

    if method in ("none", "baseline", ""):
        return raw

    if not hasattr(raw, "soil_layers_table"):
        raise AttributeError(
            "raw does not contain soil_layers_table. "
            "Store the baseline soil table in raw before applying soil scenarios."
        )

    soil = raw.soil_layers_table.copy()

    required = {"layer_ID", "OM", "rho_b"}
    missing = required.difference(soil.columns)
    if missing:
        raise ValueError(f"soil table missing required columns: {sorted(missing)}")

    # WaTRE layer convention:
    # L_01 = 0–5 cm
    # L_02 = 5–30 cm
    # L_03 = 30–60 cm
    # L_04 = 60–120 cm
    top_layers = soil["layer_ID"].isin(["L_01", "L_02"])

    # ------------------------------------------------------------------
    # Guo & Gifford (2002): reported land-use-change effects on soil C.
    # Pasture is treated as grassland in the scenario aliases.
    # These values are applied to OM in the top 30 cm.
    # ------------------------------------------------------------------
    guo_2002_om_multipliers = {
        # Declines in soil C
        "pasture_to_plantation": 0.90,          # -10%
        "grassland_to_plantation": 0.90,        # alias for pasture
        "native_forest_to_plantation": 0.87,    # -13%
        "native_forest_to_crop": 0.58,          # -42%
        "pasture_to_crop": 0.41,                # -59%
        "grassland_to_crop": 0.41,              # alias for pasture

        # Increases in soil C
        "native_forest_to_pasture": 1.08,       # +8%
        "native_forest_to_grassland": 1.08,     # alias for pasture
        "crop_to_pasture": 1.19,               # +19%
        "crop_to_grassland": 1.19,             # alias for pasture
        "crop_to_plantation": 1.18,            # +18%
        "crop_to_secondary_forest": 1.53,       # +53%
    }

    # Guo & Gifford (2002) gives soil C changes, not bulk-density changes.
    # These BD multipliers are modest modelling assumptions.
    # If a method is not listed, rho_b remains unchanged.
    guo_2002_bd_multipliers = {
        "native_forest_to_crop": 1.05,
        "pasture_to_crop": 1.05,
        "grassland_to_crop": 1.05,

        "crop_to_pasture": 0.97,
        "crop_to_grassland": 0.97,
        "crop_to_plantation": 0.97,
        "crop_to_secondary_forest": 0.95,
    }

    # ------------------------------------------------------------------
    # Guo et al. (2021): depth-specific afforestation effects.
    #
    # afforestation_average:
    #   SOC: 0–20 cm +46%, 20–60 cm +52%, 60–100 cm +20%
    #   BD:  0–20 cm -3.3%, 20–60 cm -2.5%, 60–100 cm -0.7%
    #
    # broadleaf_deciduous_afforestation:
    #   SOC: 0–20 cm +64%, 20–60 cm +76%, 60–100 cm +35%
    #   BD:  uses afforestation-average BD multipliers
    #
    # L_02 spans 5–30 cm, so it uses weighted averages.
    # ------------------------------------------------------------------
    guo_2021_layer_scenarios = {
        "afforestation_average": {
            "OM": {
                "L_01": 1.46,
                "L_02": 1.484,
                "L_03": 1.52,
                "L_04": 1.20,
            },
            "rho_b": {
                "L_01": 0.967,
                "L_02": 0.970,
                "L_03": 0.975,
                "L_04": 0.993,
            },
        },
        
        "broadleaf_deciduous_afforestation": {
            "OM": {
                "L_01": 1.64,
                "L_02": 1.688,
                "L_03": 1.76,
                "L_04": 1.35,
            },
            "rho_b": {
                "L_01": 0.967,
                "L_02": 0.970,
                "L_03": 0.975,
                "L_04": 0.993,
            },
        },
    }

    # ------------------------------------------------------------------
    # Mayer et al. (2020): additional forest-management/tree-type
    # sensitivity scenarios.
    #
    # These are applied to the top 30 cm as simple sensitivity multipliers.
    # ------------------------------------------------------------------
    mayer_2020_om_multipliers = {
        "broadleaf_moderate": 1.25,  # broadleaf afforestation after 2–3 decades
        "conifer_neutral": 1.02,     # conifer afforestation after 2–3 decades
        "n_fixing_tree": 1.12,       # N-fixing vegetation, mineral soil C
    }

    # Mayer et al. (2020) does not provide matching BD multipliers for these
    # simplified scenarios. These are modest assumptions for PTF response.
    mayer_2020_bd_multipliers = {
        "broadleaf_moderate": 0.97,
        "conifer_neutral": 1.00,
        "n_fixing_tree": 0.97,
    }

    # ------------------------------------------------------------------
    # Custom WaTRE intervention proxies.
    # These are not direct literature meta-analysis values, but practical
    # scenario representations for retention-oriented interventions.
    # ------------------------------------------------------------------
    custom_topsoil_scenarios = {
        "grassland_to_plantation_neutral": {
            "OM": 1.00,
            "rho_b": 1.00,
        },
        "wetland_organic": {
            "OM": 1.50,
            "rho_b": 0.95,
        },
    }

    # ------------------------------------------------------------------
    # Apply selected scenario.
    # ------------------------------------------------------------------
    if method in guo_2002_om_multipliers:
        soil.loc[top_layers, "OM"] *= guo_2002_om_multipliers[method]
        soil.loc[top_layers, "rho_b"] *= guo_2002_bd_multipliers.get(method, 1.00)

    elif method in guo_2021_layer_scenarios:
        layer_scenario = guo_2021_layer_scenarios[method]

        for layer_id, multiplier in layer_scenario["OM"].items():
            soil.loc[soil["layer_ID"] == layer_id, "OM"] *= multiplier

        for layer_id, multiplier in layer_scenario["rho_b"].items():
            soil.loc[soil["layer_ID"] == layer_id, "rho_b"] *= multiplier

    elif method in mayer_2020_om_multipliers:
        soil.loc[top_layers, "OM"] *= mayer_2020_om_multipliers[method]
        soil.loc[top_layers, "rho_b"] *= mayer_2020_bd_multipliers.get(method, 1.00)

    elif method in custom_topsoil_scenarios:
        soil.loc[top_layers, "OM"] *= custom_topsoil_scenarios[method]["OM"]
        soil.loc[top_layers, "rho_b"] *= custom_topsoil_scenarios[method]["rho_b"]

    elif method == "infiltration_pond":
        # Top 30 cm replaced by sandy/infiltration-enhancing material.
        if not {"sand", "silt", "clay"}.issubset(soil.columns):
            raise ValueError(
                "infiltration_pond scenario requires sand, silt and clay columns."
            )

        soil.loc[top_layers, "sand"] = 85.0
        soil.loc[top_layers, "silt"] = 10.0
        soil.loc[top_layers, "clay"] = 5.0
        soil.loc[top_layers, "rho_b"] *= 0.95

    else:
        valid_methods = sorted(
            list(guo_2002_om_multipliers.keys())
            + list(guo_2021_layer_scenarios.keys())
            + list(mayer_2020_om_multipliers.keys())
            + list(custom_topsoil_scenarios.keys())
            + ["infiltration_pond"]
        )

        raise ValueError(
            f"Unknown soil_scenario method: {method}. "
            f"Valid methods are: {valid_methods}"
        )

    raw.soil_layers_table = soil
    return raw


#============================ WETLAND SCENARIO APPLICATION ============================
def apply_wetland_scenario(raw, scenario_cfg: dict):
    """
    Attach wetland scenario parameters to raw.

    This does not run the wetland water balance. It only stores the wetland
    configuration so the model can use it during the timestep simulation.
    """
    cfg = scenario_cfg.get("wetland_scenario", {})

    enabled = bool(cfg.get("enabled", False))

    raw.wetland_enabled = enabled

    if not enabled:
        raw.wetland_params = {
            "enabled": False,
            "area_fraction": 0.0,
            "route_runoff_fraction": 0.0,
            "h_max_m": 0.0,
            "h_spill_m": 0.0,
            "k_infil_m_per_h": 0.0,
            "outflow_coeff_m_per_h": 0.0,
            "outflow_exp": 1.0,
            "open_water_evap_factor": 1.0,
            "initial_depth_m": 0.0,
        }
        return raw

    wetland_params = {
        "enabled": True,
        "area_fraction": float(cfg.get("area_fraction", 0.05)),
        "route_runoff_fraction": float(cfg.get("route_runoff_fraction", 1.0)),
        "h_max_m": float(cfg.get("h_max_m", 1.0)),
        "h_spill_m": float(cfg.get("h_spill_m", 0.4)),
        "k_infil_m_per_h": float(cfg.get("k_infil_m_per_h", 0.0003)),
        "outflow_coeff_m_per_h": float(cfg.get("outflow_coeff_m_per_h", 0.002)),
        "outflow_exp": float(cfg.get("outflow_exp", 1.5)),
        "open_water_evap_factor": float(cfg.get("open_water_evap_factor", 1.05)),
        "initial_depth_m": float(cfg.get("initial_depth_m", 0.0)),
    }

    if wetland_params["area_fraction"] <= 0:
        raise ValueError("wetland_scenario.area_fraction must be > 0.")

    if wetland_params["area_fraction"] >= 1:
        raise ValueError("wetland_scenario.area_fraction must be < 1.")

    if not 0 <= wetland_params["route_runoff_fraction"] <= 1:
        raise ValueError("wetland_scenario.route_runoff_fraction must be between 0 and 1.")

    if wetland_params["h_max_m"] <= 0:
        raise ValueError("wetland_scenario.h_max_m must be > 0.")

    if wetland_params["h_spill_m"] < 0:
        raise ValueError("wetland_scenario.h_spill_m must be >= 0.")

    if wetland_params["h_spill_m"] > wetland_params["h_max_m"]:
        raise ValueError("wetland_scenario.h_spill_m cannot exceed h_max_m.")

    if wetland_params["k_infil_m_per_h"] < 0:
        raise ValueError("wetland_scenario.k_infil_m_per_h must be >= 0.")

    if wetland_params["outflow_coeff_m_per_h"] < 0:
        raise ValueError("wetland_scenario.outflow_coeff_m_per_h must be >= 0.")

    raw.wetland_params = wetland_params

    return raw

#========================================================
