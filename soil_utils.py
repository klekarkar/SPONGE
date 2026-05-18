# soil_utils.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from pedotransfer_functions import SoilInputs, ks_all_methods, retention_characteristics

def resolve_path(base_dir: Path, file_name: str | Path) -> Path:
    """
    Resolve a file path relative to a base directory unless already absolute.
    """
    path = Path(file_name)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def require_columns(df: pd.DataFrame, columns: list[str], name: str = "table") -> None:
    """
    Check that a DataFrame contains required columns.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def load_soil_table(
    soil_cfg: dict,
    soil_dir: Path,
    soil_layer_names: list[str],
) -> pd.DataFrame:
    """
    Read, validate and reorder the soil texture/property table.

    Returns a soil table ordered according to soil_layer_names.
    """
    soil_textures_file = soil_cfg.get("soil_textures_file", None)
    if soil_textures_file is None:
        raise ValueError("Missing &soil soil_textures_file in run.nml")

    soil_textures_path = resolve_path(soil_dir, soil_textures_file)
    soil_tbl = pd.read_csv(soil_textures_path)

    # Support either 'layer_ID' or 'layer'
    if "layer_ID" not in soil_tbl.columns and "layer" in soil_tbl.columns:
        soil_tbl = soil_tbl.rename(columns={"layer": "layer_ID"})

    require_columns(
        soil_tbl,
        ["layer_ID", "sand", "silt", "clay", "rho_b", "OM", "topsoil"],
        f"soil textures table ({soil_textures_path.name})",
    )

    for c in ["sand", "silt", "clay", "rho_b", "OM"]:
        soil_tbl[c] = pd.to_numeric(soil_tbl[c], errors="raise")

    soil_tbl["topsoil"] = pd.to_numeric(
        soil_tbl["topsoil"], errors="raise"
    ).astype(int)

    soil_tbl["layer_ID"] = soil_tbl["layer_ID"].astype(str).str.strip()

    bad_top = soil_tbl.loc[~soil_tbl["topsoil"].isin([0, 1])]
    if len(bad_top) > 0:
        raise ValueError("soil_textures_file: 'topsoil' must be 0 or 1 for all rows.")

    tex_sum = soil_tbl[["sand", "silt", "clay"]].sum(axis=1)
    if not np.allclose(tex_sum.values, 100.0, atol=1e-3):
        bad = soil_tbl.loc[
            ~np.isclose(tex_sum, 100.0, atol=1e-3),
            ["layer_ID", "sand", "silt", "clay"],
        ]
        raise ValueError(f"soil textures must sum to 100 for all layers. Bad rows:\n{bad}")

    tbl_layers = soil_tbl["layer_ID"].tolist()
    missing = [x for x in soil_layer_names if x not in set(tbl_layers)]
    extra = [x for x in tbl_layers if x not in set(soil_layer_names)]

    if missing:
        raise ValueError(f"soil_textures_file missing layer_ID rows for: {missing}")

    if extra:
        raise ValueError(
            f"soil_textures_file has extra layer_ID rows not in run.nml soil_layers: {extra}"
        )

    soil_tbl = soil_tbl.set_index("layer_ID").loc[soil_layer_names].reset_index()

    return soil_tbl


def derive_soil_params_from_table(
    soil_tbl: pd.DataFrame,
    soil_layer_names: list[str],
    ptf_method: str,
    ksat_method: str,
    psi_fc_cm: float = 330.0,
    psi_pwp_cm: float = 15000.0,
) -> dict:
    """
    Derive layer-wise soil hydraulic parameters from a soil texture/property table.

    Call this after any scenario modifications to sand, silt, clay, rho_b or OM.
    """
    n_layers = len(soil_layer_names)
    soil_tbl = soil_tbl.copy()

    require_columns(
        soil_tbl,
        ["layer_ID", "sand", "silt", "clay", "rho_b", "OM", "topsoil"],
        "soil table used for PTF derivation",
    )

    soil_tbl["layer_ID"] = soil_tbl["layer_ID"].astype(str).str.strip()
    soil_tbl = soil_tbl.set_index("layer_ID").loc[soil_layer_names].reset_index()

    for c in ["sand", "silt", "clay", "rho_b", "OM"]:
        soil_tbl[c] = pd.to_numeric(soil_tbl[c], errors="raise")

    soil_tbl["topsoil"] = pd.to_numeric(
        soil_tbl["topsoil"], errors="raise"
    ).astype(int)

    tex_sum = soil_tbl[["sand", "silt", "clay"]].sum(axis=1)
    if not np.allclose(tex_sum.values, 100.0, atol=1e-3):
        bad = soil_tbl.loc[
            ~np.isclose(tex_sum, 100.0, atol=1e-3),
            ["layer_ID", "sand", "silt", "clay"],
        ]
        raise ValueError(
            f"soil textures must sum to 100 before PTF derivation. Bad rows:\n{bad}"
        )

    ptf_method = str(ptf_method).strip()
    ksat_method = str(ksat_method).strip()

    valid_ptf = {
        "Wosten_1999",
        "Weynants_2009",
        "Vereecken_1989",
        "Saxton_Rawls_2006",
        "Zacharias_Wessolek_2007",
    }

    if ptf_method not in valid_ptf:
        raise ValueError(f"Unknown ptf_method '{ptf_method}'. Valid: {sorted(valid_ptf)}")

    valid_ks = {
        "Wosten_1999_Ks",
        "Gupta_2021_Ks",
        "Cosby_1984_Ks",
        "Saxton_Rawls_2006_Ks",
    }

    if ksat_method not in valid_ks:
        raise ValueError(f"Unknown ksat_method '{ksat_method}'. Valid: {sorted(valid_ks)}")

    if ptf_method == "Saxton_Rawls_2006":
        ksat_method = "Saxton_Rawls_2006_Ks"

    theta_sat = np.zeros(n_layers, dtype=float)
    theta_fc = np.zeros(n_layers, dtype=float)
    theta_wp = np.zeros(n_layers, dtype=float)
    theta_r = np.zeros(n_layers, dtype=float)
    alpha_arr = np.full(n_layers, np.nan, dtype=float)
    n_arr = np.full(n_layers, np.nan, dtype=float)
    m_arr = np.full(n_layers, np.nan, dtype=float)
    ks_cm_day = np.zeros(n_layers, dtype=float)

    for i, row in enumerate(soil_tbl.itertuples(index=False)):
        soil_i = SoilInputs(
            sand=float(row.sand),
            silt=float(row.silt),
            clay=float(row.clay),
            rho_b=float(row.rho_b),
            OM=float(row.OM),
            topsoil=int(row.topsoil),
        )

        wr = retention_characteristics(
            soil_i,
            ptf_method,
            psi_fc_cm=psi_fc_cm,
            psi_pwp_cm=psi_pwp_cm,
        )

        theta_sat[i] = float(wr.get("sat", np.nan))
        theta_fc[i] = float(wr.get("fc", np.nan))
        theta_wp[i] = float(wr.get("pwp", np.nan))
        theta_r[i] = float(wr.get("residual", np.nan))
        alpha_arr[i] = float(wr.get("alpha", np.nan))
        n_arr[i] = float(wr.get("vG_n", np.nan))
        m_arr[i] = float(wr.get("vG_m", np.nan))

        ks_dict = ks_all_methods(soil_i)

        if ksat_method not in ks_dict:
            raise ValueError(
                f"ksat_method='{ksat_method}' not found in ks_all_methods outputs. "
                f"Available: {list(ks_dict.keys())}"
            )

        ks_cm_day[i] = float(ks_dict[ksat_method])

    soil_params_layer = {
        "theta_sat": theta_sat,
        "theta_fc": theta_fc,
        "theta_wp": theta_wp,
        "theta_r": theta_r,
        "alpha": alpha_arr,
        "vG_n": n_arr,
        "vG_m": m_arr,
        "Ks_cm_day": ks_cm_day,
    }

    for k, arr in soil_params_layer.items():
        if np.asarray(arr).shape != (n_layers,):
            raise ValueError(
                f"soil_params_layer['{k}'] wrong shape {np.asarray(arr).shape}, "
                f"expected {(n_layers,)}"
            )

    return soil_params_layer