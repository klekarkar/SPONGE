# Purpose-separated PTFs:
#   - retention: sat, residual, fc, pwp (+ optional model params)
#   - Ks: multiple alternative methods in one function
#
# Conventions:
#   - sand/silt/clay in % (0-100)
#   - rho_b in g/cm3
#   - suction head psi in cm (use magnitude; e.g., 330, 15000)
#   - Ks returned in cm/d

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Callable
import math
import numpy as np
import pandas as pd

RetentionPTFName = Literal[
    "Wosten_1999",
    "Weynants_2009",
    "Vereecken_1989",
    "Saxton_Rawls_2006",
    "Zacharias_Wessolek_2007"
]

KsMethodName = Literal[
    "Wosten_1999_Ks",            # Wösten (continuous) Ks regression
    "Gupta_2021_Ks",
    "Cosby_1984_Ks",
    "Saxton_Rawls_2006_Ks"
]

# -------------------------
# Inputs and core models
# -------------------------

@dataclass(frozen=True)
class SoilInputs:
    sand: float     # %
    silt: float     # %
    clay: float     # %
    rho_b: float    # g/cm3
    OM: float = 0.0 # % organic matter (if needed)
    OC: Optional[float] = None  # % organic carbon (optional)
    depth_cm: float = 0.0       # for OC80 only
    topsoil: int = 1            # 1 topsoil, 0 subsoil

    def porosity(self, rho_s: float = 2.65) -> float:
        return 1.0 - self.rho_b / rho_s

    def oc_percent(self) -> float:
        if self.OC is not None:
            return float(self.OC)
        # OC = OM/1.724 (common conversion used in the Appendix contexts)
        return float(self.OM) / 1.724 if self.OM is not None else 0.0


@dataclass
class VGParams:
    theta_s: float # saturation water content
    theta_r: float # residual water content
    alpha: float  # 1/cm # shape parameter related to air entry suction
    vG_n: float       # unitless # shape parameter related to pore size distribution
    vG_m: Optional[float] = None # unitless # shape parameter related to pore connectivity; if None, set to 1-1/n

    #van Genuchten m is often defined as m=1-1/n, but some PTFs (e.g., Wosten_1999) have their own regression for m.
    def __post_init__(self):
        if self.vG_m is None:
            self.vG_m = 1.0 - 1.0 / self.vG_n


@dataclass
class BCParams:
    theta_s: float
    theta_r: float
    psi_b: float  # cm (positive magnitude)
    lam: float


def theta_vg(psi_cm: np.ndarray | float, p: VGParams) -> np.ndarray:
    psi = np.asarray(psi_cm, dtype=float)
    psi = np.maximum(psi, 0.0)
    return p.theta_r + (p.theta_s - p.theta_r) / (1.0 + (p.alpha * psi) ** p.vG_n) ** p.vG_m

# -------------------------
# Retention PTFs (Appendix A Nasta et al. 2021: https://doi.org/10.1016/j.ejrh.2021.100903)
# Each returns:
#   - either model params 
# -------------------------

def Wosten_1999(soil: SoilInputs, theta_r_rule: float = 0.01) -> VGParams:
    C, Si, OM, Db, top = soil.clay, soil.silt, soil.OM, soil.rho_b, soil.topsoil

    #for this method if Clay< 18% and sand>65%, theta_r is set to 0.025 otherwise 0.01
    theta_r_rule = 0.025 if C < 18 and soil.sand > 65 else 0.01

    theta_s = (
        0.7919 + 0.001691*C - 0.29619*Db - 0.000001491*(Si**2)
        + 0.0000821*(OM**2) + 0.02427/(C if C != 0 else 1e-12)
        + 0.01113/(Si if Si != 0 else 1e-12) + 0.01472*math.log(max(Si, 1e-12))
        - 0.0000733*OM*C - 0.000619*Db*C - 0.001183*Db*OM
        - 0.0001664*top*Si
    )

    alpha = math.exp(
        -14.96 + 0.03135*C + 0.0351*Si + 0.646*OM + 15.29*Db - 0.192*top
        - 4.671*(Db**2) - 0.000781*(C**2) - 0.00687*(OM**2)
        + 0.0449/(OM if OM != 0 else 1e-12) + 0.0663*math.log(max(Si, 1e-12))
        + 0.1482*math.log(max(OM, 1e-12))
        - 0.04546*Db*Si - 0.4852*Db*OM + 0.00673*top*C
    )

    n = 1.0 + math.exp(
        -25.23 - 0.02195*C + 0.0074*Si - 0.1940*OM + 45.5*Db
        - 7.24*(Db**2) + 0.0003658*(C**2) + 0.002885*(OM**2)
        - 12.81/(Db if Db != 0 else 1e-12) - 0.1524/(Si if Si != 0 else 1e-12)
        - 0.01958/(OM if OM != 0 else 1e-12) - 0.2876*math.log(max(Si, 1e-12))
        - 0.0709*math.log(max(OM, 1e-12)) - 44.6*math.log(max(Db, 1e-12))
        - 0.02264*Db*C + 0.0896*Db*OM + 0.00718*top*C
    )

    theta_s = float(np.clip(theta_s, 0.05, 0.9))
    theta_r = float(np.clip(theta_r_rule, 0.0, theta_s - 1e-6))
    alpha = float(max(alpha, 1e-12))
    n = float(max(n, 1.01))
    return VGParams(theta_s=theta_s, theta_r=theta_r, alpha=alpha, vG_n=n)


def Weynants_2009(soil: SoilInputs) -> VGParams:
    C, S, Db = soil.clay, soil.sand, soil.rho_b
    OC = soil.oc_percent()

    theta_s = 0.6355 + 0.0013*C - 0.1631*Db
    theta_r = 0.0
    alpha = math.exp(-4.3003 - 0.0097*C + 0.0138*S - 0.0992*OC)
    n = math.exp(-1.0846 - 0.0236*C - 0.0085*S + 1.3699e-4*S**2) + 1.0

    theta_s = float(np.clip(theta_s, 0.05, 0.9))
    alpha = float(max(alpha, 1e-12))
    n = float(max(n, 1.01))
    return VGParams(theta_s=theta_s, theta_r=theta_r, alpha=alpha, vG_n=n)

def Vereecken_1989(soil: SoilInputs) -> VGParams:
    C, S, Db = soil.clay, soil.sand, soil.rho_b
    OC = soil.oc_percent()

    theta_s = 0.81 - 0.283*Db + 0.001*C
    theta_r = 0.015 + 0.005*C + 0.014*OC
    alpha = math.exp(-2.486 + 0.025*S - 0.351*OC - 2.617*Db - 0.023*C)
    n = math.exp(0.053 - 0.009*S - 0.013*C + 0.00015*S**2)

    theta_s = float(np.clip(theta_s, 0.05, 0.9))
    theta_r = float(np.clip(theta_r, 0.0, theta_s - 1e-6))
    alpha = float(max(alpha, 1e-12))

    # Appendix notes m=1 for this PTF; keep it explicit so your outputs match.
    return VGParams(theta_s=theta_s, theta_r=theta_r, alpha=alpha, vG_n=n, vG_m=1.0)


def Saxton_Rawls_2006_WR(soil: SoilInputs) -> Dict[str, float]:
    """
    Saxton & Rawls (2006) provides point estimates:
      theta33  (≈ FC at 33 kPa)
      theta1500 (≈ PWP at 1500 kPa)
      theta_s
    It does NOT define a residual water content in the vG/BC sense.
    """
    sand, clay, OM = soil.sand/100, soil.clay/100, soil.OM

    theta33t = (-0.251*sand + 0.195*clay + 0.011*OM + 0.006*(sand*OM) - 0.027*(clay*OM)
                + 0.452*(sand*clay) + 0.299)
    
    theta33 = theta33t + (1.283*(theta33t**2) - 0.374*theta33t - 0.015)

    theta1500t = (-0.024*sand + 0.487*clay + 0.006*OM + 0.005*(sand*OM) - 0.013*(clay*OM)
                  + 0.068*(sand*clay) + 0.031)
    
    theta1500 = theta1500t + (0.14*theta1500t - 0.02)

    thetaS_33t = (0.278*sand + 0.034*clay + 0.022*OM - 0.018*(sand*OM) - 0.027*(clay*OM)
                  - 0.584*(sand*clay) + 0.078)
    
    thetaS_33 = thetaS_33t + (0.636*thetaS_33t - 0.107)

    theta_s = theta33 + thetaS_33 - 0.097*sand + 0.043

    # Optional shape parameter they define for Ks
    B = (math.log(1500) - math.log(33)) / (math.log(theta33) - math.log(theta1500))

    theta_s = theta33 + thetaS_33 - 0.097*sand + 0.043

    lam = 1/B

    return dict(
        sat=float(np.clip(theta_s, 0.05, 0.9)),
        fc=float(np.clip(theta33, 0.0, 0.9)),
        pwp=float(np.clip(theta1500, 0.0, 0.9)),
        residual= 0.0,  
        lam=float(lam),
    )

def Zacharias_Wessolek_2007(soil: SoilInputs) -> VGParams:
    C, S, Db = soil.clay, soil.sand, soil.rho_b
    if S < 66.5:
        theta_r = 0.0
        theta_s = 0.788 +0.001*C -0.263*Db
        alpha = math.exp(-0.648 + 0.023*S + 0.044*C -3.168*Db)
        n = 1.392 - 0.418*(S**-0.024) + 1.212*(C**-0.704)
        m = 1.0 - 1.0/n
    elif S >= 66.5:
        theta_r = 0.0
        theta_s = 0.89 - 0.001*C - 0.322*Db
        alpha = math.exp(-4.197 + 0.013*S + 0.076*C - 0.276*Db)
        n = -2.562 + 7E-9*(S**4.004) + 3.75*(C**-0.016)
        m = 1.0 - 1.0/n

    return VGParams(theta_s=float(np.clip(theta_s, 0.05, 0.9)),
                    theta_r=float(np.clip(theta_r, 0.0, theta_s - 1e-6)),
                    alpha=float(max(alpha, 1e-12)),
                    vG_n=float(max(n, 1.01)),
                    vG_m=float(max(m, 1e-12)))

# -------------------------
# Purpose 1: Retention characteristics
# -------------------------

def retention_characteristics(
    soil: SoilInputs,
    ptf: RetentionPTFName,
    psi_fc_cm: float = 330.0,
    psi_pwp_cm: float = 15000.0, 
) -> Dict[str, float]:
    """
    Returns:
      sat, residual, fc, pwp
    plus optional keys depending on PTF:
      alpha, n, m (vG), psi_b, lam (BC)
    """
    psi_fc_cm = float(psi_fc_cm)
    psi_pwp_cm = float(psi_pwp_cm)

    # --- vG param PTFs ---
    if ptf == "Wosten_1999":
        p = Wosten_1999(soil)
        sat, res = p.theta_s, p.theta_r
        fc = float(theta_vg([psi_fc_cm], p)[0])
        pwp = float(theta_vg([psi_pwp_cm], p)[0])
        return dict(sat=sat, residual=res, fc=fc, pwp=pwp, alpha=p.alpha, vG_n=p.vG_n, vG_m=p.vG_m)

    if ptf == "Weynants_2009":
        p = Weynants_2009(soil)
        sat, res = p.theta_s, p.theta_r
        fc = float(theta_vg([psi_fc_cm], p)[0])
        pwp = float(theta_vg([psi_pwp_cm], p)[0])
        return dict(sat=sat, residual=res, fc=fc, pwp=pwp, alpha=p.alpha, vG_n=p.vG_n, vG_m=p.vG_m)

    if ptf == "Vereecken_1989":
        p = Vereecken_1989(soil)
        sat, res = p.theta_s, p.theta_r
        fc = float(theta_vg([psi_fc_cm], p)[0])
        pwp = float(theta_vg([psi_pwp_cm], p)[0])
        return dict(sat=sat, residual=res, fc=fc, pwp=pwp, alpha=p.alpha, vG_n=p.vG_n, vG_m=p.vG_m)
    
    if ptf == "Saxton_Rawls_2006":
        sr = Saxton_Rawls_2006_WR(soil)
        return dict(
            sat=sr["sat"],
            residual=sr["residual"],
            fc=sr["fc"],
            pwp=sr["pwp"],
            lam=sr["lam"],   # optional extra output
        )
    if ptf == "Zacharias_Wessolek_2007":
        p = Zacharias_Wessolek_2007(soil)
        sat, res = p.theta_s, p.theta_r
        fc = float(theta_vg([psi_fc_cm], p)[0])
        pwp = float(theta_vg([psi_pwp_cm], p)[0])
        return dict(sat=sat, residual=res, fc=fc, pwp=pwp, alpha=p.alpha, vG_n=p.vG_n, vG_m=p.vG_m)

# -------------------------
# Purpose 2: Ks methods
# -------------------------

def Wosten_1999_Ks(soil: SoilInputs) -> float:
    clay, silt, OM, Db, top = soil.clay, soil.silt, soil.OM, soil.rho_b, soil.topsoil
    return float(math.exp(
        7.755 + 0.0352*silt + 0.93*top - 0.967*(Db**2) - 0.000484*(clay**2)
        - 0.000322*(silt**2) + 0.001/(silt if silt != 0 else 1e-12) - 0.0748/(OM if OM != 0 else 1e-12)
        - 0.643*math.log(max(silt, 1e-12)) - 0.01398*Db*clay - 0.1673*Db*OM
        + 0.02986*top*clay - 0.03305*top*silt
    ))


def Guarracino_2007_Ks(vg: VGParams) -> float:
    return float(4.65e4 * vg.theta_s * (vg.alpha ** 2))


def Gupta_2021_Ks(soil: SoilInputs) -> float:
    clay, sand, Db = soil.clay, soil.sand, soil.rho_b
    exp10 = (
        1.44 + 2.053*Db - 1.256*(Db**2) - 0.0533*clay - 0.000051*Db*clay
        + 0.00055*(clay**2) + 0.0079*sand - 0.0008*Db*sand + 0.000043*clay*sand
        + 0.000052*(sand**2)
    )
    return float(10 ** exp10)

def Cosby_1984_Ks(soil: SoilInputs) -> float:
    sand, clay = soil.sand, soil.clay
    return float(60.96 * (10 ** (0.0126*sand - 0.0064*clay - 0.60)))

def Saxton_Rawls_2006_Ks(soil: SoilInputs) -> float:
    #use the previous function to get theta_s, theta1500, and lam
    sr = Saxton_Rawls_2006_WR(soil)
    theta_s = sr["sat"]
    theta33 = sr["fc"]
    lam = sr["lam"]

    Ks_mm_hr = 1930 * ((theta_s - theta33) ** (3-lam)) #mm/hr.

    #convert to cm/day
    Ks = Ks_mm_hr * 24 * 0.1 # 1 mm/hr = 0.1 cm/hr, so multiply by 24 to get cm/day

    return float(Ks)

def ks_all_methods(
    soil: SoilInputs,
) -> Dict[str, float]:

    out: Dict[str, float] = {}

    # Direct Ks PTFs
    out["Wosten_1999_Ks"] = Wosten_1999_Ks(soil)
    out["Gupta_2021_Ks"] = Gupta_2021_Ks(soil)
    out["Cosby_1984_Ks"] = Cosby_1984_Ks(soil)
    out["Saxton_Rawls_2006_Ks"] = Saxton_Rawls_2006_Ks(soil)
    out['units'] = 'cm/day'

    return out


"""--------------------------------------------------------------------------
Compute WRC from soil characteristics using different PTFs
-----------------------------------------------------------------------------
"""

def water_retention_from_df(
    soils_df: pd.DataFrame,
    method: str,
    Ks_method: str,
    *,
    psi_fc_cm: float = 330.0,
    psi_pwp_cm: float = 15000.0,
    colmap: dict | None = None,
    keep_input_cols: bool = True,
) -> pd.DataFrame:
    """
    Compute water-retention characteristics for each row in soils_df using one selected PTF method.

    Required columns by default:
      sand, silt, clay, rho_b
    Optional:
      OM (default 0.0), topsoil (default 1)

    Parameters
    ----------
    soils_df : pd.DataFrame
        Each row is a soil sample.
    method : str
        Must match your retention_characteristics() accepted method string
        e.g. "Weynants_2009", "Wosten_1999", "Vereecken_1989", "Saxton_Rawls_2006"
    colmap : dict | None
        Map your df column names to expected names, e.g. {"bulk_density":"rho_b"}.
    keep_input_cols : bool
        If True, return input cols + outputs; otherwise outputs only.

    Returns
    -------
    pd.DataFrame
        Same index as input, with columns:
          sat, residual, fc, pwp, and any extra keys your method returns (alpha, n, m, lam, psi_b).
        If a row errors, error message is stored in 'error' and outputs are NaN.
    """
    df = soils_df.copy()

    # Rename columns if needed
    if colmap:
        df = df.rename(columns=colmap)

    # Defaults for optional fields
    if "OM" not in df.columns:
        df["OM"] = 0.0
    if "topsoil" not in df.columns:
        df["topsoil"] = 1

    required = ["sand", "silt", "clay", "rho_b", "OM", "topsoil"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    results = []
    for idx, r in df.iterrows():
        soil = SoilInputs(
            sand=float(r["sand"]),
            silt=float(r["silt"]),
            clay=float(r["clay"]),
            rho_b=float(r["rho_b"]),
            OM=float(r["OM"]),
            topsoil=int(r["topsoil"]),
        )
        #error if silt+clay+sand > 100 or < 0, or if rho_b <= 0, or if OM < 0, or if topsoil not in (0,1)
        if soil.sand + soil.silt + soil.clay > 100 or soil.sand + soil.silt + soil.clay < 0:
            raise ValueError(f"Invalid soil texture percentages at index {idx}: sand + silt + clay must be between 0 and 100")
        if soil.rho_b <= 0:
            raise ValueError(f"Invalid bulk density at index {idx}: rho_b must be > 0")
        if soil.OM < 0:
            raise ValueError(f"Invalid organic matter at index {idx}: OM must be >= 0")
            continue
        if soil.topsoil not in (0, 1):
            raise ValueError(f"Invalid topsoil flag at index {idx}: topsoil must be 0 or 1")

        try:
            wr = retention_characteristics(
                soil,
                method,
                psi_fc_cm=psi_fc_cm,
                psi_pwp_cm=psi_pwp_cm,
            )
            wr_out = dict(wr)
            
        except Exception as e:
            wr_out = {"sat": np.nan, "residual": np.nan, "fc": np.nan, "pwp": np.nan}
        
                # Compute Ks using selected methods (cm/day)
        if method == 'Saxton_Rawls_2006':
            Ks = ks_all_methods(soil)['Saxton_Rawls_2006_Ks']  # Saxton_Rawls_2006_Ks is included in ks_all_methods
        else:
            Ks = ks_all_methods(soil)[Ks_method]

        #Add Ks to the output dict
        wr_out['Ks_cm_day'] = Ks

        results.append(wr_out)

        res_df = pd.DataFrame(results, index=df.index)

        results.append(wr_out)

    res_df = pd.DataFrame(results, index=df.index)

    if keep_input_cols:
        return pd.concat([df[["sand", "silt", "clay", "rho_b", "OM", "topsoil"]], res_df], axis=1)
    return res_df