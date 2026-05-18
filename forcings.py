# forcings.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import prepare_inputs as prep


def build_forcing(
    meteo_daily: pd.DataFrame,
    dn_ratios_df: pd.DataFrame,
    latitude: float,
    sim_freq: str,
    precip_method: str = "dn_ratio",
    pet_method: str = "daylength",
    fnight_pet: float = 0.05,
    day_start: int = 6,
    day_end: int = 18,
) -> pd.DataFrame:
    """
    Build model forcing at the simulation timestep.

    Inputs:
      meteo_daily: daily df indexed by date with columns P, PET (mm/day)
      dn_ratios_df: monthly day/night ratios (index 1..12, columns day_ratio/night_ratio)
      latitude: degrees
      sim_freq: "D" or "H"
      precip_method: "dn_ratio" or "uniform"
      pet_method: "daylength" or "uniform"
      fnight_pet: PET night fraction if daylength method

    Output:
      DataFrame indexed by model timestep, columns:
        P   (mm/timestep)
        PET (mm/timestep)
    """
    sim_freq = str(sim_freq).upper().strip()
    if sim_freq not in ("D", "H"):
        raise ValueError("sim_freq must be 'D' or 'H'")

    # daily base
    daily = meteo_daily.copy()
    if not isinstance(daily.index, pd.DatetimeIndex):
        daily.index = pd.to_datetime(daily.index)
    daily.index = daily.index.normalize()
    daily = daily.sort_index()

    if sim_freq == "D":
        # already mm/day
        out = daily[["P", "PET"]].rename(columns={"P": "P", "PET": "PET"}).copy()
        out.index.name = "time"
        return out

    # hourly
    # --- precipitation ---
    if precip_method == "dn_ratio":
        pre_h = prep.generate_hourly_precipitation(
            precip_daily=daily["P"],
            dn_ratios_df=dn_ratios_df,
            day_start=day_start,
            day_end=day_end,
        )  # returns df with 'pre_h'
        P_h = pre_h["pre_h"].rename("P")
    elif precip_method == "uniform":
        # uniform across 24h
        idx = pd.date_range(daily.index.min(), daily.index.max() + pd.Timedelta(hours=23), freq="h")
        vals = np.repeat((daily["P"].to_numpy() / 24.0), 24)
        P_h = pd.Series(vals, index=idx, name="P")
    else:
        raise ValueError("precip_method must be 'dn_ratio' or 'uniform'")

    # --- PET ---
    if pet_method == "daylength":
        pet_h = prep.generate_hourly_pet(
            pet_daily=daily["PET"],
            latitude=latitude,
            fnight_pet=fnight_pet,
        )  # returns df with 'pet_h'
        PET_h = pet_h["pet_h"].rename("PET")
    elif pet_method == "uniform":
        idx = pd.date_range(daily.index.min(), daily.index.max() + pd.Timedelta(hours=23), freq="h")
        vals = np.repeat((daily["PET"].to_numpy() / 24.0), 24)
        PET_h = pd.Series(vals, index=idx, name="PET")
    else:
        raise ValueError("pet_method must be 'daylength' or 'uniform'")

    # align + combine
    forcing = pd.concat([P_h, PET_h], axis=1).sort_index()

    # Safety checks
    if forcing.isna().any().any():
        bad = forcing[forcing.isna().any(axis=1)].head(10)
        raise ValueError(f"Forcing contains NaNs after disaggregation. Examples:\n{bad}")

    # Conservation check per day (important!)
    # (allow tiny numerical tolerance)
# --- Conservation check per day (align indices!) ---
    daily_sum = forcing.resample("D").sum()

    # align to the daily meteo index
    daily_sum = daily_sum.reindex(daily.index)

    diffP = (daily_sum["P"] - daily["P"]).astype(float)
    diffE = (daily_sum["PET"] - daily["PET"]).astype(float)

    maxP = float(diffP.abs().max())
    maxE = float(diffE.abs().max())

    if maxP > 1e-5:
        bad_days = diffP.abs().sort_values(ascending=False).head(10)
        raise ValueError(
            "Hourly precipitation does not conserve daily totals after alignment.\n"
            f"Max abs diff = {maxP}\n"
            "Worst days (abs diff):\n"
            f"{bad_days}"
        )

    if maxE > 1e-5:
        bad_days = diffE.abs().sort_values(ascending=False).head(10)
        raise ValueError(
            "Hourly PET does not conserve daily totals after alignment.\n"
            f"Max abs diff = {maxE}\n"
            "Worst days (abs diff):\n"
            f"{bad_days}"
        )

    forcing.index.name = "time"
    return forcing
