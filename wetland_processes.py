from __future__ import annotations


def wetland_area_from_depth(depth_m, area_fraction, h_max_m, shape_p=1.0):
    """
    Estimate active wetland area fraction from water depth.

    Parameters
    ----------
    depth_m : float
        Current wetland water depth [m].
    area_fraction : float
        Maximum wetland area as fraction of the modelled area [-].
    h_max_m : float
        Maximum wetland depth [m].
    shape_p : float
        Shape parameter for area expansion. If shape_p = 1, area increases
        linearly with depth. Larger values delay expansion at shallow depths.

    Returns
    -------
    float
        Current active wetland water surface area as fraction of modelled area [-].
    """
    if area_fraction <= 0:
        return 0.0

    if h_max_m <= 0:
        raise ValueError("h_max_m must be > 0.")

    depth_rel = max(0.0, min(float(depth_m) / float(h_max_m), 1.0))

    return float(area_fraction) * depth_rel ** float(shape_p)


def wetland_water_balance_step(
    storage_mm_grid,
    precipitation_mm,
    pet_mm,
    surface_runoff_mm,
    wetland_params,
    dt_hours=24.0,
):
    """
    Run one timestep of the wetland water balance.

    All returned fluxes are in mm over the full model/grid area, except
    depth_m and active_area_fraction.

    V1 assumption
    -------------
    Wetland infiltration is counted as additional potential recharge.
    """
    if not wetland_params.get("enabled", False):
        return {
            "storage_mm_grid": 0.0,
            "depth_m": 0.0,
            "active_area_fraction": 0.0,
            "routed_runoff_mm": 0.0,
            "wetland_precip_mm": 0.0,
            "wetland_evap_mm": 0.0,
            "wetland_infiltration_mm": 0.0,
            "wetland_overflow_mm": 0.0,
            "surface_runoff_after_wetland_mm": surface_runoff_mm,
            "wetland_recharge_mm": 0.0,
        }

    area_fraction = float(wetland_params.get("area_fraction", 0.0))
    route_runoff_fraction = float(wetland_params.get("route_runoff_fraction", 0.0))
    h_max_m = float(wetland_params.get("h_max_m", 1.0))
    h_spill_m = float(wetland_params.get("h_spill_m", h_max_m))
    shape_p = float(wetland_params.get("shape_p", 1.0))

    k_infil_m_per_h = float(wetland_params.get("k_infil_m_per_h", 0.0))
    outflow_coeff_m_per_h = float(wetland_params.get("outflow_coeff_m_per_h", 0.0))
    outflow_exp = float(wetland_params.get("outflow_exp", 1.5))
    open_water_evap_factor = float(wetland_params.get("open_water_evap_factor", 1.0))

    if area_fraction <= 0:
        raise ValueError("wetland area_fraction must be > 0 when wetland is enabled.")

    if area_fraction >= 1:
        raise ValueError("wetland area_fraction must be < 1.")

    if not 0 <= route_runoff_fraction <= 1:
        raise ValueError("route_runoff_fraction must be between 0 and 1.")

    if h_max_m <= 0:
        raise ValueError("h_max_m must be > 0.")

    if h_spill_m < 0 or h_spill_m > h_max_m:
        raise ValueError("h_spill_m must be between 0 and h_max_m.")

    if k_infil_m_per_h < 0:
        raise ValueError("k_infil_m_per_h must be >= 0.")

    if outflow_coeff_m_per_h < 0:
        raise ValueError("outflow_coeff_m_per_h must be >= 0.")

    storage_mm_grid = max(float(storage_mm_grid), 0.0)
    precipitation_mm = max(float(precipitation_mm), 0.0)
    pet_mm = max(float(pet_mm), 0.0)
    surface_runoff_mm = max(float(surface_runoff_mm), 0.0)

    # 1. Route upland runoff into wetland.
    routed_runoff_mm = route_runoff_fraction * surface_runoff_mm
    bypass_runoff_mm = surface_runoff_mm - routed_runoff_mm

    # 2. Add precipitation falling directly on wetland area.
    wetland_precip_mm = precipitation_mm * area_fraction

    storage_mm_grid += routed_runoff_mm + wetland_precip_mm

    # 3. Convert storage to local wetland depth.
    local_depth_mm = storage_mm_grid / area_fraction
    depth_m = local_depth_mm / 1000.0

    active_area_fraction = wetland_area_from_depth(
        depth_m=depth_m,
        area_fraction=area_fraction,
        h_max_m=h_max_m,
        shape_p=shape_p,
    )

    # If storage exists but the calculated active area is almost zero,
    # allow losses over the maximum wetland footprint.
    if storage_mm_grid > 0 and active_area_fraction <= 0:
        active_area_fraction = area_fraction

    # 4. Evaporation from wetland surface.
    evap_potential_mm_grid = pet_mm * open_water_evap_factor * active_area_fraction
    wetland_evap_mm = min(storage_mm_grid, evap_potential_mm_grid)
    storage_mm_grid -= wetland_evap_mm

    # 5. Infiltration from wetland bed.
    infil_potential_local_mm = k_infil_m_per_h * dt_hours * 1000.0
    infil_potential_grid_mm = infil_potential_local_mm * active_area_fraction

    wetland_infiltration_mm = min(storage_mm_grid, infil_potential_grid_mm)
    storage_mm_grid -= wetland_infiltration_mm

    # 6. Overflow when storage exceeds spill level.
    local_depth_mm = storage_mm_grid / area_fraction
    depth_m = local_depth_mm / 1000.0

    spill_storage_mm_grid = h_spill_m * 1000.0 * area_fraction

    if storage_mm_grid > spill_storage_mm_grid:
        excess_storage_mm = storage_mm_grid - spill_storage_mm_grid

        head_above_spill_m = max(depth_m - h_spill_m, 0.0)

        outlet_local_mm = (
            outflow_coeff_m_per_h
            * (head_above_spill_m ** outflow_exp)
            * dt_hours
            * 1000.0
        )
        outlet_grid_mm = outlet_local_mm * area_fraction

        if outflow_coeff_m_per_h <= 0:
            wetland_overflow_mm = excess_storage_mm
        else:
            wetland_overflow_mm = min(excess_storage_mm, outlet_grid_mm)

        storage_mm_grid -= wetland_overflow_mm
    else:
        wetland_overflow_mm = 0.0

    # 7. Emergency overflow above maximum physical capacity.
    max_storage_mm_grid = h_max_m * 1000.0 * area_fraction

    if storage_mm_grid > max_storage_mm_grid:
        emergency_overflow_mm = storage_mm_grid - max_storage_mm_grid
        wetland_overflow_mm += emergency_overflow_mm
        storage_mm_grid = max_storage_mm_grid

    surface_runoff_after_wetland_mm = bypass_runoff_mm + wetland_overflow_mm

    # V1 assumption: wetland infiltration is additional potential recharge.
    wetland_recharge_mm = wetland_infiltration_mm

    final_local_depth_mm = storage_mm_grid / area_fraction
    final_depth_m = final_local_depth_mm / 1000.0

    final_active_area_fraction = wetland_area_from_depth(
        depth_m=final_depth_m,
        area_fraction=area_fraction,
        h_max_m=h_max_m,
        shape_p=shape_p,
    )

    return {
        "storage_mm_grid": storage_mm_grid,
        "depth_m": final_depth_m,
        "active_area_fraction": final_active_area_fraction,
        "routed_runoff_mm": routed_runoff_mm,
        "wetland_precip_mm": wetland_precip_mm,
        "wetland_evap_mm": wetland_evap_mm,
        "wetland_infiltration_mm": wetland_infiltration_mm,
        "wetland_overflow_mm": wetland_overflow_mm,
        "surface_runoff_after_wetland_mm": surface_runoff_after_wetland_mm,
        "wetland_recharge_mm": wetland_recharge_mm,
    }