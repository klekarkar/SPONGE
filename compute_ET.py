import numpy as np

"""
POTENTIAL EVAPOTRANSPIRATION (PET) USING THE PENMAN-MONTEITH METHOD

This script computes the reference evapotranspiration (ETo) using the Penman-Monteith method.
"""

def compute_slope_of_vapor_pressure_curve(Tmean):
    """Computes the slope of the vapor pressure curve (delta) in kPa/C
    Args:
        T (float): temperature in degrees Celsius
    Returns:
        float: slope of the vapor pressure curve in kPa/C
    """
    delta= (4098 * (0.6108 * np.exp((17.27 * Tmean) / (Tmean + 237.3)))) / (Tmean + 237.3) ** 2
    return delta


def compute_psychrometric_constant(elevation):

    """Computes the psychrometric constant (gamma) in kPa/C
    Args:
        elevation (float): elevation in meters
    Returns:
        float: psychrometric constant in kPa/C
    """
    P= 101.3 * ((293 - (0.0065 * elevation)) / 293) ** 5.26

    gamma = 0.665 * 10**-3 * P #kPa/C
    return gamma

def compute_saturation_vapor_pressure(T):
    """Computes the saturation vapor pressure (es) in kPa
    Args:
        T (float): temperature in degrees Celsius
    Returns:
        float: saturation vapor pressure in kPa
    """
    es=0.6108 * np.exp((17.27 * T) / (T + 237.3))
    return es

def dew_point_temperature(Tmean, RH):
    """Computes the dew point temperature (Td) in degrees Celsius
    Args:
        T (float): temperature in degrees Celsius
        RH (float): relative humidity in percentage
    Returns:
        float: dew point temperature in degrees Celsius
        Reference: Lawrence, M.G., 2005. The relationship between relative humidity and the dewpoint temperature in moist air:
        A simple conversion and applications. Bulletin of the American Meteorological Society, 86(2), 225-233.
    """
    a=17.625
    b=243.04
    td = b*(np.log(RH / 100) + a * Tmean / (b + Tmean)) / (a - np.log(RH / 100) - a * Tmean / (b + Tmean))
    return td

def vapor_pressure_deficit(Tmax, Tmin, RHmean):
    """Computes the vapor pressure deficit (VPD) in kPa
    Args:
        Tmax (float): maximum temperature in degrees Celsius
        Tmin (float): minimum temperature in degrees Celsius
        RHmean (float): mean relative humidity in percentage
    Returns:
        float: vapor pressure deficit in kPa
    """
    es_tmax = compute_saturation_vapor_pressure(Tmax)
    es_tmin = compute_saturation_vapor_pressure(Tmin)
    es=(es_tmax + es_tmin) / 2
    ea = es * RHmean / 100

    return es - ea


def compute_2m_wind_speed(uz,z):
    """Converts wind speeds at different height to 2m wind speed (u2) in m/s
    Args:
        uz (float): measured wind speed at height z in m/s
    Returns:
        float: 2m wind speed in m/s
    """
    u2=uz*(4.87/(np.log(67.8*z-5.42))) #logarithmic wind profile
    return u2


def compute_net_radiation(latitude,day_of_year,elevation,Tmax,Tmin,Tmean, RHmean):
    """Computes the net radiation (Rn) in MJ/m2/day
    Args:
        latitude (float): latitude in degrees
        day_of_year (int): day of the year
        elevation (float): elevation in meters
        Tmax (float): maximum temperature in degrees Celsius
        Tmin (float): minimum temperature in degrees Celsius
        Tmean (float): mean temperature in degrees Celsius
        RHmean (float): mean relative humidity in percentage
    Returns:
        float: net radiation in MJ/m2/day
    """
    #compute extraterrestrial radiation
    Gsc=0.082 #solar constant in MJ/m2/min
    dr=1+0.033*np.cos(2*np.pi*day_of_year/365)
    phi=latitude*np.pi/180 #latitude degrees converted to radians (pi/180 converts degrees to radians) (positive for northern hemisphere)
    delta=0.409*np.sin((2*np.pi*day_of_year/365)-1.39) #solar declination in radians
    omega_ws=np.arccos(-np.tan(phi)*np.tan(delta)) #sunset hour angle in radians
    Ra=(24*60/np.pi)*Gsc*dr*(omega_ws*np.sin(phi)*np.sin(delta)+np.cos(phi)*np.cos(delta)*np.sin(omega_ws))

    #compute clear sky solar radiation (Rso) (MJ/m2/day)
    """Computes the clear sky solar radiation (Rso) in MJ/m2/day
    Args:
        Ra (MJ/m2/day)
        elevation (float): elevation in meters
    """
    Rso=(0.75+2*10**-5*elevation)*Ra

    #compute solar radiation (Rs) (MJ/m2/day)
    """Computes the solar radiation (Rs) in MJ/m2/day: 
    In absence of solar radiation data, Rs can be estimated using temperature data.
    Here the adjusted Hargreaves method is used to estimate Rs.
    The adjustment coefficient k_rs is set to 0.16 for inland areas and 0.19 for coastal areas.
    Args:   
        Tmax, Tmin (float): max, min temperature in degrees Celsius
        Ra (MJ/m2/day)
    """
    Rs=(0.16*np.sqrt(np.abs(Tmax-Tmin)))*Ra 

    #compute net shortwave radiation (Rns) (MJ/m2/day)
    """Computes the net shortwave radiation (Rns) in MJ/m2/day
    Args:
        Rso (MJ/m2/day)
        albedo (float): albedo
    """
    albedo=0.23
    Rns=(1-albedo)*Rs

    #compute net longwave radiation (Rnl) (MJ/m2/day)
    """Computes the net longwave radiation (Rnl) in MJ/m2/day
    Args:
        Tmin, Tmax (float): min, max temperature in degrees Celsius
        RHmean (float): mean relative humidity in percentage
        Rso (MJ/m2/day)
    """
    sigma=4.903*10**-9 #Stefan-Boltzmann constant in MJ/K^4/m^2/day
    ea=compute_saturation_vapor_pressure(dew_point_temperature(Tmean, RHmean))
    Rnl=sigma*((Tmax+273.16)**4+(Tmin+273.16)**4)/2*(0.34-0.14*np.sqrt(ea))*(1.35*Rns/Rso-0.35)

    #compute net radiation (Rn) (MJ/m2/day)
    Rn=Rns-Rnl

    return Rn

def compute_soil_heat_flux():

    """Computes the soil heat flux (G) in MJ/m2/day
    Args:
        Tmean (float): mean temperature in degrees Celsius
    Returns:
        float: soil heat flux in MJ/m2/day
    Since Soil Heat flux is small compared to Rn, it is often neglected in the Penman-Monteith method at daily time steps.
    
    For monthly time steps, G is often Gmonth,i = 0.14 (Tmonth,i - Tmonth,i-1 )
    """
    G=0
    return G

#-------------------------------------------------------------------------------------------------------------------------
def compute_penman_monteith_ETo(latitude, day_of_year, elevation, Tmax, Tmin, Tmean, RHmean, uz, z):
    """
    Computes the reference evapotranspiration (ET0) using the Penman-Monteith method.
    
    Args:
        latitude (float): Latitude in degrees.
        day_of_year (int): Day of the year.
        elevation (float): Elevation in meters.
        Tmax (float): Maximum daily temperature in degrees Celsius.
        Tmin (float): Minimum daily temperature in degrees Celsius.
        Tmean (float): Mean daily temperature in degrees Celsius.
        RHmean (float): Mean relative humidity in percent.
        uz (float): Wind speed at height z in m/s.
        z (float): Height at which wind speed is measured in meters.
        
    Returns:
        float: Reference evapotranspiration (ET0) in mm/day.
    """
    # Constants
    cp = 1.013e-3  # Specific heat of air at constant pressure (MJ/kg°C)
    epsilon = 0.622  # Ratio molecular weight of water vapour/dry air
    
    # Calculate the slope of vapor pressure curve
    delta = compute_slope_of_vapor_pressure_curve(Tmean)
    
    # Calculate saturation vapor pressure for Tmax and Tmin
    es_tmax = compute_saturation_vapor_pressure(Tmax)
    es_tmin = compute_saturation_vapor_pressure(Tmin)
    
    # Calculate mean saturation vapor pressure
    es = (es_tmax + es_tmin) / 2
    
    # Calculate actual vapor pressure using dew point temperature
    Td = dew_point_temperature(Tmean, RHmean)
    ea = es * RHmean / 100
    
    # Calculate psychrometric constant
    gamma = compute_psychrometric_constant(elevation)
    
    # Compute net radiation
    Rn = compute_net_radiation(latitude, day_of_year, elevation, Tmax, Tmin, Tmean, RHmean)
    
    # Compute soil heat flux (assuming daytime conditions)
    G = compute_soil_heat_flux()
    
    # Convert wind speed from height z to 2 meters above ground level
    u2 = compute_2m_wind_speed(uz, z)
    
    # Penman-Monteith equation to calculate ETo. 0.408 converts MJ/m^2/day to mm/day (1/2.45)
    ETo_PM = ((0.408 * delta * (Rn - G)) + (gamma * (900 * (es - ea) * u2) / (Tmean + 273))) / (delta + gamma * (1 + 0.34 * u2))

    return ETo_PM