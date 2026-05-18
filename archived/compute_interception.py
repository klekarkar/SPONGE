import math

#daily interception threshold D
# Sutanto et al. 2012
#Partitioning of evaporation into transpiration, soil evaporation and interception: a comparison between isotope measurements and a HYDRUS-1D model
k=0.463 #constant
LAI=5 #variable
a=4.5 #variable

def daily_interception_threshold(P, a, k, LAI):
    """
    Calculate interception threshold D in mm/day: A minimum daily threshold of precipitation required to initiate interception
    If precipitation is less than D, all precipitation is intercepted, otherwise, interception equals to D
    Args:
    k: extinction coefficient (unitless)
    LAI: leaf area index (unitless)
    P: precipitation (mm/day)
    a: interception parameter (mm)
    Returns:
    Id: interception threshold (mm/day)

    """
    b=1-math.exp(-k*LAI)
    Id=a*LAI*(1-1/(1+(b*P/a*LAI)))
    return Id