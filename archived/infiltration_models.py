import numpy as np

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Define soil data with saturated hydraulic conductivity,Ks and and sorptivity,S values for different soil types 
# (units: Ks in mm/h, S in mm/h^0.5)
#Derived  
"""Sadehi et al., (2024) A simple, accurate, and explicit form of the Green–Ampt model to
estimate infiltration, sorptivity, and hydraulic conductivity
DOI: 10.1002/vzj2.20341

Ali et al., (2016) Green-Ampt approximations: A comprehensive analysis also provide the values of Ks and S for different soil types
http://dx.doi.org/10.1016/j.jhydrol.2016.01.065

"""
# Instantiate InfiltrationModel class
class InfiltrationModel:
    """
    Class to calculate infiltration using different models
    Input: 
        Ks (saturated hydraulic conductivity in mm/h),
        S (sorptivity in mm/h^0.5 or mm/day^0.5)
        t (time in hours)
    Output:
        Cumulative infiltration in mm over time t
    """
    def __init__(self, Ks, S):
        self.Ks = Ks
        self.S = S
        self.chi = (S**2) / (2 * Ks**2)

    #dimensionless time t*
    def dimensionless_time(self, t):
        return (2 * self.Ks**2 * t) / self.S**2

    #Valiantzas (2012):New linearized two-parameter infiltration equation for direct determination of conductivity and sorptivity
    def valiantzas_model(self, t): 
        return 0.5 * self.Ks * t + self.S * np.sqrt(t * (1 + (0.5 * self.Ks / self.S)**2 * t))

    #Sadeghi et al. (2024): A simple, accurate, and explicit form of the Green–Ampt model to estimate infiltration, sorptivity, and hydraulic conductivity
    def sadeghi_model(self, t): 
        return self.Ks * t * (0.70635 + 0.32415 * np.sqrt(1 + 9.43456 * (self.S**2) / (self.Ks**2 * t)))

    #Li et al. (1976): Solutions to Green–Ampt infiltration equation
    def li_model(self, t): 
        t_star = self.dimensionless_time(t)
        return (0.5*self.S**2 / (2 * self.Ks)) * (t_star + np.sqrt(t_star**2 + 8 * t_star))
    
    #Stone et al. (1994): Approximate form of Green–Ampt infiltration equation
    def stone_model(self, t): 
        t_star = self.dimensionless_time(t)
        return (self.S**2 / (2 * self.Ks)) * (t_star + np.sqrt(2 * t_star) - 0.2987 * t_star**0.7913)

    #Salvucci and Entekhabi (1994): Explicit expressions for Green–Ampt (delta function diffusivity) infiltration rate and cumulative storage
    def salvucci_entekhabi_model(self, t):
        term1 = (1 - np.sqrt(2) / 3) * t
        term2 = (np.sqrt(2) / 3) * np.sqrt(self.chi * t + t**2)
        term3 = (np.sqrt(2) - 1) / 3 * self.chi * (np.log(t + self.chi) - np.log(self.chi))
        term4 = (np.sqrt(2) / 3) * self.chi * (np.log(t + (self.chi / 2) + np.sqrt(self.chi * t + t**2)) - np.log(self.chi / 2))
        return self.Ks * (term1 + term2 + term3 + term4)

    #Parlange et al. (2002): Explicit infiltration equations and the Lambert W-function
    def parlange_model(self, t):
        t_star = self.dimensionless_time(t)
        return (self.S**2 / (2 * self.Ks)) * (t_star + np.log(1 + t_star + np.sqrt(2 * t_star)))

    #Swamee et al. (2012):Explicit equations for infiltration
    def swamee_model(self, t):
        t_star = self.dimensionless_time(t)
        return (self.S**2 / (2 * self.Ks)) * (1.94 * t_star**0.74 + t_star**1.429)**0.7

    #Almedeij and Essen. (2014): Modified Green–Ampt infiltration model for steady rainfall
    def almedeij_model(self, t):
        t_star = self.dimensionless_time(t)
        return (self.S**2 / (2 * self.Ks)) * (0.65 * t_star + np.sqrt(0.25 * t_star**2 + 2 * t_star))

    #Vatankhah, A.R., 2015.Discussion of modified Green–Ampt infiltration model for steady rainfall by J. Almedeij and I.I. Esen
    def vatankhah_model(self, t):
        t_star = self.dimensionless_time(t)
        return (self.S**2 / (2 * self.Ks)) * (t_star + 2.693 * np.log(1 + 0.527 * np.sqrt(t_star)))

    def get_model(self, model_name, t):
        key = model_name.lower().replace(" ", "_")
        model_func = self.models.get(key)
        if model_func:
            return model_func(t)
        else:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")


    @property
    def models(self):
        return {
            "valiantzas_model": self.valiantzas_model,
            "sadeghi_model": self.sadeghi_model,
            "li_model": self.li_model,
            "stone_model": self.stone_model,
            "salvucci_entekhabi": self.salvucci_entekhabi_model,
            "parlange_model": self.parlange_model,
            "swamee_model": self.swamee_model,
            "almedeij_model": self.almedeij_model,
            "vatankhah_model": self.vatankhah_model
        }
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
