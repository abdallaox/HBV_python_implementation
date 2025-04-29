def snow_routine(precipitation, temperature, snowpack, liquid_water, params):
    """
    Snow routine: calculates snow accumulation, melt, refreezing, and release of meltwater.

    Args:
        precipitation (float): Precipitation at timestep (mm).
        temperature (float): Air temperature at timestep (°C).
        snowpack (float): Current snow water equivalent (mm).
        liquid_water (float): Liquid water stored in snowpack (mm).
        params (dict): Model parameters:
            - TT (float): Threshold temperature (°C).
            - CFMAX (float): Degree-day factor (mm/d/°C).
            - CFR (float): Refreezing factor (-).
            - CWH (float): Water holding capacity of snowpack (-, e.g., 0.1).
            - SFCF (float): Snowfall correction factor (-).
            - PCF (float): Precipitation correction factor——to account for bias in precipitation  (-).


    Returns:
        new_snowpack (float): Updated snowpack (mm).
        new_liquid_water (float): Updated liquid water in snowpack (mm).
        runoff (float): Released meltwater to soil (mm).
    """
    # Extract parameters
    TT = params['TT']['default']
    CFMAX = params['CFMAX']['default']
    CFR = params['CFR']['default']
    CWH = params['CWH']['default']
    SFCF = params['SFCF']['default']
    PCF = params['PCF']['default']
    
    # Give the model a room to account for any biases in the estimation of precipitation

    precipitation= precipitation * PCF

    # Initialize snowfall and rainfall
    snowfall = 0.0
    rainfall = 0.0
    # Calculate melting or refreezing
    melt = 0.0
    refreeze = 0.0

    # Determine if precipitation is snow or rain
    if temperature < TT:

        snowfall = precipitation * SFCF
        snowpack += snowfall

        refreeze = CFR * CFMAX * (TT - temperature)
        refreeze = min(refreeze, liquid_water)
        snowpack += refreeze
        liquid_water -= refreeze
    else:

        rainfall = precipitation 
        melt = CFMAX * (temperature - TT)
        melt = min(melt, snowpack)  # Cannot melt more than available snow
        snowpack -= melt
        liquid_water += melt + rainfall
    
    
 

    if temperature > TT:
        melt = CFMAX * (temperature - TT)
        melt = min(melt, snowpack)  # Cannot melt more than available snow
        snowpack -= melt
        liquid_water += melt + rainfall
    else:
        refreeze = CFR * CFMAX * (TT - temperature)
        refreeze = min(refreeze, liquid_water)
        snowpack += refreeze
        liquid_water -= refreeze
       

    # Water release from snowpack if above holding capacity
    holding_capacity = CWH * snowpack
    if liquid_water > holding_capacity:
        runoff = liquid_water - holding_capacity
        liquid_water = holding_capacity
    else:
        runoff = 0.0

    return snowpack, liquid_water, runoff
