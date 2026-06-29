from .snow import snow_routine
from .soil import soil_routine
from .response import response_routine_two_tanks

def hbv_step(precipitation, temperature, potential_et, params, initial_conditions):
    """
    HBV-light hydrological model (single timestep version).

    Runs one timestep of the model using snow, soil, and groundwater response routines.

    Args:
        precipitation (float): Precipitation at the timestep (mm).
        temperature (float): Air temperature at the timestep (°C).
        potential_et (float): Potential evapotranspiration at the timestep (mm).
        params (dict): Dictionary with parameters and their ranges:
            - 'snow': parameters for the snow routine
            - 'soil': parameters for the soil routine
            - 'response': parameters for the groundwater response routine
        initial_conditions (dict): Dictionary with initial storages:
            - 'snowpack': initial snowpack (mm)
            - 'liquid_water': initial liquid water in snowpack (mm)
            - 'soil_moisture': initial soil moisture (mm)
            - 'upper_storage': initial upper groundwater storage (mm)
            - 'lower_storage': initial lower groundwater storage (mm)

    Returns:
        new_states (dict): Updated storages after the timestep.
        fluxes (dict): Fluxes generated during the timestep (runoff, ET, etc.).
    """

    # Unpack initial conditions
    snowpack = initial_conditions['snowpack']
    liquid_water = initial_conditions['liquid_water']
    soil_moisture = initial_conditions['soil_moisture']
    upper_storage = initial_conditions['upper_storage']
    lower_storage = initial_conditions['lower_storage']

    # Step 1: Snow routine
    snowpack, liquid_water, runoff_from_snow = snow_routine(
        precipitation, temperature, snowpack, liquid_water, params['snow']
    )

    # Step 2: Soil routine
    soil_moisture, out_to_response, recharge, runoff_soil, actual_et = soil_routine(
        runoff_from_snow, temperature, potential_et, soil_moisture, params['soil']
    )

    # Step 3: Groundwater response routine
    upper_storage, lower_storage, discharge, quick_flow, intermediate_flow, baseflow = response_routine_two_tanks(
        out_to_response, upper_storage, lower_storage, params['response']
    )

    # Updated states
    new_states = {
        'snowpack': snowpack,
        'liquid_water': liquid_water,
        'soil_moisture': soil_moisture,
        'upper_storage': upper_storage,
        'lower_storage': lower_storage,
    }

    # Outputs/Fluxes
    fluxes = {
        'discharge': discharge,
        'quick_flow': quick_flow,
        'intermediate_flow': intermediate_flow,
        'baseflow': baseflow,
        'actual_et': actual_et,
        'recharge': recharge,
        'runoff_soil': runoff_soil,
        'runoff_from_snow': runoff_from_snow,
    }

    return new_states, fluxes