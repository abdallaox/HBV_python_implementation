def soil_routine(runoff_from_snow, temperature, potential_et, soil_moisture, params):
    """
    Soil moisture routine for HBV-light model.

    Simulates soil moisture dynamics, actual evapotranspiration (ET), 
    groundwater recharge, and surface runoff based on incoming water 
    and current soil moisture conditions.

    Parameters:
    -----------
    runoff_from_snow : float
        Incoming liquid wls
        ater from snow routine (mm/day).
    temperature : float
        Daily average temperature (°C). (Currently unused but kept for possible extensions).
    potential_et : float
        Potential evapotranspiration (mm/day).
    soil_moisture : float
        Current soil moisture storage (mm).
    params : dict
        Model parameters:
            - 'FC'   : Maximum soil moisture storage (Field Capacity) [mm].
            - 'BETA' : Shape parameter controlling recharge [-].
            - 'LP'   : Soil moisture threshold for full evapotranspiration [-].

    Returns:
    --------
    soil_moisture : float
        Updated soil moisture storage (mm).
    recharge : float
        Water recharging to the groundwater (mm/day).
    actual_et : float
        Actual evapotranspiration (mm/day).
    runoff : float
        Surface runoff due to soil overflow (mm/day).
    """

    FC = params['FC']     ['default']   # Maximum soil moisture capacity
    BETA = params['BETA'] ['default'] # Recharge curve parameter
    LP = params['LP']     ['default'] # Limit for potential ET to be fully achieved

    # --- Step 1: Calculate actual evapotranspiration ---
    if soil_moisture > LP * FC:
        actual_et = potential_et  # Full potential ET when soil is wet enough
    else:
        actual_et = potential_et * (soil_moisture / (LP * FC))  # Reduced ET if soil is dry

    # Ensure we don't evaporate more than what's available
    actual_et = min(actual_et, soil_moisture)

    # --- Step 2: Calculate groundwater recharge ---
    if soil_moisture > 0:
        recharge = runoff_from_snow * (soil_moisture / FC) ** BETA
    else:
        recharge = 0.0

    # --- Step 3: Update soil moisture balance ---
    soil_moisture = soil_moisture + runoff_from_snow - actual_et - recharge

    # --- Step 4: Handle surface runoff if soil moisture exceeds FC ---
    if soil_moisture > FC:
        runoff = soil_moisture - FC  # Excess water becomes surface runoff
        soil_moisture = FC           # Cap soil moisture at field capacity
       
    else:
        runoff = 0.0  # No surface runoff

    # --- Step 5: Prevent negative soil moisture ---
    soil_moisture = max(soil_moisture, 0.0)

    out_to_response = recharge + runoff

    return soil_moisture, out_to_response, recharge, runoff, actual_et
