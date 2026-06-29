def response_routine_two_tanks(out_to_response, upper_storage, lower_storage, params):
    """
    Groundwater response routine for HBV-light model with three flow components.
    
    Based on the HBV structure 
    
    Parameters:
    -----------
    out_to_response : float
        Water from soil routine (recharge + surface runoff) (mm/day).
    upper_storage : float
        Current storage in the upper zone reservoir (SUZ) (mm).
    lower_storage : float
        Current storage in the lower zone reservoir (SLZ) (mm).
    params : dict
        Model parameters:
            - 'K0' : Recession coefficient for quick flow (upper zone above threshold) [day^-1].
            - 'K1' : Recession coefficient for intermediate flow (upper zone) [day^-1].
            - 'K2' : Recession coefficient for baseflow (lower zone) [day^-1].
            - 'UZL' : Threshold parameter for upper zone [mm].
            - 'PERC' : Percolation rate from upper to lower zone (mm/day).
    
    Returns:
    --------
    upper_storage : float
        Updated upper zone storage (mm).
    lower_storage : float
        Updated lower zone storage (mm).
    discharge : float
        Total discharge (fast + intermediate + slow runoff) (mm/day).
    quick_flow : float
        Quick flow component from upper zone (above threshold).
    intermediate_flow : float
        Intermediate flow component from upper zone.
    baseflow : float
        Baseflow component from lower zone.
    """
    
    # Extract parameters
    K0 = params['K0'] ['default']      # Quick flow coefficient (above threshold)
    K1 = params['K1'] ['default']     # Intermediate flow coefficient
    K2 = params['K2'] ['default']     # Baseflow coefficient
    UZL = params['UZL'] ['default']   # Upper zone threshold
    PERC = params['PERC'] ['default'] # Percolation from upper to lower zone
    
    # --- Step 1: Add incoming water to upper zone reservoir ---
    upper_storage += out_to_response
    
    # --- Step 2: Percolation from upper to lower zone ---
    percolation = min(PERC, upper_storage)
    upper_storage -= percolation
    lower_storage += percolation
    
    # --- Step 3: Calculate runoff components ---
    # Quick flow (Q0): only occurs if upper zone storage exceeds threshold
    if upper_storage > UZL:
        quick_flow = K0 * (upper_storage - UZL)
    else:
        quick_flow = 0.0
    
    # Intermediate flow (Q1): from all upper zone storage
    intermediate_flow = K1 * upper_storage
    
    # Baseflow (Q2): from lower zone storage
    baseflow = K2 * lower_storage
    
    # --- Step 4: Update storages after runoff ---
    upper_storage -= (quick_flow + intermediate_flow)
    lower_storage -= baseflow
    
    # --- Step 5: Total discharge ---
    discharge = quick_flow + intermediate_flow + baseflow
    
    # --- Step 6: Prevent negative storages ---
    upper_storage = max(upper_storage, 0.0)
    lower_storage = max(lower_storage, 0.0)
    
    return upper_storage, lower_storage, discharge, quick_flow, intermediate_flow, baseflow
