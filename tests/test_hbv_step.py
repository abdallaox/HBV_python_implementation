from hbv_step import hbv_step


def test_hbv_step_returns_expected_keys():
    params = {
        'snow': {'TT': {'default': 0.0}, 'CFMAX': {'default': 2.0}, 'CFR': {'default': 0.0}, 'CWH': {'default': 0.1}, 'SFCF': {'default': 1.0}},
        'soil': {'FC': {'default': 100.0}, 'BETA': {'default': 2.0}, 'LP': {'default': 0.5}},
        'response': {'K0': {'default': 0.5}, 'K1': {'default': 0.2}, 'K2': {'default': 0.01}, 'UZL': {'default': 20.0}, 'PERC': {'default': 1.0}}
    }

    initial_conditions = {'snowpack': 0.0, 'liquid_water': 0.0, 'soil_moisture': 50.0, 'upper_storage': 10.0, 'lower_storage': 20.0}

    new_states, fluxes = hbv_step(precipitation=5.0, temperature=5.0, potential_et=2.0, params=params, initial_conditions=initial_conditions)

    # new_states contains required keys
    assert set(new_states.keys()) == {'snowpack','liquid_water','soil_moisture','upper_storage','lower_storage'}
    # fluxes contain discharge and flow components
    assert 'discharge' in fluxes and 'quick_flow' in fluxes and 'baseflow' in fluxes
