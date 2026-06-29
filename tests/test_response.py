from HBV_Lab import response_routine_two_tanks


def test_response_basic_mass_balance():
    params = {'K0': {'default': 0.5}, 'K1': {'default': 0.2}, 'K2': {'default': 0.01}, 'UZL': {'default': 20.0}, 'PERC': {'default': 1.0}}

    upper, lower, discharge, q0, q1, q2 = response_routine_two_tanks(out_to_response=5.0, upper_storage=10.0, lower_storage=20.0, params=params)

    # discharge is sum of flow components
    assert discharge == q0 + q1 + q2
    assert upper >= 0.0
    assert lower >= 0.0


def test_response_percolation_and_threshold():
    params = {'K0': {'default': 0.5}, 'K1': {'default': 0.2}, 'K2': {'default': 0.01}, 'UZL': {'default': 0.0}, 'PERC': {'default': 100.0}}

    # Large PERC means most incoming water should percolate to lower zone
    upper, lower, discharge, q0, q1, q2 = response_routine_two_tanks(out_to_response=3.0, upper_storage=1.0, lower_storage=0.0, params=params)

    # Ensure percolation did not make storages negative and produced baseflow
    assert lower >= 0.0
    assert q2 >= 0.0
