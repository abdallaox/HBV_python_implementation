import pytest
from HBV_Lab import soil_routine


def test_soil_et_and_recharge():
    params = {'FC': {'default': 100.0}, 'BETA': {'default': 2.0}, 'LP': {'default': 0.5}}

    # With moderate soil moisture and incoming runoff, we expect some recharge and ET
    soil_moisture, out_to_response, recharge, runoff, actual_et = soil_routine(
        runoff_from_snow=10.0, temperature=5.0, potential_et=2.0, soil_moisture=50.0, params=params
    )

    assert soil_moisture >= 0.0
    assert actual_et <= 2.0
    assert recharge >= 0.0
    assert out_to_response == pytest.approx(recharge + runoff)


def test_soil_overflow_produces_runoff():
    params = {'FC': {'default': 20.0}, 'BETA': {'default': 1.0}, 'LP': {'default': 0.5}}

    # Large incoming water should cause soil to exceed FC and produce runoff
    soil_moisture, out_to_response, recharge, runoff, actual_et = soil_routine(
        runoff_from_snow=100.0, temperature=5.0, potential_et=0.0, soil_moisture=10.0, params=params
    )

    assert runoff > 0.0
    assert soil_moisture <= params['FC']['default']
