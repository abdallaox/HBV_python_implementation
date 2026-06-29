from HBV_Lab import snow_routine


def test_snow_accumulates_and_runs_off():
    params = {
        'TT': {'default': 0.0},
        'CFMAX': {'default': 2.0},
        'CFR': {'default': 0.0},
        'CWH': {'default': 0.1},
        'SFCF': {'default': 1.0}
    }

    # Cold day: precipitation becomes snowfall and increases snowpack
    snowpack, liquid, runoff = snow_routine(precipitation=10.0, temperature=-5.0,
                                            snowpack=0.0, liquid_water=0.0, params=params)

    assert snowpack > 0.0
    assert liquid >= 0.0
    assert runoff == 0.0  # no holding capacity exceeded immediately


def test_snow_melts_when_warm():
    params = {
        'TT': {'default': 0.0},
        'CFMAX': {'default': 5.0},
        'CFR': {'default': 0.0},
        'CWH': {'default': 0.2},
        'SFCF': {'default': 1.0}
    }

    # Warm day with existing snowpack should generate melt and liquid water
    snowpack, liquid, runoff = snow_routine(precipitation=0.0, temperature=5.0,
                                            snowpack=20.0, liquid_water=0.0, params=params)

    assert snowpack < 20.0
    assert liquid >= 0.0
    # since holding capacity = CWH * snowpack after melt, liquid might be <= holding capacity
    assert runoff >= 0.0
