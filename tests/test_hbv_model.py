import pandas as pd
import numpy as np
import os
import tempfile
import pytest
from HBV_Lab import HBVModel


def make_sample_df(n_days=10):
    dates = pd.date_range(start='2000-01-01', periods=n_days, freq='D')
    df = pd.DataFrame({
        'Date': dates.strftime('%Y%m%d'),
        'Precipitation': np.ones(n_days) * 1.0,
        'Temperature': np.ones(n_days) * 2.0,
        'PotentialET': np.ones(n_days) * 0.5,
        'ObservedQ': np.ones(n_days) * 0.2
    })
    return df


def test_load_data_and_time_step_and_pet_expansion():
    model = HBVModel()
    df = make_sample_df(n_days=20)

    model.load_data(data=df, date_column='Date', date_format='%Y%m%d', precip_column='Precipitation', temp_column='Temperature', pet_column='PotentialET', obs_q_column='ObservedQ')

    assert model.data is not None
    assert model.start_date is not None
    assert model.time_step == 'D'


def test_set_parameters_validation():
    model = HBVModel()
    with pytest.raises(ValueError):
        model.set_parameters(None)

    # valid update
    custom = {'soil': {'FC': {'default': 200.0}}}
    model.set_parameters(custom)
    assert model.params['soil']['FC']['default'] == 200.0


def test_set_initial_conditions_and_run(tmp_path):
    model = HBVModel()
    df = make_sample_df(n_days=15)
    model.load_data(data=df, date_column='Date', date_format='%Y%m%d', precip_column='Precipitation', temp_column='Temperature', pet_column='PotentialET', obs_q_column='ObservedQ')

    model.set_initial_conditions(snowpack=5.0, soil_moisture=40.0)
    results = model.run(verbose=False)

    assert results['discharge'].shape[0] == 15
    assert np.all(results['discharge'] >= 0.0)

    # Test save_results
    out_file = tmp_path / "results_test.csv"
    model.results = results
    model.save_results(str(out_file))
    assert os.path.exists(out_file)


def test_performance_metrics_calculation():
    model = HBVModel()
    df = make_sample_df(n_days=12)
    model.load_data(data=df, date_column='Date', date_format='%Y%m%d', precip_column='Precipitation', temp_column='Temperature', pet_column='PotentialET', obs_q_column='ObservedQ')

    model.set_initial_conditions()
    results = model.run(verbose=False)
    # attach observed into results
    results['observed_q'] = np.ones(len(results['discharge'])) * 0.2
    model.results = results

    model.calculate_performance_metrics(verbose=False)
    assert hasattr(model, 'performance_metrics')
    keys = {'NSE','KGE','PBIAS','RMSE','MAE'}
    assert keys.issubset(set(model.performance_metrics.keys()))
