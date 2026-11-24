import numpy as np
import pandas as pd
import pytest

srv = pytest.importorskip("HBV_Lab.mcp_server")


def _csv(tmp_path, n_days=2 * 365):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", periods=n_days, freq="D")
    doy = dates.day_of_year.values
    df = pd.DataFrame({
        "Date": dates.strftime("%Y%m%d"),
        "Precipitation": rng.gamma(0.6, 6.0, n_days) * (rng.random(n_days) < 0.5),
        "Temperature": 9 + 7 * np.sin(2 * np.pi * (doy - 110) / 365),
        "PotentialET": 1.5 + 1.2 * np.sin(2 * np.pi * (doy - 110) / 365),
        "Q": 1.0,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_run_model_reports_store_levels_and_initial_states_carry_through(tmp_path):
    mid = srv.create_model()["model_id"]
    srv.load_data(mid, _csv(tmp_path), obs_q_column="Q", warmup_end="20001231")

    out = srv.run_model(mid)
    stores = out["stores"]
    assert set(stores) == {"snowpack", "liquid_water", "soil_moisture", "upper_storage", "lower_storage"}
    assert stores["lower_storage"]["start"] == 20.0  # default initial state
    for s in stores.values():
        assert {"start", "mean", "end"} <= set(s)

    srv.set_initial_conditions(mid, lower_storage=200.0)
    out2 = srv.run_model(mid)
    assert out2["stores"]["lower_storage"]["start"] == 200.0
    assert out2["stores"]["lower_storage"]["mean"] > stores["lower_storage"]["mean"]

    # clone_model keeps the initial states, a second run starts from them again
    clone = srv.clone_model(mid)["model_id"]
    assert srv.run_model(clone)["stores"]["lower_storage"]["start"] == 200.0
