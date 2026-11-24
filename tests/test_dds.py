import numpy as np
import pandas as pd

from HBV_Lab import HBVModel
from HBV_Lab.calibration import _dds


def test_dds_finds_minimum_within_bounds():
    target = np.array([0.3, -2.0, 7.5, 1.0])
    bounds = [(0, 1), (-5, 5), (0, 10), (1, 1)]  # last parameter fixed
    seen = []

    def fun(x):
        seen.append(x.copy())
        return float(np.sum((x - target) ** 2))

    res = _dds(fun, [0.9, 4.0, 1.0, 1.0], bounds, 500, seed=1)
    assert res.nfev == 500 and len(seen) == 500
    assert res.fun < 1e-2
    lb, ub = np.array(bounds).T
    assert all(np.all(x >= lb) and np.all(x <= ub) for x in seen)
    assert np.all(res.history[1:] <= np.array(res.history[:-1]))  # best never gets worse


def test_dds_is_reproducible_with_seed():
    fun = lambda x: float(np.sum(np.sin(3 * x) + x ** 2))
    bounds = [(-2, 2)] * 3
    a = _dds(fun, [1, 1, 1], bounds, 200, seed=7)
    b = _dds(fun, [1, 1, 1], bounds, 200, seed=7)
    assert np.array_equal(a.x, b.x)


def _synthetic_model(n_days=3 * 365):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", periods=n_days, freq="D")
    doy = dates.day_of_year.values
    df = pd.DataFrame({
        "Date": dates.strftime("%Y%m%d"),
        "Precipitation": rng.gamma(0.6, 6.0, n_days) * (rng.random(n_days) < 0.5),
        "Temperature": 9 + 7 * np.sin(2 * np.pi * (doy - 110) / 365),
        "PotentialET": 1.5 + 1.2 * np.sin(2 * np.pi * (doy - 110) / 365),
        "Q": np.nan,
    })
    m = HBVModel()
    m.load_data(data=df.copy(), obs_q_column="Q", warmup_end="20001231")
    truth = m.run(verbose=False)["discharge"].copy()
    df["Q"] = truth
    m = HBVModel()
    m.load_data(data=df, obs_q_column="Q", warmup_end="20001231")
    return m


def test_calibrate_dds_improves_and_reports(monkeypatch):
    m = _synthetic_model()
    # Start away from the parameters that generated the "observations"
    m.set_parameters({"soil": {"FC": {"default": m.params["soil"]["FC"]["max"]}},
                      "response": {"K1": {"default": m.params["response"]["K1"]["min"]}}})
    m.run(verbose=False)
    nse0 = m.performance_metrics["NSE"]
    calls = []
    out = m.calibrate(method="DDS", iterations=300, seed=0, verbose=False, plot_results=False,
                      progress_callback=lambda *a: calls.append(a))
    assert out["performance"]["NSE"] > max(nse0, 0.9)
    assert len(calls) == 300 and len(out["trajectory"]) == 300
    assert out["optimization_result"].nfev == 300
    for group in m.params.values():
        for p in group.values():
            assert p["min"] <= p["default"] <= p["max"]


def test_calibrate_dds_seed_reproducible():
    results = []
    for _ in range(2):
        m = _synthetic_model(n_days=2 * 365)
        out = m.calibrate(method="DDS", iterations=60, seed=3, verbose=False, plot_results=False)
        results.append(out["performance"]["NSE"])
    assert results[0] == results[1]
