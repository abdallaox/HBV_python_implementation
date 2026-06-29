"""
HBV_Lab MCP server — exposes the HBV hydrological model as tools an agent can call.

Run it (stdio transport, for Claude Desktop / Claude Code / any MCP client):

    pip install "HBV_Lab[mcp]"
    python -m HBV_Lab.mcp_server        # or the console script: hbv-mcp

Design notes
------------
* **Stateful models, stateless calls.** An ``HBVModel`` holds data, parameters and
  results, but MCP tools are individual calls. The server keeps a registry of live
  models; every tool references one by ``model_id``. Create one with ``create_model``,
  then load data / run / calibrate against that id.
* **No big arrays through the context window.** ``load_data`` reads a CSV or Excel
  file from disk by *path* — the multi-thousand-row time series never passes through
  the model's context. Tools return compact JSON (metrics, parameter values, output
  file paths); full result arrays are written to files via ``save_results`` /
  ``plot_results`` and referenced by path.
* Matplotlib runs headless (Agg backend) so plotting works on a server with no display.
"""

from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")  # headless: no GUI needed on a server

from mcp.server.fastmcp import FastMCP

from . import HBVModel, __version__

mcp = FastMCP("HBV_Lab")

# --- in-memory model registry -------------------------------------------------
_MODELS: dict[str, HBVModel] = {}
_COUNTER = {"n": 0}


def _new_id() -> str:
    _COUNTER["n"] += 1
    return f"model-{_COUNTER['n']}"


def _get(model_id: str) -> HBVModel:
    if model_id not in _MODELS:
        raise ValueError(
            f"Unknown model_id '{model_id}'. Call create_model first, "
            f"or list_models to see live ids."
        )
    return _MODELS[model_id]


def _metrics(model: HBVModel) -> dict:
    """Round performance metrics for compact, readable output."""
    pm = getattr(model, "performance_metrics", None)
    if not pm:
        return {}
    # cast numpy scalars to plain float so the result serializes cleanly to JSON
    return {k: (round(float(v), 4) if v is not None else None) for k, v in pm.items()}


def _find_group(model: HBVModel, name: str) -> str | None:
    for group, params in model.params.items():
        if name in params:
            return group
    return None


# --- tools --------------------------------------------------------------------
@mcp.tool()
def create_model(name: str = "") -> dict:
    """Create a new HBV model instance and return its model_id.

    All other tools operate on a model referenced by this id.
    """
    model_id = _new_id()
    _MODELS[model_id] = HBVModel()
    return {"model_id": model_id, "name": name, "hbv_lab_version": __version__}


@mcp.tool()
def list_models() -> dict:
    """List the model_ids currently held in the server registry."""
    return {"model_ids": list(_MODELS.keys())}


@mcp.tool()
def load_data(
    model_id: str,
    file_path: str,
    precip_column: str = "Precipitation",
    temp_column: str = "Temperature",
    pet_column: str = "PotentialET",
    date_column: str = "Date",
    obs_q_column: str = "",
    date_format: str = "%Y%m%d",
    warmup_end: str = "",
    start_date: str = "",
    end_date: str = "",
) -> dict:
    """Load forcing data into a model from a CSV or Excel (.xlsx) file on disk.

    Pass the column names used in the file. ``obs_q_column`` (observed discharge) is
    optional but required for calibration, uncertainty analysis and performance
    metrics. Dates use ``date_format`` (e.g. '%Y%m%d' for 19810101). ``warmup_end`` /
    ``start_date`` / ``end_date`` are optional date filters (same format).
    """
    import pandas as pd

    model = _get(model_id)
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    if file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    model.load_data(
        data=df,
        date_column=date_column,
        precip_column=precip_column,
        temp_column=temp_column,
        pet_column=pet_column,
        obs_q_column=(obs_q_column or None),
        date_format=date_format,
        warmup_end=(warmup_end or None),
        start_date=(start_date or None),
        end_date=(end_date or None),
    )
    return {
        "model_id": model_id,
        "n_timesteps": int(len(model.data)),
        "start_date": str(model.start_date),
        "end_date": str(model.end_date),
        "time_step": model.time_step,
        "has_observed_discharge": bool(obs_q_column),
    }


@mcp.tool()
def get_parameters(model_id: str) -> dict:
    """Return the model's current parameter values (the 'default' of each), grouped."""
    model = _get(model_id)
    return {
        group: {name: info["default"] for name, info in params.items()}
        for group, params in model.params.items()
    }


@mcp.tool()
def set_parameters(model_id: str, values: dict) -> dict:
    """Set parameter values from a flat mapping, e.g. {"FC": 250, "MAXBAS": 4}.

    Each named parameter's 'default' is updated; its group (snow/soil/response) is
    resolved automatically. Unknown names are reported and ignored.
    """
    model = _get(model_id)
    update: dict = {}
    unknown: list[str] = []
    for name, val in values.items():
        group = _find_group(model, name)
        if group is None:
            unknown.append(name)
            continue
        update.setdefault(group, {})[name] = {"default": val}
    if update:
        model.set_parameters(update)
    return {"updated": update, "unknown_parameters": unknown}


@mcp.tool()
def set_initial_conditions(
    model_id: str,
    snowpack: float | None = None,
    liquid_water: float | None = None,
    soil_moisture: float | None = None,
    upper_storage: float | None = None,
    lower_storage: float | None = None,
) -> dict:
    """Set initial state values (mm). Omitted states keep their current value."""
    model = _get(model_id)
    model.set_initial_conditions(
        snowpack=snowpack,
        liquid_water=liquid_water,
        soil_moisture=soil_moisture,
        upper_storage=upper_storage,
        lower_storage=lower_storage,
    )
    return {"model_id": model_id, "states": model.states}


@mcp.tool()
def run_model(model_id: str) -> dict:
    """Run the simulation over the loaded period. Returns a compact summary +
    performance metrics (if observed discharge was provided)."""
    import numpy as np

    model = _get(model_id)
    model.run(verbose=False)
    q = model.results["discharge"]
    return {
        "model_id": model_id,
        "n_timesteps": int(len(q)),
        "mean_discharge": round(float(np.mean(q)), 4),
        "max_discharge": round(float(np.max(q)), 4),
        "performance_metrics": _metrics(model),
    }


@mcp.tool()
def calibrate(
    model_id: str,
    objective: str = "NSE",
    method: str = "Nelder-Mead",
    iterations: int = 1000,
) -> dict:
    """Calibrate the model to observed discharge (requires obs data loaded).

    ``objective`` is one of NSE / KGE / RMSE / MAE. ``method`` defaults to the
    gradient-free 'Nelder-Mead' (recommended for HBV's piecewise objective).
    Returns the optimized parameters and final performance metrics.
    """
    model = _get(model_id)
    model.calibrate(
        method=method,
        objective=objective,
        iterations=iterations,
        verbose=False,
        plot_results=False,
    )
    return {
        "model_id": model_id,
        "objective": objective,
        "method": method,
        "optimized_parameters": {
            group: {name: round(info["default"], 6) for name, info in params.items()}
            for group, params in model.params.items()
        },
        "performance_metrics": _metrics(model),
    }


@mcp.tool()
def evaluate_uncertainty(
    model_id: str,
    n_runs: int = 1000,
    objective: str = "NSE",
    save_best: int = 10,
    seed: int = 42,
) -> dict:
    """Monte-Carlo uncertainty analysis (requires obs data loaded).

    Samples the parameter ranges ``n_runs`` times and keeps the ``save_best`` runs.
    Returns the best vs. current performance; full prediction intervals stay in the
    model and can be persisted with save_results.
    """
    model = _get(model_id)
    out = model.evaluate_uncertainty(
        n_runs=n_runs,
        objective=objective,
        save_best=save_best,
        seed=seed,
        plot_results=False,
        verbose=False,
    )
    return {
        "model_id": model_id,
        "objective": objective,
        "n_runs": n_runs,
        "best_performance": round(float(out["best_performance"]), 4),
        "current_performance": round(float(out["original_performance"]), 4),
    }


@mcp.tool()
def plot_results(model_id: str, output_file: str) -> dict:
    """Render the full diagnostic figure to an image file (PNG). Returns its path."""
    model = _get(model_id)
    model.plot_results(output_file=output_file, show_plots=False)
    return {"model_id": model_id, "figure_path": os.path.abspath(output_file)}


@mcp.tool()
def save_results(model_id: str, output_file: str) -> dict:
    """Write the full result time series (all fluxes and states) to a CSV file."""
    model = _get(model_id)
    model.save_results(output_file)
    return {"model_id": model_id, "results_path": os.path.abspath(output_file)}


@mcp.tool()
def save_model(model_id: str, output_path: str) -> dict:
    """Persist the entire model (data, params, results) to a file via pickle."""
    model = _get(model_id)
    model.save_model(output_path)
    return {"model_id": model_id, "model_path": os.path.abspath(output_path)}


@mcp.tool()
def load_model(model_path: str) -> dict:
    """Load a previously saved model from disk into the registry. Returns a new id."""
    if not os.path.exists(model_path):
        raise ValueError(f"File not found: {model_path}")
    model = HBVModel.load_model(model_path)
    model_id = _new_id()
    _MODELS[model_id] = model
    return {"model_id": model_id, "performance_metrics": _metrics(model)}


def main() -> None:
    """Console-script / module entry point.

    Default transport is **stdio** (for local clients like Claude Desktop / Claude Code).
    Pass ``--http`` to serve over **streamable HTTP** instead, so any remote agent can
    connect at ``http://<host>:<port>/mcp``.

        hbv-mcp                       # stdio (local)
        hbv-mcp --http                # HTTP on 127.0.0.1:8000  -> http://127.0.0.1:8000/mcp
        hbv-mcp --http --host 0.0.0.0 --port 9000   # expose on the network

    Host/port also fall back to env vars (``HBV_MCP_HOST``, ``HBV_MCP_PORT`` or ``PORT``),
    which makes it deploy-friendly on platforms like Railway/Render/Fly.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="hbv-mcp", description="HBV_Lab MCP server")
    parser.add_argument(
        "--http", action="store_true",
        help="Serve over streamable HTTP (remote agents) instead of stdio.",
    )
    parser.add_argument(
        "--host", default=os.environ.get("HBV_MCP_HOST", "127.0.0.1"),
        help="Host to bind with --http (default 127.0.0.1; use 0.0.0.0 to expose).",
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("PORT", os.environ.get("HBV_MCP_PORT", "8000"))),
        help="Port to bind with --http (default 8000, or $PORT).",
    )
    args = parser.parse_args()

    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio


if __name__ == "__main__":
    main()
