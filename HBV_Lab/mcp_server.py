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

from mcp.server.fastmcp import FastMCP, Context

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


def _downsample(seq, n: int = 20) -> list:
    """At most ``n`` evenly-spaced rounded points from ``seq`` (keeps first and last).

    Keeps the calibration trajectory compact in the tool result regardless of how many
    iterations ran.
    """
    seq = [round(float(v), 4) for v in seq]
    if len(seq) <= n:
        return seq
    step = (len(seq) - 1) / (n - 1)
    idx = sorted({int(round(k * step)) for k in range(n)})
    return [seq[i] for i in idx]


def _at_bound(model: HBVModel, rel_tol: float = 0.005) -> list:
    """List parameters whose value sits within ``rel_tol`` of their min or max bound.

    A parameter pinned to a bound usually means the search range is clipping the true
    optimum — surfacing it lets the agent widen the range with set_parameter_ranges.
    """
    flags = []
    for group, params in model.params.items():
        for name, info in params.items():
            lo, hi, val = info.get("min"), info.get("max"), info.get("default")
            if lo is None or hi is None or hi <= lo:
                continue
            margin = rel_tol * (hi - lo)
            if val <= lo + margin:
                flags.append({"parameter": name, "bound": "lower", "value": round(float(val), 4), "limit": float(lo)})
            elif val >= hi - margin:
                flags.append({"parameter": name, "bound": "upper", "value": round(float(val), 4), "limit": float(hi)})
    return flags


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
    """Return the model's current parameter values (the 'default' of each), grouped,
    plus an ``at_bound`` list flagging any parameter pinned to its min/max range."""
    model = _get(model_id)
    return {
        "parameters": {
            group: {name: float(info["default"]) for name, info in params.items()}
            for group, params in model.params.items()
        },
        "at_bound": _at_bound(model),
    }


@mcp.tool()
def get_parameter_ranges(model_id: str) -> dict:
    """Return the min / max / default of every parameter (the ranges the optimizer and
    Monte-Carlo uncertainty sampling use), grouped."""
    model = _get(model_id)
    return {
        group: {
            name: {
                "min": float(info["min"]),
                "max": float(info["max"]),
                "default": float(info["default"]),
            }
            for name, info in params.items()
        }
        for group, params in model.params.items()
    }


@mcp.tool()
def set_parameter_ranges(model_id: str, ranges: dict) -> dict:
    """Widen or narrow parameter search ranges from a flat mapping, e.g.
    {"FC": {"min": 50, "max": 500}, "CFMAX": {"max": 10}}.

    Each entry may set any of "min" / "max" / "default"; the group is resolved
    automatically. Use this when calibrate reports a parameter ``at_bound`` — widen its
    range, then calibrate again. Unknown names are reported and ignored.
    """
    model = _get(model_id)
    update: dict = {}
    unknown: list[str] = []
    for name, spec in ranges.items():
        group = _find_group(model, name)
        if group is None:
            unknown.append(name)
            continue
        allowed = {k: spec[k] for k in ("min", "max", "default") if k in spec}
        if allowed:
            update.setdefault(group, {})[name] = allowed
    if update:
        model.set_parameters(update)
    return {"updated": update, "unknown_parameters": unknown}


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
async def calibrate(
    model_id: str,
    objective: str = "NSE",
    method: str = "Nelder-Mead",
    iterations: int = 1000,
    ctx: Context = None,
) -> dict:
    """Calibrate the model to observed discharge (requires obs data loaded).

    ``objective`` is one of NSE / KGE / RMSE / MAE. ``method`` defaults to the
    gradient-free 'Nelder-Mead' (recommended for HBV's piecewise objective).

    Progress: while running, the server emits MCP progress notifications each
    optimizer iteration (visible in clients that surface them).

    Incremental use: each call continues from the model's *current* parameters, so an
    agent can call calibrate repeatedly with a small ``iterations`` budget (e.g. 50),
    inspect the improving metric and ``objective_trajectory`` between calls, and decide
    whether to keep going, widen parameter ranges, or switch objective.

    Returns the optimized parameters, final metrics, optimizer status, and the
    best-objective-per-iteration trajectory (downsampled).
    """
    import asyncio

    model = _get(model_id)
    loop = asyncio.get_running_loop()
    updates: asyncio.Queue = asyncio.Queue()

    def progress_cb(i, total, current, best):
        # Runs in the worker thread — hand the update back to the event loop safely.
        loop.call_soon_threadsafe(updates.put_nowait, (i, total, current, best))

    async def report(i, total, current, best):
        if ctx is None:
            return
        try:
            await ctx.report_progress(i, total)
            await ctx.info(
                f"calibrate iter {i}/{total}: {objective}={current:.4f} (best {best:.4f})"
            )
        except Exception:
            pass

    # Run the (blocking) calibration in a worker thread so we can stream progress.
    task = asyncio.create_task(
        asyncio.to_thread(
            model.calibrate,
            method=method,
            objective=objective,
            iterations=iterations,
            verbose=False,
            plot_results=False,
            progress_callback=progress_cb,
        )
    )
    while not task.done():
        try:
            i, total, current, best = await asyncio.wait_for(updates.get(), timeout=0.25)
            await report(i, total, current, best)
        except asyncio.TimeoutError:
            pass
    while not updates.empty():  # flush any stragglers
        await report(*updates.get_nowait())
    out = await task  # re-raises any error from the worker thread

    result = out["optimization_result"]
    traj = out.get("trajectory", [])
    success = bool(getattr(result, "success", True))
    message = str(getattr(result, "message", ""))

    # Honest convergence reporting: a maxiter-capped run isn't a "failure" — it just
    # ran out of budget and (usually) is still improving. Distinguish the cases.
    still_improving = bool(
        len(traj) >= 2 and abs(float(traj[-1]) - float(traj[-min(6, len(traj))])) > 1e-3
    )
    if success:
        status, guidance = "converged", "Optimizer converged. No further iterations needed."
    elif "iteration" in message.lower() or "maxiter" in message.lower():
        status = "hit_iteration_budget"
        guidance = (
            "Hit the iteration budget; still improving - call calibrate again to "
            "continue (it resumes from the current parameters)."
            if still_improving
            else "Hit the iteration budget but the objective has plateaued; likely converged."
        )
    else:
        status, guidance = "failed", f"Optimizer stopped without converging: {message}"

    at_bound = _at_bound(model)
    if at_bound:
        guidance += (
            " Some parameters are at their range limits "
            f"({', '.join(b['parameter'] for b in at_bound)}); consider widening them "
            "with set_parameter_ranges before re-calibrating."
        )

    return {
        "model_id": model_id,
        "objective": objective,
        "method": method,
        "n_iterations": int(getattr(result, "nit", len(traj))),
        "converged": success,
        "status": status,
        "still_improving": still_improving,
        "guidance": guidance,
        "success": success,  # kept for backward compatibility
        "message": message,
        "optimized_parameters": {
            group: {name: round(float(info["default"]), 6) for name, info in params.items()}
            for group, params in model.params.items()
        },
        "performance_metrics": _metrics(model),
        "at_bound": at_bound,
        "objective_trajectory": _downsample(traj, 20),
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
    import numpy as np

    out = model.evaluate_uncertainty(
        n_runs=n_runs,
        objective=objective,
        save_best=save_best,
        seed=seed,
        plot_results=False,
        verbose=False,
    )

    # 95% prediction-interval diagnostics from the best runs
    df = out["best_runs"]
    obs = np.asarray(df["observed"].values, dtype=float)
    lo = np.asarray(df["q5"].values, dtype=float)
    hi = np.asarray(df["q95"].values, dtype=float)
    valid = ~np.isnan(obs)
    if valid.any():
        inside = (obs[valid] >= lo[valid]) & (obs[valid] <= hi[valid])
        coverage_95 = round(float(inside.mean()), 3)
        mean_band_width = round(float(np.nanmean(hi[valid] - lo[valid])), 4)
    else:
        coverage_95 = mean_band_width = None

    # Per-parameter posterior ranges across the best parameter sets
    posterior: dict = {}
    for s in out["best_parameter_sets"]:
        for params in s["parameters"].values():
            for name, info in params.items():
                posterior.setdefault(name, []).append(info["default"])
    posterior_ranges = {
        name: {"min": round(float(min(v)), 4), "max": round(float(max(v)), 4)}
        for name, v in posterior.items()
    }

    return {
        "model_id": model_id,
        "objective": objective,
        "n_runs": n_runs,
        "save_best": save_best,
        "best_performance": round(float(out["best_performance"]), 4),
        "current_performance": round(float(out["original_performance"]), 4),
        "coverage_95": coverage_95,          # fraction of observations inside the 95% band
        "mean_band_width": mean_band_width,  # mean (q95 - q5), mm/day
        "parameter_posterior_ranges": posterior_ranges,
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


@mcp.tool()
def clone_model(model_id: str, name: str = "") -> dict:
    """Deep-copy a model (parameters, data, states, results) into a new model_id.

    The canonical split-sample pattern: calibrate ``model-A``, ``clone_model`` it, then
    ``load_data`` the validation window on the clone and ``run_model`` — the calibrated
    parameters carry over, no manual parameter transfer needed.
    """
    import copy

    src = _get(model_id)
    new_id = _new_id()
    _MODELS[new_id] = copy.deepcopy(src)
    return {"model_id": new_id, "cloned_from": model_id, "name": name}


@mcp.tool()
def copy_parameters(from_model_id: str, to_model_id: str) -> dict:
    """Copy the calibrated parameters (and ranges) from one model to another.

    Use this to transfer a calibration to a validation model without copying its data.
    """
    import copy

    src = _get(from_model_id)
    dst = _get(to_model_id)
    dst.params = copy.deepcopy(src.params)
    return {
        "from_model_id": from_model_id,
        "to_model_id": to_model_id,
        "parameters": {
            group: {name: float(info["default"]) for name, info in params.items()}
            for group, params in dst.params.items()
        },
    }


@mcp.tool()
def get_metrics(model_id: str) -> dict:
    """Return the model's current performance metrics without re-running it.

    Empty if the model hasn't been run with observed discharge yet.
    """
    model = _get(model_id)
    return {"model_id": model_id, "performance_metrics": _metrics(model)}


@mcp.tool()
def compare_models(model_ids: list) -> dict:
    """Tabulate performance metrics across several models side by side.

    Handy for comparing calibration vs validation, or several calibration variants.
    """
    rows = []
    for mid in model_ids:
        try:
            model = _get(mid)
            rows.append({"model_id": mid, "metrics": _metrics(model)})
        except ValueError as e:
            rows.append({"model_id": mid, "error": str(e)})
    return {"comparison": rows}


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
