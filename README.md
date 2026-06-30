# HBV_Lab — an agentic HBV hydrological model with an MCP server

**An intuitive, object-oriented Python implementation of a lumped conceptual HBV rainfall–runoff model — with built-in calibration and uncertainty analysis, and a Model Context Protocol (MCP) server that lets AI agents drive the whole workflow.**

[![PyPI version](https://img.shields.io/pypi/v/HBV_Lab.svg)](https://pypi.org/project/HBV_Lab/)
[![Python versions](https://img.shields.io/pypi/pyversions/HBV_Lab.svg)](https://pypi.org/project/HBV_Lab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

HBV is a widely used conceptual hydrological model that simulates the main processes governing the
transformation of precipitation into streamflow — snow accumulation and melt, soil moisture
accounting, groundwater storage, and channel routing [[1]](https://iwaponline.com/hr/article/4/3/147/1357/DEVELOPMENT-OF-A-CONCEPTUAL-DETERMINISTIC-RAINFALL).
It has been implemented in many software packages and operational products
[[2]](https://www.geo.uzh.ch/en/units/h2k/Services/HBV-Model.html) [[3]](https://hess.copernicus.org/articles/17/445/2013/).

`HBV_Lab` is a clean, transparent reimplementation in Python designed to be **easy to read, easy to
use, and easy to teach with**. Every routine is exposed as a small, well-documented function, and the
full model is wrapped in a single `HBVModel` object that handles data loading, simulation,
calibration, uncertainty analysis, plotting, and persistence. It is well suited to research
prototyping, method development, and hydrology education.

**It is also agent-ready:** a built-in [MCP](https://modelcontextprotocol.io/) server exposes the
model as tools, so an AI agent (Claude Code, Claude Desktop, or any MCP client) can build, calibrate,
validate and run uncertainty analysis on HBV models through natural language — see
[Use it as an MCP server](#use-it-as-an-mcp-server-for-ai-agents).

> **Scope at a glance:** lumped (single-cell, spatially averaged) · conceptual · daily time step ·
> requires precipitation, temperature and potential evapotranspiration as input · 14 calibratable
> parameters. Please read [Scope, Assumptions & Limitations](#scope-assumptions--limitations) before
> using results in any decision-making context.

---

## Features

- **Agent-ready MCP server** — drive the full *create → load → calibrate → validate → uncertainty*
  workflow from an AI agent over stdio or HTTP, with live calibration progress and structured results.
- **Complete HBV structure** — snow, soil, groundwater response (two reservoirs, three runoff
  components) and `MAXBAS` triangular routing.
- **Object-oriented API** — one `HBVModel` object holds data, parameters, states and results.
- **Automatic calibration** — single-objective optimisation (NSE, KGE, RMSE or MAE) via SciPy,
  using a gradient-free optimiser by default (see [note on calibration](#a-note-on-calibration)).
- **Uncertainty analysis** — Monte-Carlo sampling of the parameter space with prediction-interval
  estimation (GLUE-style).
- **Performance metrics** — NSE, KGE′, percent bias (PBIAS), RMSE, MAE and correlation, with a
  configurable warm-up period excluded from evaluation.
- **Flexible data loading** — CSV or Excel, flexible date parsing, optional expansion of 12 monthly
  mean PET values to a daily series.
- **Rich plotting** and **save/load** of both results (CSV) and the full model (pickle).
- **Interactive playground** for exploring how each parameter shapes the hydrograph.

## Model structure & parameters

The 14 parameters belong, conceptually, to four routines:

```python
parameters = {
    'snow':     ['TT', 'CFMAX', 'SFCF', 'CFR', 'CWH'],
    'soil':     ['FC', 'LP', 'BETA'],
    'response': ['K0', 'K1', 'K2', 'UZL', 'PERC'],
    'routing':  ['MAXBAS'],
}
```

In the model object the parameters are stored in **three** groups — `snow`, `soil` and `response` —
with the routing parameter `MAXBAS` held inside the `response` group. Each parameter is a dictionary
with `min` / `max` / `default` values (the ranges are used by calibration and uncertainty analysis):

```python
model.set_parameters({
    'soil':     {'FC': {'min': 50, 'max': 500, 'default': 250}},
    'response': {'MAXBAS': {'default': 4}},
})
```

A diagram of a single model time step is available
[here](https://lucid.app/publicSegments/view/a0edb3b6-8eba-4db5-9984-bfd23cc004ef/image.png).

## Installation

From PyPI:

```bash
pip install HBV_Lab
```

Inside a notebook:

```python
!pip install HBV_Lab
```

From source (latest development version):

```bash
git clone https://github.com/abdallaox/HBV_python_implementation.git
cd HBV_python_implementation
pip install -e .
```

Requires Python ≥ 3.7. Core dependencies (`numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`,
`openpyxl`) are installed automatically. The interactive playground additionally needs Bokeh
(`pip install "HBV_Lab[playground]"`).

## Quick start

```python
import pandas as pd
from HBV_Lab import HBVModel

# 1. Load forcing data: a DataFrame with date, precipitation, temperature,
#    potential ET and — optionally — observed discharge columns.
df = pd.read_excel("data/test_data_2.xlsx")
model = HBVModel()
model.load_data(
    data=df,
    date_column="Date", precip_column="P", temp_column="T",
    pet_column="PET", obs_q_column="Q",
    date_format="%Y%m%d", warmup_end="19811231",
)

# 2. (Optional) override default parameter ranges / values
# model.set_parameters({'soil': {'FC': {'default': 250}}})

# 3. Run, calibrate and analyse
model.run()
model.calibrate()              # gradient-free optimisation by default
model.evaluate_uncertainty()   # Monte-Carlo uncertainty analysis
model.plot_results()

# 4. Persist results and the model itself
model.save_results("results/run.csv")
model.save_model("models/my_model")
model = HBVModel.load_model("models/my_model")
```

> The example dataset (`data/test_data_2.xlsx`) ships with the source repository, not with the
> PyPI package. Clone the repo, or point `load_data` at your own data.

### Tutorial

A complete, annotated case study — build a model, calibrate it on one period, validate it on
another, and quantify parameter uncertainty — is provided in the notebook
[**quick_start_guide.ipynb**](https://github.com/abdallaox/HBV_python_implementation/blob/main/quick_start_guide.ipynb).

### Play with HBV

Build intuition for how each parameter shapes the hydrograph in the interactive
[**HBVLAB playground**](https://hbvpythonimplementation-production.up.railway.app/HBV_playground),
which runs a model built with this library.

### Use it as an MCP server (for AI agents)

`HBV_Lab` ships an [MCP](https://modelcontextprotocol.io/) server so **any** AI agent can build, run,
calibrate and analyse HBV models through tool calls. It runs in two modes:

- **stdio** (default) — the agent launches the server as a local subprocess. Best for desktop agents.
- **HTTP** (`--http`) — the server runs as a standalone service at `http://<host>:<port>/mcp` that any
  number of remote agents can connect to.

Install once:

```bash
pip install "HBV_Lab[mcp]"
```

> **Upgrading:** stop/close the MCP client (and any running `hbv-mcp` process) **before**
> upgrading — a running server holds `hbv-mcp.exe` open on Windows, so an in-place
> `pip install -U` can fail with a file-lock error. After upgrading, **fully restart the
> client** (or remove and re-add the server) so it re-discovers the tool list; clients fetch
> tools only once per connection, so upgrading mid-session leaves them showing the old tools.

#### Option A — local (stdio)

No need to start anything yourself; the agent runs `hbv-mcp` for you. Just add it to your client's
MCP config.

**Claude Desktop** — edit `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "hbv-lab": { "command": "hbv-mcp" }
  }
}
```

**Claude Code** — one command:

```bash
claude mcp add hbv-lab hbv-mcp
```

**Any other MCP client** — point it at the command `hbv-mcp` (transport: stdio).

#### Option B — shared HTTP server (any agent, local or remote)

Start the server once:

```bash
hbv-mcp --http                       # http://127.0.0.1:8000/mcp  (this machine only)
hbv-mcp --http --host 0.0.0.0 --port 9000   # expose on the network
```

Then connect your agent to the URL:

```bash
# Claude Code
claude mcp add --transport http hbv-lab http://127.0.0.1:8000/mcp
```

```json
// Claude Desktop / generic client config
{
  "mcpServers": {
    "hbv-lab": { "type": "http", "url": "http://127.0.0.1:8000/mcp" }
  }
}
```

Host/port also read the `PORT` / `HBV_MCP_HOST` / `HBV_MCP_PORT` environment variables, so the HTTP
mode deploys cleanly to platforms like Railway, Render or Fly.

#### What the agent can do

Tools: `create_model`, `clone_model`, `copy_parameters`, `load_data` (from a CSV/Excel file path),
`get_parameters`, `set_parameters`, `get_parameter_ranges`, `set_parameter_ranges`,
`set_initial_conditions`, `run_model`, `calibrate`, `evaluate_uncertainty`, `get_metrics`,
`compare_models`, `plot_results`, `save_results`, `save_model`, `load_model`, `list_models`. A typical
agent flow is *create → load → run → calibrate → plot*. Model state is kept server-side (each tool
takes a `model_id`) and large time series are passed by **file path**, not through the agent's
context — tools return compact metrics and output-file paths.

**Calibration progress & agent steering.** `calibrate` emits MCP progress notifications every
optimizer iteration (clients that surface them show a live progress/log view), and its result
reports an honest convergence `status` (`converged` / `hit_iteration_budget` / `failed`),
`still_improving`, the best-objective-per-iteration `objective_trajectory`, and an `at_bound` list of
parameters pinned to their range limits. Because each call continues from the model's *current*
parameters, an agent can calibrate incrementally — call `calibrate` with a small `iterations` budget,
inspect the improving metric, and decide whether to keep going, widen ranges (via
`set_parameter_ranges`), or switch objective between rounds.

**Split-sample made easy.** Calibrate one model, `clone_model` it (or `copy_parameters` to a second
model), `load_data` the validation window on the clone, and `run_model` — the calibrated parameters
carry over with no manual transfer. `compare_models` tabulates calibration vs. validation metrics.

**Transparent data loading.** `load_data` reports `pet_handling` (e.g.
`expanded_from_12_monthly_means` when PET is supplied as 12 monthly means — the HBV-light convention —
and auto-expanded to daily) and a per-column `data_quality` summary (valid/missing counts, min/max), so
an agent can sanity-check the forcing without opening the file.

**Richer uncertainty.** `evaluate_uncertainty` returns the 95% prediction-band **coverage** (fraction
of observations inside the band), **mean band width**, and per-parameter posterior **quantiles**
(p5/p25/p50/p75/p95). Pass `output_file` to write the full **per-timestep prediction band**
(Date, Observed, Calibrated, BestRun, Q5, Q95) to CSV, ready to plot. (Sampling is uniform over the
parameter ranges — parameter, not predictive, uncertainty.)

## Inputs & outputs

**Inputs** (daily, consistent units): precipitation (mm), air temperature (°C) and potential
evapotranspiration (mm). PET may be supplied as a full daily series or as 12 monthly means, which
are expanded automatically. Observed discharge (mm) is optional but required for calibration,
uncertainty analysis and performance metrics. All water fluxes and storages are expressed in
millimetres of water depth over the catchment (mm).

**Outputs**: simulated discharge and its quick / intermediate / base-flow components, plus the full
internal state and flux time series (snow pack, liquid water, soil moisture, recharge, actual ET,
upper/lower storage). When observed discharge is available, performance metrics (NSE, KGE′, PBIAS,
RMSE, MAE, correlation) are computed over the post-warm-up period.

### A note on calibration

Because the model contains many threshold operations, its objective surface is piecewise-constant.
Gradient-based optimisers (e.g. `SLSQP`, `L-BFGS-B`) see near-zero numerical gradients and tend to
stop at the starting point without improving the fit, so calibration uses a **gradient-free**
method (`Nelder-Mead`) **by default**. The optimiser performs a single-objective, local search; for
a more thorough exploration of the parameter space, run `evaluate_uncertainty()` (Monte-Carlo) or
calibrate from several starting points.

## Scope, Assumptions & Limitations

`HBV_Lab` is intended for **education and research**. To use it with confidence, please be aware of
the following:

- **Lumped & conceptual.** The catchment is treated as a single spatially averaged unit. There is no
  elevation banding, sub-basin discretisation, or explicit spatial variability. Parameters are
  effective, calibrated quantities rather than directly measurable physical properties.
- **Daily time step.** The model is designed and tested for daily forcing. Other time steps are not
  validated, and several parameters (e.g. the degree-day factor `CFMAX`, recession coefficients)
  are defined per day.
- **PET is an input.** The model does not compute potential evapotranspiration internally; you must
  supply it (a daily series or 12 monthly means).
- **Implementation variant.** This is an independent reimplementation of the HBV concept. Specific
  choices — the order in which recharge and evapotranspiration are evaluated within the soil
  routine, the two-reservoir response structure, the use of the modified KGE′ (Kling et al., 2012)
  variability ratio, and the triangular `MAXBAS` weights — mean results may differ from other HBV
  implementations such as HBV-light. It has **not** been benchmarked against an operational HBV
  code, and it is not affiliated with any official HBV product.
- **Calibration caveats.** Calibration optimises a single objective with a local search and can
  settle in local optima; good performance on a calibration period does not guarantee good
  performance elsewhere. Always validate on an independent period and inspect the hydrograph, not
  just the summary metrics.
- **Uncertainty analysis** uses uniform Monte-Carlo sampling and reports prediction intervals from
  the best-performing runs (a GLUE-style approach). These are indicative, not formal Bayesian
  posterior intervals.
- **Numerical robustness.** Inputs are assumed to be clean and gap-free; missing/NaN values in the
  forcing data are not imputed. Observed-discharge gaps (NaN) are skipped in metric calculations.
- **Not for operational decision-making** (e.g. flood forecasting or water-resource operations)
  without independent, site-specific validation by a qualified hydrologist.

> **Disclaimer.** This software is provided "as is", without warranty of any kind, under the terms of
> the [MIT License](LICENSE). The authors accept no liability for any use of the model or its
> outputs. Verify results independently before relying on them.

## Project status

- Actively maintained; current release: **v1.1.0** ([changelog via releases](https://github.com/abdallaox/HBV_python_implementation/releases)).
- Core routines and the `HBVModel` API are covered by a unit-test suite (`pytest`) and exercised
  end-to-end by the case-study notebook.
- The API is considered stable for the documented methods; internal helpers may change.

## Contributing & support

Bug reports, questions and contributions are welcome via
[GitHub issues](https://github.com/abdallaox/HBV_python_implementation/issues) and pull requests.
When reporting a problem, a minimal reproducible example (data snippet + code) is greatly
appreciated.

## Citation

If you use `HBV_Lab` in academic work, please cite this repository and the foundational HBV
references below. A `CITATION` entry can be provided on request.

## License

Released under the [MIT License](LICENSE). © Abdalla Mohammed.

## References

**[1]** Bergström, S., & Forsman, A. (1973). Development of a conceptual deterministic rainfall-runoff model. *Hydrology Research*, *4*, 147-170.

**[2]** Seibert, J., & Vis, M. J. P. (2012). Teaching hydrological modeling with a user-friendly catchment-runoff-model software package. *Hydrology and Earth System Sciences*, *16*(9), 3315-3325. [doi:10.5194/hess-16-3315-2012](https://doi.org/10.5194/hess-16-3315-2012)

**[3]** AghaKouchak, A., Nakhjiri, N., & Habib, E. (2013). An educational model for ensemble streamflow simulation and uncertainty analysis. *Hydrology and Earth System Sciences*, *17*(2), 445-452. [doi:10.5194/hess-17-445-2013](https://doi.org/10.5194/hess-17-445-2013)

**[4]** Kling, H., Fuchs, M., & Paulin, M. (2012). Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios. *Journal of Hydrology*, *424–425*, 264–277. [doi:10.1016/j.jhydrol.2012.01.011](https://doi.org/10.1016/j.jhydrol.2012.01.011)
