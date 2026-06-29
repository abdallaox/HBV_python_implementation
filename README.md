# HBV_Lab (Python implementation of a lumped conceptual HBV model)

HBV is a simple conceptual hydrological model that simulates the main hydrological processes related to snow, soil, groundwater, and routing [[1]](https://iwaponline.com/hr/article/4/3/147/1357/DEVELOPMENT-OF-A-CONCEPTUAL-DETERMINISTIC-RAINFALL). There are many software packages and off-the-shelf products that implement different versions of it [[2]](https://www.geo.uzh.ch/en/units/h2k/Services/HBV-Model.html) [[3]](https://hess.copernicus.org/articles/17/445/2013/).

I've been experimenting with the model lately and—in an endeavour to better understand the logic behind it—I decided to implement my own version—in Python, following an intuitive object-oriented programming approach.

This version implements the snow, soil, response and routing routines—controlled by 14 calibratable parameters as shown below—in addition to calibration and uncertainty analysis modules. See the [documentation for one step of the model.](https://lucid.app/publicSegments/view/a0edb3b6-8eba-4db5-9984-bfd23cc004ef/image.png)

Conceptually the parameters belong to four routines:

```python
parameters = {
    'snow':     ['TT', 'CFMAX', 'SFCF', 'CFR', 'CWH'],
    'soil':     ['FC', 'LP', 'BETA'],
    'response': ['K0', 'K1', 'K2', 'UZL', 'PERC'],
    'routing':  ['MAXBAS'],
}
```

In the model object the parameters are stored in three groups—`snow`, `soil`, and
`response`—and the routing parameter `MAXBAS` lives inside the `response` group.
Each parameter is a dict with `min` / `max` / `default` values, so a custom update
passed to `set_parameters` looks like:

```python
model.set_parameters({
    'soil':     {'FC': {'min': 50, 'max': 500, 'default': 250}},
    'response': {'MAXBAS': {'default': 4}},
})
```


This can be flexibly used for different modelling tasks, but can also be used in a classroom setup—to explain hydrological concepts (processes, calibration, uncertainty analysis, etc.).

## Get Started

### Install the Package
From PyPI:
```bash
pip install HBV_Lab
```
Or, inside a notebook:
```python
!pip install HBV_Lab
```
To work from the source / latest version:
```bash
git clone https://github.com/abdallaox/HBV_python_implementation.git
cd HBV_python_implementation
pip install -e .
```

### How to Use
It is very intuitive—you create a model like an object which has attributes (data, parameters, initial conditions, etc.) that you can assign and access. The object also performs functions (calibration, uncertainty estimation, save, load, etc.)
```python
import pandas as pd
from HBV_Lab import HBVModel

# 1. Load forcing data (a DataFrame with date, precipitation, temperature,
#    potential ET and—optionally—observed discharge columns)
df = pd.read_excel("data/test_data_2.xlsx")
model = HBVModel()
model.load_data(
    data=df,
    date_column="Date", precip_column="P", temp_column="T",
    pet_column="PET", obs_q_column="Q",
    date_format="%Y%m%d", warmup_end="19811231",
)

# 2. (Optional) override default parameter ranges/values
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
### Tutorial
Start by following a simple case study in the notebook:  [**quick_start_guide.ipynb**](https://github.com/abdallaox/HBV_python_implementation/blob/main/quick_start_guide.ipynb)
### Play with HBV 
Get a feeling of how the model works and the role of the different parameters in [**HBVLAB**](https://hbvpythonimplementation-production.up.railway.app/HBV_playground)—a playground that uses a model developed with this implementation.
### References

**[1]** Bergström, S., & Forsman, A. (1973). Development of a conceptual deterministic rainfall-runoff model. *Hydrology Research*, *4*, 147-170.

**[2]** Seibert, J., & Vis, M. J. P. (2012). Teaching hydrological modeling with a user-friendly catchment-runoff-model software package. *Hydrology and Earth System Sciences*, *16*(9), 3315-3325. [doi:10.5194/hess-16-3315-2012](https://doi.org/10.5194/hess-16-3315-2012)

**[3]** AghaKouchak, A., Nakhjiri, N., & Habib, E. (2013). An educational model for ensemble streamflow simulation and uncertainty analysis. *Hydrology and Earth System Sciences*, *17*(2), 445-452. [doi:10.5194/hess-17-445-2013](https://doi.org/10.5194/hess-17-445-2013)

