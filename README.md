# HBV_Lab (Python implementation of a lumped conceptual HBV model)

HBV is a simple conceptual hydrological model that simulates the main hydrological processes related to snow, soil, groundwater, and routing [[1]](https://iwaponline.com/hr/article/4/3/147/1357/DEVELOPMENT-OF-A-CONCEPTUAL-DETERMINISTIC-RAINFALL). There are many software packages and off-the-shelf products that implement different versions of it [[2]](https://www.geo.uzh.ch/en/units/h2k/Services/HBV-Model.html) [[3]](https://hess.copernicus.org/articles/17/445/2013/).

I've been experimenting with the model lately and—in an endeavour to better understand the logic behind it—I decided to implement my own version—in Python, following an intuitive object-oriented programming approach.

This versioin implements the snow, soil, response and routing routines—controled by 14 calibratable parameters as shown below. In addition to calibration and uncertainty analysis modules.
```python
parameters   = {
                  'snow':        ['TT', 'CFMAX', 'SFCF', 'CFR', 'CWH'],
                  'soil':        ['FC', 'LP', 'BETA'],
                  'response':    ['K0', 'K1', 'K2', 'UZL', 'PERC']
                  'routing' :    [ 'MAXBAS'],
               }
```


This can be flexibly used for different modelling tasks, but can also be used in a classroom setup—to explain hydrological concepts (processes, calibration, uncertainty analysis, etc.).

## Get Started

### Install the Package
```bash
pip install HBV_Lab
```
### How to Use
It is very intuitive—you create a model like an object which has attributes (data, parameters, initial conditions, etc.) that you can assign and access. The object also performs functions (calibration, uncertainty estimation, save, load, etc.)
```python
from HBV_Lab import HBVModel
model = HBVModel()
model.load_data("pandas dataframe")
model.set_parameters(params)
model.run()
model.calibrate()
model.evaluate_uncertainity()
model.plot_results()
model.save_results()
model.save_model("path")
model.load_model("path")
```
### Tutorial
Start by following a simple case study in the notebook:  [**quick_start_guide.ipynb**](quick_start_guide.ipynb)
### Play with HBV 
Get a feeling of how HBV model work and the role of the different parameters in [**HBVLAB**](https://www.linkedin.com/in/abdallaimam/) (which uses a model developed with this HBV implementation).
### References 
**[1]**    Bergström, S., & Forsman, A. (1973). DEVELOPMENT OF A CONCEPTUAL DETERMINISTIC RAINFALL-RUNOFF MODEL. Hydrology Research, 4, 147-170.

**[2]**    Seibert, J. and Vis, M. J. P.: Teaching hydrological modeling with a user-friendly catchment-runoff-model software package, Hydrol. Earth Syst. Sci., 16, 3315–3325, https://doi.org/10.5194/hess-16-3315-2012, 2012.

**[3]**     AghaKouchak, A., Nakhjiri, N., and Habib, E.: An educational model for ensemble streamflow simulation and uncertainty analysis, Hydrol. Earth Syst. Sci., 17, 445–452, https://doi.org/10.5194/hess-17-445-2013, 2013.

