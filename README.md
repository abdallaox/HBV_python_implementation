# HBV_Lab (Python implementation of a lumped conceptual HBV model)

HBV is a simple conceptual hydrological model that simulates the main hydrological processes related to snow, soil, groundwater, and routing [[1]](https://watermark.silverchair.com/147.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA_swggP3BgkqhkiG9w0BBwagggPoMIID5AIBADCCA90GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMKaeJWQ7VWIq_mqLkAgEQgIIDrh5vsWeaZ0wY6F-ERiovjNb4KimkGt_o5Pj5ZbNlsoszUqDq-oFtdcns5O02KBex_JakkQDDkMBBlg_PFsx9vpuBqYn1kC7EnMp274TfQ2NxQC70hO3OXoYWvMcZRf-Zl3r9w2LYzlmGc9fk2PmF173MeCsuIKaaA79mD0QvqI_9hN8Dz6P43uY3ybzhZPAWUaOFtudQTFy6iC9DZ1jChvSOBK_LpqltPDB5kbyDXkB8OYQl1RQQ3bBtO_G3pwFuat3m2YyKINkI7kLl6alxTs0VQc9cEXgxs_QkBubg_VBtVIaD8mNa7BxQ3PrWpPyrn02TfcAq67SVekWeN9K94P4B8aBBYuyi1-W8vqlp22gzHzr6FrtiP7PEE-7ymBAWQgro7XwZ4bulncCHb1seCeoLbuStdR93KGvKto6hOyN3uxuQZ5e1iP7726MkrMI5MhmCuw8awj2qQeH4LL4g1UuublCNtCX0KNkTCtdj3OUJfxORlLqcCS6PMwM7xkZpw4ytgJ1ro77v3T-yn3V7TbgriNX7UB9BWU2HGVz6SsSkZQoEcop0GPb_EpoqMMkw3NwEn8alJBXTcSbjkrAzWPm6kgZ3agvzbAI53fvueVayzDiZ49ifMnImnn3ge6Dcp3tb-M1Yw6rj8et1UhgfhJMbE0VGc_05KvEdrQYTndorlZPPUPpRK-3-jHOTDwOjz2ME-orp_oAcwoRqaja-e1aeLAuNFImu7PKsRSyn1x3zTh_2lks1_xt6T8kQgXeRluGhJwth1hXCa2ld7qrd1wo6H-6x63UFwO35ygr5MuIOuUnf0W2Rfb7tQtiPY9Vn_snj3frwtqBPOsgQUYtS5jXkSL-FngaOnUedrBvtv0iybblBtdd-LnClDE3KRDdVSjAAa0VS3v5RoUBxs3c8p44N5D3S4NQvdc8PZy9aHeJ6Tl7VNg-gWneGm3_BrOgiW1ylifGMb3X0J6NoYHSDy0mzR51VcM3w-gSCg44ejKuMPqS1yI-yTkZrAfXEKI8ECNdjLz_A9OjaMHr_saFijNhrdHX_l9_bFlfQEzh1zq6ueHZR6Bx3bMriUQYyajTQ0zpNdaQQXm1m5Caq4A-Agkmd6m9SPQNRquMCSYkP10uEw1deYjz6nAhELvTJhG7T8e2ZZ_uYantsrPO54_MkPTPI--4Mx9ePzP9modf7Dc3c98Iu1230duLUQd5sfkkRCF1Ab7P-FlJpUYt1xA2bS7csXfYc_IToq4EcNXx_Zw). There are many software packages and off-the-shelf products that implement different versions of it [[2]](https://www.geo.uzh.ch/en/units/h2k/Services/HBV-Model.html) [[3]](https://hess.copernicus.org/articles/17/445/2013/).

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

