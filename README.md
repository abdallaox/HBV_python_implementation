### HBV_Lab (Python implementation of HBV model)
```python
HBV is a simple conceptual hydrological model that simulates the main hydrological processes related to snow, soil, groundwater, and routing. There are many software packages and off-the-shelf products that implement it.

I’ve been experimenting with the model lately and——in an endeavour to better understand the logic behind it——I decided to implement my own version —— in Python, following an intuitive object-oriented programming approach.

This can be flexibly used for different modelling tasks, but can also be used in a classroom setup —— to explain hydrological concepts (processes, calibration, uncertainty analysis, etc.).

### Getting Started

```python

"""

Usage:

### ! pip install HBV_Lab  ###########

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

It is very intiuitive——the model is like an object which has attributes (data, parameters, initial_conditions, etc.) that you can assign and access. The object also performs functions (calibration, uncertainity estimation, save, load, etc.)    

Start by follwing the notebook: quic_start_guide.ipynb

"""