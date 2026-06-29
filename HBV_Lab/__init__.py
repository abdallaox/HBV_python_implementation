"""
HBV_Lab — an intuitive, object-oriented Python implementation of a lumped
conceptual HBV hydrological model for education and research.

Typical usage::

    from HBV_Lab import HBVModel
    model = HBVModel()
    model.load_data(data=df, ...)
    model.run()
    model.calibrate()
    model.evaluate_uncertainty()

The individual process routines are also importable for advanced use::

    from HBV_Lab import snow_routine, soil_routine, response_routine_two_tanks
    from HBV_Lab import hbv_step, route_with_maxbas
"""

from .HBV_model import HBVModel
from .hbv_step import hbv_step
from .snow import snow_routine
from .soil import soil_routine
from .response import response_routine_two_tanks
from .routing import route_with_maxbas

__version__ = "1.1.1"

__all__ = [
    "HBVModel",
    "hbv_step",
    "snow_routine",
    "soil_routine",
    "response_routine_two_tanks",
    "route_with_maxbas",
]
