#!/bin/bash
bokeh serve HBV_playground.py --address=0.0.0.0 --port=${PORT:-5006} --allow-websocket-origin=*