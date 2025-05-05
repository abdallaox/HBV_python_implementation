from bokeh.layouts import column, row
from bokeh.models import Slider, Button, ColumnDataSource, Div
from bokeh.models import TabPanel, Tabs
from bokeh.plotting import figure
from bokeh.io import curdoc
import numpy as np
from datetime import datetime, timedelta
from HBV_model import HBVModel



# ==== Load the real model instead of using dummy data ====
# Load the pre-calibrated model
model = HBVModel.load_model('./models/model_calibrated')

# Get the dates from the model data
dates = model.data.Date

# Initialize parameter dictionary based on the model's parameters
params = model.params

# Run the model once with default parameters to get initial data
initial_results = model.run()

# Get initial performance metrics if available
metrics_text = ""
if hasattr(model, 'performance_metrics'):
    metrics = model.performance_metrics
    metrics_text = f"NSE: {metrics.get('NSE', 'N/A'):.3f}, KGE: {metrics.get('KGE', 'N/A'):.3f}, PBIAS: {metrics.get('PBIAS', 'N/A'):.2f}%"

# Initialize sources with actual model data
sources = {
    # Discharge data
    "flow": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results['discharge']
    )),
    "observed_flow": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results.get('observed_q', np.zeros_like(initial_results['discharge']))
    )),
    
    # Snow-related data
    "snow": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results['snowpack']
    )),
    "liquid_water": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results['liquid_water']
    )),
    "temperature": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results['temperature']
    )),
    
    # Soil-related data
    "soil": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results['soil_moisture']
    )),
    "potential_et": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results['potential_et']
    )),
    "actual_et": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results['actual_et']
    )),
    
    # Response-related data
    "upper_storage": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results['upper_storage']
    )),
    "lower_storage": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results['lower_storage']
    )),
    
    # Flow components data
    "quick_flow": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results.get('quick_flow', np.zeros_like(initial_results['discharge']))
    )),
    "intermediate_flow": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results.get('intermediate_flow', np.zeros_like(initial_results['discharge']))
    )),
    "baseflow": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results.get('baseflow', np.zeros_like(initial_results['discharge']))
    )),
    
    # Stacked flow components (we'll create this properly for stacking)
    "stacked_flows": ColumnDataSource(data=dict(
        x=dates,
        baseflow=initial_results.get('baseflow', np.zeros_like(initial_results['discharge'])),
        intermediate=initial_results.get('intermediate_flow', np.zeros_like(initial_results['discharge'])),
        quick=initial_results.get('quick_flow', np.zeros_like(initial_results['discharge'])),
        # Calculate the cumulative values for proper stacking
        baseflow_top=initial_results.get('baseflow', np.zeros_like(initial_results['discharge'])),
        intermediate_top=initial_results.get('baseflow', np.zeros_like(initial_results['discharge'])) + 
                        initial_results.get('intermediate_flow', np.zeros_like(initial_results['discharge'])),
        total=initial_results['discharge']
    ))
}


# Create source for metrics
metrics_source = ColumnDataSource(data=dict(
    text=[metrics_text]
))


# ==== Parameter Definitions (using model parameters) ====
# Create empty dictionary to store sliders
sliders = {}
slider_groups = []

# Iterate through parameter groups from the model
for group_name, group_params in params.items():
    header = Div(text=f"<h3>{group_name.capitalize()} Parameters</h3>")
    group_sliders = []
    
    for key, meta in group_params.items():
        slider = Slider(
            title=key, 
            start=meta['min'], 
            end=meta['max'], 
            value=meta['default'], 
            step=0.001,
            format="0.000",
            tooltips= True,
        )
        sliders[key] = slider
        group_sliders.append(slider)
        
    
    slider_groups.append(column(header, *group_sliders))

    # Create dummy spacers (adjust heights as needed)
    top_dummy_spacer = Div(
        text="""<div style="visibility: hidden; height: 500px;"></div>""",
        width=300
    )

    bottom_dummy_spacer = Div(
        text="""<div style="visibility: hidden; height: 500px;"></div>""",
        width=300
    )

    # Create new slider groups with spacers
    slider_groups_with_spacers = (
        [top_dummy_spacer] + 
        slider_groups + 
        [bottom_dummy_spacer]
    )

# ==== Update Function using the real model ====
def update_plot(attr, old, new):
    # Create a temporary copy of the model to avoid changing the original
    temp_model = HBVModel()
    temp_model.data = model.data
    temp_model.column_names = model.column_names
    
    # Set parameters based on current slider values
    updated_params = {
        'snow': {},
        'soil': {},
        'response': {}
    }
    
    # Update parameters from sliders
    for group_name, group_params in params.items():
        for key in group_params:
            updated_params[group_name][key] = {
                'min': params[group_name][key]['min'],
                'max': params[group_name][key]['max'],
                'default': sliders[key].value
            }
    
    # Set the updated parameters to the model
    temp_model.params = updated_params
    
    # Run the model with new parameters
    results = temp_model.run()
    
    # Calculate new performance metrics
    metrics_text = ""
    if hasattr(temp_model, 'performance_metrics'):
        metrics = temp_model.performance_metrics
        metrics_text = f"NSE: {metrics.get('NSE', 'N/A'):.3f}, KGE: {metrics.get('KGE', 'N/A'):.3f}, PBIAS: {metrics.get('PBIAS', 'N/A'):.2f}%"
    
    # Update metrics source
    metrics_source.data = {'text': [metrics_text]}
    
    # Update threshold lines in plots by recreating the tabs
    tabs.tabs[0] = create_tab("snow", shared_x_range)[0]
    tabs.tabs[1] = create_tab("soil", shared_x_range)[0]
    tabs.tabs[2] = create_tab("response", shared_x_range)[0]
    tabs.tabs[3] = create_tab("Flow Components", shared_x_range)[0]
    
    # Get the flow components
    baseflow = results.get('baseflow', np.zeros_like(results['discharge']))
    intermediate_flow = results.get('intermediate_flow', np.zeros_like(results['discharge']))
    quick_flow = results.get('quick_flow', np.zeros_like(results['discharge']))
    
    # Update all data sources with new model results
    sources["flow"].data = {
        'x': dates, 
        'y': results['discharge']
    }
    
    # Snow-related data
    sources["snow"].data = {
        'x': dates, 
        'y': results['snowpack']
    }
    sources["liquid_water"].data = {
        'x': dates, 
        'y': results['liquid_water']
    }
    sources["temperature"].data = {
        'x': dates, 
        'y': results['temperature']
    }
    
    # Soil-related data
    sources["soil"].data = {
        'x': dates, 
        'y': results['soil_moisture']
    }
    sources["potential_et"].data = {
        'x': dates, 
        'y': results['potential_et']
    }
    sources["actual_et"].data = {
        'x': dates, 
        'y': results['actual_et']
    }
    
    # Response-related data
    sources["upper_storage"].data = {
        'x': dates, 
        'y': results['upper_storage']
    }
    sources["lower_storage"].data = {
        'x': dates, 
        'y': results['lower_storage']
    }
    
    # Update individual flow component sources
    sources["quick_flow"].data = {
        'x': dates, 
        'y': quick_flow
    }
    sources["intermediate_flow"].data = {
        'x': dates, 
        'y': intermediate_flow
    }
    sources["baseflow"].data = {
        'x': dates, 
        'y': baseflow
    }
    
    # Update stacked flow data for proper visualization
    sources["stacked_flows"].data = {
        'x': dates,
        'baseflow': baseflow,
        'intermediate': intermediate_flow,
        'quick': quick_flow,
        # Calculate the cumulative sums for proper stacking
        'baseflow_top': baseflow,
        'intermediate_top': baseflow + intermediate_flow,
        'total': results['discharge']
    }
    
    # Update main flow plot title with new metrics
    main_flow_plot.title.text = f"Simulated vs Observed Hydrographs ({metrics_text})"
    
    # Print parameter values for debugging
    print(f"Updated parameters:")
    for group, params_dict in updated_params.items():
        for param, values in params_dict.items():
            print(f"  {param} = {values['default']}")

# Attach update to all sliders
for s in sliders.values():
    s.on_change('value_throttled', update_plot)

# ==== Calibration Button ====
def calibrate():
    # Set the parameters to the calibrated values from the pre-loaded model
    for group_name, group_params in model.params.items():
        for key, meta in group_params.items():
            if key in sliders:
                sliders[key].value = meta['default']
    
    # Force update - this is important to ensure visual updates
    update_plot('value', None, None)
    
    print("Calibration applied - restored to pre-calibrated model parameters!")

# ==== Create Plot Tabs with Dynamic Content ====
def create_tab(tab_name, shared_x_range=None):
    """Create a tab with plots that change based on the tab name"""
    
    if shared_x_range is None:
        # Create a new x_range for the first tab
        top_plot = figure(x_axis_type="datetime", width=800, height=200,
                         tools="pan,wheel_zoom,box_zoom,reset")
    else:
        # Use the shared x_range for subsequent tabs
        top_plot = figure(x_axis_type="datetime", width=800, height=200,
                         tools="pan,wheel_zoom,box_zoom,reset",
                         x_range=shared_x_range)
    
    # Second plot with shared x_range (only for non-flow_components tabs)
    if tab_name != "Flow Components":
        bottom_plot = figure(x_axis_type="datetime", width=800, height=200,
                            tools="pan,wheel_zoom,box_zoom,reset",
                            x_range=top_plot.x_range)  # Link x-axes
    
    # Configure plots based on tab_name
    if tab_name == "snow":
        # Top plot: Temperature with TT threshold
        top_plot.title.text = "Temperature and Snow Threshold"
        top_plot.line('x', 'y', source=sources["temperature"], line_width=2, 
                     color="red", legend_label="Temperature")
        
        # Add TT threshold line
        tt_value = sliders['TT'].value
        threshold_line = top_plot.line(x=[dates[0],dates[-1:]], 
                                      y=[tt_value, tt_value], 
                                      line_width=2, line_dash="dashed", 
                                      color="gray", legend_label=f"TT Threshold")
        
        top_plot.yaxis.axis_label = "T (°C)"
        top_plot.legend.location = "top_right"
        
        # Bottom plot: Snow pack and liquid water
        bottom_plot.title.text = "Snow Pack and Liquid Water"
        bottom_plot.line('x', 'y', source=sources["snow"], line_width=2, 
                        color="blue", legend_label="Snow Pack")
        bottom_plot.line('x', 'y', source=sources["liquid_water"], line_width=2, 
                        color="lightblue", legend_label="Liquid Water")
        
        bottom_plot.yaxis.axis_label = "Water (mm)"
        bottom_plot.legend.location = "top_right"
        
        return TabPanel(child=column(top_plot, bottom_plot), title=tab_name.capitalize()), top_plot.x_range
        
    elif tab_name == "soil":
        # Top plot: Potential and Actual ET
        top_plot.title.text = "Evapotranspiration"
        top_plot.line('x', 'y', source=sources["potential_et"], line_width=2, 
                     color="orange", legend_label="Potential ET")
        top_plot.line('x', 'y', source=sources["actual_et"], line_width=2, 
                     color="green", legend_label="Actual ET")
        
        top_plot.yaxis.axis_label = "ET (mm/day)"
        top_plot.legend.location = "top_right"
        
        # Bottom plot: Soil moisture with FC threshold
        bottom_plot.title.text = "Soil Moisture"
        bottom_plot.line('x', 'y', source=sources["soil"], line_width=2, 
                        color="brown", legend_label="Soil Moisture")
        
        # Add FC threshold line
        fc_value = sliders['FC'].value
        fc_line = bottom_plot.line(x=[dates[0],dates[-1:]], 
                                  y=[fc_value, fc_value], 
                                  line_width=2, line_dash="dashed", 
                                  color="gray", legend_label=f"Field Capacity (FC)")
        
        bottom_plot.yaxis.axis_label = "Soil Moisture (mm)"
        bottom_plot.legend.location = "top_right"
        
        return TabPanel(child=column(top_plot, bottom_plot), title=tab_name.capitalize()), top_plot.x_range
        
    elif tab_name == "Flow Components":
        # Create a single, taller plot for flow components
        flow_plot = figure(x_axis_type="datetime", width=800, height=400,  # Double height
                          tools="pan,wheel_zoom,box_zoom,reset",
                          x_range=shared_x_range,
                          title="Flow Components")
        
        # Plot the stacked areas properly using varea for proper stacking
        # Baseflow (bottom layer, starts from zero)
        flow_plot.varea(x='x', y1=0, y2='baseflow_top', 
                       source=sources["stacked_flows"], 
                       fill_color="royalblue", 
                       fill_alpha=0.7,
                       legend_label="Baseflow")
        
        # Intermediate flow (middle layer, starts from top of baseflow)
        flow_plot.varea(x='x', y1='baseflow_top', y2='intermediate_top', 
                       source=sources["stacked_flows"], 
                       fill_color="darkorange", 
                       fill_alpha=0.7,
                       legend_label="Intermediate Flow")
        
        # Quick flow (top layer, starts from top of intermediate flow)
        flow_plot.varea(x='x', y1='intermediate_top', y2='total', 
                       source=sources["stacked_flows"], 
                       fill_color="tomato", 
                       fill_alpha=0.7,
                       legend_label="Quick Flow")
        
        # Add total discharge line
        flow_plot.line('x', 'total', 
                      source=sources["stacked_flows"], 
                      line_width=0.5, 
                      color="gray", 
                      #line_dash="dashed",
                      legend_label="Total Discharge")
        
        flow_plot.yaxis.axis_label = "Discharge (mm/day)"
        flow_plot.legend.location = "top_right"
        
        return TabPanel(child=flow_plot, title=tab_name.capitalize()), top_plot.x_range
        
    else:  # response tab
        # Top plot: Upper zone storage with UZL threshold
        top_plot.title.text = "Upper Tank Storage"
        top_plot.line('x', 'y', source=sources["upper_storage"], line_width=2, 
                     color="purple", legend_label="Upper Storage")
        
        # Add UZL threshold line
        uzl_value = sliders['UZL'].value
        uzl_line = top_plot.line(x=[dates[0], dates[-1:]], 
                                y=[uzl_value, uzl_value], 
                                line_width=2, line_dash="dashed", 
                                color="gray", legend_label=f"UZL Threshold")
        
        top_plot.yaxis.axis_label = "Storage (mm)"
        top_plot.legend.location = "top_right"
        
        # Bottom plot: Lower zone storage
        bottom_plot.title.text = "Lower Tank Storage"
        bottom_plot.line('x', 'y', source=sources["lower_storage"], line_width=2, 
                        color="darkblue", legend_label="Lower Storage")
        
        bottom_plot.yaxis.axis_label = "Storage (mm)"
        bottom_plot.legend.location = "top_right"
        
        return TabPanel(child=column(top_plot, bottom_plot), title=tab_name.capitalize()), top_plot.x_range

# Create the first tab and get its x_range to share with other tabs
first_tab, shared_x_range = create_tab("snow")

# Create remaining tabs using the shared x_range
second_tab, _ = create_tab("soil", shared_x_range)
third_tab, _ = create_tab("response", shared_x_range)
fourth_tab, _ = create_tab("Flow Components", shared_x_range)  # New tab for flow components

tabs = Tabs(tabs=[first_tab, second_tab, third_tab, fourth_tab])

# ==== Create a taller main flow plot at the bottom ====
main_flow_plot = figure(x_axis_type="datetime", width=800, height=350,
                       title=f"Simulated vs Observed Hydrographs ({metrics_text})",
                       tools="pan,wheel_zoom,box_zoom,reset,save",
                       x_range=shared_x_range)  # Link x-axes with tabs

# Plot simulated discharge
main_flow_plot.line('x', 'y', source=sources["flow"], line_width=3, color="crimson", 
                   legend_label="Simulated Flow")

# Add observed discharge (always plot, it will just be zeros if no observed data)
main_flow_plot.line('x', 'y', source=sources["observed_flow"], line_width=2, 
                   color="blue", legend_label="Observed Flow", line_dash="dashed")

main_flow_plot.yaxis.axis_label = "Discharge (mm/day)"
main_flow_plot.legend.location = "top_left"
main_flow_plot.grid.grid_line_alpha = 0.3

# ==== FINAL LAYOUT WITH ALWAYS-VISIBLE SCROLLBAR ====
# Create the title DIV (fixed, outside scroll area)
title_div = Div(
    text="<h2 style='text-align: left; margin-bottom: 10px; padding-left: 18px;'>HBV Model Parameters</h2>",
    width=320,
    styles={'margin-left': '0px'}
)

# Create the scrollable content panel (just sliders now)
scrollable_content = column(
    *slider_groups_with_spacers,
    width=310,  # Narrower to account for scrollbar
    height=790,  # Reduced height since button is outside
    sizing_mode="inherit",
    styles={
        'overflow-y': 'scroll',
        'overflow-x': 'hidden',
        'padding-left': '25px',  # Space between scrollbar and content
        'padding-right': '25px', 
        'direction': 'rtl',  # Scrollbar on left
    },
    stylesheets=[
        """
        :host {
            scrollbar-width: thin;
            scrollbar-color: #c1c1c1 #f1f1f1;
        }
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        
        /* Content alignment */
        .bk-panel-models-column-Column {
            direction: ltr;
            text-align: left;
        }
        
        /* Slider titles and values */
        .bk-slider-title {
            text-align: left !important;
            padding-left: 5px !important;
            margin-left: 0px !important;
        }
        .bk-slider-value {
            text-align: left !important;
            margin-left: 0px !important;
            padding-left: 5px !important;
        }
        
        /* Parameter group headers */
        .bk-Div {
            text-align: left !important;
            padding-left: 5px !important;
        }
        
        /* Slider widget alignment */
        .bk-input-group {
            margin-left: 0px !important;
            padding-left: 5px !important;
        }
        """
    ]
)

# Create Calibrate button (now outside scroll area)
calibrate_button = Button(
    label="Calibrate Model", 
    button_type="success",
    width=270,
    styles={'margin-left': '18px', 'margin-top': '10px'}
)
# Connect the button to the calibrate function
calibrate_button.on_click(calibrate)

# Combine all elements
parameters_panel = column(
    title_div,
    scrollable_content,
    calibrate_button,  # Now below scroll area
    width=320,
    sizing_mode="fixed",
    styles={'margin-right': '10px'}
)

# Adjust right panel spacing
right_panel = column(
    Div(text="<h2 style='text-align: center'>Processes Output</h2>"), 
    tabs,
    Div(text="<h3 style='text-align: center'>Hydrographs</h3>"),
    main_flow_plot,
    sizing_mode="fixed",
    styles={'margin-left': '10px'}
)
layout = row(
    parameters_panel, 
    right_panel,
    sizing_mode="fixed"
)

# Add to document
curdoc().add_root(layout)
curdoc().title = "HBV Model Dashboard"