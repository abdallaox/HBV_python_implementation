from bokeh.layouts import column, row
from bokeh.models import Slider, Button, ColumnDataSource, Div, Toggle
from bokeh.models import TabPanel, Tabs
from bokeh.plotting import figure
from bokeh.io import curdoc
import numpy as np
from datetime import datetime, timedelta
from HBV_model import HBVModel

# ==== Load the pre-calibrated model ====
model = HBVModel.load_model('./models/model_calibrated')
dates = model.data.Date
params = model.params
initial_results = model.run()

# Get initial performance metrics
metrics_text = ""
if hasattr(model, 'performance_metrics'):
    metrics = model.performance_metrics
    metrics_text = f"NSE: {metrics.get('NSE', 'N/A'):.3f} | KGE: {metrics.get('KGE', 'N/A'):.3f} | PBIAS: {metrics.get('PBIAS', 'N/A'):.2f}%"

# ==== Initialize data sources ====
sources = {
    "flow": ColumnDataSource(data=dict(x=dates, y=initial_results['discharge'])),
    "observed_flow": ColumnDataSource(data=dict(
        x=dates, 
        y=initial_results.get('observed_q', np.zeros_like(initial_results['discharge']))
    )),
    "snow": ColumnDataSource(data=dict(x=dates, y=initial_results['snowpack'])),
    "liquid_water": ColumnDataSource(data=dict(x=dates, y=initial_results['liquid_water'])),
    "temperature": ColumnDataSource(data=dict(x=dates, y=initial_results['temperature'])),
    "soil": ColumnDataSource(data=dict(x=dates, y=initial_results['soil_moisture'])),
    "potential_et": ColumnDataSource(data=dict(x=dates, y=initial_results['potential_et'])),
    "actual_et": ColumnDataSource(data=dict(x=dates, y=initial_results['actual_et'])),
    "upper_storage": ColumnDataSource(data=dict(x=dates, y=initial_results['upper_storage'])),
    "lower_storage": ColumnDataSource(data=dict(x=dates, y=initial_results['lower_storage'])),
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
    "stacked_flows": ColumnDataSource(data=dict(
        x=dates,
        baseflow=initial_results.get('baseflow', np.zeros_like(initial_results['discharge'])),
        intermediate=initial_results.get('intermediate_flow', np.zeros_like(initial_results['discharge'])),
        quick=initial_results.get('quick_flow', np.zeros_like(initial_results['discharge'])),
        baseflow_top=initial_results.get('baseflow', np.zeros_like(initial_results['discharge'])),
        intermediate_top=initial_results.get('baseflow', np.zeros_like(initial_results['discharge'])) + 
                        initial_results.get('intermediate_flow', np.zeros_like(initial_results['discharge'])),
        total=initial_results['discharge']
    ))
}

# Threshold sources
threshold_sources = {
    "tt_threshold": ColumnDataSource(data=dict(
        x=[dates.iloc[0], dates.iloc[-1]],
        y=[params['snow']['TT']['default'], params['snow']['TT']['default']]
    )),
    "fc_threshold": ColumnDataSource(data=dict(
        x=[dates.iloc[0], dates.iloc[-1]],
        y=[params['soil']['FC']['default'], params['soil']['FC']['default']]
    )),
    "uzl_threshold": ColumnDataSource(data=dict(
        x=[dates.iloc[0], dates.iloc[-1]],
        y=[params['response']['UZL']['default'], params['response']['UZL']['default']]
    ))
}

metrics_source = ColumnDataSource(data=dict(text=[metrics_text]))

# ==== Create Parameter Sliders with Minimal Styling ====
sliders = {}
slider_groups = []

# Minimal icons
group_icons = {
    'snow': '❄',
    'soil': '🌱',
    'response': '💧'
}

for group_name, group_params in params.items():
    icon = group_icons.get(group_name, '·')
    header = Div(text=f"""
        <div style='
            color: #1f1f1f;
            padding: 12px 0 8px 0;
            margin: 16px 0 8px 0;
            font-size: 13px;
            font-weight: 500;
            border-bottom: 1px solid #e5e5e5;
            letter-spacing: 0.3px;
        '>
            <span style='margin-right: 6px;'>{icon}</span>
            {group_name.upper()}
        </div>
    """, width=280)
    
    group_sliders = []
    for key, meta in group_params.items():
        slider = Slider(
            title=key, 
            start=meta['min'], 
            end=meta['max'], 
            value=meta['default'], 
            step=0.001,
            format="0.000",
            tooltips=True,
            width=280,
            styles={'margin': '6px 0'}
        )
        sliders[key] = slider
        group_sliders.append(slider)
    
    slider_groups.append(column(header, *group_sliders, spacing=2))

# Spacers
top_spacer = Div(text="<div style='height: 16px;'></div>", width=300)
bottom_spacer = Div(text="<div style='height: 16px;'></div>", width=300)

slider_groups_with_spacers = [top_spacer] + slider_groups + [bottom_spacer]

# ==== Update Function ====
def update_plot(attr, old, new):
    temp_model = HBVModel()
    temp_model.data = model.data
    temp_model.column_names = model.column_names
    
    updated_params = {'snow': {}, 'soil': {}, 'response': {}}
    for group_name, group_params in params.items():
        for key in group_params:
            updated_params[group_name][key] = {
                'min': params[group_name][key]['min'],
                'max': params[group_name][key]['max'],
                'default': sliders[key].value
            }
    
    temp_model.params = updated_params
    results = temp_model.run()
    
    # Calculate and update metrics
    metrics_text = ""
    if hasattr(temp_model, 'performance_metrics'):
        metrics = temp_model.performance_metrics
        metrics_text = f"NSE: {metrics.get('NSE', 'N/A'):.3f} | KGE: {metrics.get('KGE', 'N/A'):.3f} | PBIAS: {metrics.get('PBIAS', 'N/A'):.2f}%"
    
    # Update the metrics display
    metrics_display.text = f"""
        <div style='
            background: #f9f9f9;
            padding: 10px 16px;
            border-radius: 6px;
            margin: 0 0 16px 0;
            border: 1px solid #e5e5e5;
        '>
            <div style='font-size: 11px; color: #666; margin-bottom: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;'>
                Performance Metrics
            </div>
            <div style='font-size: 13px; color: #1f1f1f; font-weight: 400; font-family: monospace;'>
                {metrics_text}
            </div>
        </div>
    """
    
    # Update thresholds
    threshold_sources["tt_threshold"].data = {
        'x': [dates.iloc[0], dates.iloc[-1]],
        'y': [sliders['TT'].value, sliders['TT'].value]
    }
    threshold_sources["fc_threshold"].data = {
        'x': [dates.iloc[0], dates.iloc[-1]],
        'y': [sliders['FC'].value, sliders['FC'].value]
    }
    threshold_sources["uzl_threshold"].data = {
        'x': [dates.iloc[0], dates.iloc[-1]],
        'y': [sliders['UZL'].value, sliders['UZL'].value]
    }
    
    baseflow = results.get('baseflow', np.zeros_like(results['discharge']))
    intermediate_flow = results.get('intermediate_flow', np.zeros_like(results['discharge']))
    quick_flow = results.get('quick_flow', np.zeros_like(results['discharge']))
    
    # Update all sources
    sources["flow"].data = {'x': dates, 'y': results['discharge']}
    sources["snow"].data = {'x': dates, 'y': results['snowpack']}
    sources["liquid_water"].data = {'x': dates, 'y': results['liquid_water']}
    sources["temperature"].data = {'x': dates, 'y': results['temperature']}
    sources["soil"].data = {'x': dates, 'y': results['soil_moisture']}
    sources["potential_et"].data = {'x': dates, 'y': results['potential_et']}
    sources["actual_et"].data = {'x': dates, 'y': results['actual_et']}
    sources["upper_storage"].data = {'x': dates, 'y': results['upper_storage']}
    sources["lower_storage"].data = {'x': dates, 'y': results['lower_storage']}
    sources["quick_flow"].data = {'x': dates, 'y': quick_flow}
    sources["intermediate_flow"].data = {'x': dates, 'y': intermediate_flow}
    sources["baseflow"].data = {'x': dates, 'y': baseflow}
    
    sources["stacked_flows"].data = {
        'x': dates,
        'baseflow': baseflow,
        'intermediate': intermediate_flow,
        'quick': quick_flow,
        'baseflow_top': baseflow,
        'intermediate_top': baseflow + intermediate_flow,
        'total': results['discharge']
    }

for s in sliders.values():
    s.on_change('value_throttled', update_plot)

# ==== Calibration Button ====
def calibrate():
    for group_name, group_params in model.params.items():
        for key, meta in group_params.items():
            if key in sliders:
                sliders[key].value = meta['default']
    update_plot('value', None, None)

# ==== Create Minimal Plots ====
def create_plot(width=850, height=200, shared_x_range=None):
    """Helper to create minimal styled plots"""
    if shared_x_range is None:
        p = figure(x_axis_type="datetime", width=width, height=height,
                  tools="pan,wheel_zoom,box_zoom,reset,save",
                  toolbar_location="above")
    else:
        p = figure(x_axis_type="datetime", width=width, height=height,
                  tools="pan,wheel_zoom,box_zoom,reset,save",
                  x_range=shared_x_range,
                  toolbar_location="above")
    
    # Minimal styling
    p.background_fill_color = "#ffffff"
    p.border_fill_color = "#ffffff"
    p.outline_line_color = "#e5e5e5"
    p.outline_line_width = 1
    p.grid.grid_line_color = "#f0f0f0"
    p.grid.grid_line_alpha = 1
    p.grid.grid_line_width = 1
    
    return p

def create_tab(tab_name, shared_x_range=None):
    """Create minimal tabs"""
    top_plot = create_plot(shared_x_range=shared_x_range)
    
    if tab_name != "Flow Components":
        bottom_plot = create_plot(shared_x_range=top_plot.x_range)
    
    if tab_name == "snow":
        top_plot.title.text = "Temperature & Snow Threshold"
        top_plot.line('x', 'y', source=sources["temperature"], line_width=2, 
                     color="#ef4444", legend_label="Temperature", alpha=0.85)
        top_plot.line('x', 'y', source=threshold_sources["tt_threshold"], 
                     line_width=1.5, line_dash="dashed", color="#a1a1aa", 
                     legend_label="TT Threshold", alpha=0.7)
        top_plot.yaxis.axis_label = "Temperature (°C)"
        top_plot.legend.location = "top_right"
        top_plot.legend.background_fill_alpha = 0.9
        top_plot.legend.border_line_color = "#e5e5e5"
        
        bottom_plot.title.text = "Snow Pack & Liquid Water"
        bottom_plot.line('x', 'y', source=sources["snow"], line_width=2, 
                        color="#3b82f6", legend_label="Snow Pack", alpha=0.85)
        bottom_plot.line('x', 'y', source=sources["liquid_water"], line_width=2, 
                        color="#93c5fd", legend_label="Liquid Water", alpha=0.85)
        bottom_plot.yaxis.axis_label = "Water Equivalent (mm)"
        bottom_plot.legend.location = "top_right"
        bottom_plot.legend.background_fill_alpha = 0.9
        bottom_plot.legend.border_line_color = "#e5e5e5"
        
        return TabPanel(child=column(top_plot, bottom_plot, spacing=12), 
                       title="Snow"), top_plot.x_range
        
    elif tab_name == "soil":
        top_plot.title.text = "Evapotranspiration"
        top_plot.line('x', 'y', source=sources["potential_et"], line_width=2, 
                     color="#f97316", legend_label="Potential ET", alpha=0.85)
        top_plot.line('x', 'y', source=sources["actual_et"], line_width=2, 
                     color="#22c55e", legend_label="Actual ET", alpha=0.85)
        top_plot.yaxis.axis_label = "ET (mm/day)"
        top_plot.legend.location = "top_right"
        top_plot.legend.background_fill_alpha = 0.9
        top_plot.legend.border_line_color = "#e5e5e5"
        
        bottom_plot.title.text = "Soil Moisture"
        bottom_plot.line('x', 'y', source=sources["soil"], line_width=2, 
                        color="#92400e", legend_label="Soil Moisture", alpha=0.85)
        bottom_plot.line('x', 'y', source=threshold_sources["fc_threshold"], 
                        line_width=1.5, line_dash="dashed", color="#a1a1aa", 
                        legend_label="Field Capacity", alpha=0.7)
        bottom_plot.yaxis.axis_label = "Soil Moisture (mm)"
        bottom_plot.legend.location = "top_right"
        bottom_plot.legend.background_fill_alpha = 0.9
        bottom_plot.legend.border_line_color = "#e5e5e5"
        
        return TabPanel(child=column(top_plot, bottom_plot, spacing=12), 
                       title="Soil"), top_plot.x_range
        
    elif tab_name == "Flow Components":
        flow_plot = create_plot(height=420, shared_x_range=shared_x_range)
        flow_plot.title.text = "Flow Component Breakdown"
        
        flow_plot.varea(x='x', y1=0, y2='baseflow_top', 
                       source=sources["stacked_flows"], 
                       fill_color="#3b82f6", fill_alpha=0.6,
                       legend_label="Baseflow")
        flow_plot.varea(x='x', y1='baseflow_top', y2='intermediate_top', 
                       source=sources["stacked_flows"], 
                       fill_color="#f97316", fill_alpha=0.6,
                       legend_label="Intermediate")
        flow_plot.varea(x='x', y1='intermediate_top', y2='total', 
                       source=sources["stacked_flows"], 
                       fill_color="#ef4444", fill_alpha=0.6,
                       legend_label="Quick Flow")
        flow_plot.line('x', 'total', source=sources["stacked_flows"], 
                      line_width=1.5, color="#52525b", alpha=0.7,
                      legend_label="Total")
        
        flow_plot.yaxis.axis_label = "Discharge (mm/day)"
        flow_plot.legend.location = "top_right"
        flow_plot.legend.background_fill_alpha = 0.9
        flow_plot.legend.border_line_color = "#e5e5e5"
        
        return TabPanel(child=flow_plot, title="Flow"), top_plot.x_range
        
    else:  # response
        top_plot.title.text = "Upper Zone Storage"
        top_plot.line('x', 'y', source=sources["upper_storage"], line_width=2, 
                     color="#a855f7", legend_label="Upper Storage", alpha=0.85)
        top_plot.line('x', 'y', source=threshold_sources["uzl_threshold"], 
                     line_width=1.5, line_dash="dashed", color="#a1a1aa", 
                     legend_label="UZL Threshold", alpha=0.7)
        top_plot.yaxis.axis_label = "Storage (mm)"
        top_plot.legend.location = "top_right"
        top_plot.legend.background_fill_alpha = 0.9
        top_plot.legend.border_line_color = "#e5e5e5"
        
        bottom_plot.title.text = "Lower Zone Storage"
        bottom_plot.line('x', 'y', source=sources["lower_storage"], line_width=2, 
                        color="#14b8a6", legend_label="Lower Storage", alpha=0.85)
        bottom_plot.yaxis.axis_label = "Storage (mm)"
        bottom_plot.legend.location = "top_right"
        bottom_plot.legend.background_fill_alpha = 0.9
        bottom_plot.legend.border_line_color = "#e5e5e5"
        
        return TabPanel(child=column(top_plot, bottom_plot, spacing=12), 
                       title="Response"), top_plot.x_range

# Create tabs
first_tab, shared_x_range = create_tab("snow")
second_tab, _ = create_tab("soil", shared_x_range)
third_tab, _ = create_tab("response", shared_x_range)
fourth_tab, _ = create_tab("Flow Components", shared_x_range)

tabs = Tabs(tabs=[first_tab, second_tab, third_tab, fourth_tab])

# ==== Main Hydrograph Plot ====
main_flow_plot = create_plot(height=380, shared_x_range=shared_x_range)
main_flow_plot.title.text = "Simulated vs Observed Discharge"

main_flow_plot.line('x', 'y', source=sources["flow"], line_width=2.5, 
                   color="#ef4444", legend_label="Simulated", alpha=0.85)
main_flow_plot.line('x', 'y', source=sources["observed_flow"], line_width=2, 
                   color="#3b82f6", legend_label="Observed", line_dash="dashed", alpha=0.75)

main_flow_plot.yaxis.axis_label = "Discharge (mm/day)"
main_flow_plot.legend.location = "top_left"
main_flow_plot.legend.background_fill_alpha = 0.9
main_flow_plot.legend.border_line_color = "#e5e5e5"

# ==== Minimal Layout ====
title_div = Div(text="""
    <div style='
        color: #1f1f1f;
        padding: 16px 20px;
        margin: 0;
        border-bottom: 1px solid #e5e5e5;
        background: #ffffff;
    '>
        <h2 style='margin: 0; font-size: 15px; font-weight: 500; letter-spacing: 0.3px;'>
            Model Parameters
        </h2>
    </div>
""", width=320)

# Metrics display (minimal design)
metrics_display = Div(text=f"""
    <div style='
        background: #f9f9f9;
        padding: 10px 16px;
        border-radius: 6px;
        margin: 0 0 16px 0;
        border: 1px solid #e5e5e5;
    '>
        <div style='font-size: 11px; color: #666; margin-bottom: 4px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;'>
            Performance Metrics
        </div>
        <div style='font-size: 13px; color: #1f1f1f; font-weight: 400; font-family: monospace;'>
            {metrics_text}
        </div>
    </div>
""", width=850)

scrollable_content = column(
    *slider_groups_with_spacers,
    width=310,
    height=725,
    sizing_mode="inherit",
    styles={
        'overflow-y': 'scroll',
        'overflow-x': 'hidden',
        'padding': '0 20px',
        'direction': 'rtl',
        'background': '#ffffff',
    },
    stylesheets=["""
        :host {
            scrollbar-width: thin;
            scrollbar-color: #d4d4d8 #fafafa;
        }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { 
            background: #fafafa; 
        }
        ::-webkit-scrollbar-thumb { 
            background: #d4d4d8; 
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover { background: #a1a1aa; }
        .bk-panel-models-column-Column { direction: ltr; }
    """]
)

calibrate_button = Button(
    label="Restore Calibrated Values", 
    button_type="light",
    width=280,
    height=36,
    styles={
        'margin': '12px 20px',
        'font-size': '13px',
        'font-weight': '400',
    }
)
calibrate_button.on_click(calibrate)

parameters_panel = column(
    title_div,
    scrollable_content,
    calibrate_button,
    width=320,
    sizing_mode="fixed",
    styles={
        'margin-right': '20px',
        'background': '#ffffff',
        'border': '1px solid #e5e5e5',
        'border-radius': '8px',
    }
)

# Minimal toggle button
processes_toggle = Toggle(
    label="Hide Details", 
    button_type="light",
    width=100,
    height=32,
    active=True,
    styles={
        'font-size': '13px',
        'font-weight': '400',
        'margin-bottom': '16px',
    }
)

processes_header = Div(text="""
    <div style='
        color: #1f1f1f;
        padding: 0;
        margin: 0 0 16px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    '>
        <h2 style='margin: 0; font-size: 15px; font-weight: 500; letter-spacing: 0.3px;'>
            Model Outputs
        </h2>
    </div>
""", width=750)

# Collapsible content container
collapsible_content = column(
    tabs,
    spacing=0,
    visible=True,
    name="collapsible_section"
)

# Toggle callback
def toggle_callback(attr, old, new):
    collapsible_content.visible = new
    if new:
        processes_toggle.label = "Hide Details"
    else:
        processes_toggle.label = "Show Details"

processes_toggle.on_change('active', toggle_callback)

# Header row with toggle
header_row = row(processes_header, processes_toggle, spacing=10)

right_panel = column(
    header_row,
    collapsible_content,
    metrics_display,
    main_flow_plot,
    sizing_mode="fixed",
    styles={'margin-left': '0px'}
)

layout = row(
    parameters_panel, 
    right_panel,
    sizing_mode="fixed",
    styles={
        'padding': '24px',
        'background': '#fafafa',
    }
)

curdoc().add_root(layout)
curdoc().title = "HBV Model Dashboard"