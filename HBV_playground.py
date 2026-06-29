from bokeh.layouts import column, row
from bokeh.models import (
    Slider, Button, ColumnDataSource, Div, Toggle,
    LinearAxis, Range1d, TabPanel, Tabs, Title
)
from bokeh.plotting import figure
from bokeh.io import curdoc
import numpy as np
from HBV_Lab import HBVModel

# ── Load pre-calibrated model ─────────────────────────────────────────────────
model = HBVModel.load_model('./models/model_calibrated')
dates  = model.data.Date
params = model.params
initial_results = model.run()

# ── Plot factory ──────────────────────────────────────────────────────────────
def make_plot(width=860, height=210, shared_x_range=None):
    kwargs = dict(
        x_axis_type="datetime", width=width, height=height,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above",
    )
    if shared_x_range is not None:
        kwargs["x_range"] = shared_x_range
    p = figure(**kwargs)
    p.background_fill_color = "#ffffff"
    p.border_fill_color     = "#ffffff"
    p.outline_line_color    = "#e2e8f0"
    p.outline_line_width    = 1
    p.grid.grid_line_color  = "#f1f5f9"
    p.grid.grid_line_width  = 1
    p.axis.axis_label_text_font_size  = "11px"
    p.axis.major_label_text_font_size = "10px"
    p.title.text_font_size  = "12px"
    p.title.text_font_style = "normal"
    p.title.text_color      = "#334155"
    return p

# ── Data sources ──────────────────────────────────────────────────────────────
precip_vals = initial_results['precipitation']
zeros       = np.zeros_like(precip_vals)

sources = {
    "flow":             ColumnDataSource(dict(x=dates, y=initial_results['discharge'])),
    "observed_flow":    ColumnDataSource(dict(x=dates, y=initial_results.get('observed_q', zeros))),
    "precipitation":    ColumnDataSource(dict(x=dates, top=precip_vals)),
    "snow":             ColumnDataSource(dict(x=dates, y=initial_results['snowpack'])),
    "liquid_water":     ColumnDataSource(dict(x=dates, y=initial_results['liquid_water'])),
    "temperature":      ColumnDataSource(dict(x=dates, y=initial_results['temperature'])),
    "soil":             ColumnDataSource(dict(x=dates, y=initial_results['soil_moisture'])),
    "potential_et":     ColumnDataSource(dict(x=dates, y=initial_results['potential_et'])),
    "actual_et":        ColumnDataSource(dict(x=dates, y=initial_results['actual_et'])),
    "upper_storage":    ColumnDataSource(dict(x=dates, y=initial_results['upper_storage'])),
    "lower_storage":    ColumnDataSource(dict(x=dates, y=initial_results['lower_storage'])),
    "quick_flow":       ColumnDataSource(dict(x=dates, y=initial_results.get('quick_flow',        zeros))),
    "intermediate_flow":ColumnDataSource(dict(x=dates, y=initial_results.get('intermediate_flow', zeros))),
    "baseflow":         ColumnDataSource(dict(x=dates, y=initial_results.get('baseflow',          zeros))),
    "stacked_flows":    ColumnDataSource(dict(
        x=dates,
        baseflow         = initial_results.get('baseflow',          zeros),
        intermediate     = initial_results.get('intermediate_flow', zeros),
        quick            = initial_results.get('quick_flow',        zeros),
        baseflow_top     = initial_results.get('baseflow',          zeros),
        intermediate_top = (initial_results.get('baseflow', zeros) +
                            initial_results.get('intermediate_flow', zeros)),
        total            = initial_results['discharge'],
    )),
}

threshold_sources = {
    "tt_threshold":  ColumnDataSource(dict(x=[dates.iloc[0], dates.iloc[-1]], y=[params['snow']['TT']['default']]     * 2)),
    "fc_threshold":  ColumnDataSource(dict(x=[dates.iloc[0], dates.iloc[-1]], y=[params['soil']['FC']['default']]     * 2)),
    "uzl_threshold": ColumnDataSource(dict(x=[dates.iloc[0], dates.iloc[-1]], y=[params['response']['UZL']['default']]* 2)),
}

# ── Parameter sliders (explicit order: snow → soil → response) ────────────────
GROUP_ORDER = ['snow', 'soil', 'response']
ICONS       = {'snow': '❄', 'soil': '🌱', 'response': '💧'}

sliders       = {}
slider_groups = []

for group_name in GROUP_ORDER:
    group_params = params[group_name]
    icon = ICONS[group_name]
    header = Div(
        text=f"""
        <div style='color:#1e293b; padding:10px 0 6px; margin:12px 0 4px;
                    font-size:11px; font-weight:600; letter-spacing:0.5px;
                    text-transform:uppercase; border-bottom:1px solid #e2e8f0;'>
            <span style='margin-right:5px;'>{icon}</span>{group_name}
        </div>""",
        width=264,
    )
    group_sliders = []
    for key, meta in group_params.items():
        s = Slider(
            title=key,
            start=meta['min'], end=meta['max'], value=meta['default'],
            step=0.001, format="0.000",
            tooltips=True, width=264,
            styles={'margin': '3px 0'},
        )
        sliders[key] = s
        group_sliders.append(s)
    slider_groups.append(column(header, *group_sliders, spacing=1))

# ── Update callback ───────────────────────────────────────────────────────────
def fmt_metrics(pm):
    if not pm:
        return ""
    nse   = pm.get('NSE',   float('nan'))
    kge   = pm.get('KGE',   float('nan'))
    pbias = pm.get('PBIAS', float('nan'))
    return f"NSE {nse:.3f}  ·  KGE {kge:.3f}  ·  PBIAS {pbias:+.1f}%"


def update_plot(attr, old, new):
    temp = HBVModel()
    temp.data         = model.data
    temp.column_names = model.column_names
    temp.warmup_end   = model.warmup_end

    up = {g: {} for g in GROUP_ORDER}
    for g in GROUP_ORDER:
        for key, meta in params[g].items():
            up[g][key] = {'min': meta['min'], 'max': meta['max'], 'default': sliders[key].value}
    temp.params = up
    res = temp.run(verbose=False)

    pm = getattr(temp, 'performance_metrics', {})
    metrics_title.text = fmt_metrics(pm)

    threshold_sources["tt_threshold"].data  = {'x': [dates.iloc[0], dates.iloc[-1]], 'y': [sliders['TT'].value]  * 2}
    threshold_sources["fc_threshold"].data  = {'x': [dates.iloc[0], dates.iloc[-1]], 'y': [sliders['FC'].value]  * 2}
    threshold_sources["uzl_threshold"].data = {'x': [dates.iloc[0], dates.iloc[-1]], 'y': [sliders['UZL'].value] * 2}

    bf = res.get('baseflow',          np.zeros_like(res['discharge']))
    mf = res.get('intermediate_flow', np.zeros_like(res['discharge']))
    qf = res.get('quick_flow',        np.zeros_like(res['discharge']))

    sources["flow"].data              = {'x': dates, 'y': res['discharge']}
    sources["snow"].data              = {'x': dates, 'y': res['snowpack']}
    sources["liquid_water"].data      = {'x': dates, 'y': res['liquid_water']}
    sources["temperature"].data       = {'x': dates, 'y': res['temperature']}
    sources["soil"].data              = {'x': dates, 'y': res['soil_moisture']}
    sources["potential_et"].data      = {'x': dates, 'y': res['potential_et']}
    sources["actual_et"].data         = {'x': dates, 'y': res['actual_et']}
    sources["upper_storage"].data     = {'x': dates, 'y': res['upper_storage']}
    sources["lower_storage"].data     = {'x': dates, 'y': res['lower_storage']}
    sources["quick_flow"].data        = {'x': dates, 'y': qf}
    sources["intermediate_flow"].data = {'x': dates, 'y': mf}
    sources["baseflow"].data          = {'x': dates, 'y': bf}
    sources["stacked_flows"].data     = {
        'x': dates, 'baseflow': bf, 'intermediate': mf, 'quick': qf,
        'baseflow_top': bf, 'intermediate_top': bf + mf, 'total': res['discharge'],
    }

for s in sliders.values():
    s.on_change('value_throttled', update_plot)


def restore_calibrated():
    for g in GROUP_ORDER:
        for key, meta in params[g].items():
            if key in sliders:
                sliders[key].value = meta['default']
    update_plot('value', None, None)


# ── Detail tabs ───────────────────────────────────────────────────────────────
leg = dict(location="top_right", background_fill_alpha=0.9,
           border_line_color="#e2e8f0", label_text_font_size="10px")

def build_tab(name, shared_x_range=None):
    top = make_plot(shared_x_range=shared_x_range)
    if name != "flow":
        bot = make_plot(shared_x_range=top.x_range)

    if name == "snow":
        top.title.text = "Temperature & snow-threshold (TT)"
        top.line('x', 'y', source=sources["temperature"],
                 line_width=2, color="#ef4444", legend_label="Temperature", alpha=0.85)
        top.line('x', 'y', source=threshold_sources["tt_threshold"],
                 line_width=1.5, line_dash="dashed", color="#94a3b8",
                 legend_label="TT threshold", alpha=0.8)
        top.yaxis.axis_label = "Temperature (°C)"
        top.legend.update(**leg)

        bot.title.text = "Snowpack & liquid water in snow"
        bot.line('x', 'y', source=sources["snow"],
                 line_width=2, color="#3b82f6", legend_label="Snowpack", alpha=0.85)
        bot.line('x', 'y', source=sources["liquid_water"],
                 line_width=2, color="#93c5fd", legend_label="Liquid water", alpha=0.85)
        bot.yaxis.axis_label = "Water equivalent (mm)"
        bot.legend.update(**leg)
        return TabPanel(child=column(top, bot, spacing=10), title="Snow"), top.x_range

    elif name == "soil":
        top.title.text = "Potential vs actual evapotranspiration"
        top.line('x', 'y', source=sources["potential_et"],
                 line_width=2, color="#f97316", legend_label="Potential ET", alpha=0.85)
        top.line('x', 'y', source=sources["actual_et"],
                 line_width=2, color="#22c55e", legend_label="Actual ET", alpha=0.85)
        top.yaxis.axis_label = "ET (mm/day)"
        top.legend.update(**leg)

        bot.title.text = "Soil moisture & field capacity (FC)"
        bot.line('x', 'y', source=sources["soil"],
                 line_width=2, color="#92400e", legend_label="Soil moisture", alpha=0.85)
        bot.line('x', 'y', source=threshold_sources["fc_threshold"],
                 line_width=1.5, line_dash="dashed", color="#94a3b8",
                 legend_label="Field capacity (FC)", alpha=0.8)
        bot.yaxis.axis_label = "Soil moisture (mm)"
        bot.legend.update(**leg)
        return TabPanel(child=column(top, bot, spacing=10), title="Soil"), top.x_range

    elif name == "response":
        top.title.text = "Upper zone storage & quick-flow threshold (UZL)"
        top.line('x', 'y', source=sources["upper_storage"],
                 line_width=2, color="#a855f7", legend_label="Upper storage", alpha=0.85)
        top.line('x', 'y', source=threshold_sources["uzl_threshold"],
                 line_width=1.5, line_dash="dashed", color="#94a3b8",
                 legend_label="UZL threshold", alpha=0.8)
        top.yaxis.axis_label = "Storage (mm)"
        top.legend.update(**leg)

        bot.title.text = "Lower zone storage (baseflow reservoir)"
        bot.line('x', 'y', source=sources["lower_storage"],
                 line_width=2, color="#14b8a6", legend_label="Lower storage", alpha=0.85)
        bot.yaxis.axis_label = "Storage (mm)"
        bot.legend.update(**leg)
        return TabPanel(child=column(top, bot, spacing=10), title="Response"), top.x_range

    else:  # flow components
        p = make_plot(height=430, shared_x_range=shared_x_range)
        p.title.text = "Runoff components (stacked)"
        p.varea(x='x', y1=0,              y2='baseflow_top',     source=sources["stacked_flows"],
                fill_color="#3b82f6", fill_alpha=0.55, legend_label="Baseflow")
        p.varea(x='x', y1='baseflow_top', y2='intermediate_top', source=sources["stacked_flows"],
                fill_color="#f97316", fill_alpha=0.55, legend_label="Intermediate")
        p.varea(x='x', y1='intermediate_top', y2='total',        source=sources["stacked_flows"],
                fill_color="#ef4444", fill_alpha=0.55, legend_label="Quick flow")
        p.line('x', 'total', source=sources["stacked_flows"],
               line_width=1.5, color="#334155", alpha=0.6, legend_label="Total")
        p.yaxis.axis_label = "Discharge (mm/day)"
        p.legend.update(**leg)
        return TabPanel(child=p, title="Flow components"), top.x_range


tab_snow,     shared_x = build_tab("snow")
tab_soil,     _        = build_tab("soil",     shared_x)
tab_response, _        = build_tab("response", shared_x)
tab_flow,     _        = build_tab("flow",     shared_x)
tabs = Tabs(tabs=[tab_snow, tab_soil, tab_response, tab_flow])

# ── Main hydrograph ───────────────────────────────────────────────────────────
main_plot = make_plot(height=420, shared_x_range=shared_x)
main_plot.title.text  = "Simulated vs Observed Discharge"
main_plot.title.align = "left"

# Metrics as a right-aligned subtitle inside the figure
initial_pm     = getattr(model, 'performance_metrics', {})
metrics_title  = Title(
    text           = fmt_metrics(initial_pm),
    text_font_size = "11px",
    text_color     = "#64748b",
    text_font_style= "normal",
    align          = "right",
)
main_plot.add_layout(metrics_title, 'above')

# Precipitation — inverted bars, secondary right axis
max_precip = float(np.nanmax(precip_vals)) if len(precip_vals) > 0 else 10.0
main_plot.extra_y_ranges = {"precip": Range1d(start=max_precip * 3.5, end=0)}
main_plot.add_layout(LinearAxis(
    y_range_name="precip",
    axis_label="Precipitation (mm)",
    axis_label_text_color="#94a3b8",
    major_label_text_color="#94a3b8",
    axis_line_color="#e2e8f0",
    major_tick_line_color="#e2e8f0",
    minor_tick_line_color=None,
), 'right')
main_plot.vbar(
    x='x', top='top', source=sources["precipitation"],
    width=86400000 * 0.8, y_range_name="precip",
    fill_color="#bfdbfe", line_color=None, alpha=0.5,
    legend_label="Precipitation",
)

# Observed — solid, prominent dark blue
main_plot.line('x', 'y', source=sources["observed_flow"],
               line_width=3, color="#1d4ed8",
               legend_label="Observed", alpha=0.9)

# Simulated — solid red, slightly thinner so observed reads first
main_plot.line('x', 'y', source=sources["flow"],
               line_width=2, color="#ef4444",
               legend_label="Simulated", alpha=0.85)

main_plot.yaxis.axis_label             = "Discharge (mm/day)"
main_plot.legend.location              = "top_left"
main_plot.legend.background_fill_alpha = 0.9
main_plot.legend.border_line_color     = "#e2e8f0"
main_plot.legend.label_text_font_size  = "10px"

# ── Left panel ────────────────────────────────────────────────────────────────
# Heights: right panel ≈ header(68) + toggle-row(38) + tabs(460) + gap(8) + main_plot(420) = ~994px
# Left panel: header(58) + scrollable(880) + button(54) ≈ 992px
SCROLL_H   = 880
TOP_SPACER = 320   # blank space above and below the sliders

sidebar_header = Div(
    text="""
    <div style='color:#1e293b; padding:14px 20px 10px; margin:0;
                border-bottom:1px solid #e2e8f0; background:#ffffff;'>
        <div style='font-size:13px; font-weight:600;'>Parameters</div>
        <div style='font-size:11px; color:#94a3b8; margin-top:2px;'>
            Updates on slider release
        </div>
    </div>""",
    width=304,
)

scrollable = column(
    Div(text="<div style='height:80px'></div>", width=264),
    *slider_groups,
    Div(text=f"<div style='height:{TOP_SPACER * 2}px'></div>", width=264),
    width=294, height=SCROLL_H,
    sizing_mode="inherit",
    styles={
        'overflow-y': 'scroll',
        'overflow-x': 'hidden',
        'padding': '0 20px',
        'direction': 'rtl',
        'background': '#ffffff',
    },
    stylesheets=["""
        :host { scrollbar-width: thin; scrollbar-color: #cbd5e1 #f8fafc; }
        ::-webkit-scrollbar       { width: 5px; }
        ::-webkit-scrollbar-track { background: #f8fafc; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
        .bk-panel-models-column-Column { direction: ltr; }
    """],
)

restore_btn = Button(
    label="↺  Restore calibrated values",
    button_type="primary",
    width=264, height=36,
    styles={
        'margin': '10px 20px',
        'font-size': '12px',
        'font-weight': '500',
        'letter-spacing': '0.2px',
    },
)
restore_btn.on_click(restore_calibrated)

from bokeh.models import CustomJS

# ── Scroll-to-middle on page load ─────────────────────────────────────────────
# Attach to sources["flow"] — its data is set by the server on every page load
# AND on every slider update, so js_on_change fires reliably.
# window._scrollInit guards against re-running after the first time.
SCROLL_INIT_CODE = """
if (window._scrollInit) return;
window._scrollInit = true;

function initScroll() {
    var all = Array.from(document.querySelectorAll('*'));
    for (var i = 0; i < all.length; i++) {
        var el = all[i];
        try {
            if (Math.abs(el.clientHeight - SCROLL_H) < 10 &&
                el.scrollHeight > el.clientHeight + 80) {
                el.scrollTop = TOP_SPACER;
                return true;
            }
        } catch(e) {}
    }
    return false;
}

var attempt = 0;
(function retry() {
    if (!initScroll() && attempt++ < 20) setTimeout(retry, 200);
})();
""".replace("SCROLL_H", str(SCROLL_H)).replace("TOP_SPACER", str(TOP_SPACER))

sources["flow"].js_on_change('data', CustomJS(code=SCROLL_INIT_CODE))

left_panel = column(
    sidebar_header, scrollable, restore_btn,
    width=304, sizing_mode="fixed",
    styles={
        'margin-right': '20px',
        'background': '#ffffff',
        'border': '1px solid #e2e8f0',
        'border-radius': '8px',
    },
)

# ── Right panel ───────────────────────────────────────────────────────────────
app_header = Div(
    text="""
    <div style='margin-bottom:16px; padding-bottom:14px; border-bottom:1px solid #e2e8f0;'>
        <div style='font-size:16px; font-weight:600; color:#0f172a; margin-bottom:3px;'>
            HBV Model Playground
        </div>
        <div style='font-size:12px; color:#64748b;'>
            Ramundberget catchment, Sweden &nbsp;·&nbsp; 2015–2020 &nbsp;·&nbsp;
            14 calibratable parameters
        </div>
    </div>""",
    width=860,
)

details_toggle = Toggle(
    label="Hide details", button_type="light",
    width=100, height=30, active=True,
    styles={'font-size': '12px'},
)
details_label = Div(
    text="<div style='font-size:13px; font-weight:600; color:#1e293b; padding-top:4px;'>Model internals</div>",
    width=740,
)
detail_section = column(tabs, spacing=0, visible=True)


def on_toggle(attr, old, new):
    detail_section.visible = new
    details_toggle.label   = "Hide details" if new else "Show details"

details_toggle.on_change('active', on_toggle)

right_panel = column(
    app_header,
    row(details_label, details_toggle, spacing=10),
    detail_section,
    Div(text="<div style='height:8px'></div>", width=860),
    main_plot,
    sizing_mode="fixed",
)

# ── Root ──────────────────────────────────────────────────────────────────────
layout = row(
    left_panel, right_panel,
    sizing_mode="fixed",
    styles={'padding': '24px', 'background': '#f8fafc'},
)

curdoc().add_root(layout)
curdoc().title = "HBV Model Playground"

