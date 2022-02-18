from glob import glob
from datetime import datetime
import pytz
from clearml import Dataset, Model
from tensorflow import keras

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd

from amplify.data import PredictData
from amplify.data import DataSplit
from amplify.models import YeetLSTMv2

app = Dash(
    __name__, title="Amplify Microgrid AI", external_stylesheets=[dbc.themes.ZEPHYR]
)

server = app.server


# Load the model
model = YeetLSTMv2(
    n_series_len=48,
    n_series_ft=7,
    n_series_out=2,
    n_lstm=400,
    model_weights_path=Model(
        model_id="f6b26b93ecc842319d0733711523f22e"
    ).get_local_copy(),
    production_mode=True,
)

# Create prediction engine
predict_data = PredictData(model)

# Get historical data
xy_data = pd.read_csv(
    glob(
        Dataset.get(
            dataset_name="xy_data",
            dataset_project="amplify",
        ).get_local_copy()
        + "/**"
    )[0],
    index_col=0,
)

# TODO: make output a figure
# TODO: add buttons and inputs
@app.callback(
    Output("charging_prediction", "figure"),
    Input("predict-optimal-charging-input", "n_clicks"),
    State("num-cars-input", "value"),                   # input_1
    State("charging-time-input", "value"),              # input_2
    State("kw-per-hour-input", "value"),                # input_3
    State("num-window-input", "value"))                 # input_4

def calculate_optimal_charging(n_clicks, input_1, input_2, input_3, input_4):
    if n_clicks <= 0:
        raise PreventUpdate
    else:
        # make calculation
        num_cars = int(input_1)
        hrs_to_charge = int(input_2)
        charging_rate_kwh = int(input_3)
        num_window = int(input_4)
        #charging_df = predict_data.calculate_charging(
        #    preds_df=predict_data.pred_out,
        #    num_cars=num_cars,
        #    hrs_to_charge=hrs_to_charge,
        #    charging_rate_kwh=charging_rate_kwh,
        #    num_charging_windows=num_window
        #)
        #TODO: add correct calculation output
        #TODO: add plot
        fig = px.line(x=[1,2,3,4], y = [num_cars, hrs_to_charge, charging_rate_kwh, num_window])
        #fig = px.line(
        #charging_df,
        #labels={
        #    "value": "Power (kW)",
        #    "dt": "Date and Time (Local)",
        #    "variable": "",
        #},
        #color_discrete_sequence=px.colors.qualitative.D3,
        #)
        #fig.add_vline(
        #    x=datetime.now(pytz.timezone("US/Eastern")),
        #    line_width=2,
        #    line_color="red",
        #    line_dash="dash",
        #)
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig


@app.callback(
    Output("forcast_data", "figure"), Input("interval-component", "n_intervals")
)
def forcast_data(n):
    """This will get data from OpenWeather OnceCall API"""
    preds = predict_data.forecast()
    preds.index = preds.index.tz_convert("US/Eastern")

    fig = px.line(
        preds,
        labels={
            "value": "Power (kW)",
            "dt": "Date and Time (Local)",
            "variable": "",
        },
        color_discrete_sequence=px.colors.qualitative.D3,
    )
    fig.add_vline(
        x=datetime.now(pytz.timezone("US/Eastern")),
        line_width=2,
        line_color="red",
        line_dash="dash",
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


@app.callback(Output("historical_data", "figure"), Input("date-slider", "value"))
def historical_data(i):
    """This uses validation and test portion of the existing historical data"""
    y_preds = model.predict(
        np.reshape(
            np.array(
                xy_data.iloc[i : i + 48].drop(
                    ["True Power (kW) solar", "True Power (kW) usage"], axis=1
                )
            ),
            (1, 48, 7),
        )
    )

    temp_df = pd.DataFrame()
    temp_df["Predicted Solar"] = y_preds[0, :, 0]
    temp_df["Predicted Usage"] = y_preds[0, :, 1]
    temp_df["Actual Solar"] = xy_data["True Power (kW) solar"].iloc[i : i + 48].values
    temp_df["Actual Usage"] = xy_data["True Power (kW) usage"].iloc[i : i + 48].values
    temp_df.index = pd.to_datetime(xy_data.iloc[i : i + 48].index)
    temp_df.index = temp_df.index.tz_convert("US/Eastern")

    fig = px.line(
        temp_df,
        labels={
            "value": "Power (kW)",
            "index": "Date and Time (Local)",
            "variable": "",
        },
        color_discrete_sequence=px.colors.qualitative.D3,
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


### Dash App Stuff ###
### -------------- ###

LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

nav = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("Prediction", active=True, href="#")),
        # dbc.NavItem(dbc.NavLink("Historical", active=False, href="#")),
    ],
    pills=True,
    className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=LOGO, height="30px")),
                        dbc.Col(
                            dbc.NavbarBrand("Amplify Microgrid AI", className="ms-2")
                        ),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="#",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                nav,
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
)

# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


dashboard = dbc.Container(
    [
        html.Div(
            id="graph_wrapper",
            children=[
                html.H4(
                    "Forcast Power Generation and Usage Predictions",
                    className="mt-3",
                    style={"textAlign": "center"},
                ),
                dcc.Graph(id="forcast_data"),
                html.H4(
                    "Charging Prediction",
                    className="mt-3",
                    style={"textAlign": "center"},
                ),
                dcc.Graph(id="charging_prediction"),
                html.H4(
                    "Historical Power Generation and Usage Predictions",
                    className="mt-3",
                    style={"textAlign": "center"},
                ),
                dcc.Graph(id="historical_data"),
            ],
        ),
        html.Div(
            [
                dcc.Slider(
                    min=int(xy_data.shape[0] - 0.2 * xy_data.shape[0]),
                    max=int(xy_data.shape[0] - 48),
                    value=int(xy_data.shape[0] * 0.9 - 24),
                    step=3,
                    id="date-slider",
                    marks=None,
                )
            ],
            className="d-grid gap-2 col-6 mx-auto",
        ),
    ]
)

footer = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                dbc.Col(html.Footer(html.P("© 2022 Amplify", className="ms-2"))),
                align="center",
                className="g-0",
            )
        ]
    ),
    color="light",
    className="fixed-bottom",
)

counter = dcc.Interval(
    id="interval-component",
    interval=15 * 60 * 1000,  # update every 5 minutes
    n_intervals=0,
)

## Sidebar (issue with using on phone)
# the style arguments for the sidebar. We use position:fixed and a fixed width
# SIDEBAR_STYLE = {
#    "position": "fixed",
#    "top": 0,
#    "left": 0,
#    "bottom": 0,
#    "width": "16rem",
#    "padding": "2rem 1rem",
#    "background-color": "#f8f9fa",
# }

# sidebar = html.Div(
#    [
#        html.H2("Sidebar", className="display-4"),
#        html.Hr(),
#        html.P("A simple sidebar layout with navigation links", className="lead"),
#    ],
#    style=SIDEBAR_STYLE,
# )
num_cars_dropdown = [1, 2, 3]
kwh_per_hour_dropdown = [6.5, 7.5, 50, 100]
charging_time_input = [3, 6, 1]
num_best_charging_options_input = [1, 12, 1]


controls = dbc.Card(
    [
        html.Div(
            [
                html.H1([dbc.Badge("Charging Calculator", className="ms-1", color="primary")]),
                dbc.FormText("Enter your information to calculate the optimal windows to charge your car(s)!")
            ]
        ),
        html.Div(
            [
                dbc.Label("Number of Cars:"),
                dcc.Dropdown(
                    id="num-cars-input",
                    options=[{"label": x, "value": x} for x in num_cars_dropdown],
                    value=num_cars_dropdown[0],
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Charging Rate (kWh):"),
                dcc.Dropdown(
                    id="kw-per-hour-input",
                    options=[{"label": x, "value": x} for x in kwh_per_hour_dropdown],
                    value=kwh_per_hour_dropdown[0],
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Charging Time:"),
                dbc.Input(
                    id="charging-time-input",
                    type="number",
                    min=charging_time_input[0],
                    max=charging_time_input[1],
                    step=charging_time_input[2],
                    value=charging_time_input[0],
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Number of Optimal Charging Windows:"),
                dbc.Input(
                    id="num-window-input",
                    type="number",
                    min=num_best_charging_options_input[0],
                    max=num_best_charging_options_input[1],
                    step=num_best_charging_options_input[2],
                    value=num_best_charging_options_input[0],
                ),
            ]
        ),
        html.Div(
            [
                dbc.Button("Predict", size='lg', id='predict-optimal-charging-input', n_clicks=0)
            ], className="d-grid gap-2 col-6 mx-auto"
        )
    ],
    body=True,
)

middle = dbc.Row([dbc.Col(controls, md=3), dbc.Col(dashboard, md=9)])

app.layout = html.Div([navbar, middle, footer, counter])

if __name__ == "__main__":
    app.run_server(debug=True)
