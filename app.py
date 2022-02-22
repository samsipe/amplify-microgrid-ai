from glob import glob
from datetime import datetime
import pytz
from clearml import Dataset, Model

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd

from amplify.data import PredictData
from amplify.data import DataSplit
from amplify.models import YeetLSTMv2

from flask_caching import Cache

app = Dash(
    __name__,
    title="Amplify Microgrid AI",
    external_stylesheets=[dbc.themes.ZEPHYR],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
    suppress_callback_exceptions=True,
)

server = app.server
cache = Cache(
    app.server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "./data/cache"}
)

TIMEOUT = 180


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

# (x_train, y_train), (x_val, y_val), (x_test, y_test), norm_layer = DataSplit(
#     xy_data,
# ).split_data()

# model.evaluate(x_val, y_val, verbose=1)
# model.evaluate(x_test, y_test, verbose=1)


@cache.memoize(timeout=TIMEOUT)
def query_data():
    preds = PredictData(model, num_cars=2, hrs_to_charge=3, kw_to_charge=7).forecast()
    return preds.to_json(date_format="iso", orient="split")


def data_cache():
    return pd.read_json(query_data(), orient="split").tz_convert("US/Eastern")


@app.callback(
    Output("forecast_data", "figure"), Input("interval-component", "n_intervals")
)
def forecast_data(n):
    """This will get data from OpenWeather OnceCall API"""

    preds = data_cache()

    fig = px.line(
        preds,
        y=[
            "Predicted Solar",
            "Predicted Usage",
        ],
        labels={
            "value": "Power (kW)",
            "index": "Date and Time (Local)",
            "variable": "",
        },
        color_discrete_sequence=px.colors.qualitative.D3,
    )
    fig.add_bar(x=preds.index, y=preds["Optimal Charging"], name="Optimal EV Charging")
    fig.add_vline(
        x=datetime.now(pytz.timezone("US/Eastern")),
        line_width=2,
        line_color="red",
        line_dash="dash",
    )
    fig.update_xaxes(
        fixedrange=True,
        showgrid=True,
        range=[preds.index.min(), preds.index.max()],
    )
    fig.update_yaxes(
        fixedrange=True,
        showgrid=True,
    )
    fig.update_layout(
        margin=dict(l=0, r=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
    fig.update_xaxes(fixedrange=True, showgrid=True)
    fig.update_yaxes(fixedrange=True, showgrid=True)
    fig.update_layout(
        margin=dict(l=0, r=5),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


### Dash App Stuff ###
### -------------- ###

LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

nav = dbc.Nav(
    [
        dbc.NavItem(
            dbc.NavLink(
                "Forecast",
                href="#forecast",
                id="forecast_link",
                style={"textAlign": "center"},
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Historical",
                href="#historical",
                id="historical_link",
                style={"textAlign": "center"},
            )
        ),
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


forecast_dashboard = dbc.Container(
    [
        html.Div(
            id="graph_wrapper",
            children=[
                html.H4(
                    "Power Predictions from Forecast Weather",
                    className="mt-5",
                    style={"textAlign": "center"},
                ),
                dcc.Graph(id="forecast_data"),
            ],
        ),
        html.Div(className="pb-5"),
    ]
)

historical_dashboard = dbc.Container(
    [
        html.Div(
            id="graph_wrapper",
            children=[
                html.H4(
                    "Power Predictions from Historical Weather",
                    className="mt-5",
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
        html.Div(className="pb-5"),
    ]
)


@app.callback(
    [
        Output("dashboard", "children"),
        Output("forecast_link", "active"),
        Output("historical_link", "active"),
    ],
    [Input("url", "hash")],
)
def dashboard(hash):
    if hash == "#forecast":
        return forecast_dashboard, True, False
    elif hash == "#historical":
        return historical_dashboard, False, True
    else:
        return forecast_dashboard, True, False


footer = dbc.Navbar(
    dbc.Container(
        [
            html.Div("© 2022 Amplify"),
            html.Div(
                [
                    "Made with ⚡️ by ",
                    html.A(
                        "John",
                        href="https://www.linkedin.com/in/john-droescher/",
                        style={"textDecoration": "none"},
                    ),
                    ", ",
                    html.A(
                        "Christian",
                        href="https://www.linkedin.com/in/christianwelling/",
                        style={"textDecoration": "none"},
                    ),
                    ", and ",
                    html.A(
                        "Sam",
                        href="https://www.linkedin.com/in/samsipe/",
                        style={"textDecoration": "none"},
                    ),
                ]
            ),
        ],
    ),
    color="light",
    className="fixed-bottom",
)

location = dcc.Location(id="url", refresh=False)

counter = dcc.Interval(
    id="interval-component",
    interval=15 * 60 * 1000,  # update every 5 minutes
    n_intervals=0,
)

app.layout = html.Div([location, navbar, html.Div(id="dashboard"), footer, counter])

if __name__ == "__main__":
    app.run_server(debug=True)
