from glob import glob
from datetime import datetime
import pytz
from clearml import Dataset, Model

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
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

TIMEOUT = 300  # 5 minutes


# Load the model
model = YeetLSTMv2(
    n_series_len=48,
    n_series_ft=7,
    n_series_out=2,
    n_lstm=400,
    # model_weights_path=Model(
    #     model_id="f6b26b93ecc842319d0733711523f22e"
    # ).get_local_copy(),
    model_weights_path="models/lstm_weights.hdf5",
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
def forecast_power(n):
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
        automargin=False,
    )
    fig.update_yaxes(
        fixedrange=True,
        showgrid=True,
    )
    fig.update_layout(
        margin=dict(l=0, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@app.callback(
    Output("forecast_energy", "figure"), Input("interval-component", "n_intervals")
)
def forecast_energy(n):
    """This will get data from OpenWeather OnceCall API"""

    preds = data_cache()
    net_usage = preds["Predicted Usage"] - preds["Predicted Solar"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=net_usage.sum(),
            number={"suffix": " kWh"},
        )
    )
    fig.update_traces(
        gauge_axis_range=[0, 1000],
        gauge_bar_color=px.colors.qualitative.D3[2],
        selector=dict(type="indicator"),
    )
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
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
    fig.update_xaxes(
        fixedrange=True,
        showgrid=True,
        automargin=False,
    )
    fig.update_yaxes(
        fixedrange=True,
        showgrid=True,
    )
    fig.update_layout(
        margin=dict(l=0, r=10),
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
        dbc.NavItem(
            dbc.NavLink(
                "Slide Deck",
                href="https://docs.google.com/presentation/d/e/2PACX-1vT06gWZCeAlgXYaDCISYzmNJQR6VNKkXUM7l-DlcfSascUbV9HWGY0d4SDm3y9iT8KtHvaZfa62lyVj/pub",
                target="blank",
                id="slide_deck",
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
                    "Predicted Power from Forecast Weather",
                    className="mt-5",
                    style={"textAlign": "center"},
                ),
                dcc.Graph(id="forecast_data"),
                html.H4(
                    "Predicted Net Energy Usage over the Next 48 Hours",
                    style={"textAlign": "center"},
                ),
                dcc.Graph(id="forecast_energy"),
            ],
        ),
        html.Div(
            [
                html.H5("About"),
                dcc.Markdown(
                    """
                    The **top figure** above displays predicted solar power generation and predicted total building power usage.
                    The green bars represent optimal times to charge an electric vehicle based on minimizing peak electrical demand and electricity cost.
                    The **bottom figure** is the predicted net energy usage for the building over the next 48 hours.
                    These predictions could be used in other ways, such as controlling reserve battery storage, timing hot water heaters, and reducing air conditioner usage.

                    These predictions are made in real time using a Long Short-Term Memory (LSTM) Deep Learning Model on current forecast weather data.
                    This model was trained on historical weather data from the same building.

                    Please go to the [historical](#historical) data page to compare the model's predictions versus actual solar and usage data from the test set.
                    """
                ),
            ],
        ),
    ]
)

historical_dashboard = dbc.Container(
    [
        html.Div(
            id="graph_wrapper",
            children=[
                html.H4(
                    "Predicted Power from Historical Weather",
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
            className="pt-3 d-grid gap-2 col-10 mx-auto",
        ),
        html.Div(
            [
                html.H5("About"),
                dcc.Markdown(
                    """
                    The figure above displays predicted solar power generation and predicted total building power usage as well as actual solar power generation and actual total building usage for the same time period.
                    Moving the slider at the bottom of the figure will select a different 48 hour window from the test set.

                    These predictions are made in real time, when the slider is moved, using a Long Short-Term Memory (LSTM) Deep Learning Model trained on historical weather data from the same building.
                    There is no overlap between the training data and the test data seen above.

                    Please go to the [forecast](#forecast) data page to see the current power and energy predictions for the building.
                    """
                ),
            ],
        ),
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
            html.Div(f"© {datetime.now().year} Amplify"),
            html.Div(
                [
                    "Made with ⚡️ by ",
                    html.A(
                        "John",
                        href="https://www.linkedin.com/in/john-droescher/",
                        target="blank",
                        style={"textDecoration": "none"},
                    ),
                    ", ",
                    html.A(
                        "Christian",
                        href="https://www.linkedin.com/in/christianwelling/",
                        target="blank",
                        style={"textDecoration": "none"},
                    ),
                    ", and ",
                    html.A(
                        "Sam",
                        href="https://samsipe.com/",
                        target="blank",
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
    interval=TIMEOUT * 1000,  # update every 5 minutes
    n_intervals=0,
)

app.layout = html.Div(
    [
        location,
        navbar,
        html.Div(id="dashboard", className="pb-3 mb-5"),
        footer,
        counter,
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
