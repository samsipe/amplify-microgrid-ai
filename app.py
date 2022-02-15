from glob import glob
from datetime import datetime
import pytz
from clearml import Dataset, Model
from tensorflow import keras

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd

from amplify.data import PredictData
from amplify.data import DataSplit
from amplify.models import YeetLSTMv2


def clearml_setup():
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

    return model, xy_data


def historical_data(xy_data, model):
    """ "This uses validation and test portion of the existing historical data"""
    i = np.random.default_rng().integers(
        xy_data.shape[0] - 0.2 * xy_data.shape[0], xy_data.shape[0] - 48
    )
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

    fig = px.line(
        xy_data.iloc[i : i + 48],
        x=xy_data.iloc[i : i + 48].index,
        y=[
            y_preds[0, :, 0].T,
            y_preds[0, :, 1].T,
            "True Power (kW) solar",
            "True Power (kW) usage",
        ],
        labels={
            "y": "Power (kW)",
            "x": "Date and Time",
            "variable": "",
        },
        color_discrete_sequence=px.colors.qualitative.D3,
    )

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def forcast_data(model):
    """This will get data from OpenWeather OnceCall API"""
    preds = PredictData(model).forecast()
    preds.index = preds.index.tz_convert("US/Eastern")

    fig = px.line(
        preds,
        labels={
            "value": "Power (kW)",
            "dt": "Date and Time",
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


### Dash App Stuff ###
### -------------- ###

app = Dash(
    __name__, title="Amplify Microgrid AI", external_stylesheets=[dbc.themes.ZEPHYR]
)

server = app.server

LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

nav = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("Forcast", active=True, href="#")),
        dbc.NavItem(dbc.NavLink("Historical", active=False, href="#")),
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


model, xy_data = clearml_setup()
dashboard = dbc.Container(
    [
        html.Div(
            html.H4("Power Generation and Usage Predictions"),
            className="mt-5 mx-auto",
        ),
        html.Div(
            id="graph_wrapper",
            children=[
                dcc.Graph(id="forcast-data", figure=forcast_data(model)),
                dcc.Graph(id="historical-data", figure=historical_data(xy_data, model)),
            ],
        ),
        html.Div(
            [
                dbc.Button("Predict", color="primary", href="/"),
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


def serve_layout():
    return html.Div(
        [
            navbar,
            dashboard,
            footer,
        ]
    )


app.layout = serve_layout

if __name__ == "__main__":
    app.run_server(debug=True)
