from glob import glob

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

data_dir = glob(
    Dataset.get(
        dataset_name="xy_data",
        dataset_project="amplify",
    ).get_local_copy()
    + "/**"
)[0]

xy_data = pd.read_csv(data_dir, index_col=0)

#TODO: remove?
(x_train, y_train), (x_val, y_val), (x_test, y_test), norm_layer = DataSplit(
    xy_data,
    sequence=False,
).split_data()


model_path = Model(model_id="f6b26b93ecc842319d0733711523f22e").get_local_copy()
print(model_path)
model = YeetLSTMv2(n_series_len=48, n_series_ft=7, n_series_out=2, n_lstm=400, model_weights_path=model_path)

#TODO: remove?
# model.evaluate(x_val, y_val, verbose=1)
# model.evaluate(x_test, y_test, verbose=1)

def add_data(xy_data, model):
    i = np.random.default_rng().integers(0, xy_data.shape[0] - 48)
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

def predict_data(model):

    preds = PredictData(model).forecast()
    preds.index = preds.index.tz_convert("US/Eastern")

    fig = px.line(
        preds,
        labels={
            "value": "Power (kW)",
            "dt": "Date and Time",
            "variable": "Features",
        },
    )

    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig


app = Dash(
    __name__, title="Amplify Microgrid AI", external_stylesheets=[dbc.themes.ZEPHYR]
)

server = app.server

LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

nav = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("Real Time", active=True, href="#")),
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
                        dbc.Col(dbc.NavbarBrand("Amplify Microgrid AI", className="ms-2")),
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
            html.H4("Power Generation and Usage Predictions"),
            className="mt-5 mx-auto",
        ),
        html.Div(
            id="graph_wrapper",
            children=[dcc.Graph(id="predict-data", figure=predict_data(model))],
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
