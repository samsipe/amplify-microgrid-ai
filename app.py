from glob import glob

from clearml import Dataset, Model
from amplify.data import DataSplit
from amplify.models import YeetLSTMv2
import tensorflow as tf

from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd
import flask

#data_dir = glob(
#    Dataset.get(
#        dataset_name="xy_data",
#        dataset_project="amplify",
#    ).get_local_copy()
#    + "/**"
#)[0]

#xy_data = pd.read_csv(data_dir, index_col=0)

#_, _, _, norm_layer = DataSplit(
#    xy_data,
#    series_length=48,
#    stride=1,
#).split_data()

# Define the model
#inputs = tf.keras.layers.Input(shape=(48, 7))
#x = norm_layer(inputs)
#x = tf.keras.layers.LSTM(400, return_sequences=True, dropout=0.2)(x)
#outputs = tf.keras.layers.TimeDistributed(
#    tf.keras.layers.Dense(2, activation="relu", kernel_regularizer="l2")
#)(x)
#model = tf.keras.models.Model(inputs, outputs)
#
#model.compile(
#    optimizer=tf.keras.optimizers.Adam(),
#    loss=tf.keras.losses.MeanSquaredError(),
#    metrics=[tf.keras.metrics.RootMeanSquaredError()],
#)
model_path = Model(model_id="f6b26b93ecc842319d0733711523f22e").get_local_copy()
model = YeetLSTMv2(n_series_len=48, n_series_ft=7, n_series_out=2, n_lstm=200, model_weights_path=model_path)


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
            "variable": "Features",
        },
    )

    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig


server = flask.Flask(__name__)

app = Dash(
    __name__,
    title="Amplify Microgrid AI",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    server=server,
)

LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

nav = dbc.Nav(
    [
        dbc.NavItem(dbc.NavLink("Prediction", active=True, href="#")),
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
    children=[
        html.H4(
            children="Power Generation and Usage Predictions",
        ),
        dcc.Loading(
            html.Div(
                id="graph_wrapper",
                children=[
                    dcc.Graph(id="example-graph", figure=add_data(xy_data, model))
                ],
            )
        ),
        html.Div(
            [
                dbc.Button("Predict", color="primary", href="/"),
            ],
            className="d-grid gap-2 col-6 mx-auto",
        ),
    ]
)

app.layout = html.Div([navbar, dashboard])


if __name__ == "__main__":
    app.run_server(debug=True)
