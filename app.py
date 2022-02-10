from glob import glob

from clearml import Dataset
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import flask


def add_data():
    data_dir = glob(
        Dataset.get(
            dataset_name="xy_data",
            dataset_project="amplify",
        ).get_local_copy()
        + "/**"
    )[0]

    xy_data = pd.read_csv(data_dir, index_col=0)

    fig = px.line(
        xy_data,
        x=xy_data.index,
        y=["True Power (kW) solar", "True Power (kW) usage"],
        labels={
            "value": "Power (kW)",
            "index": "Date and Time",
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
                children=[dcc.Graph(id="example-graph", figure=add_data())],
            )
        ),
        html.Div(
            [
                dbc.Button("Predict", color="primary"),
            ],
            className="d-grid gap-2 col-6 mx-auto",
        ),
    ]
)

app.layout = html.Div([navbar, dashboard])


if __name__ == "__main__":
    app.run_server(debug=True)
