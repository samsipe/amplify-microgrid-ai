# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

app = Dash(__name__,
           title='Amplify Microgrid AI',
           external_stylesheets=[dbc.themes.BOOTSTRAP])

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
xy_data = pd.read_csv('./data/xy_data.csv')

fig = px.line(
    xy_data,
    x=xy_data.index,
    labels={"value": "Power (kW)", "index": "Date and Time"},
    y=["True Power (kW) solar", "True Power (kW) usage"],
    title="Solar Generation and Building Usage (Power)",
)

app.layout = html.Div(
    children=[
        html.H1(children='Amplify Microgrid AI', style={'textAlign': 'center'}),
        html.Div(
            children='''
        Power Generation and Usage Predictions
    ''',
            style={'textAlign': 'center'},
        ),
        dcc.Graph(id='example-graph', figure=fig),
    ],
)

if __name__ == '__main__':
    app.run_server(debug=True)
