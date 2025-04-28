
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import os

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1("Wi-Fi IDS Real-Time Monitoring Dashboard", style={'textAlign': 'center'}),
    
    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0),
    
    html.Div(id='live-update-text'),
    dcc.Graph(id='live-update-graph'),
])

@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    if os.path.exists('../datasets/preprocessed_features.csv'):
        data = pd.read_csv('../datasets/preprocessed_features.csv')
        attack_count = data[data['label'] == 1].shape[0]
        legitimate_count = data[data['label'] == 0].shape[0]

        return [
            html.Span('Total Packets Captured: {}'.format(len(data)), style={'padding': '10px'}),
            html.Span('Legitimate Packets: {}'.format(legitimate_count), style={'padding': '10px'}),
            html.Span('Detected Attacks: {}'.format(attack_count), style={'padding': '10px', 'color': 'red'})
        ]
    else:
        return "Dataset not available yet."

@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph(n):
    if os.path.exists('../datasets/preprocessed_features.csv'):
        data = pd.read_csv('../datasets/preprocessed_features.csv')
        fig = px.histogram(data, x='label', title='Attack vs Legitimate Packets',
                           labels={'label': 'Packet Type (0: Legitimate, 1: Attack)'}, color='label')
        return fig
    else:
        return go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
