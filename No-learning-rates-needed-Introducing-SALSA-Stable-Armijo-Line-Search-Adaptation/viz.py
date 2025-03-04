import os
import pickle
from itertools import cycle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)
# get all pickle files
directory = "../results" #/line_search c=0.5.pickle

all_files = os.listdir(directory)

pickle_files = [file for file in all_files if file[-7:] == ".pickle"]

loaded_files_dict = {}

for fil e in pickle_files:
    file_path = os.path.join(directory, file)
    with open(file_path, 'rb') as handle:
        loaded_files_dict[file] = pickle.load(handle)

# get all result dicts
result_dicts = []
for key in loaded_files_dict.keys():
    result_dicts.append(loaded_files_dict[key])


#setup directory for setting them in plotly
c_values= [result_dict["c_value"] for result_dict in result_dicts]

# get proper data format
data = []
for result in result_dicts:
    for metric in ["learning_rate","train_loss", "val_loss", "test_loss"]:
        for i,value in enumerate(result[metric]):
            data.append({
                "c_value":result["c_value"],
                "metric": metric,
                "value": value,
                "optimizer": result["optimizer"],
                "learning_rate": result["learning_rate"][i],
                "iteration": i,
                "title": result["title"]
            })
df_long = pd.DataFrame(data)
#tesr

# determine color maps for each title(optmizer + cvalue combination)
unique_titles = df_long["title"].unique()

colors = px.colors.qualitative.Plotly
#make generator
color_cycle = cycle(colors)
next(color_cycle) # akward way to skip first color(blue)
color_map = {title: next(color_cycle) for title in unique_titles}


app.layout = html.Div([
html.Label('Optimizer:'),
    dcc.Dropdown(
        id='optimizer-selector',
        options=[
            {"label":"Adam","value":"Adam"},
            {"label":"AdamSLS","value":"AdamSLS"},
            {"label":"SaLSA","value":"SaLSA"}
        ],
        value=list(set([result_dict["optimizer"] for result_dict in result_dicts])),#defaults
        multi=True
    ),
    html.Label("c_value:"),
    dcc.Dropdown(
        id='cvalue_selector',
        options=list(set(c_values)),
        value=[-1,0.7],#c_values,
        multi=True
    ),
    html.H1("Line Search Comparison"),
    dcc.Graph(id="learning_rate_comparison"),
    dcc.Graph(id="train_loss_comparison"),
    html.Center(dcc.Graph(id="c_value_comparison")),
])

# Dash callback to update graph
@app.callback(
    [Output("learning_rate_comparison", "figure"),
     Output("train_loss_comparison", "figure")],
    [Input("optimizer-selector", "value"),
     Input("cvalue_selector", "value")]
)
def update_line_chart(selected_optimizers,selected_cvalues):
    if not selected_optimizers:  # If no optimizers selected, return empty figure
        return px.line()

    # Filter df_long for learning_rate and selected optimizers
    # todo could do logical or with learning rate + optimizer in brackets
    filtered_df = df_long[(df_long['optimizer'].isin(selected_optimizers)) & (df_long["c_value"]).isin(selected_cvalues)]

    # sort by title for alphabetica legend order(grouped colors)
    filtered_df = filtered_df.sort_values(by='iteration')

    """ lr_fig = px.line(filtered_df, x='iteration', y='value', color='title',line_dash='metric',
                  labels={'value': 'Metric Value', 'iteration': 'Iteration'},
                  title='Learning Rate Comparison:',log_y=True,)"""
    #define plot dataframes
    filtered_df_lr = filtered_df[filtered_df['metric']== "learning_rate"]
    filtered_df_loss = filtered_df[filtered_df['metric'].isin(["train_loss","val_loss"])]
    #filtered_df_test = filtered_df[filtered_df["metric"]=="test_loss"].groupby(["c_value","optimizer","title"])["value"].first().reset_index()

    #sort titles
    sorted_titles = sorted(filtered_df['title'].unique())

    lr_fig = px.line(filtered_df_lr, x='iteration', y='value', color='title',
                  category_orders={'title': sorted_titles},
                  labels={'value': 'learning_rate', 'iteration': 'Iteration'},
                  title='Learning Rate Comparison:',log_y=True,color_discrete_map=color_map)
    loss_fig = px.line(filtered_df_loss, x='iteration', y='value', color='title',category_orders={'title': sorted_titles},line_dash='metric',
                     labels={'value': 'Metric Value', 'iteration': 'Iteration'},
                     title='Loss Comparison:', color_discrete_map=color_map, log_y=True)
    #loss_fig.add_trace(go.Scatter(y=[0.5,0.5], mode="lines"), row=1, col=1)

    # Maybe jesus can tell me why the graph is displayed in the slider in only one of them :)
    lr_fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True),
            #range=[df_long['iteration'].min(), df_long['iteration'].max()],
            type="linear"
        ))

    loss_fig.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True),
            type="linear"
        ))

    return lr_fig,loss_fig


@app.callback(
    Output('c_value_comparison', "figure"),
    [Input("optimizer-selector", "value"),
     Input("cvalue_selector", "value")]
)
def update_grouped_bar_chart(selected_optimizers, selected_cvalues):
        # Filter the DataFrame based on the selected optimizers and c_values
        filtered_df = df_long[
            (df_long['optimizer'].isin(selected_optimizers)) &
            (df_long["c_value"].isin(selected_cvalues)) &
            (df_long["c_value"] != -1) & # -1 is default value for no c_value
            (df_long["metric"] != "learning_rate")
            ]

        color_map = {
            'train_loss': '#a6cee3',
            'val_loss': '#1f78b4',
            'test_loss': '#b2df8a'  #
        }
        # get latest value
        latest_values = filtered_df[
            filtered_df['iteration'] == filtered_df.groupby(['c_value', 'metric'])['iteration'].transform('max')]

        # data per optimizer
        df_adam_sls = latest_values[latest_values['optimizer'] == 'AdamSLS']
        df_SaLSA = latest_values[latest_values['optimizer'] == 'SaLSA']

        fig = make_subplots(rows=1, cols=2, subplot_titles=("AdamSLS", "SaLSA"), shared_yaxes=True)

        for metric in df_adam_sls['metric'].unique():
            metric_data = df_adam_sls[df_adam_sls['metric'] == metric]
            fig.add_trace(
                go.Bar(
                    x=metric_data['c_value'],
                    y=metric_data['value'],
                    name=metric,
                    marker_color=color_map.get(metric)
                ),
                row=1, col=1
            )

        for metric in df_SaLSA['metric'].unique():
            metric_data = df_SaLSA[df_SaLSA['metric'] == metric]
            fig.add_trace(
                go.Bar(
                    x=metric_data['c_value'],
                    y=metric_data['value'],
                    name=metric,
                    marker_color=color_map.get(metric),
                    showlegend=False  # Avoid duplicate legend entries
                ),
                row=1, col=2
            )

        fig.update_layout(
            title="Comparison of Loss Values per C Value for AdamSLS and SaLSA",
            title_x=0.5,
            width=1000,
            barmode='group',
            bargap=0.1
        )
        fig.update_xaxes(title_text="C value", row=1, col=1)
        fig.update_xaxes(title_text="C value", row=1, col=2)
        fig.update_yaxes(title_text="Loss ", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        fig.update_xaxes(showticklabels=False)

        return fig


if __name__ == '__main__':
    app.run_server(debug=True)