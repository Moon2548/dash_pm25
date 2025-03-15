from pages.function_eng_model import eng_model_predict_2_5

from dash import (
    html,
    dcc,
    callback,
    Input,
    Output,
    register_page,
)
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from pycaret.regression import load_model, predict_model

eng = pd.read_excel("export-pm25_eng-1d.xlsx")

fig = go.Figure()

fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": "blue"},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 60], "color": "yellow"},
                {"range": [60, 100], "color": "red"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 80,
            },
        },
        value=eng["pm_2_5"].mean(),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Test  Avg PM 2.5  Test"},
    )
)

register_page(__name__, path_template="/location/<city>")

layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(
            className="title",
            children=[
                html.H3(id="location-title"),
            ],
        ),
        html.Div(
            className="cards",
            style={"display": "block"},
            children=[
                dcc.Tabs(
                    id="tabs",
                    value="tab-1",
                    children=[
                        dcc.Tab(
                            label="Data",
                            value="tab-1",
                            className="my-tab",
                            selected_className="my-tab--selected",
                        ),
                        dcc.Tab(
                            label="Predictions",
                            value="tab-2",
                            className="my-tab",
                            selected_className="my-tab--selected",
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="tabs-content"),
    ],
)


@callback(Output("location-title", "children"), Input("url", "pathname"))
def update_title(pathname):
    city_name = pathname.split("/")[-1]
    return f"{city_name}"


def create_line_graph(df, y_column, title):
    """สร้างกราฟเส้นจาก DataFrame"""
    if y_column in df.columns:
        fig = px.line(df, x="timestamp", y=y_column)
        fig.update_layout(title=title)
        return dcc.Graph(figure=fig)
    else:
        return html.Div(f"No data for {title}")


@callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_content(tab):
    if tab == "tab-1":
        return html.Div(
            children=[
                html.Div(
                    className="cards",
                    style={"text-align": "center"},
                    children=[
                        html.H1("Data"),
                        dcc.Checklist(
                            id="checklist",
                            options=[
                                {"label": "pm 10", "value": "pm_10"},
                                {"label": "pm 2.5", "value": "pm_2_5"},
                                {"label": "Temperature", "value": "temperature"},
                                {"label": "Humidity", "value": "humidity"},
                            ],
                            value=["pm_10", "pm_2_5", "temperature", "humidity"],
                            labelStyle={"display": "block"},
                        ),
                        dcc.DatePickerRange(
                            id="date-range",
                            min_date_allowed=eng["timestamp"].min().date(),
                            max_date_allowed=eng["timestamp"].max().date(),
                            start_date=eng["timestamp"].min().date(),
                            end_date=eng["timestamp"].max().date(),
                            display_format="YYYY-MM-DD",
                        ),
                    ],
                ),
                html.Div(id="graph-container"),
            ],
        )
    elif tab == "tab-2":
        return html.Div(
            children=[
                html.H1("Predictions"),
                dcc.Graph(figure=fig),
                html.Div(
                    className="cards",
                    children=[
                        html.H3("Predict PM 2.5"),
                        html.Button("Predict", id="predict-button"),
                        dcc.Input(
                            id="input-day",
                            type="number",
                            placeholder="days",
                        ),
                        html.Div(id="prediction-output"),
                    ],
                ),
            ],
        )


@callback(
    Output("graph-container", "children"),
    Input("checklist", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_graphs(selected_values, start_date, end_date):
    mask = eng["timestamp"].between(start_date, end_date)
    filtered_df = eng[mask]
    graphs = []
    for value in selected_values:
        title = value.replace("_", " ").title()
        graph = create_line_graph(filtered_df, value, title)
        graphs.append(html.Div(className="cards", children=[graph]))
    return graphs


@callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    Input("input-day", "value"),
)

def predic(n_click, day):
    if n_click:
        result = eng_model_predict_2_5(int(day)+2)

        # สร้างกราฟแสดงเฉพาะ Predictions
        fig = px.line(
            result,
            x=result.index,
            y="Predictions",  # แสดงแค่คอลัมน์ 'Predictions'
            labels={"index": "Time", "Predictions": "PM2.5"},
            title="Predictions of PM2.5",
            markers=True
        )

        return dcc.Graph(figure=fig)
