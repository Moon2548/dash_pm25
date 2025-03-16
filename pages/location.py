from pages.function_eng_model import eng_model_predict_2_5

from dash import (
    html,
    dcc,
    callback,
    Input,
    Output,
    State,
    register_page,
    callback_context,
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
                {"range": [0, 100], "color": "gray"},
            ],
        },
        value=0,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "---"},
    )
)

register_page(__name__, path_template="/location/<city>")

layout = html.Div(
    style={},
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


@callback(
    Output("tabs-content", "children"), Input("tabs", "value"), Input("url", "pathname")
)
def render_content(tab, pathname):
    city_name = pathname.split("/")[-1]
    if city_name == "eng_psu":
        data = pd.read_csv("export_data/filtered_data_3_best.csv")
    if tab == "tab-1":
        return html.Div(
            children=[
                html.Div(
                    className="cards",
                    style={"text-align": "center"},
                    children=[
                        html.H1("Data"),
                        html.Div(
                            style={
                                "display": "grid",
                                "grid-template-columns": "1fr 1fr",
                            },
                            children=[
                                dcc.Checklist(
                                    style={"display": "flex"},
                                    id="checklist",
                                    options=[
                                        {"label": "pm 10", "value": "pm_10"},
                                        {"label": "pm 2.5", "value": "pm_2_5"},
                                        {
                                            "label": "Temperature",
                                            "value": "temperature",
                                        },
                                        {"label": "Humidity", "value": "humidity"},
                                    ],
                                    # value=["pm_10", "pm_2_5", "temperature", "humidity"],
                                    labelStyle={"display": "block"},
                                ),
                                dcc.DatePickerRange(
                                    id="date-range",
                                    min_date_allowed=eng["timestamp"].min().date(),
                                    max_date_allowed=eng["timestamp"].max().date(),
                                    start_date=eng["timestamp"].min().date(),
                                    end_date=eng["timestamp"].max().date(),
                                    display_format="MM-DD-YYYY",
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(id="graph-container"),
            ],
        )
    elif tab == "tab-2":
        return html.Div(
            children=[
                dcc.Store(id="n-clicks-store", data=0),
                dcc.Store(id="show-data", data=0),
                html.H1("Predictions"),
                dcc.Graph(id="indicator-pre", figure=fig),
                html.Div(
                    id="pre",
                    children=[
                        html.Button(
                            "Previous",
                            style={"display": "none"},
                            id="prev-button",
                            n_clicks=0,
                        ),
                        html.Button(
                            "Next",
                            style={"display": "none"},
                            id="next-button",
                            n_clicks=0,
                        ),
                    ],
                ),
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
                        dcc.Loading(
                            children=[
                                html.Div(id="prediction-output"),
                            ]
                        ),
                    ],
                ),
            ],
        )


@callback(
    Output("graph-container", "children"),
    Input("checklist", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("url", "pathname"),
)
def update_graphs(selected_values, start_date, end_date, pathname):
    city_name = pathname.split("/")[-1]
    if city_name == "eng_psu":
        data = pd.read_csv("export_data/filtered_data_3_best.csv")
    mask = data["timestamp"].between(start_date, end_date)
    filtered_df = data[mask]
    graphs = []
    if selected_values is None:
        return html.Div()
    else:
        for value in selected_values:
            title = value.replace("_", " ").title()
            graph = create_line_graph(filtered_df, value, title)
            graphs.append(html.Div(className="cards", children=[graph]))
        return graphs


@callback(
    # Output("prediction-output", "children"),
    # Output("pre", "children"),
    Output("show-data", "data"),
    Input("predict-button", "n_clicks"),
    Input("input-day", "value"),
    Input("url", "pathname"),
)
def predic(n_click, day, pathname):
    city_name = pathname.split("/")[-1]

    ctx = callback_context

    if not ctx.triggered:
        return 0

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    global result_clean

    if button_id == "predict-button":
        if city_name == "eng_psu":
            result = eng_model_predict_2_5(int(day) + 2)

        # ลบ NaN ออกจาก Predictions ก่อนพล็อต
        result_clean = result.dropna(subset=["Predictions"])

        # fig_pre = px.line(
        #     result_clean,  # ใช้ข้อมูลที่ลบ NaN แล้ว
        #     x=result_clean.index,
        #     y="Predictions",
        #     labels={"index": "Time", "Predictions": "PM2.5"},
        #     title="Predictions of PM2.5",
        #     markers=True,
        # )

        # selected_index = 0
        # x_point = result_clean.index[selected_index]
        # y_point = result_clean["Predictions"][selected_index]

        # fig_pre.add_trace(
        #     go.Scatter(
        #         x=[x_point],
        #         y=[y_point],
        #         mode="markers",
        #         marker=dict(size=10, color="green"),  # ปรับขนาดและสีของจุดได้
        #         name=f"จุดที่ {selected_index + 1}",  # กำหนดชื่อของ trace
        #     )
        # )

        return 1

    return 0


@callback(
    Output("n-clicks-store", "data"),
    Input("prev-button", "n_clicks"),
    Input("next-button", "n_clicks"),
    Input("input-day", "value"),
    State("n-clicks-store", "data"),
)
def update_n_clicks_store(prev_clicks, next_clicks, day, stored_n_clicks):
    ctx = callback_context

    if not ctx.triggered:
        return stored_n_clicks  # ไม่มีการ trigger อะไรเลย

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "prev-button":
        if stored_n_clicks <= 0:
            return 0
        else:
            return stored_n_clicks - 1
    elif button_id == "next-button":
        if stored_n_clicks >= int(day):
            return int(day)
        else:
            return stored_n_clicks + 1
    else:
        return stored_n_clicks  # กรณี input-day เปลี่ยนโดยไม่มีการกดปุ่ม


@callback(
    Output("pre", "children"),
    Output("prediction-output", "children"),
    Output("indicator-pre", "figure"),
    Input("n-clicks-store", "data"),
    Input("show-data", "data"),
)
def last_pre(click, data):
    if data == 1:
        fig_pre = px.line(
            result_clean,  # ใช้ข้อมูลที่ลบ NaN แล้ว
            x=result_clean.index,
            y="Predictions",
            labels={"index": "Time", "Predictions": "PM2.5"},
            title="Predictions of PM2.5",
            markers=True,
        )

        x_point = result_clean.index[click]
        y_point = result_clean["Predictions"][click]

        indi = go.Figure()

        indi.add_trace(
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
                value=y_point,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "PM 2.5"},
            )
        )

        fig_pre.add_trace(
            go.Scatter(
                x=[x_point],
                y=[y_point],
                mode="markers",
                marker=dict(size=10, color="green"),  # ปรับขนาดและสีของจุดได้
                name=f"จุดที่ {click + 1}",  # กำหนดชื่อของ trace
            )
        )
        return (
            [
                html.Button(
                    "Previous",
                    id="prev-button",
                    n_clicks=0,
                ),
                html.Button(
                    "Next",
                    id="next-button",
                    n_clicks=0,
                ),
            ],
            dcc.Graph(figure=fig_pre),
            indi,
        )

    else:
        return (
            [
                html.Button(
                    "Previous",
                    style={"display": "none"},
                    id="prev-button",
                    n_clicks=0,
                ),
                html.Button(
                    "Next",
                    style={"display": "none"},
                    id="next-button",
                    n_clicks=0,
                ),
            ],
            [],
            fig,
        )
