from pages.function_eng_model import (
    eng_model_predict_2_5,
    eng_model_temp,
    eng_model_humidity,
)
from pages.function_surat_model import (
    surat_model_predict_2_5,
    surat_model_temp,
    surat_model_humidity,
)

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
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from pycaret.regression import load_model, predict_model

# Load data
eng = pd.read_csv("export_data/filtered_data_3_best.csv")
surat = pd.read_csv("export_data/clean_data_jsps001_1d.csv")

# Initialize empty figure
fig = go.Figure()
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": "#800020"},
            "steps": [
                {"range": [0, 33], "color": "rgba(0,255,0,0.2)"},
                {"range": [33, 66], "color": "rgba(255,255,0,0.2)"},
                {"range": [66, 100], "color": "rgba(255,0,0,0.2)"},
            ],
        },
        value=0,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "---"},
    )
)

register_page(__name__, path_template="/location/<city>")

# Improved layout with consistent styling
layout = html.Div(
    style={
        "background": "linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%)",
        "font-family": "'Segoe UI', Roboto, 'Helvetica Neue', sans-serif",
        "padding": "40px",
        "border-radius": "20px",
        "box-shadow": "0 10px 25px rgba(0, 0, 0, 0.08)",
        "max-width": "1400px",
        "margin": "0 auto",
    },
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(
            className="title-container",
            style={
                "text-align": "center",
                "margin-bottom": "30px",
            },
            children=[
                html.H1(
                    id="location-title",
                    style={
                        "color": "#800020",
                        "font-size": "36px",
                        "font-weight": "700",
                        "letter-spacing": "0.5px",
                        "margin": "0",
                        "padding-bottom": "5px",
                        "border-bottom": "4px solid #800020",
                        "display": "inline-block",
                    },
                ),
                html.P(
                    "Detailed air quality analytics and predictions",
                    style={
                        "color": "#555",
                        "font-size": "18px",
                        "margin-top": "10px",
                        "font-style": "italic",
                    },
                ),
                html.A(
                    "← Back to Dashboard",
                    href="/",
                    style={
                        "color": "#800020",
                        "text-decoration": "none",
                        "font-weight": "600",
                        "display": "block",
                        "margin-top": "15px",
                    },
                ),
            ],
        ),
        html.Div(
            className="content-container",
            style={
                "background-color": "#ffffff",
                "border-radius": "16px",
                "padding": "25px",
                "box-shadow": "0 8px 20px rgba(0, 0, 0, 0.06)",
                "margin-bottom": "30px",
            },
            children=[
                dcc.Tabs(
                    id="tabs",
                    value="tab-1",
                    style={
                        "border-bottom": "1px solid #ddd",
                        "margin-bottom": "20px",
                    },
                    children=[
                        dcc.Tab(
                            label="Historical Data",
                            value="tab-1",
                            style={
                                "border-radius": "8px 8px 0 0",
                                "padding": "12px 24px",
                                "font-weight": "600",
                                "color": "#666",
                            },
                            selected_style={
                                "border-top": "3px solid #800020",
                                "border-left": "1px solid #ddd",
                                "border-right": "1px solid #ddd",
                                "border-radius": "8px 8px 0 0",
                                "padding": "12px 24px",
                                "font-weight": "600",
                                "color": "#800020",
                            },
                        ),
                        dcc.Tab(
                            label="Predictions",
                            value="tab-2",
                            style={
                                "border-radius": "8px 8px 0 0",
                                "padding": "12px 24px",
                                "font-weight": "600",
                                "color": "#666",
                            },
                            selected_style={
                                "border-top": "3px solid #800020",
                                "border-left": "1px solid #ddd",
                                "border-right": "1px solid #ddd",
                                "border-radius": "8px 8px 0 0",
                                "padding": "12px 24px",
                                "font-weight": "600",
                                "color": "#800020",
                            },
                        ),
                    ],
                ),
                html.Div(id="tabs-content"),
            ],
        ),
        html.Footer(
            style={
                "margin-top": "40px",
                "text-align": "center",
                "color": "#666",
                "font-size": "14px",
                "padding-top": "20px",
                "border-top": "1px solid #ddd",
            },
            children=["© 2025 PM 2.5 Monitoring System | Data updated daily"],
        ),
    ],
)


@callback(Output("location-title", "children"), Input("url", "pathname"))
def update_title(pathname):
    city_name = pathname.split("/")[-1]

    # Format the city name to be more readable
    if city_name == "eng_psu":
        return "Engineering PSU Station"
    elif city_name == "surat":
        return "Surat Thani Station"
    else:
        return f"{city_name.replace('_', ' ').title()}"


def create_line_graph(df, y_column, title):
    """Create a line graph from DataFrame with improved styling"""
    if y_column in df.columns:
        color_map = {
            "pm_10": "#FF6B6B",
            "pm_2_5": "#800020",
            "temperature": "#4ECDC4",
            "humidity": "#45B7D1",
        }

        color = color_map.get(y_column, "#800020")

        fig = px.line(
            df,
            x="timestamp",
            y=y_column,
            labels={"timestamp": "Date", y_column: title},
        )

        fig.update_traces(line=dict(color=color, width=3))

        fig.update_layout(
            title={
                "text": title,
                "font": {
                    "size": 24,
                    "color": "#333",
                    "family": "Segoe UI, Roboto, sans-serif",
                },
                "x": 0.5,
                "xanchor": "center",
            },
            paper_bgcolor="white",
            plot_bgcolor="rgba(240, 240, 240, 0.5)",
            margin=dict(l=40, r=40, t=80, b=40),
            xaxis=dict(
                title="Date",
                titlefont=dict(size=14, color="#333"),
                showgrid=True,
                gridcolor="rgba(0, 0, 0, 0.1)",
            ),
            yaxis=dict(
                title=title,
                titlefont=dict(size=14, color="#333"),
                showgrid=True,
                gridcolor="rgba(0, 0, 0, 0.1)",
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Segoe UI, Roboto, sans-serif",
            ),
        )

        return dcc.Graph(figure=fig)
    else:
        return html.Div(f"No data for {title}")


@callback(
    Output("tabs-content", "children"), Input("tabs", "value"), Input("url", "pathname")
)
def render_content(tab, pathname):
    city_name = pathname.split("/")[-1]
    if city_name == "eng_psu":
        data = eng
    elif city_name == "surat":
        data = surat
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    if tab == "tab-1":
        return html.Div(
            children=[
                html.Div(
                    className="filter-container",
                    style={
                        "display": "flex",
                        "justify-content": "space-between",
                        "margin-bottom": "20px",
                        "padding": "20px",
                        "background-color": "rgba(0, 0, 0, 0.03)",
                        "border-radius": "12px",
                        "align-items": "center",
                    },
                    children=[
                        html.Div(
                            style={
                                "width": "50%",
                            },
                            children=[
                                html.H3(
                                    "Select Parameters",
                                    style={
                                        "color": "#800020",
                                        "margin-bottom": "15px",
                                        "font-weight": "600",
                                    },
                                ),
                                dcc.Checklist(
                                    id="checklist",
                                    options=[
                                        # {"label": " PM 10", "value": "pm_10"},
                                        {"label": " PM 2.5", "value": "pm_2_5"},
                                        {
                                            "label": " Temperature",
                                            "value": "temperature",
                                        },
                                        {"label": " Humidity", "value": "humidity"},
                                    ],
                                    value=["pm_2_5"],
                                    labelStyle={
                                        "display": "flex",
                                        "align-items": "center",
                                        "margin-bottom": "10px",
                                        "font-size": "16px",
                                        "color": "#333",
                                    },
                                    inputStyle={"margin-right": "10px"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "width": "50%",
                                "text-align": "right",
                            },
                            children=[
                                html.H3(
                                    "Select Date Range",
                                    style={
                                        "color": "#800020",
                                        "margin-bottom": "15px",
                                        "font-weight": "600",
                                    },
                                ),
                                dcc.DatePickerRange(
                                    id="date-range",
                                    min_date_allowed="2023-09-01",
                                    max_date_allowed=(
                                        data["timestamp"].max().date()
                                        - relativedelta(months=1)
                                    ),
                                    start_date="2023-09-01",
                                    end_date=(
                                        data["timestamp"].max().date()
                                        - relativedelta(months=1)
                                    ),
                                    display_format="MMM DD, YYYY",
                                    style={"font-size": "16px"},
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="graph-container",
                    style={"margin-top": "20px"},
                ),
            ],
        )
    elif tab == "tab-2":
        return html.Div(
            children=[
                dcc.Store(id="n-clicks-store", data=0),
                dcc.Store(id="mode", data=1),
                html.Div(
                    className="prediction-controls",
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                        "align-items": "center",
                        "margin-bottom": "40px",
                        "padding": "30px",
                        "background-color": "rgba(0, 0, 0, 0.03)",
                        "border-radius": "12px",
                    },
                    children=[
                        html.H2(
                            "Prediction Parameters",
                            style={
                                "color": "#800020",
                                "margin-bottom": "25px",
                                "font-weight": "600",
                            },
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "justify-content": "center",
                                "margin-bottom": "25px",
                                "width": "100%",
                            },
                            children=[
                                html.Button(
                                    "PM 2.5",
                                    id="select-pm",
                                    style={
                                        "background-color": "#800020",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "12px 24px",
                                        "margin": "0 10px",
                                        "border-radius": "8px",
                                        "font-weight": "600",
                                        "font-size": "16px",
                                        "cursor": "pointer",
                                        "transition": "background-color 0.3s",
                                    },
                                ),
                                html.Button(
                                    "Temperature",
                                    id="select-temp",
                                    style={
                                        "background-color": "#4ECDC4",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "12px 24px",
                                        "margin": "0 10px",
                                        "border-radius": "8px",
                                        "font-weight": "600",
                                        "font-size": "16px",
                                        "cursor": "pointer",
                                        "transition": "background-color 0.3s",
                                    },
                                ),
                                html.Button(
                                    "Humidity",
                                    id="select-humi",
                                    style={
                                        "background-color": "#45B7D1",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "12px 24px",
                                        "margin": "0 10px",
                                        "border-radius": "8px",
                                        "font-weight": "600",
                                        "font-size": "16px",
                                        "cursor": "pointer",
                                        "transition": "background-color 0.3s",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "align-items": "center",
                                "justify-content": "center",
                                "margin-top": "15px",
                                "width": "100%",
                            },
                            children=[
                                html.H3(
                                    id="predict-name",
                                    style={
                                        "color": "#800020",
                                        "margin-right": "20px",
                                        "font-weight": "500",
                                    },
                                ),
                                dcc.Input(
                                    id="input-day",
                                    type="number",
                                    min=1,
                                    # max=30,
                                    value=7,
                                    placeholder="days",
                                    style={
                                        "padding": "10px 15px",
                                        "border-radius": "8px",
                                        "border": "1px solid #ccc",
                                        "width": "100px",
                                        "margin-right": "15px",
                                        "font-size": "16px",
                                    },
                                ),
                                html.Button(
                                    "Generate Prediction",
                                    id="predict-button",
                                    style={
                                        "background-color": "#800020",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "12px 24px",
                                        "border-radius": "8px",
                                        "font-weight": "600",
                                        "font-size": "16px",
                                        "cursor": "pointer",
                                        "transition": "background-color 0.3s",
                                        "box-shadow": "0 4px 6px rgba(128, 0, 32, 0.2)",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            id="loading-container",
                            style={"margin-top": "15px", "width": "100%"},
                            children=[
                                dcc.Loading(
                                    id="loading-1",
                                    type="circle",
                                    color="#800020",
                                    children=[
                                        dcc.Store(id="show-data", data=0),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="results-container",
                    style={
                        "display": "grid",
                        "grid-template-columns": "1fr 1fr",
                        "gap": "30px",
                        "margin-top": "10px",
                    },
                    children=[
                        html.Div(
                            className="indicator-card",
                            style={
                                "background-color": "#fff",
                                "border-radius": "16px",
                                "padding": "20px",
                                "box-shadow": "0 8px 20px rgba(0, 0, 0, 0.06)",
                                "transition": "transform 0.3s ease",
                                "height": "400px",
                            },
                            children=[
                                dcc.Graph(
                                    id="indicator-pre",
                                    figure=fig,
                                    style={"height": "100%"},
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),
                        html.Div(
                            className="chart-card",
                            style={
                                "background-color": "#fff",
                                "border-radius": "16px",
                                "padding": "20px",
                                "box-shadow": "0 8px 20px rgba(0, 0, 0, 0.06)",
                                "transition": "transform 0.3s ease",
                            },
                            children=[
                                html.Div(id="prediction-output"),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="navigation-controls",
                    style={
                        "display": "flex",
                        "justify-content": "center",
                        "margin-top": "30px",
                        "gap": "20px",
                    },
                    children=[
                        html.Div(
                            id="pre",
                            style={"display": "flex", "gap": "15px"},
                            children=[
                                html.Button(
                                    "Previous Day",
                                    style={
                                        "display": "none",
                                        "background-color": "#555",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "12px 24px",
                                        "border-radius": "8px",
                                        "font-weight": "500",
                                        "cursor": "pointer",
                                    },
                                    id="prev-button",
                                    n_clicks=0,
                                ),
                                html.Button(
                                    "Next Day",
                                    style={
                                        "display": "none",
                                        "background-color": "#555",
                                        "color": "white",
                                        "border": "none",
                                        "padding": "12px 24px",
                                        "border-radius": "8px",
                                        "font-weight": "500",
                                        "cursor": "pointer",
                                    },
                                    id="next-button",
                                    n_clicks=0,
                                ),
                            ],
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
        data = eng
    elif city_name == "surat":
        data = surat

    # Convert timestamps if needed
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    mask = data["timestamp"].between(start_date, end_date)
    filtered_df = data[mask]

    graphs = []

    if selected_values is None or len(selected_values) == 0:
        return html.Div(
            html.P(
                "Please select at least one parameter to display",
                style={
                    "text-align": "center",
                    "color": "#666",
                    "font-size": "18px",
                    "margin": "40px 0",
                },
            )
        )
    else:
        for value in selected_values:
            title = value.replace("_", " ").title()
            graph = create_line_graph(filtered_df, value, title)
            graphs.append(
                html.Div(
                    className="graph-card",
                    style={
                        "background-color": "#fff",
                        "border-radius": "16px",
                        "padding": "20px",
                        "box-shadow": "0 8px 20px rgba(0, 0, 0, 0.06)",
                        "margin-bottom": "25px",
                        "transition": "transform 0.3s ease",
                    },
                    children=[graph],
                )
            )
        return graphs


@callback(
    Output("mode", "data"),
    Output("predict-name", "children"),
    Input("select-pm", "n_clicks"),
    Input("select-temp", "n_clicks"),
    Input("select-humi", "n_clicks"),
)
def mode(a, b, c):
    ctx = callback_context

    if not ctx.triggered:
        return 1, "Predict PM 2.5"

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "select-pm":
        return 1, "Predict PM 2.5 for"
    elif button_id == "select-temp":
        return 2, "Predict Temperature for"
    elif button_id == "select-humi":
        return 3, "Predict Humidity for"


@callback(
    Output("show-data", "data"),
    Input("predict-button", "n_clicks"),
    State("input-day", "value"),
    State("url", "pathname"),
    State("mode", "data"),
)
def predic(n_click, day, pathname, mode):
    city_name = pathname.split("/")[-1]

    ctx = callback_context

    if not ctx.triggered:
        return 0

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    global result_clean

    if button_id == "predict-button" and day is not None:
        if city_name == "eng_psu":
            if mode == 1:
                result = eng_model_predict_2_5(int(day) + 2)
            elif mode == 2:
                result = eng_model_temp(int(day) + 2)
            elif mode == 3:
                result = eng_model_humidity(int(day) + 2)
        elif city_name == "surat":
            if mode == 1:
                result = surat_model_predict_2_5(int(day) + 2)
            elif mode == 2:
                result = surat_model_temp(int(day) + 2)
            elif mode == 3:
                result = surat_model_humidity(int(day) + 2)

        # Remove NaN values from predictions
        result_clean = result.dropna(subset=["Predictions"])
        return 1

    return 0


@callback(
    Output("n-clicks-store", "data"),
    Input("prev-button", "n_clicks"),
    Input("next-button", "n_clicks"),
    Input("input-day", "value"),
    State("n-clicks-store", "data"),
    Input("predict-button", "n_clicks"),
)
def update_n_clicks_store(prev_clicks, next_clicks, day, stored_n_clicks, pre):
    ctx = callback_context

    if not ctx.triggered or day is None:
        return stored_n_clicks

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "prev-button":
        if stored_n_clicks <= 0:
            return 0
        else:
            return stored_n_clicks - 1
    elif button_id == "next-button":
        if stored_n_clicks >= (int(day) - 1):
            return int(day) - 1
        else:
            return stored_n_clicks + 1
    elif button_id == "predict-button":
        return 0
    else:
        return stored_n_clicks


@callback(
    Output("pre", "children"),
    Output("prediction-output", "children"),
    Output("indicator-pre", "figure"),
    Input("n-clicks-store", "data"),
    Input("show-data", "data"),
    Input("mode", "data"),
)
def last_pre(click, data, mode):
    # Set parameter name based on mode
    if mode == 1:
        name = "PM 2.5"
        color = "#800020"
        unit = "µg/m³"
    elif mode == 2:
        name = "Temperature"
        color = "#4ECDC4"
        unit = "°C"
    elif mode == 3:
        name = "Humidity"
        color = "#45B7D1"
        unit = "%"

    # Default empty figure
    empty_fig = go.Figure()
    empty_fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 33], "color": "rgba(0,255,0,0.2)"},
                    {"range": [33, 66], "color": "rgba(255,255,0,0.2)"},
                    {"range": [66, 100], "color": "rgba(255,0,0,0.2)"},
                ],
            },
            value=0,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": f"{name}"},
        )
    )

    if data == 1:
        # Create line chart for predictions
        fig_pre = px.line(
            result_clean,
            x=result_clean.index,
            y="Predictions",
            labels={"index": "Date", "Predictions": f"{name} ({unit})"},
            title=f"{name} Forecast",
        )

        # Improve line chart styling
        fig_pre.update_traces(
            line=dict(color=color, width=3),
            mode="lines+markers",
            marker=dict(size=8, line=dict(width=2, color="white")),
        )

        fig_pre.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="rgba(240, 240, 240, 0.5)",
            margin=dict(l=40, r=40, t=80, b=40),
            title={
                "font": {
                    "size": 22,
                    "color": "#333",
                    "family": "Segoe UI, Roboto, sans-serif",
                },
                "x": 0.5,
                "xanchor": "center",
            },
            xaxis=dict(
                title="Date",
                titlefont=dict(size=14, color="#333"),
                showgrid=True,
                gridcolor="rgba(0, 0, 0, 0.1)",
            ),
            yaxis=dict(
                title=f"{name} ({unit})",
                titlefont=dict(size=14, color="#333"),
                showgrid=True,
                gridcolor="rgba(0, 0, 0, 0.1)",
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Segoe UI, Roboto, sans-serif",
            ),
        )

        # Highlight the selected day
        x_point = result_clean.index[click]
        y_point = result_clean["Predictions"][click]

        fig_pre.add_trace(
            go.Scatter(
                x=[x_point],
                y=[y_point],
                mode="markers",
                marker=dict(
                    size=14,
                    color="white",
                    line=dict(width=3, color=color),
                ),
                name=f"Day {click + 1}",
                hovertemplate=f"<b>Date</b>: %{x_point}<br><b>{name}</b>: %{y_point:.1f} {unit}<extra></extra>",
            )
        )

        # Create gauge indicator
        indi = go.Figure()

        # Set max range based on parameter
        if mode == 1:  # PM 2.5
            max_range = 150
            steps = [
                {"range": [0, 12], "color": "rgba(0,255,0,0.4)", "name": "Good"},
                {
                    "range": [12, 35.4],
                    "color": "rgba(255,255,0,0.4)",
                    "name": "Moderate",
                },
                {
                    "range": [35.4, 55.4],
                    "color": "rgba(255,165,0,0.4)",
                    "name": "Unhealthy for Sensitive Groups",
                },
                {
                    "range": [55.4, 150],
                    "color": "rgba(255,0,0,0.4)",
                    "name": "Unhealthy",
                },
            ]
        elif mode == 2:  # Temperature
            max_range = 50
            steps = [
                {"range": [0, 15], "color": "rgba(0,0,255,0.4)", "name": "Cool"},
                {"range": [15, 25], "color": "rgba(0,255,0,0.4)", "name": "Pleasant"},
                {"range": [25, 35], "color": "rgba(255,255,0,0.4)", "name": "Warm"},
                {"range": [35, 50], "color": "rgba(255,0,0,0.4)", "name": "Hot"},
            ]
        else:  # Humidity
            max_range = 100
            steps = [
                {"range": [0, 30], "color": "rgba(255,165,0,0.4)", "name": "Dry"},
                {
                    "range": [30, 60],
                    "color": "rgba(0,255,0,0.4)",
                    "name": "Comfortable",
                },
                {"range": [60, 100], "color": "rgba(0,0,255,0.4)", "name": "Humid"},
            ]

        indi.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=y_point,
                domain={"y": [0, 1], "x": [0.05, 0.95]},
                title={
                    "text": f"{name} - Day {click + 1}",
                    "font": {
                        "size": 24,
                        "color": "#333",
                        "family": "Segoe UI, Roboto, sans-serif",
                    },
                },
                gauge={
                    "axis": {
                        "range": [None, max_range],
                        "tickwidth": 1,
                        "tickcolor": "#333",
                    },
                    "bar": {"color": color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 2,
                    "bordercolor": "#333",
                    "steps": steps,
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": y_point,
                    },
                },
                number={"suffix": f" {unit}", "font": {"size": 26}},
            )
        )

        # Update layout
        indi.update_layout(
            paper_bgcolor="white",
            margin=dict(l=30, r=30, t=100, b=30),
        )

        # Buttons for navigation
        navigation_buttons = [
            html.Button(
                "← Previous Day",
                id="prev-button",
                n_clicks=0,
                style={
                    "background-color": "#666",
                    "color": "white",
                    "border": "none",
                    "padding": "12px 24px",
                    "border-radius": "8px",
                    "font-weight": "500",
                    "cursor": "pointer",
                    "transition": "background-color 0.3s",
                    "box-shadow": "0 4px 6px rgba(0,0,0,0.1)",
                },
            ),
            html.Button(
                "Next Day →",
                id="next-button",
                n_clicks=0,
                style={
                    "background-color": "#800020",
                    "color": "white",
                    "border": "none",
                    "padding": "12px 24px",
                    "border-radius": "8px",
                    "font-weight": "500",
                    "cursor": "pointer",
                    "transition": "background-color 0.3s",
                    "box-shadow": "0 4px 6px rgba(128,0,32,0.2)",
                },
            ),
        ]

        # Add summary stats
        prediction_summary = html.Div(
            style={
                "height": "100%",
                "display": "flex",
                "flex-direction": "column",
                "justify-content": "space-between",
            },
            children=[
                dcc.Graph(
                    figure=fig_pre,
                    style={"height": "100%"},
                ),
                html.Div(
                    style={
                        "margin-top": "15px",
                        "padding": "15px",
                        "background-color": "rgba(0,0,0,0.03)",
                        "border-radius": "8px",
                    },
                    children=[
                        html.H4(
                            f"Forecast Summary",
                            style={"margin-bottom": "10px", "color": "#800020"},
                        ),
                        html.Div(
                            style={
                                "display": "grid",
                                "grid-template-columns": "1fr 1fr",
                                "gap": "10px",
                            },
                            children=[
                                html.Div(
                                    [
                                        html.P(
                                            f"Average: {result_clean['Predictions'].mean():.1f} {unit}",
                                            style={
                                                "margin": "5px 0",
                                                "font-size": "14px",
                                            },
                                        ),
                                        html.P(
                                            f"Maximum: {result_clean['Predictions'].max():.1f} {unit}",
                                            style={
                                                "margin": "5px 0",
                                                "font-size": "14px",
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.P(
                                            f"Minimum: {result_clean['Predictions'].min():.1f} {unit}",
                                            style={
                                                "margin": "5px 0",
                                                "font-size": "14px",
                                            },
                                        ),
                                        html.P(
                                            f"Selected: {y_point:.1f} {unit} (Day {click + 1})",
                                            style={
                                                "margin": "5px 0",
                                                "font-size": "14px",
                                                "font-weight": "bold",
                                                "color": color,
                                            },
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )

        return navigation_buttons, prediction_summary, indi

    else:
        # Return empty/default components when no prediction data is available
        return (
            [
                html.Button(
                    "Previous Day",
                    style={"display": "none"},
                    id="prev-button",
                    n_clicks=0,
                ),
                html.Button(
                    "Next Day",
                    style={"display": "none"},
                    id="next-button",
                    n_clicks=0,
                ),
            ],
            html.Div(
                html.P(
                    "Generate a prediction to see results",
                    style={
                        "text-align": "center",
                        "color": "#666",
                        "font-size": "18px",
                        "margin": "40px 0",
                    },
                )
            ),
            empty_fig,
        )
