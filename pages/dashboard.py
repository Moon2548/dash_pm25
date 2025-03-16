from dash import (
    Dash,
    html,
    dcc,
    callback,
    Input,
    Output,
    register_page,
    callback_context,
)
import dash_leaflet as dl
import pandas as pd
import plotly.express as px

register_page(__name__, path="/")

eng = pd.read_excel("export-pm25_eng-1d.xlsx")

markers = [
    {"id": "eng_psu", "position": [7.0070763, 100.5021177], "name": "Eng PSU"},
    {"id": "surat", "position": [9.139554, 99.3301465], "name": "Surat"},
]

thailand_bounds = [[25.5, 92.3], [0.6, 110.7]]

# Layout with improved aesthetics
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
        html.Div(
            className="title-container",
            style={
                "text-align": "center",
                "margin-bottom": "30px",
            },
            children=[
                html.H1(
                    "PM 2.5 Predictions Dashboard",
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
                    "Interactive air quality monitoring system for Thailand",
                    style={
                        "color": "#555",
                        "font-size": "18px",
                        "margin-top": "10px",
                        "font-style": "italic",
                    },
                ),
            ],
        ),
        html.Div(
            style={
                "display": "grid",
                "grid-template-columns": "1fr 1.5fr",
                "gap": "35px",
                "margin-top": "20px",
            },
            children=[
                html.Div(
                    className="map-container",
                    style={
                        "background-color": "#ffffff",
                        "border-radius": "16px",
                        "padding": "20px",
                        "box-shadow": "0 8px 20px rgba(0, 0, 0, 0.06)",
                        "transition": "transform 0.3s ease",
                    },
                    children=[
                        html.H3(
                            "Monitoring Locations",
                            style={
                                "color": "#800020",
                                "margin-top": "0",
                                "margin-bottom": "15px",
                                "font-weight": "600",
                                "border-left": "4px solid #800020",
                                "padding-left": "10px",
                            },
                        ),
                        html.P(
                            "Click on a marker to view detailed statistics",
                            style={
                                "color": "#666",
                                "margin-bottom": "15px",
                                "font-size": "14px",
                            },
                        ),
                        dl.Map(
                            center=[13.7563, 100.5018],
                            zoom=5,
                            minZoom=5,
                            children=[
                                dl.TileLayer(
                                    url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                                ),
                                dl.LayerGroup(
                                    [
                                        dl.Marker(
                                            id=m["id"],
                                            position=m["position"],
                                            children=dl.Tooltip(
                                                m["name"],
                                                permanent=False,
                                                direction="top",
                                            ),
                                            icon={
                                                "iconUrl": "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
                                                "iconSize": [25, 41],
                                                "iconAnchor": [12, 41],
                                            },
                                        )
                                        for m in markers
                                    ]
                                ),
                            ],
                            id="map",
                            maxBounds=thailand_bounds,
                            style={
                                "width": "100%",
                                "height": "500px",
                                "border-radius": "12px",
                                "border": "3px solid #800020",
                                "box-shadow": "0 4px 12px rgba(128, 0, 32, 0.15)",
                            },
                        ),
                    ],
                ),
                html.Div(
                    className="stats-container",
                    style={
                        "background-color": "#ffffff",
                        "border-radius": "16px",
                        "padding": "25px",
                        "box-shadow": "0 8px 20px rgba(0, 0, 0, 0.06)",
                        "display": "flex",
                        "flex-direction": "column",
                        "justify-content": "center",
                    },
                    children=[
                        html.H3(
                            "Location Statistics",
                            style={
                                "color": "#800020",
                                "margin-top": "0",
                                "margin-bottom": "25px",
                                "font-weight": "600",
                                "border-left": "4px solid #800020",
                                "padding-left": "10px",
                            },
                        ),
                        html.Div(
                            id="output",
                            style={
                                "color": "#444",
                                "font-size": "18px",
                                "margin-bottom": "30px",
                                "padding": "15px",
                                "background-color": "#f9f9f9",
                                "border-radius": "8px",
                                "border-left": "4px solid #800020",
                            },
                        ),
                        html.Div(
                            className="stats-cards",
                            style={
                                "display": "grid",
                                "grid-template-columns": "repeat(3, 1fr)",
                                "gap": "20px",
                                "margin-bottom": "30px",
                            },
                            children=[
                                html.Div(
                                    className="stat-card",
                                    style={
                                        "background": "linear-gradient(135deg, #fff8e1 0%, #ffe0b2 100%)",
                                        "border-radius": "12px",
                                        "padding": "20px",
                                        "text-align": "center",
                                        "box-shadow": "0 4px 10px rgba(255, 215, 0, 0.2)",
                                    },
                                    children=[
                                        html.H4(
                                            "PM 2.5",
                                            style={
                                                "margin": "0 0 10px 0",
                                                "color": "#b4530a",
                                                "font-size": "16px",
                                            },
                                        ),
                                        html.Div(
                                            id="average_pm25",
                                            style={
                                                "color": "#b4530a",
                                                "font-size": "24px",
                                                "font-weight": "bold",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="stat-card",
                                    style={
                                        "background": "linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)",
                                        "border-radius": "12px",
                                        "padding": "20px",
                                        "text-align": "center",
                                        "box-shadow": "0 4px 10px rgba(66, 165, 245, 0.2)",
                                    },
                                    children=[
                                        html.H4(
                                            "Temperature",
                                            style={
                                                "margin": "0 0 10px 0",
                                                "color": "#0d47a1",
                                                "font-size": "16px",
                                            },
                                        ),
                                        html.Div(
                                            id="average_temperature",
                                            style={
                                                "color": "#0d47a1",
                                                "font-size": "24px",
                                                "font-weight": "bold",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="stat-card",
                                    style={
                                        "background": "linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)",
                                        "border-radius": "12px",
                                        "padding": "20px",
                                        "text-align": "center",
                                        "box-shadow": "0 4px 10px rgba(76, 175, 80, 0.2)",
                                    },
                                    children=[
                                        html.H4(
                                            "Humidity",
                                            style={
                                                "margin": "0 0 10px 0",
                                                "color": "#2e7d32",
                                                "font-size": "16px",
                                            },
                                        ),
                                        html.Div(
                                            id="average_humidity",
                                            style={
                                                "color": "#2e7d32",
                                                "font-size": "24px",
                                                "font-weight": "bold",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="link",
                            style={
                                "text-align": "center",
                                "margin-top": "20px",
                            },
                        ),
                    ],
                ),
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


@callback(
    Output("output", "children"),
    Output("average_pm25", "children"),
    Output("average_temperature", "children"),
    Output("average_humidity", "children"),
    Output("link", "children"),
    [Input(m["id"], "n_clicks") for m in markers],
)
def display_marker_info(*args):
    ctx = callback_context
    if not ctx.triggered:
        return (
            html.Div(
                [
                    html.I(
                        className="fas fa-map-marker-alt", style={"margin-right": "8px"}
                    ),
                    "Select a monitoring station on the map to view data",
                ]
            ),
            "—",
            "—",
            "—",
            "",
        )

    clicked_id = ctx.triggered[0]["prop_id"].split(".")[0]
    marker_info = next((m for m in markers if m["id"] == clicked_id), None)

    if clicked_id == "eng_psu":
        data = eng
    if clicked_id == "surat":
        data = pd.read_csv("export_data\\clean_data_jsps001_1d.csv")

    if marker_info:
        pm25_value = round(data["pm_2_5"].mean(), 2)
        temp_value = round(data["temperature"].mean(), 2)
        humidity_value = round(data["humidity"].mean(), 2)

        # Add air quality assessment based on PM2.5 value
        pm25_status = ""
        if pm25_value <= 12:
            pm25_status = "Good"
        elif pm25_value <= 35.4:
            pm25_status = "Moderate"
        elif pm25_value <= 55.4:
            pm25_status = "Unhealthy for Sensitive Groups"
        elif pm25_value <= 150.4:
            pm25_status = "Unhealthy"
        elif pm25_value <= 250.4:
            pm25_status = "Very Unhealthy"
        else:
            pm25_status = "Hazardous"

        return (
            html.Div(
                [
                    html.Div(
                        style={
                            "display": "flex",
                            "align-items": "center",
                            "margin-bottom": "15px",
                        },
                        children=[
                            html.I(
                                className="fas fa-map-marker-alt",
                                style={
                                    "color": "#800020",
                                    "font-size": "24px",
                                    "margin-right": "10px",
                                },
                            ),
                            html.Span(
                                f"{marker_info['name']}",
                                style={
                                    "font-size": "22px",
                                    "font-weight": "bold",
                                    "color": "#333",
                                },
                            ),
                        ],
                    ),
                    html.Div(
                        f"Coordinates: {marker_info['position'][0]:.6f}, {marker_info['position'][1]:.6f}",
                        style={
                            "color": "#666",
                            "font-size": "14px",
                            "margin-left": "34px",
                        },
                    ),
                ]
            ),
            html.Div(
                [
                    f"{pm25_value}",
                    html.Span(
                        " µg/m³", style={"font-size": "14px", "font-weight": "normal"}
                    ),
                    html.Div(
                        pm25_status,
                        style={
                            "font-size": "14px",
                            "margin-top": "5px",
                            "font-weight": "normal",
                            "color": "#666",
                        },
                    ),
                ]
            ),
            html.Div(
                [
                    f"{temp_value}",
                    html.Span(
                        " °C", style={"font-size": "14px", "font-weight": "normal"}
                    ),
                ]
            ),
            html.Div(
                [
                    f"{humidity_value}",
                    html.Span(
                        " %", style={"font-size": "14px", "font-weight": "normal"}
                    ),
                ]
            ),
            html.Div(
                html.A(
                    children=[
                        html.I(
                            className="fas fa-chart-line", style={"margin-right": "8px"}
                        ),
                        "View Detailed Analytics",
                    ],
                    href=f"/location/{marker_info['id']}",
                    target="_blank",
                    style={
                        "display": "inline-block",
                        "padding": "12px 24px",
                        "color": "white",
                        "background-color": "#800020",
                        "border-radius": "8px",
                        "text-decoration": "none",
                        "font-size": "16px",
                        "font-weight": "500",
                        "transition": "background-color 0.3s ease",
                        "box-shadow": "0 4px 6px rgba(128, 0, 32, 0.2)",
                    },
                ),
                style={"text-align": "center", "margin-top": "20px"},
            ),
        )
    return "Error: ไม่พบข้อมูล Marker", "", "", "", ""
