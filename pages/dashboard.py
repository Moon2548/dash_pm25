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
    # {"id": "bangkok", "position": [13.7563, 100.5018], "name": "Bangkok"},
    {"id": "eng_psu", "position": [7.0070763, 100.5021177], "name": "Eng PSU"},
    {"id": "surat", "position": [9.139554, 99.3301465], "name": "Surat"},
]
thailand_bounds = [[25.5, 92.3], [0.6, 110.7]]

# Layout ของ Dash App
layout = html.Div(
    style={
        "background-color": "black",
        "border-radius": "10px",
        "padding": "20px",
    },
    children=[
        html.Div(
            className="title",
            children=[
                html.H1("🌪️ PM 2.5 Predictions 💨"),
            ],
        ),
        html.Div(
            style={"display": "grid", "grid-template-columns": "1fr 1.5fr"},
            children=[
                html.Div(
                    className="cards",
                    children=[
                        dl.Map(
                            center=[13.7563, 100.5018],
                            zoom=5,
                            minZoom=5,
                            children=[
                                dl.TileLayer(),
                                dl.LayerGroup(
                                    [
                                        dl.Marker(
                                            id=m["id"],
                                            position=m["position"],
                                            children=dl.Tooltip(m["name"]),
                                        )
                                        for m in markers
                                    ]
                                ),
                            ],
                            id="map",
                            maxBounds=thailand_bounds,  # กำหนดขอบเขตประเทศไทย
                            style={"width": "500px", "height": "500px"},
                        ),
                    ],
                ),
                html.Div(
                    className="cards",
                    style={"font-size": "20px"},
                    children=[
                        html.Div(
                            id="output",
                            style={"color": "white", "font-size": "20px"},
                        ),  # แสดงข้อมูล Marker ที่ถูกคลิก
                        # html.Div(
                        #     className="aver",
                        #     style={"color": "lightgreen"},
                        #     id="average_pm10",
                        # ),
                        html.Div(
                            className="aver",
                            style={"color": "yellow"},
                            id="average_pm25",
                        ),
                        html.Div(
                            className="aver",
                            style={"color": "orange"},
                            id="average_temperature",
                        ),
                        html.Div(
                            className="aver",
                            style={"color": "red"},
                            id="average_humidity",
                        ),
                        html.Div(id="link"),
                    ],
                ),
            ],
        ),
    ],
)


# Callback ทำงานเมื่อคลิกที่ Marker
@callback(
    Output("output", "children"),
    # Output("average_pm10", "children"),
    Output("average_pm25", "children"),
    Output("average_temperature", "children"),
    Output("average_humidity", "children"),
    Output("link", "children"),
    [Input(m["id"], "n_clicks") for m in markers],  # ดักจับการคลิกทุก Marker
)
def display_marker_info(*args):
    ctx = callback_context  # ตรวจสอบว่า input ไหนถูกกด
    if not ctx.triggered:
        return "คลิกที่ Marker เพื่อดูข้อมูล", "", "", "", ""

    clicked_id = ctx.triggered[0]["prop_id"].split(".")[0]  # หาค่า ID ที่ถูกคลิก
    marker_info = next((m for m in markers if m["id"] == clicked_id), None)

    if clicked_id == "eng_psu":
        data = eng
    if clicked_id == "surat":
        data = pd.read_csv("export_data\clean_data_jsps001_1d.csv")

    if marker_info:
        return (
            f"คุณกดที่ {marker_info['name']} (Lat: {marker_info['position'][0]}, Lon: {marker_info['position'][1]})",
            # f"ค่าเฉลี่ย PM10: {data['pm_10'].mean()}",
            f"ค่าเฉลี่ย PM2.5: {data['pm_2_5'].mean()}",
            f"ค่าเฉลี่ยอุณหภูมิ: {data['temperature'].mean()}",
            f"ค่าเฉลี่ยความชื้น: {data['humidity'].mean()}",
            html.A(
                "🔗 ดูข้อมูลเพิ่มเติม",
                href=f"/location/{marker_info['id']}",
                target="_blank",  # ✅ ให้เปิดในแท็บใหม่
                style={
                    "display": "inline-block",
                    "margin-top": "10px",
                    "color": "black",
                    "text-decoration": "none",
                    "background-color": "lightblue",
                    "border-radius": "5px",
                },
            ),
        )
    return "Error: ไม่พบข้อมูล Marker", "", "", "", ""
