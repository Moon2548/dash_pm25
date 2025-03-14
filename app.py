from dash import Dash, html, dcc, page_registry, page_container

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)

app.layout = html.Div(
    [
        # html.Div(
        #     className="bar",
        #     children=[
        #         html.Div(html.A(f"{page['name']}", href=page["relative_path"]))
        #         for page in page_registry.values()
        #     ],
        # ),
        page_container,
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
