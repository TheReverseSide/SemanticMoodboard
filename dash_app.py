import pandas as pd
import dash
from dash import dcc, html
import plotly.express as px

# Load and prepare data
df = pd.read_csv('outputs/freedom_viz_ready.csv')
df_agg = df.groupby(['english_coword', 'lang_name'])['count'].sum().reset_index()
# print(f"df agg {df_agg}")
heatmap_df = df_agg.pivot(index='english_coword', columns='lang_name', values='count').fillna(0)
print(f"pivoted agg")
print(heatmap_df.head(30))

# Create Dash app
app = dash.Dash(__name__)

fig = px.imshow(
    heatmap_df,
    labels=dict(x="Language", y="Co-Word", color="Frequency"),
    x=heatmap_df.columns,
    y=heatmap_df.index,
    aspect="auto",
    color_continuous_scale="Teal"
)
fig.update_xaxes(side="top")

app.layout = html.Div([
    html.H1("Cross-Language Freedom Heatmap"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run(debug=True)
