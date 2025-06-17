import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import os

# Load and prepare data
df = pd.read_csv('outputs/freedom_viz_ready.csv')
df = df[df['english_coword'] != 'freedom'] # Removing freedom entries
df_agg = df.groupby(['english_coword', 'lang_name'])['count'].first().reset_index()

# Find top words by total count across all languages
top_words = df_agg.groupby('english_coword')['count'].sum().sort_values(ascending=False)
top_word_list = top_words.index.tolist()

# Filter aggregated data to only include top words
df_final = df_agg[df_agg['english_coword'].isin(top_word_list)]

#! Heatmap
heatmap_df = df_final.pivot(index='english_coword', columns='lang_name', values='count').fillna(0)
heatmap_df['total'] = heatmap_df.sum(axis=1)
heatmap_df = heatmap_df.sort_values('total', ascending=False).drop(columns='total')
heatmap_subset = heatmap_df.iloc[:10, :10]
heatmap_fig = px.imshow(
    heatmap_subset,
    labels=dict(x="Language", y="Co-Word", color="Frequency"),
    x=heatmap_subset.columns,
    y=heatmap_subset.index,
    title="Top Co-Words by Total Frequency Across Languages",
    aspect="auto",
    color_continuous_scale="Teal"
)
heatmap_fig.update_xaxes(side="top")

#! Barchart
bar_fig_subset = top_words.head(25)
bar_fig = px.bar(
    x=bar_fig_subset.values[::-1],
    y=bar_fig_subset.index[::-1],
    orientation='h',
    labels={'x': 'Total Frequency', 'y': 'Co-Words'},
    title="Top Co-Words by Total Frequency",
    color=bar_fig_subset.values[::-1],
    color_continuous_scale="Teal" 
)
bar_fig.update_layout(
    plot_bgcolor="#fcfbfb",
    paper_bgcolor='white',
    height=700
)

# Create separate bar charts for each language
languages = df_agg['lang_name'].unique()

#! Bar charts for each language
language_figs = {}
for lang in languages:
    # Filter data for this language
    lang_data = df_agg[df_agg['lang_name'] == lang]
    
    # Get top 10 words for this language
    top_10_lang = lang_data.nlargest(10, 'count')
    
    # Create bar chart for this language
    lang_fig = px.bar(
        x=top_10_lang['count'][::-1],  # Reverse for descending order
        y=top_10_lang['english_coword'][::-1],  # Reverse for descending order
        orientation='h',
        labels={'x': 'Frequency', 'y': 'Co-Words'},
        title=f"Top 10 Co-Words in {lang}",
        color=top_10_lang['count'][::-1],
        color_continuous_scale="Teal"
    )
    
    lang_fig.update_layout(
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='white',
        height=700  # Set consistent height
    )
    
    language_figs[lang] = lang_fig


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Cross-Language Freedom Analysis"),
    
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Heatmap across languages', value='tab-1'),
        dcc.Tab(label='Bar Chart across languages', value='tab-2'),
        dcc.Tab(label='Bar charts for each language', value='tab-3'),
    ]),
    
    html.Div(id='tab-content')
])

@app.callback(Output('tab-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            dcc.Graph(figure=heatmap_fig),
            html.Div([
                html.H2("Observations"),
                html.B("This heatmap shows the frequency of co-words across different languages. Darker colors indicate higher frequency of usage."),
                html.P("We can see that 'be' is the most common word across 3 langauges, likely involved in sentences such as 'Freedom is important', etc etc."),
                html.P("English is responsible for a bit over 80% of the uses of the word 'fight' and 'more', implying that freedom is something that should be expanded, and fought for. English also has a strong presence of 'do', perhaps hinting that English's relationship to freedom is more action-oriented (Freedom to do X or Y)."),
                html.P("An interesting note is that in  Spanish, 'expression' is a commonly used word - frequently used as part of the expression 'la libertad de expresión'. Italian also has some uses of 'expression' as well, so perhaps this is a phrase common in romance languages."),
            ], style={
                'padding': '10px', 
                'margin': '10px 0',
                'borderRadius': '5px',
                'fontSize': '16px',
                'color': '#333'
            })
        ])
    elif tab == 'tab-2':
        return html.Div([
            dcc.Graph(figure=bar_fig),
            html.Div([
                html.H2("Observations"),
                html.B("This bar chart displays the most frequently used co-words with 'freedom' across all languages combined."),
                html.P("As seen before, 'be' is overwhelmingly the most common word - no surprises here. A few other expected words are present: 'religious', 'equality', and 'more'."),
                html.P("It seems that beyond 'be' and 'have', there is alot of diversity in how different langauges conceptualize freedom."),
                html.P("A few interesting unexpected ones were: 'sell', 'conditional' (what is conditional freedom? Again, this is romance language specific), and 'leave' (Are people leaving freedom behind? Are they being left out?)"),
            ], style={
                'padding': '10px', 
                'margin': '10px 0',
                'borderRadius': '5px',
                'fontSize': '16px',
                'color': '#333'
            })
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H2("Top Words by Language"),
            html.Div([
                html.Div([
                    dcc.Graph(figure=language_figs[lang])
                ], style={'width': '45%', 'display': 'inline-block'})
                for lang in languages
            ]),
            html.Div([
            html.B("This table shows the raw aggregated data used to create the visualizations. Each row represents a unique co-word and language combination with its frequency count."),
            html.P("I found it very interesting that only English ('religious') and Italian ('worship') mention spirituality in their 10 most frequent words."),
            html.P("Another interesting note that while German has 'earn freedom', English has 'right (to freedom)', possibly hinting that for English speakers, freedom in an unassailable right, and for German speakers it is something to be worked for."),
            ], style={
                'padding': '10px', 
                'margin': '10px 0',
                'borderRadius': '5px',
                'fontSize': '16px',
                'color': '#333'
            })
        ])

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)), debug=False) # host on render