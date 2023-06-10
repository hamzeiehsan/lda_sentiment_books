import numpy as np
import matplotlib
import seaborn as sns
import plotly.express as px
import pyLDAvis.gensim_models
import pyLDAvis

class Vis:
    def __init__(self, df, notebook=False):
        if notebook:
            pyLDAvis.enable_notebook()  # Visualise inside a notebook
        self.df = df
        self.colorset_treemap = [
            "#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7",
            "#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"
        ]

    def treemap_chapter_sentiments(self, file_name='chapter_sentiments.html'):
        df_tmp = self.df.\
            groupby(['book', 'chapter', 'sentiment'], as_index=False).\
            agg({'paragraph_number': ['count']})
        df_tmp.columns = df_tmp.columns.droplevel(1)
        fig = px.treemap(df_tmp, path=[px.Constant("bookshelf"), 'book', 'chapter', 'sentiment'],
                         values='paragraph_number',
                         color='sentiment',
                         color_discrete_map={'(?)': 'lightgrey',
                                             'POSITIVE': "#b2e061",
                                             'NEUTRAL': "#7eb0d5",
                                             'NEGATIVE': "#fd7f6f"})
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.write_html(file_name)


    def treemap_chapter_theme(self, file_name='chapter_topic.html'):
        df_tmp = self.df. \
            groupby(['book', 'chapter', 'dominant_topic'], as_index=False). \
            agg({'paragraph_number': ['count']})
        df_tmp.columns = df_tmp.columns.droplevel(1)
        df_tmp['dominant_topic'] = df_tmp['dominant_topic'].astype(str)

        topics = list(self.df['dominant_topic'].unique())
        color_map = {'(?)':'lightgrey'}

        for tid, t in enumerate(topics):
            color_map[str(t)] = self.colorset_treemap[tid]
        print(color_map)
        fig = px.treemap(df_tmp, path=[px.Constant("bookshelf"), 'book', 'chapter', 'dominant_topic'],
                         values='paragraph_number',
                         color='dominant_topic',
                         color_discrete_map=color_map)
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.write_html(file_name)

    def treemap_topic_sentiment(self, file_name='topic_sentiment.html'):
        df_tmp = self.df. \
            groupby(['book', 'dominant_topic', 'sentiment'], as_index=False). \
            agg({'paragraph_number': ['count']})
        df_tmp.columns = df_tmp.columns.droplevel(1)

        fig = px.treemap(df_tmp, path=[px.Constant("bookshelf"), 'book',
                                       'dominant_topic',
                                       'sentiment'],
                         values='paragraph_number',
                         color='sentiment',
                         color_discrete_map={'(?)': 'lightgrey',
                                             'POSITIVE': "#b2e061",
                                             'NEUTRAL': "#7eb0d5",
                                             'NEGATIVE': "#fd7f6f"})
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.write_html(file_name)
