import logging
import math
import numpy as np
import pandas as pd
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pyLDAvis.gensim_models
import pyLDAvis
from plotnine import *


class Vis:
    def __init__(self, sentiment, lda, notebook=False):
        if notebook:
            pyLDAvis.enable_notebook()  # Visualise inside a notebook
        self.sentiment = sentiment
        self.lda = lda
        self.df = sentiment.df
        self.colorset_treemap = [
            "#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7",
            "#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"
        ]
        self.sent_sentiment = self.create_detailed_sentiment_info()

    def treemap_chapter_sentiments(self, file_name='figures/chapter_sentiments.html'):
        df_tmp = self.df. \
            groupby(['book', 'chapter', 'sentiment'], as_index=False). \
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
        logging.info('chapter-sentiment treemap html file is saved in {}'.format(file_name))

    def treemap_chapter_topic(self, file_name='figures/chapter_topic.html'):
        df_tmp = self.df. \
            groupby(['book', 'chapter', 'dominant_topic'], as_index=False). \
            agg({'paragraph_number': ['count']})
        df_tmp.columns = df_tmp.columns.droplevel(1)
        df_tmp['dominant_topic'] = df_tmp['dominant_topic'].astype(str)

        topics = list(self.df['dominant_topic'].unique())
        color_map = {'(?)': 'lightgrey'}

        for tid, t in enumerate(topics):
            color_map[str(t)] = self.colorset_treemap[tid]

        fig = px.treemap(df_tmp, path=[px.Constant("bookshelf"), 'book', 'chapter', 'dominant_topic'],
                         values='paragraph_number',
                         color='dominant_topic',
                         color_discrete_map=color_map)
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.write_html(file_name)
        logging.info('chapter-topic treemap html is saved in: \n\t{}'.format(file_name))

    def treemap_topic_sentiment(self, file_name='figures/topic_sentiment.html'):
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
        logging.info('topic-sentiment treemap html is saved in: \n\t{}'.format(file_name))

    def pyviz_topics(self, file_name='figures/pyvis_topics.html'):
        lda_display = pyLDAvis.gensim_models.prepare(self.lda.lda_model,
                                                     self.lda.bow_corpus,
                                                     self.lda.dictionary,
                                                     sort_topics=False)
        pyLDAvis.save_html(lda_display, file_name)
        logging.info('pyvis html visualization is saved in: \n\t{}'.format(file_name))

    @staticmethod
    def generate_topic_words(topic_text, stopwords):
        w_dict = {}
        words = topic_text.replace("\n", ' ').replace(".", " ").replace(", ", " ").replace("\t", ' ').split(' ')
        for w in words:
            if len(w) > 2 and w.lower() not in stopwords and "'" not in w:
                if w not in w_dict.keys():
                    w_dict[w] = 0
                w_dict[w] += 1
        return w_dict

    def generate_topics_wordcloud(self, file_name='figures/wordclouds.svg'):
        self.df["processed_text"] = self.df['processed'].apply(" ".join)
        topic_texts_df = self.df.groupby(['dominant_topic'])['processed_text'].apply(
            ' '.join).reset_index()

        cloud = WordCloud(background_color='white',
                          width=2500,
                          height=1800,
                          max_words=60,
                          prefer_horizontal=1.0)

        fig, axes = plt.subplots(math.ceil(len(topic_texts_df) / 2), 2, figsize=(10, 10),
                                 sharex=True, sharey=True)
        for idx, row in topic_texts_df.iterrows():
            # axs[0, 0].plot(x, y)
            # axs[0, 0].set_title('Axis [0, 0]')
            fig.add_subplot(axes[math.floor(idx / 2), idx % 2])
            topic_words = row['processed_text']
            cloud.generate_from_frequencies(Vis.generate_topic_words(
                topic_words, self.lda.processor.stopwords),
                max_font_size=300)

            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic ' + str(row['dominant_topic']), fontdict=dict(size=16))
            plt.gca().axis('off')
        if len(topic_texts_df) % 2 == 1:  # remove extra empty plot in case of odd number of topics
            fig.delaxes(axes[math.floor(len(topic_texts_df) / 2)][1])
        plt.savefig(file_name, format='svg', dpi=1200)
        logging.info("wordclouds are generated and saved in:\t{}".format(file_name))

    def generate_sentiments_wordcloud(self, file_name='figures/wordclouds_sentiments.svg'):
        self.df["processed_text"] = self.df['processed'].apply(" ".join)
        sentiment_texts_df = self.df.groupby(['sentiment'])['processed_text'].apply(
            ' '.join).reset_index()

        cloud = WordCloud(background_color='white',
                          width=2500,
                          height=1800,
                          max_words=60,
                          prefer_horizontal=1.0)

        fig, axes = plt.subplots(3, 1, figsize=(10, 10),
                                 sharex=True, sharey=True)
        for idx, row in sentiment_texts_df.iterrows():
            fig.add_subplot(axes[idx])
            sentiment_words = row['processed_text']
            cloud.generate_from_frequencies(Vis.generate_topic_words(
                sentiment_words, self.lda.processor.stopwords),
                max_font_size=300)

            plt.gca().imshow(cloud)
            plt.gca().set_title('Sentiment: ' + str(row['sentiment']), fontdict=dict(size=16))
            plt.gca().axis('off')
        plt.savefig(file_name, format='svg', dpi=1200)
        logging.info("wordclouds are generated and saved in:\t{}".format(file_name))

    def generate_topic_sentiments_wordcloud(self, file_name='figures/wordclouds_topic_sentiments.svg'):
        self.df["processed_text"] = self.df['processed'].apply(" ".join)
        st_texts_df = self.df.groupby(['dominant_topic', 'sentiment'])['processed_text'].apply(
            ' '.join).reset_index()

        cloud = WordCloud(background_color='white',
                          width=2500,
                          height=1800,
                          max_words=60,
                          prefer_horizontal=1.0)

        fig, axes = plt.subplots(len(st_texts_df['dominant_topic'].unique()), 3, figsize=(10, 10),
                                 sharex=True, sharey=True)
        for idx, row in st_texts_df.iterrows():
            col = 0
            if row['sentiment'] == 'NEUTRAL':
                col = 1
            elif row['sentiment'] == 'NEGATIVE':
                col = 2
            fig.add_subplot(axes[int(row['dominant_topic']) - 1, col])
            st_words = row['processed_text']
            cloud.generate_from_frequencies(Vis.generate_topic_words(
                st_words, self.lda.processor.stopwords),
                max_font_size=300)

            plt.gca().imshow(cloud)
            plt.gca().set_title('Topic {0}:{1}'.format(str(row['dominant_topic']),
                                                       str(row['sentiment'])), fontdict=dict(size=16))
            plt.gca().axis('off')
        plt.savefig(file_name, format='svg', dpi=1200)
        logging.info("wordclouds are generated and saved in:\t{}".format(file_name))

    def create_detailed_sentiment_info(self):
        list_dict_info = []
        for idx, row in self.df.iterrows():
            if isinstance(row['sentiment_details'], dict):
                sent_dict = row['sentiment_details']
            else:
                string_json = str(row['sentiment_details'])
                string_json = string_json.replace("'", '"')
                sentence_counter = math.ceil((len(string_json.split(":")) - 2) / 4) + 1
                for i in range(sentence_counter, -1, -1):
                    string_json = string_json.replace("{}:".format(i), '"{}":'.format(i))
                try:
                    sent_dict = json.loads(string_json)
                except:
                    print("error \n{0} \n{1}\n\n".format(row['sentiment_details'], string_json))
            for key, sent_info in sent_dict.items():
                tmp = {'book': row['book'],
                       'chapter': row['chapter'],
                       'paragraph_number': row['paragraph_number'],
                       'dominant_topic': row['dominant_topic'],
                       'perc_contribution': row['perc_contribution'],
                       'sentence_number': key,
                       'sentence_prec': sent_info['len'],
                       'sentence_sentiment': sent_info['tag'],
                       'sentence_sentiment_score': sent_info['score']
                       }
                list_dict_info.append(tmp)
        sentence_level_df = pd.DataFrame(list_dict_info)
        return sentence_level_df

    def plot_sentiment_score_distributions(self,
                                           file_name='figures/sentiment_score_distribution.png',
                                           alpha=0.1):
        fig = (ggplot(self.sent_sentiment,
                      aes(x='sentence_sentiment_score', color='sentence_sentiment', fill='sentence_sentiment'))
               + geom_density(alpha=alpha)
               )
        fig.save(file_name, dpi=300)
        logging.info("sentiment distribution plot is generated and saved in:\t{}".format(file_name))

    def plot_sentiment_score_distributions_violin(self, file_name='figures/sentiment_score_distribution_violin.png'):

        sns.boxplot(x='dominant_topic', y='sentence_sentiment_score', notch=True,
                    data=self.sent_sentiment, showfliers=True)
        plt.xticks(rotation=90)
        plt.savefig(file_name, dpi=300)
        logging.info("sentiment distribution - violin plot is generated and saved in:\t{}".format(file_name))

    def plot_sentiment_distributions_per_topic(self,
                                               file_name='figures/sentiment_distribution_per_topic.png'):
        df_tmp = self.sent_sentiment. \
            groupby(['dominant_topic', 'sentence_sentiment'], as_index=False). \
            agg({'sentence_number': ['count']})
        df_tmp.columns = df_tmp.columns.droplevel(1)
        fig = (ggplot(df_tmp, aes('dominant_topic', 'sentence_number', fill='sentence_sentiment'))
               + geom_col()
               + scale_x_continuous()
               )
        fig.save(file_name, dpi=300)
        logging.info("sentiment distribution plot is generated and saved in:\t{}".format(file_name))

    def generate_stat_report_sentiment_per_sentence(self):
        df_tmp = self.sent_sentiment. \
            groupby(['sentence_sentiment'], as_index=False). \
            agg({'sentence_sentiment_score': ['count', 'median', 'mean']})
        return df_tmp

    def generate_stat_report_sentiment_per_sentence_per_book(self):
        df_tmp = self.sent_sentiment. \
            groupby(['book', 'sentence_sentiment'], as_index=False). \
            agg({'sentence_sentiment_score': ['count', 'median', 'mean']})
        return df_tmp

    def generate_stat_report_sentiment_per_sentence_per_chapter(self):
        df_tmp = self.sent_sentiment. \
            groupby(['book', 'chapter', 'sentence_sentiment'], as_index=False). \
            agg({'sentence_sentiment_score': ['count', 'median', 'mean']})
        return df_tmp

    def generate_stat_report_sentiment_per_sentence_per_topic(self):
        df_tmp = self.sent_sentiment. \
            groupby(['dominant_topic', 'sentence_sentiment'], as_index=False). \
            agg({'sentence_sentiment_score': ['count', 'median', 'mean']})
        return df_tmp

    def generate_stat_report_sentiment_per_paragraph(self):
        df_tmp = self.df. \
            groupby(['sentiment'], as_index=False). \
            agg({'paragraph_number': ['count']})
        return df_tmp

    def generate_stat_report_sentiment_per_paragraph_per_book(self):
        df_tmp = self.df. \
            groupby(['book', 'sentiment'], as_index=False). \
            agg({'paragraph_number': ['count']})
        return df_tmp

    def generate_stat_report_sentiment_per_paragraph_per_chapter(self):
        df_tmp = self.df. \
            groupby(['book', 'chapter', 'sentiment'], as_index=False). \
            agg({'paragraph_number': ['count']})
        return df_tmp

    def generate_stat_report_sentiment_per_paragraph_per_topic(self):
        df_tmp = self.df. \
            groupby(['dominant_topic', 'sentiment'], as_index=False). \
            agg({'perc_contribution': ['count', 'median', 'mean', 'min', 'max']})
        return df_tmp
