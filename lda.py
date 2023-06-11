from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore, HdpModel
from gensim.models import CoherenceModel
from preprocess import Processor
import matplotlib.pyplot as plt
import pandas as pd


import logging

class LDAModel:
    def __init__(self, df):
        self.df = df
        self.corpus = df['paragraph']
        self.processor = None  # lazy loading
        self.dictionary = None
        self.processed = []
        self.bow_corpus = []
        self.lda_model = None
        self.hdp_model = None
        self.corpus_topics = []

    def preprocess(self, simple_tokenizer=True, custom_stopwords=[]):
        logging.info("Calling preprocessing in LDA model:\n")
        self.processor = Processor(self.df, simple_tokenizer)
        # here, I can run any fancy preprocessing workflow that I need :)
        self.processor.preprocess()
        self.processed = self.processor.processed
        self.processor.map_to_df()
        self.df = self.processor.df
        logging.info("Preprocessing is done - from LDA model:\n")

    def create_bow_model(self, filter_below=5, filter_above=200, keep_n=4000):
        self.dictionary = Dictionary(self.processed)
        self.dictionary.filter_extremes(no_below=filter_below, no_above=filter_above, keep_n=keep_n)
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.processed]
        logging.info("Bag of word representations are constructed...")

    def plot_optimal_coherent_scores(self, scoring_method="c_v", start_range=1, end_range=15):
        topics = []
        score = []
        for i in range(start_range, end_range):
            lda_model = LdaMulticore(corpus=self.bow_corpus,
                                     id2word=self.dictionary,
                                     iterations=10,
                                     num_topics=i,
                                     workers=8,
                                     passes=10,
                                     random_state=100)
            if scoring_method == "u_mass":
                cm = CoherenceModel(model=lda_model,
                                    corpus=self.bow_corpus,
                                    dictionary=self.dictionary,
                                    coherence=scoring_method)
            elif scoring_method == "c_v":
                cm = CoherenceModel(model=lda_model, texts=self.processed,
                                    corpus=self.bow_corpus,
                                    dictionary=self.dictionary,
                                    coherence=scoring_method)
            topics.append(i)
            score.append(cm.get_coherence())
        plt.plot(topics, score)
        plt.xlabel('Number of Topics (from {0} to {1})'.format(start_range, end_range))
        plt.ylabel('Coherence Score {}'.format(scoring_method))
        plt.show()

    def create_lda_model(self, num_topic, iteration=100):
        lda_model = LdaMulticore(corpus=self.bow_corpus,
                                 id2word=self.dictionary,
                                 iterations=iteration,
                                 num_topics=num_topic,
                                 workers=4,
                                 passes=100)
        lda_model.print_topics(-1)
        self.lda_model = lda_model

    def create_hdp_model(self):
        hdp_model = HdpModel(corpus=self.bow_corpus,
                             id2word=self.dictionary,)
        hdp_model.print_topics(-1)
        self.hdp_model = hdp_model


    def format_topics_sentences(self, use_lda_model=True):
        model = None
        if use_lda_model:
            model = self.lda_model
            logging.info("using LDA model to label topics for paragraphs")
        else:
            model = self.hdp_model
            logging.info("using HDP model to label topics for paragraphs")
        # empty data frame
        sent_topics_df = pd.DataFrame()
        # retrieve main topic for each document
        for i, row_list in enumerate(model[self.bow_corpus]):
            row = row_list[0] if model.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the dominant topic, percentage of contribution and the set of keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = model.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = pd.concat([sent_topics_df, pd.DataFrame(
                        [pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords])])], ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['dominant_topic', 'perc_contribution', 'topic_keywords']

        self.df = pd.concat([self.df, sent_topics_df], axis=1)
        logging.info("tagging topics are done, check the dataframe in LDAModel object\n")
