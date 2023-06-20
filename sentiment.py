import logging

import pandas as pd
from flair.data import Sentence
from flair.nn import Classifier
from spacy.lang.en import English


class Sentiment:
    def __init__(self, df, fast=False):
        self.df = df
        self.nlp = English()
        self.nlp.add_pipe('sentencizer')
        # load the model
        if fast:
            self.tagger = Classifier.load('sentiment-fast')
        else:
            self.tagger = Classifier.load('sentiment')

    def predict(self, load_from_file=False, file_name='out/sentiment.csv'):
        if load_from_file:
            logging.info("loading sentiments from already exported file:\n\t{}".format(file_name))
            tmp_df = pd.read_csv(file_name)
            len_before = len(self.df)
            tmp_df = tmp_df[['book', 'chapter', 'paragraph_number', 'sentiment',
                             'sentiment_details']]
            self.df = pd.merge(self.df, tmp_df, how='left',
                               left_on=['book', 'chapter', 'paragraph_number'],
                               right_on=['book', 'chapter', 'paragraph_number'])
            logging.info("left join done and sentiments are loaded - check: {0}=={1}?".format(len_before, len(self.df)))
        else:
            logging.info('start computing the sentiments - the process will take time, be patient')
            self.df['sentiment'], self.df['sentiment_details'] = \
                zip(*self.df['paragraph'].apply(lambda x: self.single_predict(x)))

    def single_predict(self, paragraph):
        sentiments = {}
        tags = {'NEGATIVE': 0, 'POSITIVE': 0}
        nlp_paragraph = self.nlp(paragraph)
        idx = 0
        for sent in nlp_paragraph.sents:
            sentence = Sentence(sent.text)
            self.tagger.predict(sentence)
            sentiments[idx] = {'len': len(sent.text) / len(paragraph),
                               'tag': sentence.tag,
                               'score': sentence.score}
            tags[sentence.tag] += sentiments[idx]['len'] * sentiments[idx]['score']
            idx += 1
        if tags['POSITIVE'] == 0 and tags['NEGATIVE'] == 0:
            return 'NEUTRAL', sentiments
        elif tags['POSITIVE'] == 0:
            return 'NEGATIVE', sentiments
        elif tags['NEGATIVE'] == 0:
            return 'POSITIVE', sentiments
        elif tags['POSITIVE'] / tags['NEGATIVE'] >= 1.5:
            return 'POSITIVE', sentiments
        elif tags['NEGATIVE'] / tags['POSITIVE'] >= 1.5:
            return 'NEGATIVE', sentiments
        return 'NEUTRAL', sentiments

    def export_to_csv(self, file_address='out/sentiment.csv'):
        self.df.to_csv(file_address)
        logging.info('file is writen successfully')
