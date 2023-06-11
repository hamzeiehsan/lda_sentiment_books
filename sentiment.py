import logging

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

    def predict(self):
        logging.info('start computing the sentiments - the process will take time, be patient')
        self.df['sentiment'] = self.df['paragraph'].apply(lambda x: self.single_predict(x))

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
            return 'NEUTRAL'
        elif tags['POSITIVE'] == 0:
            return 'NEGATIVE'
        elif tags['NEGATIVE'] == 0:
            return 'POSITIVE'
        elif tags['POSITIVE'] / tags['NEGATIVE'] >= 1.5:
            return 'POSITIVE'
        elif tags['NEGATIVE'] / tags['POSITIVE'] >= 1.5:
            return 'NEGATIVE'
        return 'NEUTRAL'

    def export_to_csv(self, file_address='out/sentiment.csv'):
        self.df.to_csv(file_address)
        logging.info('file is writen successfully')
