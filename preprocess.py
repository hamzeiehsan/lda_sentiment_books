from collections import defaultdict
from gensim import corpora
import re


RE_COMBINE_WHITESPACE = re.compile(r"\s+")


class Processor:
    def __init__(self, corpus):
        self.raw = corpus
        self.corpus = []
        self.pre_tokenize()
        self.processed = None
        self.custom_stopwords = []

        self.wfrequencies = defaultdict(int)
        self.tokens = [[word for word in document.lower().split() if word not in self.custom_stopwords]
                       for document in self.corpus]
        self.count_word_frequencies()

    def add_stop_words(self, stop_words_list):
        self.custom_stopwords.extend(stop_words_list)

    def pre_tokenize(self):
        for text in self.raw:
            tmp = text.translate({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+1234567890"})
            tmp = RE_COMBINE_WHITESPACE.sub(" ", tmp)
            self.corpus.append(tmp)


    def preprocess(self, force=False, min_freq=5, max_freq=200):
        if self.processed is None or force:
            print('preprocessing workflow started...')
            # Only keep words that appear more than five times and repeated less than 200 times
            self.processed = [[token for token in text if max_freq > self.wfrequencies[token] > min_freq]
                              for text in self.tokens]
        return self.processed

    def count_word_frequencies(self, refresh=False):
        if refresh:
            self.wfrequencies = defaultdict(int)
        for text in self.tokens:
            for token in text:
                self.wfrequencies[token] += 1
