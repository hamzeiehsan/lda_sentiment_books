from collections import defaultdict
import re
import en_core_web_md
import pandas as pd
from gensim.models import Phrases
from gensim.models.phrases import Phraser

import logging

RE_COMBINE_WHITESPACE = re.compile(r"\s+")


class Processor:
    def __init__(self, corpus, simple_tokenizer=True, custom_stopwords=[]):
        logging.info("Preprocessor initiated...")
        if isinstance(corpus, list):
            self.raw = corpus
            self.df = None
        elif isinstance(corpus, pd.DataFrame):
            self.raw = list(corpus['paragraph'])
            self.df = corpus
        self.corpus = []
        self.custom_stopwords = custom_stopwords

        self.nlp = en_core_web_md.load()

        # adding custom stopwords to the set
        stopwords = self.nlp.Defaults.stop_words
        logging.info("\tstopwords length (standard): {}".format(len(stopwords)))
        stopwords = list(stopwords)
        stopwords.extend(self.custom_stopwords)
        self.stopwords = stopwords
        logging.info("\tstopwords length: {}".format(len(self.stopwords)))

        self.pre_tokenize()
        self.processed = None

        self.wfrequencies = defaultdict(int)
        if simple_tokenizer:
            self.tokens = [[word.strip() for word in document.lower().split() if
                            word.strip() not in self.stopwords and len(word.strip()) > 2]
                           for document in self.corpus]
        else:
            self.spacy_tokenizer()
        logging.info("\ttokenization and initial preprocessing is done")
        self.count_word_frequencies()
        logging.info("\tword count is done:")
        logging.info("\t\t{}\n".format(self.filter_wfrequencies()))

    def add_stop_words(self, stop_words_list):
        self.custom_stopwords.extend(stop_words_list)

    def pre_tokenize(self):
        for text in self.raw:
            tmp = text.translate({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\\|`~-=_+1234567890"})
            tmp = RE_COMBINE_WHITESPACE.sub(" ", tmp)
            self.corpus.append(tmp)
        logging.info("\tAll paragraph are normalized by removing unwanted characters")

    def spacy_tokenizer(self):
        # POS tags to remove
        removal = ['ADV', 'PRON', 'CCONJ', 'PUNCT', 'PART', 'DET', 'ADP', 'SPACE', 'NUM', 'SYM']

        self.tokens = []
        for summary in self.nlp.pipe(self.corpus):
            proj_tok = [token.text.lower() for token in summary if
                        token.pos_ not in removal and not token.is_stop and token.is_alpha and
                        token.text.lower() not in self.stopwords]
            self.tokens.append(proj_tok)
        logging.info("Spacy listed POS removal and stopwords removal are done...")

    def preprocess(self, force=False, min_freq=5, max_freq=200):
        if self.processed is None or force:
            print('preprocessing workflow started...')
            # Only keep words that appear more than five times and repeated less than 200 times
            self.processed = [[token for token in text if max_freq > self.wfrequencies[token] > min_freq]
                              for text in self.tokens]
            # bigrams and trigrams
            bigram = Phrases(self.tokens, min_count=5, threshold=100)  # higher threshold fewer phrases.
            trigram = Phrases(bigram[self.tokens], threshold=100)
            bigram_mod = Phraser(bigram)
            trigram_mod = Phraser(trigram)
            logging.info("analyzing bigrams and trigrams")
            self.processed = [bigram_mod[doc] for doc in self.processed]
            self.processed = [trigram_mod[bigram_mod[doc]] for doc in self.processed]
        logging.info("\tPreprocessing done...\n")
        return self.processed

    def count_word_frequencies(self, refresh=False):
        if refresh:
            self.wfrequencies = defaultdict(int)
        for text in self.tokens:
            for token in text:
                if token.lower not in self.stopwords:
                    self.wfrequencies[token] += 1

    def map_to_df(self):
        if self.df is not None:
            self.df['tokens'] = self.tokens
            if self.processed is not None:
                self.df['processed'] = self.processed

    def filter_wfrequencies(self, threshold=150):
        return {k: v for (k, v) in self.wfrequencies.items() if v > threshold}
