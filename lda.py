from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
from preprocess import Processor
import matplotlib.pyplot as plt
import seaborn as sns


class LDAModel:
    def __init__(self, corpus, corpuse_info):
        self.corpus = corpus
        self.corpus_info = corpuse_info
        self.processor = None  # lazy loading
        self.dictionary = None
        self.tokens = []
        self.bow_corpus = []
        self.lda_model = None
        self.corpus_topics = []

    def preprocess(self, simple_tokenizer=True):
        self.processor = Processor(self.corpus, simple_tokenizer)
        # here, I can run any fancy preprocessing workflow that I need :)
        self.processor.preprocess()
        self.tokens = self.processor.processed

    def create_bow_model(self, filter_below=5, filter_above=200, keep_n=4000):
        self.dictionary = Dictionary(self.tokens)
        self.dictionary.filter_extremes(no_below=filter_below, no_above=filter_above, keep_n=keep_n)
        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.tokens]

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
                cm = CoherenceModel(model=lda_model, texts=self.tokens,
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
        self.corpus_topics = [sorted(self.lda_model[self.bow_corpus][text])[0][0]
                              for text in range(len(self.tokens))]
