from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from textstat.textstat import textstat
import spacy
import numpy as np

class BaselineNew(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = LogisticRegression()
        # self.model =RandomForestClassifier(n_estimators=100, max_depth=5,min_samples_split=2, random_state=0)
        # print(self.language)
        if self.language == 'english':
            self.nlp = spacy.load('en_core_web_md')
        if self.language == 'spanish':
            self.nlp = spacy.load('es_core_news_sm')

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' ')) 
        # syllable_count = self.getSyllable(word)/len_tokens
        # easy_difficult = textstat.flesch_reading_ease(word)
        a = 0
        for item in word:
            b = self.nlp(item).vector_norm
            a += b
        return [len_chars,len_tokens,a/len_tokens]

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])
        # print(sent)

        self.model.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))

        return self.model.predict(X)

    # def getSyllable(self, word):
    #     syllable = 0
    #     for item in word:
    #         syllable_count = textstat.syllable_count(item)
    #         syllable += syllable_count
    #     return syllable