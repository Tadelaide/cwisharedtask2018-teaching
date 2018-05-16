from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import preprocessing
import numpy as np
from nltk import Counter 


class BaselineTf(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
        else:  # spanish
            self.avg_word_length = 6.2

        self.model = LogisticRegression()
        #for tf model
        self.wordList = []
        self.maxNumber = 0
        self.tfList = None
        #a try to tfidf, use the tfidf matrix
        self.tfidfWordList = []
        self.tfidf = TfidfVectorizer()

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        tf = 0
        for text in word.split(' '):
            tf += self.tf(text)
        # tfidf = self.gettfidf(word)

        return [len_chars,len_tokens, tf]

    def train(self, trainset):
        X = []
        y = []
        self.tfidf = TfidfVectorizer()
        for sent in trainset:
            self.tfidfWordList.append(sent['target_word'])
            for item in sent['target_word'].split(" "):
                self.wordList.append(item)
        #tf model
        self.tfList = Counter(self.wordList)
        max = np.array([item for item in self.tfList.values()])
        self.maxNumber = np.max(max)
        #tfidf model
        weightTfidf = self.tfidf.fit_transform(self.tfidfWordList).toarray()

        # zeroVector = np.zeros(len(re[0]))
        # for item in re:
        #     itemVector = np.array(item)
        #     zeroVector += itemVector
        # self.tfidfResult = dict(zip(tfidf.get_feature_names(),zeroVector))
        # self.tfidfResult = {key:value for key,value in tfidf.vocabulary_.items()}
        # self.normal = np.max(np.array([item for item in self.tfidfResult.values()]))

        for sent in trainset:
            X.append(self.extract_features(sent['target_word'])) 
            y.append(sent['gold_label'])
        self.model.fit(X, y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))
        return self.model.predict(X)

    def tf(self, word):
        if word in self.tfList.keys():
            tf = self.tfList[word]/self.maxNumber
        else:
            tf = 0
        return tf

    def gettfidf(self, word):        
        tfidf = 0
        wordList = []
        wordList.append(word)
        tfidfList = self.tfidf.transform(wordList).toarray()
        for item in tfidfList[0]:
            tfidf += item
        return tfidf
        # for item in word.split(' '):
        #     if item in self.tfidfResult.keys():
        #         tfidf += self.tfidfResult[item]
        #     else:
        #         tfidf += 0.5