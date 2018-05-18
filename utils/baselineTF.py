from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import preprocessing
import numpy as np
from nltk import Counter 
from sklearn.ensemble import RandomForestClassifier
from textstat.textstat import textstat
import matplotlib.pyplot as plt

from sklearn.learning_curve import learning_curve

class BaselineTf(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            self.avg_syllable_length = 5
        else:  # spanish
            self.avg_word_length = 6.2
            self.avg_syllable_length = 12
        #self.model = LogisticRegression()
        self.model =RandomForestClassifier(n_estimators=200, max_depth=None,min_samples_split=2, random_state=0)
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
        tfidf = self.gettfidf(word)

        syllable_count = self.getSyllable(word)/self.avg_syllable_length
        # print(syllable_count)

        return [len_chars,len_tokens,tf,tfidf]
        #return [len_chars,len_tokens]

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

        zeroVector = np.zeros(len(weightTfidf[0]))
        for item in weightTfidf:
            itemVector = np.array(item)
            zeroVector += itemVector
        self.tfidfResult = dict(zip(self.tfidf.get_feature_names(),zeroVector))
        # self.tfidfResult = {key:value for key,value in self.tfidf.vocabulary_.items()}
        # self.normal = np.max(np.array([item for item in self.tfidfResult.values()]))
        # print(self.tfList)
        for sent in trainset:
            X.append(self.extract_features(sent['target_word'])) 
            y.append(sent['gold_label'])
        self.model.fit(X, y)
        title = "TF+TFIDF " + self.language.capitalize()
        self.plot_learning_curve(self.model,title,X,y)

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))
        # Y = {}
        # cont = 0
        # for sent in testset:
        #     if self.model.predict(X)[cont] == sent['gold_label']:
        #         if not sent['target_word'] in sent.keys():
        #             Y[sent['target_word']] = 1
        #     cont += 1
        # print(Y)
        
        return self.model.predict(X)

    def tf(self, word):
        if word in self.tfList.keys():
            tf = self.tfList[word]/self.maxNumber
        else:
            tf = 0
        return tf

    def gettfidf(self, word):        
        tfidf = 0
        # wordList = []
        # wordList.append(word)
        # tfidfList = self.tfidf.transform(wordList).toarray()
        # for item in tfidfList[0]:
        #     tfidf += item
        for item in word.split(' '):
            if item in self.tfidfResult.keys():
                tfidf += self.tfidfResult[item]
            else:
                tfidf += 1
        return tfidf/len(word.split(' '))


    def getSyllable(self, word):
        syllable = 0
        for item in word:
            syllable_count = textstat.syllable_count(item)
            syllable += syllable_count
        return syllable


    def plot_learning_curve(self,estimator, title, X, y,ylim=None, cv=None,train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=1, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.fill_between(train_sizes, train_scores_mean -   train_scores_std,
                         train_scores_mean + train_scores_std,  alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean -    test_scores_std,
                         test_scores_mean + test_scores_std,    alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid("on") 
        if ylim:
            plt.ylim(ylim)
        plt.title(title)
        plt.show()