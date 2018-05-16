from sklearn.linear_model import LogisticRegression
from nltk import Counter
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer


class Baseline(object):
    def __init__(self, language):
        self.language = language        
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            self.wordNumber = 0
            self.wordbackup = []
            self.keyWord = []
            self.posdic = {}

        else:  # spanish
            self.avg_word_length = 6.2
            self.wordNumber = 0
            self.wordbackup = []
            self.keyWord = []
            self.posdic = {}

        self.model = LogisticRegression()

    def extract_features(self, sent):
        #所挑选出的词或者词词组的长度除以平均长度
        len_chars = len(sent['target_word']) / self.avg_word_length
        #所套选出的词或者词组中包含多少个词
        len_tokens = len(sent['target_word'].split(' '))
        #wordbags = sent['target_word'].split(' ')
        wordbags = re.sub("[^\w'|^\w-]"," ", sent['target_word']).split()
        pos = {word:pos for (word, pos) in nltk.pos_tag(nltk.word_tokenize(sent['sentence'])) if word in wordbags}
        for item in wordbags:
            if item in pos.keys():
                #print(wordbags)
                #print(pos)
                #print(pos[item])
                #print(self.posdic[item])
                if item in self.posdic.keys():
                    if pos[item] in self.posdic[item]:
                        return [10, len_chars, len_tokens]
                    else:
                        return [0, len_chars, len_tokens]
                else:
                    return [0, len_chars, len_tokens]
            else:
                return [0, len_chars, len_tokens]
        """ for tf method
        for item in wordbags :
            if item in self.keyWord:
                return [1, len_chars, len_tokens]
            else:
                return [0, len_chars, len_tokens]
        """

    def statistics(self, trainset):
        for sent in trainset:
            self.wordNumber += len(re.sub("[^\w']"," ", sent['sentence']).split())
            self.wordbackup += re.sub("[^\w']"," ", sent['sentence']).split()
        self.wordCounter = Counter(self.wordbackup)

    def tf(self, trainset):
        for sent in trainset:
            self.keyWord +=  sent['target_word'].split(' ')
            
    #this only use English, not Spanish
    def pos(self, trainset):
        keywords = []
        tempdic = {}
        for sent in trainset:
            #keywords = sent['target_word'].split(' ')
            keywords = re.sub("[^\w'|^\w-]"," ", sent['target_word']).split()
            tempdic = { word:pos for (word, pos) in nltk.pos_tag(nltk.word_tokenize(sent['sentence'])) if word in keywords}
            for key in tempdic.keys():
                if key not in self.posdic.keys():
                    self.posdic[key] = [tempdic[key]]
                else:
                    if not tempdic[key] in self.posdic[key] :
                        self.posdic[key].append(tempdic[key])


    def train(self, trainset):
        X = []
        y = []
        #self.statistics(trainset)
        #self.tf(trainset)
        self.pos(trainset)
        for sent in trainset:
            #X.append(self.extract_features(sent['target_word']))
            X.append(self.extract_features(sent))
            #和是否有 binary 存在的难易度
            y.append(sent['gold_label'])
        #使用逻辑模型
        self.model.fit(X, y)
        

    def test(self, testset):
        X = []
        for sent in testset:
            #X.append(self.extract_features(sent['target_word']))
            X.append(self.extract_features(sent))

        return self.model.predict(X)
