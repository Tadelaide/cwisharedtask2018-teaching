# from sklearn.feature_extraction.text import TfidfTransformer    
# from sklearn.feature_extraction.text import CountVectorizer    
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn import preprocessing
import numpy as np
from nltk import Counter
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math



corpus=["I come to China to travel and",  
    "This is a car polupar in China",            
    "I love tea and Apple ",     
    "The work is to write some papers in science",
    "science science write write write write"]  
  
# test = ["I love China"]
# a = [["I love China"],["I love China"]]
# print(Counter(test))

# aa = {'a':1,'b':2}
# aaa = np.array([item for item in aa.values()])
# print(np.max(aaa))

# transformer = TfidfTransformer()  
# tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))    
# print tfidf  
tfidf2 = TfidfVectorizer()
weight_2 = tfidf2.fit_transform(corpus).toarray()
# test1 = tfidf2.transform(test).toarray()


p = 0

# for item in test1[0]:
#     p += item 

tfidf = TfidfVectorizer()
re = tfidf.fit_transform(corpus).toarray()


print(re)


# c = np.array(re[0])
b = np.zeros(len(re[0]))
for item in re:
    a = np.array(item)
    b += a 
qwer = dict(zip(tfidf.get_feature_names(),b))
print(qwer)

word=tfidf2.get_feature_names()
# print(word)
print(tfidf2.vocabulary_)
A = {}
for key,value in tfidf.vocabulary_.items():
    A[key] = value
A = {key:value for key,value in tfidf.vocabulary_.items()}
B = [value for key,value in tfidf.vocabulary_.items()]

a = {'a':1,'b':2,'f':6,'c':7}
print(a.values())
aa = np.array([item for item in a.values()])
print(np.max(aa))

log_reg= LogisticRegression(class_weight="balanced")
# 训练模型
log_reg.fit(data, numpy.asarray(labels))
# 根据输入预测
log_reg.predict_proba(input)

# data = [ ("me gusta comer en la cafeteria".split(), "SPANISH"),
#          ("Give it to me".split(), "ENGLISH"),
#          ("No creo que sea una buena idea".split(), "SPANISH"),
#          ("No it is not a good idea to get lost at sea".split(), "ENGLISH") ]

# test_data = [("Yo creo que si".split(), "SPANISH"),
#               ("it is lost on me".split(), "ENGLISH")]

# word_to_ix = {}
# for sent, _ in data + test_data:
#     for word in sent:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)


