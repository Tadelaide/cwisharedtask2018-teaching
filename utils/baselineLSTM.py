from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import time
import math
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class BaselineLSTM(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        # if language == 'english':
        #     self.avg_word_length = 5.3
        # else:  # spanish
        #     self.avg_word_length = 6.2

        self.model = LogisticRegression()
        
        self.all_letters = string.ascii_letters + " .,;'"
        self.n_letters = len(self.all_letters)

        self.category_lines = {'easy':[],'hard':[]}
        self.all_categories = ['easy','hard']

        self.n_categories = len(self.all_categories)
        self.rnn = None
        self.criterion = None
        self.optimizer = None
        self.n_hidden = 256

        self.learning_rate = 0.001 # If you set this too high, it might explode. If too low, it might not learn
        self.predictions = []

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        return [len_chars, len_tokens]

    def train(self, trainset):
        # X = []
        # y = []
        for sent in trainset:
            # X.append(self.extract_features(sent['target_word']))
            # y.append(sent['gold_label'])
            # print(sent['gold_label'],'111111',sent['target_word'])
            if sent['gold_label']=='1':
                self.category_lines['hard'].append(sent['target_word'])
            else:
                self.category_lines['easy'].append(sent['target_word'])
        
        self.rnn = RNN(self.n_letters, self.n_hidden, self.n_categories)
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=self.learning_rate)
        
        n_epochs = 100000
        current_loss = 0
        for epoch in range(1, n_epochs + 1):
        # Get a random training input and target
            category, line, category_tensor, line_tensor = self.random_training_pair()
            output, loss = self.trainRnn(category_tensor, line_tensor)
            current_loss += loss
        # self.model.fit(X, y)

    def test(self, testset):
        predictions = []
        for sent in testset:
            # X.append(self.extract_features(sent['target_word']))
            output = self.evaluate(Variable(self.line_to_tensor(sent['target_word'])))
            topv, topi = output.data.topk(1, 1, True)
            category_index = topi[0][0]
            if self.all_categories[category_index] == 'hard':
                predictions.append('1')
            else :
                predictions.append('0')

        
        # return self.model.predict(X)
        return predictions
#调用
    def trainRnn(self, category_tensor, line_tensor):
        self.rnn.zero_grad()
        hidden = self.rnn.init_hidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = self.rnn(line_tensor[i], hidden)
        loss = self.criterion(output, category_tensor)
        loss.backward()
        self.optimizer.step()
        return output, loss.data[0]

    def evaluate(self, line_tensor):
        hidden = self.rnn.init_hidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = self.rnn(line_tensor[i], hidden)
        return output

    def predict(self,input_line, n_predictions=1):
        output = self.evaluate(Variable(self.line_to_tensor(input_line)))
        # Get top N categories
        topv, topi = output.data.topk(n_predictions, 1, True)
        self.predictions = []

        for i in range(n_predictions):
            value = topv[0][i]
            category_index = topi[0][i]
            print('(%.2f) %s' % (value, self.all_categories[category_index]))
            self.predictions.append([value, self.all_categories[category_index]])

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letter_to_tensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        letter_index = self.all_letters.find(letter)
        tensor[0][letter_index] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def line_to_tensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            letter_index = self.all_letters.find(letter)
            tensor[li][0][letter_index] = 1
        return tensor

    def category_from_output(self, output):
        top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
        category_i = top_i[0][0]
        return self.all_categories[category_i], category_i

    def random_training_pair(self):                                                                                                               
        category = random.choice(self.all_categories)
        line = random.choice(self.category_lines[category])
        category_tensor = Variable(torch.LongTensor([self.all_categories.index(category)]))
        line_tensor = Variable(self.line_to_tensor(line))
        return category, line, category_tensor, line_tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))
    
