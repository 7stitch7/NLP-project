import nltk
import pandas as pd

nltk.download('punkt')
from nltk import word_tokenize

nltk.download('treebank')
from nltk.corpus import treebank

import numpy as np
from sklearn.model_selection import train_test_split

# Retrieve tagged sentences from treebank corpus
tagged_sentences = nltk.corpus.treebank.tagged_sents()

print(tagged_sentences[0])
print("Tagged sentences: ", len(tagged_sentences))
print("Tagged words:", len(nltk.corpus.treebank.tagged_words()))
# tagged_words(): list of (str,str) tuple

sentences, sentence_tags = [], []
for tagged_sentence in tagged_sentences:
    # The zip() function returns a zip object, which is an iterator of tuples where the first item in each passed iterator is paired together,
    # and then the second item in each passed iterator are paired together etc.
    sentence, tags = zip(*tagged_sentence)
    sentence = [word.lower() for word in sentence]
    sentences.append(np.array(sentence))
    sentence_tags.append(np.array(tags))

print(sentences[5])
print(sentence_tags[5])

(train_sentences,
 test_sentences,
 train_tags,
 test_tags) = train_test_split(sentences, sentence_tags, test_size=0.2)

words, tags = set([]), set([])

for s in train_sentences:
    for w in s:
        words.add(w.lower())

for ts in train_tags:
    for t in ts:
        tags.add(t)

word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs

tag2index = {t: i + 2 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding
tag2index['-OOV-'] = 1  # The special value used for OOVs


def tag_to_index(tag):
    if tag in tag2index:
        return tag2index[tag]
    else:
        return tag2index['-OOV-']

train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

for s in train_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    train_sentences_X.append(s_int)

for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w.lower()])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    test_sentences_X.append(s_int)

for s in train_tags:
    train_tags_y.append([tag_to_index(t) for t in s])

for s in test_tags:
    test_tags_y.append([tag_to_index(t) for t in s])

print(train_sentences_X[0])
print(test_sentences_X[0])
print(train_tags_y[0])
print(test_tags_y[0])

MAX_LENGTH = len(max(train_sentences_X, key=len))
print(MAX_LENGTH)

import torch
new_train_sentences_X = [torch.from_numpy(np.array(l)) for l in train_sentences_X]
new_test_sentences_X = [torch.from_numpy(np.array(l)) for l in test_sentences_X]
new_train_tags_y = [torch.from_numpy(np.array(l)) for l in train_tags_y]
new_test_tags_y = [torch.from_numpy(np.array(l)) for l in test_tags_y]

from torch.nn.utils.rnn import pad_sequence
after_pad = pad_sequence(new_train_sentences_X+new_test_sentences_X+new_train_tags_y+new_test_tags_y,batch_first=True)
train_sentences_X_pad = after_pad[:len(new_train_sentences_X)]
test_sentences_X_pad = after_pad[len(new_train_sentences_X):len(new_train_sentences_X)+len(new_test_sentences_X)]
train_tags_y_pad = after_pad[len(new_train_sentences_X)+len(new_test_sentences_X):-len(new_test_tags_y)]
test_tags_y_pad = after_pad[-len(new_test_tags_y):]

#More detailed info about the TensorDataset, https://pytorch.org/docs/1.1.0/_modules/torch/utils/data/dataset.html#TensorDataset
from torch.utils.data import TensorDataset
train_data = TensorDataset(train_sentences_X_pad, train_tags_y_pad)

batch_size = 128
#More detailed info about the dataLoader, https://pytorch.org/docs/1.1.0/_modules/torch/utils/data/dataloader.html
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,batch_first=True, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=-1)
        return tag_scores


EMBEDDING_DIM = 128
HIDDEN_DIM = 256

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2index), len(tag2index)).cuda()
loss_function = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(40):
    loss_now = 0.0
    acc = 0

    for sentence,targets in train_loader:
        sentence = sentence.cuda()
        targets = targets.cuda()

        model.zero_grad()
        model.train()
        tag_scores = model(sentence)

        # loss = loss_function(tag_scores, targets)
        loss = loss_function(tag_scores.view(-1,tag_scores.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        loss_now+=loss.item()

        model.eval()
        tag_scores = model(sentence)
        _, predicted = torch.max(tag_scores, -1)
        prediction = predicted.view(-1).cpu().numpy()
        t = targets.view(-1).cpu().numpy()
        # Note: The training accuracy here is calculated with "PAD", which means most of pos tag will be "0".
        acc = acc+accuracy_score(prediction,t)*len(prediction)
    print('Epoch: %d, training loss: %.4f, training acc: %.2f%%'%(epoch+1,loss_now,100*acc/len(train_sentences_X)/MAX_LENGTH))

model.eval()
sentence = test_sentences_X_pad.cuda()
tag_scores = model(sentence)
_, predicted = torch.max(tag_scores, -1)
predicted = predicted.cpu().numpy()

# cut off the PAD part
test_len_list = [len(s) for s in test_sentences_X]
actual_predicted_list= []
for i in range(predicted.shape[0]):
    actual_predicted_list+=list(predicted[i])[:test_len_list[i]]

# get actual tag list
actual_tags = sum(test_tags_y, [])

print('Test Accuracy: %.2f%%'%(accuracy_score(actual_predicted_list,actual_tags)*100))

def read_data(file_name):
    df = pd.read_csv(file_name)
    Sentences = ' '.join(df.Sentence.tolist()).split(' . ')
    word = ' '.join(df.Sentence.tolist()).split(' ')
    try:
        target = ' '.join(df.NER.tolist()).split(' ')

    except:
        target = df.NER.tolist()
    train = [sentence.strip().split() + ['.'] for sentence in Sentences]
    train[-1] = train[-1][:-1]
    i = 0
    tags = []
    for sentence in train:
        tag = target[i:i + len(sentence)]
        tags.append(tag)
        i += len(sentence)
    print("data has been loaded!")
    return train, tags


def read_test_data(file_name):
    df = pd.read_csv(file_name)
    Sentences = ' '.join(df.Sentence.tolist()).split(' . ')
    word = ' '.join(df.Sentence.tolist()).split(' ')
    train = [sentence.strip().split() + ['.'] for sentence in Sentences]
    train[-1] = train[-1][:-1]
    print("test data has been loaded!")
    return train


train_data, target_y_train = read_data("train.csv")
validation_data, target_y_validation = read_data("val.csv")
test_data = read_test_data('test.csv')

pre_data = train_data+validation_data+test_data

#decode the result to have actual tags
def decode_result(predictions, test_samples_X, index2tag):
    token_sequences = []
    ## write your codes here
    # cut off the PAD part
    test_len_list = [len(s) for s in test_samples_X]
    actual_predicted_list= []
    for i in range(predictions.shape[0]):
        actual_predicted_tagindex = list(predictions[i])[:test_len_list[i]]
        actual_predicted_tag = [index2tag[index] for index in actual_predicted_tagindex]
        token_sequences.append(actual_predicted_tag)

    return token_sequences


test_samples = [
    word_tokenize("This race is awesome, I want to race too."),
    word_tokenize("That race is silly, I do not want to race.")
]
batch = 100
batch_num = len(pre_data)//batch+1
predict = []
for i in range(batch_num):
  test_samples = pre_data[i*batch:min((i+1)*batch,len(pre_data))]
  MAX_LENGTH = len(max(pre_data, key=len))
  print(test_samples)
  # Converting sentence (tokens) word to index
  test_samples_X = []
  for s in test_samples:
      s_int = []
      for w in s:
          try:
              s_int.append(word2index[w.lower()])
          except KeyError:
              s_int.append(word2index['-OOV-'])
      test_samples_X.append(s_int)

  # manually add PAD
  test_samples_X_pad = []
  for l in test_samples_X:
      test_samples_X_pad.append(l+[0]*(MAX_LENGTH-len(l)))

  index2tag = {i: t for t, i in tag2index.items()}

  model.eval()
  sentence = torch.from_numpy(np.array(test_samples_X_pad)).cuda()
  predictions = model(sentence)
  _, predictions = torch.max(predictions, -1)
  predictions = predictions.cpu().numpy()
  POS = decode_result(predictions, test_samples_X, index2tag )
  predict = predict + POS

  POS_tag = [' '.join(predict[i]) for i in range(len(predict))]
  tag = pd.DataFrame(POS_tag)
  tag.to_csv('POS_tag_sentence.csv')

  print(POS_tag[0])