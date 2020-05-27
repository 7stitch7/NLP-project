import pandas as pd
import  numpy as np
import re
""""
df_train = pd.read_csv("imdb_train.csv")
df_test = pd.read_csv("imdb_test.csv")

reviews_train = df_train['review'].tolist()
sentiments_train = df_train['sentiment'].tolist()
reviews_test = df_test['review'].tolist()
sentiments_test = df_test['sentiment'].tolist()

print("Training set number:",len(reviews_train))
print("Testing set number:",len(reviews_test))
# extract the text from HTML, remove some irrelevant tags(e.g. <br />)
import urllib
from bs4 import BeautifulSoup
def remove_html_tags(x):
    soup = BeautifulSoup(x, "html.parser")
    x = soup.get_text()
    return x

# expand common contractions
# contraction will be considered as a new word compared to the original phrase,
# then increases irrelevant context or center words when doing word embedding. Therefore, we need expand contractions.
# These are just common English contractions. There are many edge cases. i.e. University's working on it.
contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have",
                    "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                    "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did",
                    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have",
                    "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                    "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us",
                    "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have",
                    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                    "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                    "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                    "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                    "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                    "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                    "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                    "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have",
                    "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
                    "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have",
                    "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                    "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
def expand_contractions(x, contraction_mapping=contraction_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)   if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, x)
    expanded_text = re.sub("'", "", expanded_text)

    return expanded_text


# remove punctuations
# as the dataset is the Large Movie Review Dataset from IMDB, it contains many additional characters,
# we need remove these symbols before training models.
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '_',
          'ª', '³', 'º', '½', '¾', 'ß', 'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó',
          'č', 'ô', 'õ', 'ö', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ğ', 'ı', 'ō', 'ż', 'א', 'ג', 'ו', 'י', 'כ', 'ל', 'מ', 'ן', 'ר','à', 'æ', 'ř']


import unicodedata
def remove_punctuation_re(x):
    import re
    x = re.sub(r'[^\w\s]','',x)
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, '')
    return x

# the meaning of numbers is not quite related to sentiment,
# and it can not be directly judged by amount
def remove_numbers_re(x):
    import re
    x = re.sub("\d+", " ", x)
    return x


# converting to lowercase
def Decapitalisation(x):
    x = x.lower()
    return x

# removing stop words
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize
stopword_list = sw.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
def remove_stopwords(x):
    tokens = word_tokenize(x)
    x = [w for w in tokens if not w in stopword_list]
    return x

# Grouping the inflected forms of a word together
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatisation(x):
    x = [lemmatizer.lemmatize(plural) for plural in x]
    return x
#
def data_processing(sentense_list, cleanhtml=True,Decapitalite=True, clean_punctuation=True, clean_numbers=True, stopword=True
                    ,lemmatize = True,contractions=True):

    processed = []
    i = 0
    for sentense in sentense_list:
        # print(sentense)
        if (cleanhtml):
            sentense = remove_html_tags(sentense)
        if (contractions):
            sentense = expand_contractions(sentense)
        if (clean_punctuation):
            sentense = remove_punctuation_re(sentense)
        if (clean_numbers):
            sentense = remove_numbers_re(sentense)
        if (Decapitalite):
            sentense = Decapitalisation(sentense)
        if (stopword):
            sentense = remove_stopwords(sentense)
        if (lemmatize):
            sentense = lemmatisation(sentense)

        # i += 1
        precessed_sentense = ' '.join(sentense)
        processed.append(precessed_sentense)
        # print(sentense)
        # print(i)
        print(precessed_sentense)
    print("processing have been done!")
    return processed
#
#
#
x_train = data_processing(reviews_train)
x_test = data_processing(reviews_test)

df = pd.DataFrame(columns=['train_data','train_label','test_data','test_label'])
df['train_data'] = x_train
df['train_label'] = sentiments_train
df['test_data'] = x_test
df['test_label'] = sentiments_test
df.to_csv("./processed data.csv", sep=',')
data = pd.read_csv('processed data.csv')
x_train = data['train_data'].tolist()
x_test =data['test_data'].tolist()
"""
data = pd.read_csv('processed data.csv')
x_train = data['train_data'].tolist()
x_test =data['test_data'].tolist()

sentences = x_train + x_test


word_sequence = " ".join(sentences).split()
words = word_sequence
word_size = {}
for i in word_sequence:
    if i in word_size:
        word_size[i]+=1
    else:
        word_size[i] = 1
words = []
for word in word_size:
    if word_size[word]>10:
        words.append(word)

word_list = sorted(list(set(words)))
# make dictionary so that we can be reference each index of unique word
word_dict = {w: i for i, w in enumerate(word_list)}
new_sentences = []
for sentence in sentences:
    sentence = sentence.split()
    new_words = []
    for word in sentence:
        if word in word_dict:
            new_words.append(word)
    new_words = new_words[:500]
    sentence = ' '.join(new_words)
    new_sentences.append(sentence)


sentences = new_sentences

window_size = 4

def make_target_context(sentences,window_size):
    skip_grams = []
    # window size is 2

    for sentence in sentences:
        sentence = sentence.split()
        for i in range(len(sentence)):
            target_word = word_dict[sentence[i]]
            context_words = sentence[max(i-window_size,0):i]+sentence[i+1:min(i+window_size+1,len(sentence))]
            for w in context_words:
                skip_grams.append([target_word,word_dict[w]])

    return skip_grams


skip_grams = make_target_context(sentences,window_size)
print('data preparing has been done!')


# Please comment your code
voc_size = len(word_list)
print(voc_size)


def prepare_batch(data, size, current_batch):
    random_inputs = []
    random_labels = []
    input = data[(current_batch-1)*size:min(current_batch*size,len(data))]
    for i in range(len(input)):
        input_temp = [0] * voc_size
        input_temp[input[i][0]] = 1
        random_inputs.append(input_temp)  # target
        random_labels.append(input[i][1])  # context word

    return np.array(random_inputs), np.array(random_labels)


learning_rate = 0.001
batch_size = 256
embedding_size = 100
no_of_epochs = 2000

# Please comment your code
# Please comment your code
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SkipGram(nn.Module):
    def __init__(self):
        super(SkipGram, self).__init__()
        # You need to use "bias=False" when you define Linear functions
        # ***************put your code here***************
        self.linear1 = nn.Linear(voc_size, embedding_size, bias=False)
        self.linear2 = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, x):
        hidden = self.linear1(x)
        out = self.linear2(hidden)
        return out


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#skip_gram_model = torch.load('wordembedding14.pt').to(device)
skip_gram_model = SkipGram().to(device)
criterion = nn.CrossEntropyLoss()  # please note we are using "CrossEntropyLoss" here
optimiser = optim.Adam(skip_gram_model.parameters(), lr=learning_rate)
print('start training now!')
batch_num = len(skip_grams)//batch_size
print(batch_num)
for epoch in range(no_of_epochs):
    current_batch = 1
    for batch in range(batch_num):

        inputs, labels = prepare_batch(skip_grams, batch_size,current_batch=current_batch)
        inputs_torch = torch.from_numpy(inputs).float().to(device)
        labels_torch = torch.from_numpy(labels).to(device)

        # ***************put your code here***************
        # 1. zero grad
        # 2. forword propagation
        # 3. calculate loss
        # 4. back propagation
        optimiser.zero_grad()
        outputs = skip_gram_model(inputs_torch)
        loss = criterion(outputs, labels_torch)  # We don't need to calcualte logsoftmax here
        loss.backward()
        optimiser.step()
        if current_batch%20==19:
            print('Epoch: %d, Batch: %d,Left Batch: %d, loss: %.4f' % (epoch + 1,current_batch , batch_num-current_batch,loss))
        current_batch+=1
        if current_batch%500==499:
            torch.save(skip_gram_model, "wordembedding17.pt")

    #if epoch % 100 == 99:
    print('Epoch: %d, loss: %.4f' % (epoch + 1, loss))

# Hint: you can refer lab1 to know how to get the weight from a Model Linear layer
    if epoch%100==99:
        torch.save(skip_gram_model, "wordembedding17.pt")

#wordembedding 1, batch_size = 256, embedding_size = 16
#wordembedding 2, batch_size = 64, embedding_size = 16
#wordembedding 3, batch_size = 256, embedding_size = 16
#wordembedding 4, batch_size = 64, embedding_size = 16
#wordembedding 5, batch_size = 256, embedding_size = 50
#wordembedding 6, batch_size = 256, embedding_size = 50 new data
#wordembedding 7, batch_size = 256, embedding_size = 50 small word size
#wordembedding 9, batch_size = 1024, embedding_size = 50 small word size >100
#wordembedding 10, batch_size = 1024, embedding_size = 50 small word size >100, sentence len <= 500
#wordembedding 11, batch_size = 1024, embedding_size = 50 small word size >100,window_size = 4.lr=0.001,sentence len <= 500
#wordembedding 12, batch_size = 1024, embedding_size = 50 small word size >150,window_size = 4,lr=0.001
#wordembedding 13, batch_size = 1024, embedding_size = 50 small word size >150,window_size = 5,lr=0.001
#wordembedding 14, batch_size = 1024, embedding_size = 100 small word size >100,window_size = 4,lr=0.001
#wordembedding 15, batch_size = 256, embedding_size = 100 small word size >100,window_size = 4,lr=0.001
#wordembedding 16, batch_size = 256, embedding_size = 100 small word size >100,window_size = 4,lr=0.001


# # Please comment your code
# import pandas as pd
# import  numpy as np
# data = pd.read_csv('processed data.csv')
# data.loc[data['test_label']=='pos',['test_label']] = 1
# data.loc[data['test_label']=='neg',['test_label']] = 0
# data.loc[data['train_label']=='pos',['train_label']] = 1
# data.loc[data['train_label']=='neg',['train_label']] = 0
# x_train = data['train_data'].tolist()
# x_test =data['test_data'].tolist()
# y_train = data['train_label'].tolist()
# y_test = data['test_label'].tolist()
# sentences = x_train + x_test
# word_sequence = " ".join(sentences).split()
# words = word_sequence
# word_size = {}
# for i in word_sequence:
#     if i in word_size:
#         word_size[i]+=1
#     else:
#         word_size[i] = 1
# words = []
# for word in word_size:
#     if word_size[word]>100:
#         words.append(word)
#
# word_list = sorted(list(set(words)))
# # make dictionary so that we can be reference each index of unique word
# word_dict = {w: i+1 for i, w in enumerate(word_list)}
# new_sentences = []
# char = []
# for word in word_list:
#     for i in word:
#         char.append(i)
# print(len(set(char)))
# print(sorted(set(char)))
# print(len(word_list))
# for sentence in sentences:
#     sentence = sentence.split()
#     new_words = []
#     for word in sentence:
#         if word in word_dict:
#             new_words.append(word)
#     sentence = ' '.join(new_words)
#     new_sentences.append(sentence)
#
# sentences = new_sentences
# print(len(sentences))
# x_train = sentences[:len(x_train)]
# x_test = sentences[len(x_train):]
# print(x_train[1])
# # make dictionary so that we can be reference each index of unique word
#
# voc_size = len(word_list)
# embedding_size = 50
#
#
# # Assume that we have the following character instances
# char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
#             'h', 'i', 'j', 'k', 'l', 'm', 'n',
#
#             'o', 'p', 'q', 'r', 's', 't', 'u',
#             'v', 'w', 'x', 'y', 'z']
#
# # one-hot encoding and decoding
# # {'a': 0, 'b': 1, 'c': 2, ..., 'j': 9, 'k', 10, ...}
# num_dic = {n: i for i, n in enumerate(char_arr)}
# dic_len = len(num_dic)
#
# # a list words for sequence data (input and output)
# seq_data = word_list
#
#
# # Make a batch to have sequence data for input and ouput
# # wor -> X, d -> Y
# # dee -> X, p -> Y
# max_seq = 0
# for sentence in sentences:
#     if len(sentence.split()) > max_seq:
#         max_seq = len(sentence.split())
# topword = 500
# def padding(sentences,max_seq=topword):
#     seq = []
#     print(len(sentences))
#     for sentence in sentences:
#         sentence = sentence.split()
#         if len(sentence) < min(max_seq,topword):
#             sentence = sentence + ['<padding>']*(max_seq-len(sentence))
#         if len(sentence) >= min(max_seq,topword):
#             sentence = sentence[:min(max_seq,topword)]
#             print(sentence)
#         seq.append(sentence)
#     return seq
#
# # word_sequence = []
# # for sentence in sentences:
# #     for word in sentence:
# #         word_sequence.append(word)
#
# # Please comment your code
# # convert all sentences to unique word list
#
# char = []
# for word in word_list:
#     for i in word:
#         char.append(i)
# print(len(set(char)))
# print(sorted(set(char)))
# print(len(word_list))
# word_dict['<padding>']=0
# X_train = padding(x_train)
# X_test = padding(x_test)
# new_sentences = []
# for sentence in X_train:
#   new_sentences.append([word_dict[word] for word in sentence])
# X_train=np.array(new_sentences)
# new_sentences = []
# for sentence in X_test:
#   new_sentences.append([word_dict[word] for word in sentence])
# X_test=np.array(new_sentences)