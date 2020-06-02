import pandas as pd
df = pd.read_csv("POS_tag_sentence.csv")
POS = df['0'].tolist()
POS_tag = [tag.split() for tag in POS]


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

def tf_idf(data):
    import numpy as np
    DF = {}

    for tokensized_doc in data:
        # get each unique word in the doc - we need to know whether the word is appeared in the document
        for term in np.unique(tokensized_doc):
            try:
                DF[term] += 1
            except:
                DF[term] = 1

    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from collections import Counter
    import math

    tf_idf = {}

    # total number of documents
    N = len(data)

    doc_id = 0
    # get each tokenised doc
    for tokensized_doc in data:
        # initialise counter for the doc
        counter = Counter(tokensized_doc)
        # calculate total number of words in the doc
        total_num_words = len(tokensized_doc)

        # get each unique word in the doc
        for term in np.unique(tokensized_doc):
            # calculate Term Frequency
            tf = counter[term] / total_num_words

            # calculate Document Frequency
            df = DF[term]

            # calculate Inverse Document Frequency
            idf = math.log(N / (df + 1)) + 1

            # calculate TF-IDF
            tf_idf[doc_id, term] = tf * idf

        doc_id += 1

    TF_doc=[]
    for i in range(N):
        temp=[]
        for word in data[i]:
            temp.append(tf_idf[(i,word)])
        TF_doc.append(temp)

    return TF_doc

word_to_ix = {}
for sentence in train_data+validation_data+test_data:
    for word in sentence:
        word = word.lower()
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
word_list = list(word_to_ix.keys())

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {START_TAG:0, STOP_TAG:1}
for tags in target_y_train+target_y_validation:
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

pos_to_ix = {START_TAG:0, STOP_TAG:1}
for tags in POS_tag:
    for tag in tags:
        if tag not in pos_to_ix:
            pos_to_ix[tag] = len(pos_to_ix)

import gensim.downloader as api
import numpy as np
word_emb_model = api.load("glove-twitter-50")
print("="*89)
print("pre-trained word embedding model has been loaded!")
print("="*89)
EMBEDDING_DIM = 50

embedding_matrix = []
for word in word_list:
    try:
        embedding_matrix.append(word_emb_model.wv[word])
    except:
        embedding_matrix.append([0]*EMBEDDING_DIM)
embedding_matrix = np.array(embedding_matrix)
embedding_matrix.shape

train_pos = POS_tag[:len(train_data)]
val_pos = POS_tag[len(train_data):(len(train_data+validation_data))]
test_pos = POS_tag[-len(test_data):]

def to_index(data, to_ix):
    input_index_list = []
    for sent in data:
        input_index_list.append([to_ix[w] for w in sent])
    return input_index_list

train_input_index =  to_index(train_data,word_to_ix)
train_pos_index =  to_index(train_pos,pos_to_ix)
train_output_index = to_index(target_y_train,tag_to_ix)
val_input_index = to_index(validation_data,word_to_ix)
val_pos_index =  to_index(val_pos,pos_to_ix)
val_output_index = to_index(target_y_validation,tag_to_ix)
test_input_index = to_index(test_data,word_to_ix)
test_pos_index =  to_index(test_pos,pos_to_ix)

train_tf_idf_index = tf_idf(train_data)
val_tf_idf_index = tf_idf(validation_data)
test_tf_idf_index = tf_idf(test_data)
#test_output_index = to_index(target_y_test,tag_to_ix)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, n_labels,dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, n_labels)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        #triu returns the upper triangular part of a matrix (2-D tensor) or batch of matrices (see section below)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) #0::2 means starting with index 0, step = 2
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[0, :]
        return self.dropout(x)

# import torchtext
# from torchtext.data.utils import get_tokenizer
# TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
#                             init_token='<sos>',
#                             eos_token='<eos>',
#                             lower=True)
# train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
# TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def batchify(data, bsz):
#     data = TEXT.numericalize([data.examples[0].text])
#     # Divide the dataset into bsz parts.
#     nbatch = data.size(0) // bsz
#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.narrow(0, 0, nbatch * bsz)
#     # Evenly divide the data across the bsz batches.
#     data = data.view(bsz, -1).t().contiguous()
#     return data.to(device)
#
# batch_size = 20
# eval_batch_size = 10
# train_data = batchify(train_txt, batch_size)
# val_data = batchify(val_txt, eval_batch_size)
# test_data = batchify(test_txt, eval_batch_size)
#
# bptt = 35
# def get_batch(source, i):
#     seq_len = min(bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target
#
# ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
ntokens = len(word_list)
emsize = 500 # embedding dimension
nhid = 500 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 10 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 10 # the number of heads in the multiheadattention models
dropout = 0.05 # the dropout value
n_labels=len(tag_to_ix)
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, n_labels,dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

import time
def train():

    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(word_list)
    for i, idxs in enumerate(train_input_index):
        data = train_input_index[i]
        targets = train_output_index[i]
        pos_index = train_pos_index[i]
        tf_idf_index = train_tf_idf_index[i]

        data = torch.tensor(data, dtype=torch.long).to(device).unsqueeze(0)
        pos_in = torch.tensor(pos_index, dtype=torch.long).to(device)
        tf_idf_in = torch.tensor(tf_idf_index,dtype=torch.float).to(device)
        targets = torch.tensor(targets, dtype=torch.long).to(device)


        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(0), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
    cur_loss = total_loss / len(train_input_index)
    elapsed = time.time() - start_time
    print('| epoch {:3d} | '
          'lr {:02.2f} | ms/batch {:5.2f} | '
          'loss {:5.2f} | ppl {:8.2f}'.format(
            epoch, scheduler.get_lr()[0],
            elapsed,
            cur_loss, math.exp(cur_loss)))


def evaluate(eval_model, val_data,val_target):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    ground_truth=[]
    predicted = []
    ntokens = len(word_list)
    with torch.no_grad():
        for i, idxs in enumerate(val_data):
            data = val_data[i]
            tf_idf_index = val_tf_idf_index[i]
            pos_index = val_pos_index[i]
            targets = val_target[i]
            data = torch.tensor(data, dtype=torch.long).to(device).unsqueeze(0)
            pos_in = torch.tensor(pos_index, dtype=torch.long).to(device)
            tf_idf_in = torch.tensor(tf_idf_index, dtype=torch.float).to(device)
            targets = torch.tensor(targets, dtype=torch.long).to(device)
            output = eval_model(data)
            output_flat = output.squeeze(0)
            ground_truth+=targets.tolist()
            predicted+=output_flat.argmax(dim=1).tolist()
            total_loss += len(data) * criterion(output_flat, targets).item()
    accuracy = float((np.array(predicted) == np.array(ground_truth)).astype(int).sum()) / float(len(ground_truth))
    return ground_truth, predicted,accuracy,total_loss / (len(val_data) - 1)

best_val_loss = float("inf")
epochs = 10 # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    _,_,accuracy,val_loss = evaluate(model, val_input_index,val_output_index)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f} | accuracy {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss),accuracy))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

_,_,test_accuracy,test_loss = evaluate(best_model, val_input_index,val_output_index)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test accuracy {:8.2f}'.format(
    test_loss, math.exp(test_loss),test_accuracy))
print('=' * 89)
#
# bptt = 35
# def get_batch(source, i):
#     seq_len = min(bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target