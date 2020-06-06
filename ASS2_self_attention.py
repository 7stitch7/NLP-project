import math

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

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(1)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.05 # the dropout value
class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, pos_to_ix,nhead,nlayers, embedding_dim, hidden_dim,dropout,num_layers=2,method='ATTN_TYPE_DOT_PRODUCT'):
        super(BiLSTM_CRF, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead, embedding_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

        self.init_weights()

        self.method = method
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.pos_to_ix = pos_to_ix
        self.posset_size = len(pos_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        """Here we use the embedding matrix as the initial weights of nn.Embedding"""
        self.word_embeds.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.lstm = nn.LSTM(embedding_dim + len(pos_to_ix) + 1, hidden_dim // 2,
                            num_layers=self.num_layers, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2*self.num_layers, 1, self.hidden_dim // 2).to(device),
                torch.randn(2*self.num_layers, 1, self.hidden_dim // 2).to(device))
    def _generate_square_subsequent_mask(self, sz):
        #triu returns the upper triangular part of a matrix (2-D tensor) or batch of matrices (see section below)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)


    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence, pos_tagging,tf_idf):
        embeds = self.word_embeds(sentence).view(len(sentence), -1)
        pos = torch.eye(self.posset_size).to(device)[pos_tagging]
        tf_idf = torch.tensor(np.array(tf_idf),dtype=torch.float).unsqueeze(1)
        combined = torch.cat((embeds, pos,tf_idf), 1).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(combined, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

        return lstm_out

    def _cal_attention(self, lstm_out):
        self.hidden = self.init_hidden()
        device = lstm_out.device
        if self.src_mask is None or self.src_mask.size(0) != len(lstm_out):
            mask = self._generate_square_subsequent_mask(len(lstm_out)).to(device)
            self.src_mask = mask
        src = lstm_out * math.sqrt(lstm_out.size()[-1])
        src = self.pos_encoder(src)
        self_atten_output = self.transformer_encoder(src.unsqueeze(1), self.src_mask).view(len(lstm_out),-1)
        attention_out = self.hidden2tag(self_atten_output)
        return attention_out

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, pos_tagging,tf_idf, tags):
        lstm_out = self._get_lstm_features(sentence, pos_tagging,tf_idf)

        attention_feats = self._cal_attention(lstm_out)
        forward_score = self._forward_alg(attention_feats)
        gold_score = self._score_sentence(attention_feats, tags)
        return forward_score - gold_score

    def forward(self, sentence, pos_tagging,tf_idf):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_out = self._get_lstm_features(sentence, pos_tagging,tf_idf)

        attention_feats = self._cal_attention(lstm_out)

        # Find the best path, given the features.

        score, tag_seq = self._viterbi_decode(attention_feats)
        return score, tag_seq,attention_feats

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




import numpy as np
from sklearn.metrics import accuracy_score
def cal_acc(model, input_index,pos_index,tf_idf_index, output_index):
    ground_truth = []
    predicted = []
    for i in range(len(input_index)):
        input_sent = input_index[i]
        output_tag = output_index[i]
        pos_tag = pos_index[i]
        tf_idf = tf_idf_index[i]
        input_sent = torch.tensor(input_sent, dtype=torch.long).to(device)
        pos_tag = torch.tensor(pos_tag, dtype=torch.long).to(device)
        tf_idf = torch.tensor(tf_idf,dtype=torch.long).to(device)
        _,prediction,_ = model(input_sent,pos_tag,tf_idf)
        predicted = predicted + prediction
        ground_truth = ground_truth + output_tag
    accuracy = float((np.array(predicted)==np.array(ground_truth)).astype(int).sum())/float(len(ground_truth))
    return ground_truth, predicted, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 128
HIDDEN_LAYER=1
method = 'ATTN_TYPE_DOT_PRODUCT'
LR = 5

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, pos_to_ix, nhead, nlayers ,EMBEDDING_DIM, HIDDEN_DIM,dropout, HIDDEN_LAYER,method).to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
"""Each epoch will take about 1-2 minutes"""
print("="*89)
print("start training!")
print("="*89)
import datetime

for epoch in range(20):
    time1 = datetime.datetime.now()
    train_loss = 0

    model.train()
    for i, idxs in enumerate(train_input_index):
        tags_index = train_output_index[i]
        pos_index = train_pos_index[i]
        tf_idf_index = train_tf_idf_index[i]

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = torch.tensor(idxs, dtype=torch.long).to(device)
        pos_in = torch.tensor(pos_index, dtype=torch.long).to(device)
        tf_idf_in = torch.tensor(tf_idf_index,dtype=torch.float).to(device)
        targets = torch.tensor(tags_index, dtype=torch.long).to(device)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in,pos_in,tf_idf_in , targets)


        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()

    model.eval()
    _, _, train_acc = cal_acc(model,train_input_index,train_pos_index,train_tf_idf_index,train_output_index)
    _, _, val_acc = cal_acc(model,val_input_index,val_pos_index,val_tf_idf_index,val_output_index)

    val_loss = 0
    for i, idxs in enumerate(val_input_index):
        tags_index = val_output_index[i]
        sentence_in = torch.tensor(idxs, dtype=torch.long).to(device)
        pos_index = val_pos_index[i]
        pos_in = torch.tensor(pos_index, dtype=torch.long).to(device)
        tf_idf_index = val_tf_idf_index[i]
        tf_idf_in = torch.tensor(tf_idf_index,dtype=torch.float).to(device)
        targets = torch.tensor(tags_index, dtype=torch.long).to(device)
        loss = model.neg_log_likelihood(sentence_in,pos_in,tf_idf_in, targets)
        val_loss+=loss.item()
    time2 = datetime.datetime.now()

    print("Epoch:%d, Training loss: %.2f, train acc: %.4f, val loss: %.2f, val acc: %.4f, time: %.2fs" %(epoch+1, train_loss,train_acc, val_loss, val_acc, (time2-time1).total_seconds()))
torch.save(model,"NER_pos_tfidf.pt")
y_true,y_pred,_ = cal_acc(model,val_input_index,val_pos_index,val_tf_idf_index,val_output_index)

def decode_output(output_list):
    ix_to_tag = {v:k for k,v in tag_to_ix.items()}
    return [ix_to_tag[output] for output in output_list]

y_true_decode = decode_output(y_true)
y_pred_decode = decode_output(y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_true_decode,y_pred_decode,digits=4))
test_pos_index[-1]=test_pos_index[-1][:-1]
ground_truth = []
predicted = []
for i in range(len(test_input_index)):
    input_sent = test_input_index[i]

    pos_tag = test_pos_index[i]
    tf_idf = test_tf_idf_index[i]
    input_sent = torch.tensor(input_sent, dtype=torch.long).to(device)
    pos_tag = torch.tensor(pos_tag, dtype=torch.long).to(device)
    tf_idf = torch.tensor(tf_idf, dtype=torch.long).to(device)
    _, prediction, _ = model(input_sent, pos_tag, tf_idf)
    predicted = predicted + prediction

print(predicted)
test_output = decode_output(predicted)

output_file = pd.DataFrame(columns=['Id','Predicted'])
output_file['Predicted'] = test_output
output_file['Id'] = np.arange(0,len(test_output))
output_file.to_csv('predicted.csv', index=False)