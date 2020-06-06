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
word_emb_model = api.load("glove-twitter-200")
print("="*89)
print("pre-trained word embedding model has been loaded!")
print("="*89)
EMBEDDING_DIM = 200

embedding_matrix = []
for word in word_list:
    try:
        embedding_matrix.append(word_emb_model.wv[word])
    except:
        embedding_matrix.append([0]*EMBEDDING_DIM)
embedding_matrix = np.array(embedding_matrix)


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

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale

        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output).squeeze(dim=1)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class LayerNorm(nn.Module):
    """实现LayerNorm。其实PyTorch已经实现啦，见nn.LayerNorm。"""

    def __init__(self, features, epsilon=1e-6):
        """Init.

        Args:
            features: 就是模型的维度。论文默认512
            epsilon: 一个很小的数，防止数值计算的除0错误
        """
        super(LayerNorm, self).__init__()
        # alpha
        self.gamma = nn.Parameter(torch.ones(features))
        # beta
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        """前向传播.

        Args:
            x: 输入序列张量，形状为[B, L, D]
        """
        # 根据公式进行归一化
        # 在X的最后一个维度求均值，最后一个维度就是模型的维度
        mean = x.mean(-1, keepdim=True)
        # 在X的最后一个维度求方差，最后一个维度就是模型的维度
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta


def padding_mask(seq_k, seq_q):
    # seq_k和seq_q的形状都是[B,L]
    seq_k=seq_q=seq_k.unsqueeze(0)
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                      diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        """初始化。

        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model]).type(torch.float64)
        position_encoding = torch.from_numpy(position_encoding)
        position_encoding = torch.cat((pad_row, position_encoding))

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        # 找出这一批序列的最大长度
        # max_len = torch.max(input_len)
        # tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # # 这里range从1开始也是因为要避开PAD(0)的位置
        # input_pos = tensor(
        #     [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        input_pos = torch.Tensor(list(range(1,input_len+1))).type(torch.LongTensor)
        return self.position_encoding(input_pos)



class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.unsqueeze(dim=2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(x + output.squeeze(dim=2))
        return output



class EncoderLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):

    def __init__(self,
               vocab_size,
               max_seq_len,
               num_layers=6,
               model_dim=240,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        output = inputs
        output += self.pos_embedding(inputs_len)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)

        return output, attentions

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, pos_to_ix, embedding_dim, hidden_dim,num_layers=2,method='ATTN_TYPE_DOT_PRODUCT'):
        super(BiLSTM_CRF, self).__init__()
        self.method = method
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.pos_to_ix = pos_to_ix
        self.posset_size = len(pos_to_ix)
        self.mulithead_attention =Encoder(vocab_size=len(word_list),max_seq_len=1232,num_heads=8,num_layers=6)

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
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), -1)
        pos = torch.eye(self.posset_size).to(device)[pos_tagging]
        tf_idf = torch.tensor(np.array(tf_idf),dtype=torch.float).unsqueeze(1)
        combined = torch.cat((embeds, pos,tf_idf), 1).view(len(sentence), -1)

        combined,_ = self.mulithead_attention(combined,combined.size(0))
        combined = combined.unsqueeze(dim=1)

        lstm_out, self.hidden = self.lstm(combined, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

        return lstm_out

    def _cal_attention(self, lstm_out, method):
        attention_result = torch.zeros(lstm_out.size()[0], self.hidden_dim * 2, device=device)
        if method == 'ATTN_TYPE_DOT_PRODUCT':
            # bmm: https://pytorch.org/docs/master/generated/torch.bmm.html
            for i in range(lstm_out.size()[0]):
                hidden = lstm_out[i]
                attn_weights = F.softmax(torch.bmm(hidden.unsqueeze(0).unsqueeze(0), lstm_out.T.unsqueeze(0)), dim=-1)
                attn_output = torch.bmm(attn_weights, lstm_out.unsqueeze(0))
                concat_output = torch.cat((hidden.unsqueeze(0),attn_output[0]), 1)
                attention_result[i] = concat_output.squeeze(0)
        elif method == 'ATTN_TYPE_SCALE_DOT_PRODUCT':
            for i in range(lstm_out.size()[0]):
                hidden = lstm_out[i]
                attn_weights = F.softmax(1/np.sqrt(self.hidden_dim)*torch.bmm(hidden.unsqueeze(0).unsqueeze(0), lstm_out.T.unsqueeze(0)), dim=-1)
                attn_output = torch.bmm(attn_weights, lstm_out.unsqueeze(0))
                concat_output = torch.cat((hidden.unsqueeze(0),attn_output[0]), 1)
                attention_result[i] = concat_output.squeeze(0)
        attention_out = self.hidden2tag(attention_result)
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
        lstm_feats = self.hidden2tag(lstm_out)
        #attention_feats = self._cal_attention(lstm_out, self.method)
        forward_score = self._forward_alg(lstm_feats)
        gold_score = self._score_sentence(lstm_feats, tags)
        return forward_score - gold_score

    def forward(self, sentence, pos_tagging,tf_idf):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_out = self._get_lstm_features(sentence, pos_tagging,tf_idf)
        lstm_feats = self.hidden2tag(lstm_out)
        #attention_feats = self._cal_attention(lstm_out, self.method)

        # Find the best path, given the features.


        # score, tag_seq = self._viterbi_decode(lstm_feats)
        return lstm_feats



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
        tf_idf = torch.tensor(tf_idf,dtype=torch.float64).to(device)
        # _,prediction,_ = model(input_sent,pos_tag,tf_idf)
        output = model(input_sent,pos_tag,tf_idf)
        prediction = torch.argmax(output,dim=1).tolist()
        predicted = predicted + prediction
        ground_truth = ground_truth + output_tag
    accuracy = float((np.array(predicted)==np.array(ground_truth)).astype(int).sum())/float(len(ground_truth))
    return ground_truth, predicted, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 128
HIDDEN_LAYER= 2
method = 'ATTN_TYPE_DOT_PRODUCT'


model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, pos_to_ix ,EMBEDDING_DIM, HIDDEN_DIM,HIDDEN_LAYER,method).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4)
loss_func = nn.CrossEntropyLoss()
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
        # loss = model.neg_log_likelihood(sentence_in,pos_in,tf_idf_in , targets)
        output = model.forward(sentence_in,pos_in,tf_idf_in)
        loss = loss_func(output, targets)


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
        tf_idf_in = torch.tensor(tf_idf_index,dtype=torch.float64).to(device)
        targets = torch.tensor(tags_index, dtype=torch.long).to(device)
        output = model(sentence_in, pos_in, tf_idf_in)
        prediction = torch.argmax(output, dim=1).tolist()
        # loss = model.neg_log_likelihood(sentence_in,pos_in,tf_idf_in, targets)
        loss = loss_func(output, targets)
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
    tf_idf = torch.tensor(tf_idf, dtype=torch.float64).to(device)
    # _, prediction, _ = model(input_sent, pos_tag, tf_idf)
    output = model(input_sent, pos_tag, tf_idf)
    prediction = torch.argmax(output, dim=1).tolist()
    predicted = predicted + prediction

print(predicted)
test_output = decode_output(predicted)

output_file = pd.DataFrame(columns=['Id','Predicted'])
output_file['Predicted'] = test_output
output_file['Id'] = np.arange(0,len(test_output))
output_file.to_csv('predicted.csv', index=False)