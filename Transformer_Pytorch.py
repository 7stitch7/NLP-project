# Get Data
import datetime

import pandas as pd
from torch import optim

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
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0,
               tag_size = 7):
        super(Encoder, self).__init__()
        self.tag_size = tag_size

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.result = nn.Linear(model_dim,tag_size)

    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)
        result = self.result(output)


        return result, output, attentions

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
        output,_,_ = model.forward(input_sent,len(input_sent))
        output = torch.argmax(output,dim=1).tolist()
        predicted = predicted + output
        ground_truth = ground_truth + output_tag
    accuracy = float((np.array(predicted)==np.array(ground_truth)).astype(int).sum())/float(len(ground_truth))
    return ground_truth, predicted, accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Encoder(vocab_size=len(word_list),max_seq_len=1232)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
loss_func = nn.CrossEntropyLoss()

print("="*89)
print("start training!")
print("="*89)

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
        optimizer.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = torch.tensor(idxs, dtype=torch.long).to(device)
        pos_in = torch.tensor(pos_index, dtype=torch.long).to(device)
        tf_idf_in = torch.tensor(tf_idf_index,dtype=torch.float).to(device)
        targets = torch.tensor(tags_index, dtype=torch.long).to(device)

        # Step 3. Run our forward pass.
        output,_,_ = model.forward(sentence_in,len(sentence_in))
        loss = loss_func(output,targets)

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
        output, _, _ = model.forward(sentence_in, len(sentence_in))
        loss = loss_func(output, targets)
        val_loss+=loss.item()
    time2 = datetime.datetime.now()

    print("Epoch:%d, Training loss: %.2f, train acc: %.4f, val loss: %.2f, val acc: %.4f, time: %.2fs" %(epoch+1, train_loss,train_acc, val_loss, val_acc, (time2-time1).total_seconds()))
torch.save(model,"NER_pos_tfidf.pt")
y_true,y_pred,_ = cal_acc(model,val_input_index,val_pos_index,val_tf_idf_index,val_output_index)
