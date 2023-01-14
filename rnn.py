"""
Adapted from https://towardsdatascience.com/language-translation-with-rnns-d84d43b40571
"""

import numpy as np
import random

#Preprocess

def loadData():
    with open('small_vocab_en.txt','rt') as f:
        english_sentences=f.read().split('\n')
        f.close()
    with open('small_vocab_fr.txt','rt') as f:
        french_sentences=f.read().split('\n')
        f.close()
    return (english_sentences,french_sentences)

def tokenize(data):
    token_dict=dict()
    token_out=[]
    for i in data:
        sentence=i.split(' ')
        token_out.append([])
        for word in sentence:
            if word not in token_dict.keys():
                token_dict[word]=len(token_dict.keys())+1
            token_out[-1].append(token_dict[word])
    return (token_out,token_dict)

def pad(data):
    length=0
    for i in data:
        if len(i)>length:
            length=len(i)
    out=[]
    for i in data:
        out.append(i+[0 for _ in range(length-len(i))])
    return out

#Define model

def rectLinear(xx):
    x=np.array(xx,dtype=float)
    for i in range(len(x)):
            x[i]=max(0,x[i]) #apply activation function
    return x

def rectLinearDerivative(xx):
    x=np.array(xx,dtype=float)
    for i in range(len(x)):
            x[i]=int(x[i]>0) #apply activation function
    return np.diag(x)
    
def crossEntropy(q,p):
    z=np.array(q,dtype=float)+1e-7 #For numerical stability
    return -sum(p*np.log(z))

def crossEntropyGradient(q,p): #w.r.t q
    z=np.array(q,dtype=float)+1e-7
    return [[-p[i]/z[i] for i in range(len(p))]]

def softmax(x):
    z=np.array(x,dtype=float)-max(x)
    return np.exp(z) / np.sum(np.exp(z), axis=0)

def softmaxDerivative(x):
    z=np.array(x,dtype=float)-max(x)
    v=np.sum(np.exp(z), axis=0)
    u=np.array([np.exp(z)]).transpose()
    du=np.diag(u.transpose()[0])
    dv=u.transpose()[0]
    return (np.multiply(v,du)-np.multiply(u,dv))/v**2

def rnn_hidden(xt,wx,wh,ht,bh,act=rectLinear):
    h=wx.dot(xt)+wh.dot(ht)+bh
    h=np.array(act(h),dtype=float)
    return h
    
def rnn_output(ht,wy,by,act=softmax):
    y=wy.dot(ht)+by
    y=np.array(act(y),dtype=float)
    return y

def training(data,encoder_params,decoder_params):
    for eng,fr in data:
        pass

oneHot=lambda i,l: [1 if _==i else 0 for _ in range(l)]

def encodeSentence(sentence,wx,wh,bh):
    ht=np.zeros((h_dim,1))
    context=[]
    for token in sentence:
        xt=np.array(oneHot(token,vocab_size)).reshape((vocab_size,1))
        ht=rnn_hidden(xt,wx,wh,ht,bh)
        context.append(ht)
    return context

def decodeSentence(context,wx,wh,bh,wy,by):
    y=[]
    ht=context[-1]
    while (len(y)==0 or y[-1].argmax()!=td['.']) and len(y)<max_len:
        y.append(rnn_output(ht,wy,by))
        ht=rnn_hidden(y[-1],wx,wh,ht,bh)
    return y

def detokenize(tokens,td):
    sentence=[]
    for i in tokens:
        sentence.append(td[i.argmax()])
    return sentence

#Initialise data

e,f=loadData()
to,td=tokenize(e[:10]+f[:10])
td_inverse={v: k for k, v in td.items()}
pto=pad(to)

h_dim=80
vocab=sorted(list(td.keys()))
vocab_size=len(td.keys())
max_len=10

#Initialise parameters
ht=np.random.uniform(0,1,(h_dim,1))
xt=np.array(oneHot(random.randint(0,vocab_size-1),vocab_size)).reshape((vocab_size,1))
wx_enc,wx_dec,wh_enc,wh_dec,wy=np.random.uniform(0, 1, (h_dim,vocab_size)),np.random.uniform(0, 1, (h_dim,vocab_size)),np.random.uniform(0, 1, (h_dim,h_dim)),np.random.uniform(0, 1, (h_dim,h_dim)),np.random.uniform(0, 1, (vocab_size,h_dim))
bh_enc,bh_dec,by=np.random.uniform(0,1,(h_dim,1)),np.random.uniform(0,1,(h_dim,1)),np.random.uniform(0,1,(vocab_size,1))

c=encodeSentence(to[0],wx_enc,wh_enc,bh_enc)
d=decodeSentence(c,wx_dec,wh_dec,bh_dec,wy,by)