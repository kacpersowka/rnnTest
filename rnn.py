"""
Adapted from https://towardsdatascience.com/language-translation-with-rnns-d84d43b40571
"""

import numpy as np
import random
import matplotlib.pyplot as plt

#Preprocess

def loadData(dataset):
    if dataset=='translate':
        with open('small_vocab_en.txt','rt') as f:
            english_sentences=f.read().split('\n')
            f.close()
        with open('small_vocab_fr.txt','rt') as f:
            french_sentences=f.read().split('\n')
            f.close()
        return (english_sentences,french_sentences)
    elif dataset=='english':
        with open('small_vocab_en.txt','rt') as f:
            english_sentences=f.read().split('\n')
            f.close()
        return (english_sentences,None)
    elif dataset=='sine':
        x=np.array([[[_]] for _ in range(0,360,5)])
        xx=np.array([[[_*(np.pi/180)]] for _ in range(0,360,5)])
        y=np.sin(xx)
        return (x,y)
    else:
        return (None,None)

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

def tanh(xx):
    x=np.array(xx,dtype=float)
    return np.tanh(x)
    
def tanhDerivative(xx):
    x=np.array(xx,dtype=float)
    return np.diag((1-np.tanh(x)**2).reshape((len(x))))

def rectLinear(xx):
    x=np.array(xx,dtype=float)
    for i in range(len(x)):
            x[i]=max(0,x[i]) #apply activation function
    return x

def rectLinearDerivative(xx):
    x=np.array(xx,dtype=float)
    for i in range(len(x)):
            x[i]=int(x[i]>0) #apply activation function
    return np.diag(x.reshape((len(x))))
    
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
    u=np.exp(z)
    du=np.diag(u.reshape(len(x)))
    dv=u.reshape(len(x))
    return (np.multiply(v,du)-np.multiply(u,dv))/v**2

def mse(x,y):
    s=0
    for i in range(len(x)):
        s+=(x[i]-y[i])**2
    return s/len(x)

def mseGradient(x,y): #dL/dX (w.r.t predicted)
    return np.array([2*(x[i]-y[i])/len(x) for i in range(len(x))])

def rnn_hidden(xt,wx,wh,ht,bh,act=tanh):
    a=wx.dot(xt)+wh.dot(ht)+bh
    h=np.array(act(a),dtype=float)
    return h
    
def rnn_hidden_derivative(xt,wx,wh,ht_1,bh,actDer=tanhDerivative):
    a=wx.dot(xt)+wh.dot(ht_1)+bh
    dhtda=np.array(actDer(a),dtype=float) #h=act(a), dhtda=act'(a)
    dadwx=np.array([xt.reshape((len(xt))) for _ in a])
    dadwh=np.array([ht_1.reshape((len(ht_1))) for _ in a])
    dadb=np.diag([1 for _ in a])
    dadh=wh
    dadx=wx
    return (dhtda.dot(dadwh),dhtda.dot(dadwx),dhtda.dot(dadb),dhtda.dot(dadh),dhtda.dot(dadx)) #dht/dwh, dht/dwx, dht/db, dht/dht-1, dht/dxt
    
def rnn_output(ht,wy,by,act=softmax):
    y=wy.dot(ht)+by
    y=np.array(act(y),dtype=float)
    return y

def rnn_output_derivative(ht,wy,by,actDer=softmaxDerivative):
    a=wy.dot(ht)+by #y=act(a)
    dyda=np.array(actDer(a),dtype=float) #Get proper jacobians from actDer
    #Should write down the derivation of these
    dadw=np.array([ht.reshape((len(ht))) for _ in a])
    dadb=np.diag([1 for _ in a])
    dadh=wy
    return (dyda.dot(dadw),dyda.dot(dadb),dyda.dot(dadh)) #dy/dwy, dy/db, dy/dh

def training(data,encoder_params,decoder_params):
    for eng,fr in data:
        pass

oneHot=lambda i,l: [1 if _==i else 0 for _ in range(l)]

def rnnPass(x,h,wh,wy,wx,bh,by,hidden_act=rectLinear,out_act=softmax):
    ht=rnn_hidden(x,wx,wh,h,bh,act=hidden_act)
    yt=rnn_output(h,wy,by,act=out_act)
    return (ht,yt)

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

def backpropThroughTime(x,y,wh,wx,wy):
    pass

#Initialise data

data='sine'
x,y=loadData(data)
if data=='translate':
    to,td=tokenize(x[:10]+y[:10])
    td_inverse={v: k for k, v in td.items()}
    pto=pad(to)
    vocab=sorted(list(td.keys()))
    vocab_size=len(td.keys())
    max_len=10
    x_dim=vocab_size
    pairs=[]
    for i in range(0,len(to),2):
        pairs.append((pto[i],pto[i+1]))
elif data=='english':
    to,td=tokenize(x)
    td_inverse={v: k for k, v in td.items()}
    pto=pad(to)
    vocab=sorted(list(td.keys()))
    vocab_size=len(td.keys())
    max_len=10
    x_dim=vocab_size
elif data=='sine':
    pairs=[]
    for i in range(len(x)):
        pairs.append((x[i],y[i]))
    x_dim=1
h_dim=80

if 1:
    #Initialise parameters
    h=[np.zeros((h_dim,1))]
    ht=h[0] #Hidden units at time t
    if data=='translate':
        xt=np.array(oneHot(random.randint(0,vocab_size-1),vocab_size)).reshape((vocab_size,1)) #Input word at time t (one-hot vector)
        #Weight matrices for encoder and decoder
        wx_enc,wx_dec,wh_enc,wh_dec,wy=np.random.uniform(0, 1, (h_dim,x_dim)),np.random.uniform(0, 1, (h_dim,x_dim)),np.random.uniform(0, 1, (h_dim,h_dim)),np.random.uniform(0, 1, (h_dim,h_dim)),np.random.uniform(0, 1, (x_dim,h_dim))
        #Biases for encoder and decoder
        bh_enc,bh_dec,by=np.random.uniform(0,1,(h_dim,1)),np.random.uniform(0,1,(h_dim,1)),np.random.uniform(0,1,(x_dim,1))
    elif data=='english':
        xt=np.array(oneHot(random.randint(0,vocab_size-1),vocab_size)).reshape((vocab_size,1))
        wx,wh,wy=np.random.uniform(0, 1, (h_dim,x_dim)),np.random.uniform(0, 1, (h_dim,h_dim)),np.random.uniform(0, 1, (x_dim,h_dim))
        bh,by=np.random.uniform(0,1,(h_dim,1)),np.random.uniform(0,1,(x_dim,1))
        t=0
        xt=np.array(oneHot(pto[0][t],vocab_size)).reshape((vocab_size,1))
        yt=np.array(oneHot(pto[0][t+1],vocab_size)).reshape((vocab_size,1))
        
        ht,y_=rnnPass(xt,ht,wh,wy,wx,bh,by)
        l=crossEntropy(y_,yt)
        dldy_=crossEntropyGradient(y_,yt)
    elif data=='sine':
        lr=1e-3
        losses=[]
        verbose=False
        epochs=200
        T=16
        wx,wh,wy=np.random.uniform(0, 1, (h_dim,x_dim)),np.random.uniform(0, 1, (h_dim,h_dim)),np.random.uniform(0, 1, (x_dim,h_dim))
        bh,by=np.random.uniform(0,1,(h_dim,1)),np.random.uniform(0,1,(x_dim,1))
        #Forward pass
        for epoch in range(epochs):
            h=[np.zeros((h_dim,1))]
            YPred=[]
            Lt=[]
            Dfht_Dht_1=[] #df(h0)_dh0, df(h1)_dh1, df(h2)_dh2 ....
            Dfht_Dwh=[] #df(h0)_dwh, df(h1)_dwh, df(h2)_dwh ....
            Dfht_Dbh=[] #df(h0)_dbh, df(h1)_dbh, df(h2)_dbh ....
            Dfht_Dwx=[] #df(h0)_dwx, df(h1)_dwx, df(h2)_dwx ....
            DLt_Dwh=[]
            DLt_Dbh=[]
            DLt_Dby=[]
            DLt_Dwy=[]
            DLt_Dwx=[]
            if verbose:
                print('Epoch',epoch+1,'out of',epochs)
                print('Running for',T,'timesteps...')
            for t in range(T):
                ht_new,yPred=rnnPass(x[t],h[t],wh,wy,wx,bh,by,tanh,tanh)
                Lt.append(mse(yPred,y[t]))
                dLt_dyPredt=mseGradient(yPred,y[t])
                dyPredt_dwy,dyPredt_dby,dyPredt_dht=rnn_output_derivative(h[t],wy,by,actDer=tanhDerivative)
                DLt_Dwy.append(np.dot(dLt_dyPredt,dyPredt_dwy))
                DLt_Dby.append(np.dot(dLt_dyPredt,dyPredt_dby))
                dLt_dht=np.dot(dLt_dyPredt,dyPredt_dht)
                dfht_dwh, dfht_dwx, dfht_dbh, dfht_dht_1, dht_dxt=rnn_hidden_derivative(x[t],wx,wh,h[t],bh,actDer=tanhDerivative)
                Dfht_Dht_1.append(dfht_dht_1)
                Dfht_Dwh.append(dfht_dwh)
                Dfht_Dwx.append(dfht_dwx)
                Dfht_Dbh.append(dfht_dbh)
                #To get dht_dwh you need to incorporate ht-1, ht-2 etc etc, truncate?
                dlt_dwh=0
                dlt_dbh=0
                dlt_dwx=0
                for i in range(len(h)):
                    dh=1
                    for j in range(i): #Incorporate truncation?
                        dh=np.dot(dh,Dfht_Dht_1[-j])
                    #Need to consider the role of dot product and incorporating weights not accounted for previously
                    dlt_dwh+=np.multiply(dLt_dht.transpose(),np.dot(dh,Dfht_Dwh[-i]))
                    dlt_dbh+=np.multiply(dLt_dht.transpose(),np.dot(dh,Dfht_Dbh[-i]))
                    dlt_dwx+=np.multiply(dLt_dht.transpose(),np.dot(dh,Dfht_Dwx[-i]))
                DLt_Dwh.append(dlt_dwh)
                DLt_Dbh.append(dlt_dbh)
                DLt_Dwx.append(dlt_dwx)
                h.append(ht_new)
                YPred.append(yPred[0][0])
            L=sum(Lt)/T
            if verbose:
                print('Predictions:',YPred)
                print('Actual:',[_[0][0] for _ in list(y)[:T]])
                print('Loss:',L)
            losses.append(L[0])
            DL_Dwh=sum(DLt_Dwh)/T
            DL_Dbh=np.diag(sum(DLt_Dbh)/T).reshape((h_dim,1))
            DL_Dwx=sum(DLt_Dwx)/T
            DL_Dwy=sum(DLt_Dwy)/T
            DL_Dby=np.diag(sum(DLt_Dby)/T).reshape((x_dim,1))
            if verbose:
                print('Updating weights...')
            """print(wh.shape,DL_Dwh.shape)
            print(wy.shape,DL_Dwy.shape)
            print(wx.shape,DL_Dwx.shape)
            print(bh.shape,DL_Dbh.shape)
            print(by.shape,DL_Dby.shape)"""
            wh=wh-lr*DL_Dwh
            wy=wy-lr*DL_Dwy
            wx=wx-lr*DL_Dwx
            bh=bh-lr*DL_Dbh
            by=by-lr*DL_Dby
            print('Epoch: ',epoch+1,'/',epochs,' Loss: ',losses[-1],' '*20,sep='',end='\r',flush=True)
        print()
        print('Done')
        plt.plot(losses)
        plt.show()

#c=encodeSentence(pairs[0][0],wx_enc,wh_enc,bh_enc)
#d=decodeSentence(c,wx_dec,wh_dec,bh_dec,wy,by)

