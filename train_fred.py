#!/usr/bin/env python
# coding: utf-8


import os  
from spacy.lang.en import English
import pandas as pd
import numpy as np

from fred import S2S, compute_loss, compute_apply_gradients, pad

from tensorflow.keras import layers,Model
from tensorflow.keras.initializers import Constant
import tensorflow as tf

nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

dir="C:\\Users\\EnzoT\\Documents\\datasets\\gutenberg\\export"
authors = os.listdir(dir)   

n_sentences = 200
data = []
for author in authors:
    books=os.listdir(os.path.join(dir, author))
    for book in books:
        count=0
        with open(os.path.join(dir,author, book), 'r',encoding="utf-8") as fp:
            lines=fp.readlines()
            for line in lines[100:]:
                # if len(line)<=500:
                count=count+1
                sent=line.replace("\n","")
                tok = ['<S>'] + [token.string.strip() for token in tokenizer(sent.lower()) if token.string.strip() != ''] + ['</S>']
                data.append((author,sent,tok))

                if count==n_sentences:
                    break
                
df = pd.DataFrame(data, columns =['Author', 'Raw', 'Tokens']) 

from nltk.probability import FreqDist
raw_data = list(df['Tokens'])
flat_list = [item for sublist in raw_data for item in sublist]
freq = FreqDist(flat_list)

# ### Training Word2Vec and USE
print("USE encoding")
import tensorflow_hub as hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
USE = hub.load(module_url)
print ("module %s loaded" % module_url)
D = np.asarray(USE(df["Raw"]),dtype=np.float32)

from gensim.models import Word2Vec
import numpy as np

EMBEDDING_SIZE = 300
w2v = Word2Vec(list(df['Tokens']), size=EMBEDDING_SIZE, window=10, min_count=1, negative=10, workers=10)
word_map = {}
word_map["<PAD>"] = 0
word_vectors = [np.zeros((EMBEDDING_SIZE,))]
for i, w in enumerate([w for w in w2v.wv.vocab]):
    word_map[w] = i+1
    word_vectors.append(w2v.wv[w])
word_vectors = np.vstack(word_vectors)
i2w = dict(zip([*word_map.values()],[*word_map]))
nw = word_vectors.shape[0]
na = len(df.Author.unique())
print("%d auteurs et %d mots" % (na,nw))


# ### Padding, word2id and shifting
def pad(a,shift = False):
    shape = len(a)
    max_s = max([len(x) for x in a])
    token = np.zeros((shape,max_s+1),dtype = np.int)
    mask  =  np.zeros((shape,max_s+1),dtype = np.int)
    for i,o in enumerate(a):
        token[i,:len(o)] = o
        mask[i,:len(o)] = 1
    if shift:
        return token[:,:-1],mask[:,:-1],token[:,1:],mask[:,1:],max_s
    else:
        return token[:,:-1],mask[:,:-1],max_s


ang_tok,mask_ang_tok,ang_tok_shift,mask_ang_tok_shift,ang_pl = pad([[word_map[w] for w in text] for text in raw_data],shift = True)

aut2id = {i:auth for i, auth in enumerate(df.Author.unique())}
authors_id = np.asarray([aut2id[i] for i in list(df['Author'])])
authors_id = np.expand_dims(authors_id, 1)

batch_size = 32

X = np.hstack([authors_id,D,ang_tok,mask_ang_tok])
Y = np.hstack([ang_tok_shift,mask_ang_tok_shift])

X = X.astype(np.float32)

from sklearn.model_selection import train_test_split
import tensorflow as tf

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, random_state=101)
train_data = tf.data.Dataset.from_tensor_slices((X_train,Y_train)).batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices((X_test,Y_test)).batch(batch_size)

# ### Model declaration and training

model = S2S(na,word_vectors,i2w,ang_pl)

from tqdm import tqdm

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

loss_f = tf.keras.losses.CategoricalCrossentropy()
    
optimizer = tf.keras.optimizers.Adam()
loss_f = tf.keras.losses.CategoricalCrossentropy()
epochs = 140

checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 model=model)

tr_loss = []
te_loss = []
tr_acc = []
te_acc = []

    
for epoch in range(1, epochs + 1):
    print(epoch,flush=True,)

    for x,y in tqdm(train_data):
        
        a,x_topic,x,x_mask = tf.split(x,[1,512,ang_pl,ang_pl],axis=1)

        y,y_mask = tf.split(y,2,axis=1)
        
        y = tf.one_hot(y,depth = nw)
        
        loss,label,prediction = compute_apply_gradients(model,loss_f,a,x_topic,x,x_mask,y,y_mask,optimizer)

        train_loss(loss)
        train_accuracy(label, prediction)
        
    for x,y in tqdm(test_data):

        a,x_topic,x,x_mask = tf.split(x,[1,512,ang_pl,ang_pl],axis=1)
        y,y_mask = tf.split(y,2,axis=1)
        
        y = tf.one_hot(y,depth = nw)
        
        loss,label,prediction = compute_loss(model,loss_f,a,x_topic,x,x_mask,y,y_mask)

        test_loss(loss)
        test_accuracy(label, prediction)
    #print(" ".join(model.generate(aut2id["radiohead"],word_map['<S>'],word_map['</S>'])))
    #print(" ".join(model.generate(aut2id["disney"],word_map['<S>'],word_map['</S>'])))
    print(
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}')
    
    tr_loss.append(train_loss.result())
    te_loss.append(test_loss.result())
    tr_acc.append(train_accuracy.result())
    te_acc.append(test_accuracy.result())

    if epoch % 10 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

A = []
for i in range(model.na):
    A.append(model.A(i).numpy())
A = np.vstack(A)
    
np.save("author_embeddings.npy", A)