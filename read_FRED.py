#!/usr/bin/env python
# coding: utf-8

import os  
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime

from spacy.lang.en import English
from gensim.models import Word2Vec
from nltk.probability import FreqDist

from sklearn.model_selection import train_test_split

import tensorflow as tf

from fred import S2S, compute_loss, pad

def build_dataset(dir, batch_size):
    
    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    authors = os.listdir(dir)   

    max_length=512
    n_sentences = 200
    data = []
    for author in authors:
        books=os.listdir(os.path.join(dir, author))
        for book in books:
            count=0
            with open(os.path.join(dir,author, book), 'r',encoding="utf-8") as fp:
                lines=fp.readlines()
                for line in lines[100:]:
                    count=count+1
                    sent=line.replace("\n","")
                    tok = ['<S>'] + [token.string.strip() for token in tokenizer(sent.lower()) if token.string.strip() != ''][:(max_length-2)] + ['</S>']
                    
                    data.append((author,sent,tok))

                    if count==n_sentences:
                        break
                    
    df = pd.DataFrame(data, columns =['Author', 'Raw', 'Tokens']) 

    raw_data = list(df['Tokens'])
    flat_list = [item for sublist in raw_data for item in sublist]
    freq = FreqDist(flat_list)

    # ### Training Word2Vec and USE
    # print("USE encoding")
    # import tensorflow_hub as hub
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
    # USE = hub.load(module_url)
    # print ("module %s loaded" % module_url)
    # D = np.asarray(USE(df["Raw"]),dtype=np.float32)

    print("Training Word2Vec")

    EMBEDDING_SIZE = 100
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

    ang_tok,mask_ang_tok,ang_tok_shift,mask_ang_tok_shift,ang_pl = pad([[word_map[w] for w in text] for text in raw_data],shift = True)

    aut2id = {auth:i for i, auth in enumerate(df.Author.unique())}
    authors_id = np.asarray([aut2id[i] for i in list(df['Author'])])
    authors_id = np.expand_dims(authors_id, 1)

    D=np.load("use_embeddings_512_200.npy")
    X = np.hstack([authors_id,D,ang_tok,mask_ang_tok])
    Y = np.hstack([ang_tok_shift,mask_ang_tok_shift])

    X = X.astype(np.float32)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, random_state=101)
    train_data = tf.data.Dataset.from_tensor_slices((X_train,Y_train)).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((X_test,Y_test)).batch(batch_size)

    return train_data, test_data, na, nw, word_vectors, i2w, ang_pl, aut2id

if __name__=="__main__":

    dir="..\\..\\datasets\\gutenberg\\export"
    batch_size=32

    train_data, test_data, na, nw, word_vectors, i2w, ang_pl, aut2id = build_dataset(dir, batch_size)
    word_map=dict(zip([*i2w.values()],[*i2w]))

    # ### Model declaration and training

    print("Building model ... \n", flush=True)
    model = S2S(na,word_vectors,i2w,ang_pl)
    optimizer = tf.optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint("training_checkpoints\\"))

    A = []
    for i in range(model.na):
        A.append(model.A(i).numpy())
    A = np.vstack(A)

    print("Free Generation")
    for test in range(5):
        print("Molière")
        print(" ".join(model.generate(aut2id["Molière"],word_map['<S>'],word_map['</S>'])))
        print("Cervantes")
        print(" ".join(model.generate(aut2id["Cervantes Saavedra, Miguel de"],word_map['<S>'],word_map['</S>'])))
        print("Dickens")
        print(" ".join(model.generate(aut2id["Dickens, Charles"],word_map['<S>'],word_map['</S>'])))

    print("USE encoding")
    import tensorflow_hub as hub
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
    USE = hub.load(module_url)
    print ("module %s loaded" % module_url)

    vec = USE(["But I'm a creep, I'm a weirdo newline What the hell am I doing here?"])
    vec = tf.constant(np.asarray(vec,dtype=np.float32))

    print("Creep in the style of ")
    print("Molière")
    print(" ".join(model.translate(aut2id["Molière"],vec,word_map['<S>'],word_map['</S>'])))
    print("Cervantes")
    print(" ".join(model.translate(aut2id["Cervantes Saavedra, Miguel de"],vec,word_map['<S>'],word_map['</S>'])))
    print("Dickens")
    print(" ".join(model.translate(aut2id["Dickens, Charles"],vec,word_map['<S>'],word_map['</S>'])))
