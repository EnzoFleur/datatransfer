#!/usr/bin/env python
# coding: utf-8


import os  
import pandas as pd
import numpy as np
from tqdm import tqdm

from spacy.lang.en import English
from gensim.models import Word2Vec
from nltk.probability import FreqDist

import tensorflow as tf

from fred import pad

if __name__=="__main__":

    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    dir="../../datasets/export"
    authors = os.listdir(dir)   

    n_sentences = 200
    data = []
    print("Iterating through authors ... ", flush=True)
    for author in tqdm(authors):
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

    ang_tok,mask_ang_tok,ang_tok_shift,mask_ang_tok_shift,ang_pl = pad([[word_map[w] for w in text] for text in raw_data],shift = True)

    aut2id = {auth:i for i, auth in enumerate(df.Author.unique())}
    authors_id = np.asarray([aut2id[i] for i in list(df['Author'])])
    authors_id = np.expand_dims(authors_id, 1)

    batch_size = 32

    D=np.load("use_embeddings.npy")
    X = np.hstack([authors_id,D,ang_tok,mask_ang_tok])
    Y = np.hstack([ang_tok_shift,mask_ang_tok_shift])

    X = X.astype(np.float32)

    print("Saving data ... ")
    np.save('dataset_X.npy', X)
    np.save('dataset_Y.npy', Y)
    print("Preprocessing done !")