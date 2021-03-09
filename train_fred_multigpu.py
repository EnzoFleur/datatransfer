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

from sklearn.model_selection import train_test_split

import horovod.tensorflow as hvd

from fred import S2S, compute_loss, pad

if __name__=="__main__":

     
    # Initialize Horovod
    hvd.init()
    
    # Pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    

    nlp = English()
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    dir="../../datasets/export"
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
                    tok = ['<S>'] + [token.string.strip() for token in tokenizer(sent.lower()) if token.string.strip() != ''] + ['</S>']
                    
                    data.append((author,sent,tok[:512]))

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

    batch_size = 32

    D=np.load("use_embeddings_512_200.npy")
    X = np.hstack([authors_id,D,ang_tok,mask_ang_tok])
    Y = np.hstack([ang_tok_shift,mask_ang_tok_shift])

    X = X.astype(np.float32)

    del df, ang_tok, mask_ang_tok, ang_tok_shift, mask_ang_tok_shift

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, random_state=101)
    train_data = tf.data.Dataset.from_tensor_slices((X_train,Y_train)).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((X_test,Y_test)).batch(batch_size)

    # ### Model declaration and training

    model = S2S(na,word_vectors,i2w,ang_pl)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    loss_f = tf.keras.losses.CategoricalCrossentropy()
        
    opt = tf.optimizers.Adam(learning_rate=0.01 * hvd.size())

    # Horovod: add Horovod DistributedOptimizer.
    opt = hvd.DistributedOptimizer(opt)

    loss_f = tf.keras.losses.CategoricalCrossentropy()
    epochs = 100

    checkpoint_dir = 'training_checkpoints' if hvd.rank() == 0 else None
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt") if hvd.rank() == 0 else None
    checkpoint = tf.train.Checkpoint(optimizer=opt,
                                    model=model)

    @tf.function
    def compute_apply_gradients_multigpu(a,x_topic,x,x_mask,y,y_mask, first_batch):

        with tf.GradientTape() as tape:
            
            loss,label,prediction= compute_loss(model, loss_f,a,x_topic,x,x_mask,y,y_mask)
            
        tape = hvd.DistributedGradientTape(tape)

        grads = tape.gradient(loss, model.trainable_variables)

        opt.apply_gradients(zip(grads, model.trainable_variables))

        if first_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(opt.variables(), root_rank=0)
        
        return loss,label,prediction

    tr_loss = []
    te_loss = []
    tr_acc = []
    te_acc = []

    for epoch in range(1, epochs + 1):
        print(epoch,flush=True,)

        for batch, (x,y) in enumerate(tqdm(train_data.take(len(train_data) // hvd.size()))):
            
            a,x_topic,x,x_mask = tf.split(x,[1,512,ang_pl,ang_pl],axis=1)

            y,y_mask = tf.split(y,2,axis=1)
            
            y = tf.one_hot(y,depth = nw)
            
            loss,label,prediction = compute_apply_gradients_multigpu(a,x_topic,x,x_mask,y,y_mask, batch==0)

            train_loss(loss)
            train_accuracy(label, prediction)

        if (hvd.rank()==0):   
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

            with open("loss_results_multi.txt", "a") as ff:
                ff.write('%06f | %06f | %06f | %06f' % (tr_loss, te_loss, tr_acc, te_acc))
            
            tr_loss.append(train_loss.result())
            te_loss.append(test_loss.result())
            tr_acc.append(train_accuracy.result())
            te_acc.append(test_accuracy.result())

            if (epoch % 10 == 0):
                checkpoint.save(file_prefix=checkpoint_prefix)
    
    if (hvd.rank()==0):
        A = []
        for i in range(model.na):
            A.append(model.A(i).numpy())
        A = np.vstack(A)

        np.save("author_embeddings_multigpu.npy", A)
