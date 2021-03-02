#!/usr/bin/env python
# coding: utf-8

import numpy as np

from tensorflow.keras import layers,Model
from tensorflow.keras.initializers import Constant
import tensorflow as tf

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

# ### Seq2Seq model
class S2S(tf.keras.Model):
    def __init__(self,na,W,i2w,pl):
      
        super(S2S, self).__init__() 
        
        self.nw = W.shape[0]
        self.r = W.shape[1]
        self.na = na
        self.pl = pl
        
        self.i2w = i2w
        
        self.W = layers.Embedding(self.nw,self.r)
        self.W.build((None, ))
        self.W.set_weights([W])
        self.W.trainable = True

        self.A = layers.Embedding(self.na,self.r)
                
        self.decoder = layers.GRU(512, return_sequences=True, return_state=True,dropout=0.2)
        
        self.mapper = layers.Dense(self.nw,activation = "softmax")

    @tf.function
    def call(self,a,x_topic,x,x_mask):
        
        a = tf.tile(a, [1,self.pl])
       
        x = self.W(x)
        a = self.A(a)
        
        x = tf.concat([a, x], 2)
        
        x_mask = tf.cast(x_mask,dtype=bool)
        
        out, _ = self.decoder(x,mask=x_mask,initial_state = x_topic)   

        probs = self.mapper(out)

        return probs
    
    def generate(self,a,start_emb,stop_emb):
        
        a = tf.constant([[a]])
        
        a = self.A(a)

        x_emb = tf.expand_dims(tf.expand_dims(self.W(tf.constant(start_emb)),axis = 0),axis = 0)  
        
        x_emb = tf.concat([a, x_emb], 2)

        _,out = self.decoder(x_emb) 

        probs = tf.squeeze(self.mapper(out))
        
        #x = tf.math.argmax(probs).numpy()
        val,argval = tf.nn.top_k(probs, k=5, sorted=True, name=None)
        val = val.numpy()
        val = val / np.sum(val)
        x = argval.numpy()[np.random.choice(5, 1, p=val)[0]]
        
        aout = [self.i2w[x]]

        
        for i in range(30):
            x_emb = tf.expand_dims(tf.expand_dims(self.W(tf.constant(x)),axis = 0),axis = 0) 
            x_emb = tf.concat([a, x_emb], 2)
            _,out = self.decoder(x_emb, initial_state=out)  
            
            probs = tf.squeeze(self.mapper(out))
            #x = tf.math.argmax(probs).numpy()
            val,argval = tf.nn.top_k(probs, k=5, sorted=True, name=None)
            val = val.numpy()
            val = val / np.sum(val)
            x = argval.numpy()[np.random.choice(5, 1, p=val)[0]]
            aout.append(self.i2w[x])
            
            if self.i2w[x] == stop_emb:
                break
            
        return aout
    
    def complete(self,a,start_emb,stop_emb):
        
        a = tf.constant([[a]])
        
        a1 = self.A(a)
        
        a = tf.tile(a, [1,len(start_emb)])
        
        a = self.A(a)
        
        aout = [self.i2w[x] for x in start_emb[1:]]

        x_emb = tf.expand_dims(self.W(tf.constant(start_emb)),axis = 0)
        
        x_emb = tf.concat([a, x_emb], 2)

        _,out = self.decoder(x_emb) 

        probs = tf.squeeze(self.mapper(out))
        
        #x = tf.math.argmax(probs).numpy()
        val,argval = tf.nn.top_k(probs, k=5, sorted=True, name=None)
        val = val.numpy()
        val = val / np.sum(val)
        x = argval.numpy()[np.random.choice(5, 1, p=val)[0]]
        
        aout.append(self.i2w[x])

        
        for i in range(30):
            x_emb = tf.expand_dims(tf.expand_dims(self.W(tf.constant(x)),axis = 0),axis = 0) 
            x_emb = tf.concat([a1, x_emb], 2)
            _,out = self.decoder(x_emb, initial_state=out)  
            
            probs = tf.squeeze(self.mapper(out))
            #x = tf.math.argmax(probs).numpy()
            val,argval = tf.nn.top_k(probs, k=5, sorted=True, name=None)
            val = val.numpy()
            val = val / np.sum(val)
            x = argval.numpy()[np.random.choice(5, 1, p=val)[0]]
            aout.append(self.i2w[x])
            
            if self.i2w[x] == stop_emb:
                break
            
        return aout
    
    def translate(self,a,sentence,start_emb,stop_emb):
        
        a = tf.constant([[a]])
        
        a = self.A(a)

        x_emb = tf.expand_dims(tf.expand_dims(self.W(tf.constant(start_emb)),axis = 0),axis = 0)  
        
        x_emb = tf.concat([a, x_emb], 2)

        _,out = self.decoder(x_emb,initial_state = sentence) 

        probs = tf.squeeze(self.mapper(out))
        
        #x = tf.math.argmax(probs).numpy()
        val,argval = tf.nn.top_k(probs, k=5, sorted=True, name=None)
        val = val.numpy()
        val = val / np.sum(val)
        x = argval.numpy()[np.random.choice(5, 1, p=val)[0]]
        
        aout = [self.i2w[x]]

        
        for i in range(30):
            x_emb = tf.expand_dims(tf.expand_dims(self.W(tf.constant(x)),axis = 0),axis = 0) 
            x_emb = tf.concat([a, x_emb], 2)
            _,out = self.decoder(x_emb, initial_state=out)  
            
            probs = tf.squeeze(self.mapper(out))
            #x = tf.math.argmax(probs).numpy()
            val,argval = tf.nn.top_k(probs, k=5, sorted=True, name=None)
            val = val.numpy()
            val = val / np.sum(val)
            x = argval.numpy()[np.random.choice(5, 1, p=val)[0]]
            aout.append(self.i2w[x])
            
            if self.i2w[x] == stop_emb:
                break
            
        return aout


# ### Function for optim

@tf.function
def compute_loss(model,loss_f,a,x_topic,x,x_mask,y,y_mask):
   
    probs = model(a,x_topic,x,x_mask)
    
    y_true= tf.boolean_mask(y,y_mask)
    y_pred = tf.boolean_mask(probs,y_mask) 
    
    return loss_f(y_true,y_pred),y_true,y_pred


@tf.function
def compute_apply_gradients(model,loss_f,a,x_topic,x,x_mask,y,y_mask, optimizer):

    with tf.GradientTape() as tape:
        
        loss,label,prediction= compute_loss(model, loss_f,a,x_topic,x,x_mask,y,y_mask)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss,label,prediction