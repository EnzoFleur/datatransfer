from __future__ import division
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import map_tag
import nltk
import os
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import spacy
import re
import pickle
import tqdm
 
import pandas as pd

np.random.seed(13)

nlp=spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

gut_dir="C:\\Users\\EnzoT\\Documents\\datasets\\gutenberg"

authorship=pd.read_pickle(os.path.join(gut_dir, 'authorship_5.pickle'))

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.:;$!?\'\`]", " ", string)
    # string = re.sub(r"\d+", " ", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
def read(file_path):
    try:
        with open(file_path, mode='r', encoding='utf-8') as f_in:
            content=re.sub('\r\n', '', f_in.read())
            content=clean_str(content)
        if len(content)==0:
            content=None
    except:
        content=None

    return(content)

def read_lyrics(file_path):
    try:
        with open(file_path, mode='r', encoding='utf-8') as f_in:
            content=re.sub('\n', ' newLine ', f_in.read())
            content=clean_str(content)
        if len(content)==0:
            content=None
    except:
        content=None

    return(content)

def get_pgids(value, axis='author'):

    pgids={row.id: row.title for _, row in authorship[authorship[axis]==value][['id','title']].iterrows()}

    return pgids

def get_pgfiles(value, text='text', axis='author'):

    pgids=get_pgids(value, axis)

    pg_files=[os.path.join(gut_dir, text, 'PG%d_%s.txt') % (pgid, text) for pgid in pgids.keys()]
    pg_files=[pg_file for pg_file in pg_files if os.path.exists(pg_file)]

    return pg_files

def read_gutenberg(value, text='text', axis='author', clean=True):

    pgids=get_pgids(value, axis)

    pg_files=[os.path.join(gut_dir, text, 'PG%d_%s.txt') % (pgid, text) for pgid in pgids.keys()]
    pg_files=[pg_file for pg_file in pg_files if os.path.exists(pg_file)]
    
    if clean:
        content=read(pg_files[0])
    else:
        with open(pg_files[0], 'r', encoding='utf-8') as ff:
            content=ff.read()

    return content

with open("authors.txt", "r") as ff:
    authors=[line.replace('\n', '') for line in ff.readlines()]

authorship=authorship[authorship.author.isin(authors)]
authorship=authorship[['author', 'id', 'title']].groupby('author').sample(n=10)

for author in tqdm.tqdm(authorship.author.unique()):

    pg_files=get_pgfiles(author, text='text')
    os.mkdir(os.path.join(gut_dir, 'export', author))

    for pg_file in pg_files:

        content=read(pg_file)[:500000]
        doc=nlp(content)
        content=list(doc.sents)[:500]

        new_file=os.path.join(gut_dir, 'export', author, pg_file.split('\\')[-1].replace('sentence', 'export'))

        with open(new_file, 'w', encoding='utf-8') as ff:
            for line in content:
                ff.write("%s\n" % line)

poem=read_gutenberg('Baudelaire, Charles', text='sentence', clean=False)

roman=read_gutenberg('Zola, Émile', text='sentence', clean=False)



