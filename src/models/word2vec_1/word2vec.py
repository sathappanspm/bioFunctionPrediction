#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

__author__ = "Debanjan Datta"
__email__ = "ddatta@vt.edu"
__version__ = "0.0.1"

import gensim
import json
import spacy
import textacy
import numpy as np
import pickle
import os
from bioFunctionPrediction.src.utils.dataloader import GODAG

# ---Model config--- #
Word_Embed_Size = 512
epochs = 250
# ---------------- #
# MODE = 0 train the model
# MODE = 1 fetch the embedding dict
# ---------------- #
MODE = 1
# this file should have :
# { _id[0...k]: np.array[shape = [Word_Embed_Size]] , ... }

EMDED_FILE = 'GO_word_embed_dict.pkl'
Word2vec_MODEL_FILE = 'word2vec_1.bin'


# ------------------ #
def get_data():
    temp_json_2 = 'temp_data_2.json'
    with open(temp_json_2) as tmp_file:
        data_dict_2 = json.loads(tmp_file.read())
    return data_dict_2


def train():
    global Word_Embed_Size
    global Word2vec_MODEL_FILE
    global epochs
    data_dict_2 = get_data()
    sentences = []
    for k, v in data_dict_2.items():
        sentences.append(v)
    model = gensim.models.Word2Vec(
        sentences,
        iter=epochs,
        window=4,
        size=Word_Embed_Size,
        workers=8,
        min_count=1
    )
    print('Model', model)
    model.save(Word2vec_MODEL_FILE)


def load_model():
    global Word2vec_MODEL_FILE
    # load model
    model = gensim.models.Word2Vec.load(Word2vec_MODEL_FILE)
    print(model)
    return model


def create_embed_dict():
    global Word_Embed_Size
    global MODE

    GODAG_obj = GODAG()
    GODAG_obj.initialize_idmap(None, None)
    idmap = GODAG_obj.idmap

    def _format(k):
        return k.replace('GO:', '')

    idmap = {_format(k): v for k, v in idmap.items()}

    if MODE == 0:
        train()

    model = load_model()
    emb_dict = {}
    data_dict = get_data()
    words = model.wv.vocab

    for k, sent in data_dict.items():
        k = str(k).zfill(7)
        sent_vec = np.zeros([Word_Embed_Size])
        for w in sent:
            try:
                vec = np.array(model.wv.word_vec(w))
                sent_vec = sent_vec + vec
            except:
                test = w in words
                print('Word not found ', w, 'in Vocab ', test)
        try:
            key_id = idmap[k]
            emb_dict[key_id] = sent_vec
        except:
            print('Key not found in idmap ', k)
    print(emb_dict.keys())
    return emb_dict


def initialize():
    global EMDED_FILE
    global MODE

    if MODE == 0:
        res = create_embed_dict()
        with open(EMDED_FILE, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    elif MODE == 1:
        if os.path.isfile(EMDED_FILE) :
            with open(EMDED_FILE, 'rb') as handle:
                res = pickle.load(handle)
        else :
            res = create_embed_dict()
    return res

def setup():
    global MODE
    initialize()
    MODE = 1

setup()

# ------------------------ #
# Use this function to extract the embedding dictionary

def get_id_embed_dict():
    return initialize()

z = get_id_embed_dict()



