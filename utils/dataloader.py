#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"

import tarfile
import gzip
from Bio import SeqIO
import random
import logging
import obonet
import numpy as np
import networkx as nx
from functools import partial
import glob
import json
import os
import sys
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('DataLoader')


DATADIR = os.path.join(os.path.dirname(__file__), '../resources/')
def has_path(a, b, go_dag=None):
    """
    return True if the GODAG has a path from a to b.
    """
    try:
        if nx.has_path(go_dag, a, b):
            return True
    except Exception as e:
        log.exception(e)

    return False


class GODAG(object):
    net = obonet.read_obo(os.path.join(DATADIR, 'go.obo'))
    isagraph = nx.DiGraph(net.edges(keys=['is_a', 'part-of'])).reverse()
    vfunc = np.vectorize(partial(has_path, go_dag=isagraph), otypes=[np.bool_])
    alt_id = {val: key for key, vals in nx.get_node_attributes(net, "alt_id").items() for val in vals}
    idmap = None
    GOIDS = None

    @staticmethod
    def initialize_idmap(idlist=None, root=None):
        allnodes = set(GODAG.isagraph.nodes)
        if root:
            allnodes = set(nx.descendants(GODAG.isagraph, root))
            
        if idlist is not None :
            log.info('loading go funcs of len-{}'.format(len(idlist)))
            GODAG.GOIDS = list(set([GODAG.get(id) for id in idlist]))
            updatedIdlist = (GODAG.GOIDS).copy()
            GODAG.idmap = {id: index for index, id in enumerate(GODAG.GOIDS)}
            goset = allnodes - GODAG.idmap.keys()
        else:
            with open(os.path.join(DATADIR, 'funcCounts.json')) as inpf:
                funcweights = json.load(inpf)
                
            tmp = sorted(funcweights.items(), reverse=True, key=lambda x: x[1])
            GODAG.idmap = {GODAG.get(tmp[index][0]): index for index in range(len(tmp))}
            goset = allnodes - funcweights.keys()
            GODAG.GOIDS = [item[0] for item in tmp]
            updatedIdlist = GODAG.GOIDS
        
        GODAG.GOIDS += list(goset)
        for id in goset:
            GODAG.idmap[GODAG.get(id)] = len(GODAG.idmap)
        
        return updatedIdlist

    @staticmethod
    def get_leafnodes(labels):
        """
        function to get only unique paths in GO DAG (children and its ancestors must not be in the list at the same time)
        """
        ulabels = []
        for seq in labels:
            uniqset = set(seq)
            for i in seq:
                ## remove any ancestors present in label set
                uniqset.difference_update(nx.ancestors(GODAG.isagraph, i))

            ulabels.append(list(uniqset))

        return ulabels

    @staticmethod
    def get_ancestors(node):
        return nx.ancestors(GODAG.isagraph, GODAG.get(node))

    @staticmethod
    def get_children(node):
        return nx.descendants(GODAG.isagraph, GODAG.get(node))

    @staticmethod
    def get_id(node):
        return GODAG.idmap[GODAG.get(node)]

    @staticmethod
    def id2node(id):
        return GODAG.GOIDS[id]

    @staticmethod
    def get(id):
        return GODAG.alt_id.get(id, id)

    @staticmethod
    def to_npy(funcs):
        mat = np.zeros(len(GODAG.idmap), dtype=np.int32)
        indices = [GODAG.get_id(node) for node in funcs]
        mat[indices] = 1
        return mat

    @staticmethod
    def get_fullmat(funcs):
        return np.any(vfunc(GOIDS, funcs), axis=0)


class FeatureExtractor():
    ngrammap = dict()
    aminoacidmap = dict()

    @staticmethod
    def to_ngrams(seq, ngram=3):
        return [FeatureExtractor._ngram2id(seq[i: i + ngram]) for i in range(len(seq) -ngram + 1)]

    @staticmethod
    def _ngram2id(ngram):
        if ngram not in FeatureExtractor.ngrammap:
            FeatureExtractor.ngrammap[ngram] = len(FeatureExtractor.ngrammap) + 1

        return FeatureExtractor.ngrammap[ngram]

    @staticmethod
    def to_onehot(seq):
        return [FeatureExtractor._aminoacid2id(aacid) for aacid in seq]

    @staticmethod
    def _aminoacid2id(aacid):
        if aacid not in FeatureExtractor.aminoacidmap:
            FeatureExtractor.aminoacidmap[aacid] = len(FeatureExtractor.aminoacidmap) + 1

        return FeatureExtractor.aminoacidmap[aacid]

    @staticmethod
    def dump(datadir):
        with open(os.path.join(DATADIR, 'aminoAcidMap.json'), 'w') as outf:
            json.dump(outf, FeatureExtractor.aminoacidmap)

        with open(os.path.join(DATADIR, 'ngramMap.json'), 'w') as outf:
            json.dump(outf, FeatureExtractor.ngrammap)

    @staticmethod
    def load(datadir):
        with open(os.path.join(DATADIR, 'aminoacids.txt'), 'r') as inpf:
            FeatureExtractor.aminoacidmap = {key: (index + 1) for index, key in enumerate(json.load(inpf))}

        with open(os.path.join(DATADIR, 'ngrams.txt'), 'r') as inpf:
            ngrams = json.load(inpf)
            FeatureExtractor.ngrammap = {key: (index + 1) for index, key in enumerate(ngrams)}


class DataLoader(object):
    def __init__(self, filename='/groups/fungcat/datasets/current/fasta/AllSeqsWithGO_expanded.tar'):
        self.dir = os.path.isdir(filename)
        if self.dir:
            self.tarobj = filename
            self.members = glob.glob(os.path.join(filename, '*'))
        else:
            self.tarobj = tarfile.open(filename)
            self.members = self.tarobj.getmembers()
        
        self.openfiles = set()

    def getmember(self, member):
        """
        return the gzip file obj of member
        """
        if member.name in self.openfiles:
            return self.getmember(self.members[random.randint(0, len(self.members))])
            
        self.openfiles.add(member.name)    
        if self.dir:
            fobj = gzip.open(member, 'rt')
        else:
            fobj = gzip.open(self.tarobj.extractfile(member),
                              mode='rt')
        
        return (member.name, fobj)

    def getrandom(self):
        return self.getmember(self.members[random.randint(0, len(self.members))])
    
    def close(self):
        if not self.dir:
            self.tarobj.close()


class DataIterator(object):
    def __init__(self, batchsize=1,
                 functype='', size=100, seqlen=2000,
                 featuretype='onehot', dataloader=None,
                 numfiles=None, ngramsize=3, **kwargs):
        self.fobj = []
        self.fnames = []
        self.current = 0
        self.numfiles = None
        self.batchsize = batchsize
        self.functype = functype
        self.maxdatasize = size
        self.maxseqlen = seqlen
        self.itersize = 0
        self.featuretype = featuretype
        self.loader = dataloader
        self.ngramsize = ngramsize
        self.expectedshape = ((self.maxseqlen - self.ngramsize + 1) 
                              if self.featuretype == 'ngrams' else self.maxseqlen)
        
        self.featurefunc = getattr(FeatureExtractor, 'to_{}'.format(featuretype))
        
        if featuretype == 'ngrams':
            self.featureExt = lambda x: self.featurefunc(x, ngram=ngramsize)
            log.info('using ngrams of size {} as feature representation'.format(self.ngramsize))
        else:
            self.featureExt = lambda x: self.featurefunc(x)

    def __next__(self):
        if self.fobj == []:
            self.loadfile()

        if self.itersize >= self.maxdatasize:
            raise StopIteration

        inputs, labels = [], []
        for fastaObj in SeqIO.parse(self.fobj[self.current], 'fasta'):
            seq = str(fastaObj.seq)
            if len(seq) > self.maxseqlen:
                continue

            desc = json.loads(fastaObj.description.split(' ', 1)[1])
            funcs = [GODAG.get(func['go_id']) for func in desc['go_ids']
                     if((func.get('qualifier', '') != 'NOT') and
                        ((self.functype == '') or
                         (self.functype[-1].lower() == func['aspect'].lower()))
                        )]

            inputs.append(self.featureExt(seq))
            labels.append(GODAG.to_npy(funcs))
            if len(labels) >= self.batchsize:
                log.info('sending batch')
                self.itersize += 1
                inputs, labels = self._format(inputs, labels)
                return inputs, labels


        if self.itersize >= self.maxdatasize:
            raise StopIteration
        
        self.current = (self.current + 1)
        if self.numfiles:
            self.current = self.current % len(self.numfiles)
        else:
            self.fobj[-1].close()
        
        if self.current < len(self.fnames):
            self.loadfile()
        
        if inputs:
            self.itersize += 1
            inputs, labels = self._format(inputs, labels)
        else:
            inputs, labels = self.__next__()
            
        return inputs, labels

    def _format(self, inputs, labels):
        inputs, labels = pd.DataFrame(inputs, dtype=np.int32).fillna(0), np.vstack(labels)
        log.info('{}'.format(str(inputs.shape)))
        if inputs.shape[1] < self.expectedshape:
            inputs = np.concatenate([inputs.as_matrix(), 
                                      np.zeros((inputs.shape[0],
                                                self.expectedshape - inputs.shape[1]))], axis=1)
        return inputs, labels
    
    def loadfile(self):
        flhandler = self.loader.getrandom()
        self.fnames.append(flhandler[0])
        self.fobj.append(flhandler[1])
        log.info('read file - {}'.format(flhandler[0]))
    
    def __iter__(self, inputs=[], labels=[]):
        return self

    def reset(self):
        for fno in range(len(self.fobj)):
            self.fobj[fno].seek(0)

    def close(self):
        for fno in range(len(self.fobj)):
            self.fobj[fno].close()
