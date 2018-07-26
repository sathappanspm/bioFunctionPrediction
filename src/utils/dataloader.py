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
import ipdb

random.seed(1)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
log = logging.getLogger('root.DataLoader')

DATADIR = os.path.join(os.path.dirname(__file__), '../../resources/')

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {'cc': CELLULAR_COMPONENT,
                'mf': MOLECULAR_FUNCTION,
                'bp': BIOLOGICAL_PROCESS}


if nx.__version__ == '2.0':
    filterbyEdgeType = lambda etypes, net: net.edges(keys=etypes)
else:
    filterbyEdgeType = lambda etypes, net: [(e[0], e[1]) for e in net.edges(keys=True) if e[2] in etypes]


def load_labelembedding(path, goids):
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format(path)
    modelroot = model.wv if hasattr(model, 'wv') else model
    # try:
    model_worddict = modelroot.index2word
    # except AttributeError:
    #     model_worddict = model.index2word

    ## reorder the embeddings to follow the same order as goids list
    # neworder = [model.wv.vocab[goid].index for goid in goids]
    neworder = [modelroot.vocab[goid].index for goid in goids]
    # reorderedmat = model.wv.syn0[neworder, :]
    reorderedmat = modelroot.syn0[neworder, :]
    return reorderedmat #(np.vstack([np.zeros(reorderedmat.shape[1]), reorderedmat])).astype(np.float32)


def load_pretrained_embedding(path):
    import pandas as pd
    df = pd.read_csv(path, sep='\t', index_col=0)
    npmat = df.as_matrix().astype(np.float32)
    indexmap = {s: (i + 1) for i, s in enumerate(df.index)}
    npmat = np.concatenate([np.zeros((1, npmat.shape[1]), dtype=np.float32), npmat], axis=0)
    return npmat, indexmap


def has_path(a, b, go_dag=None):
    """
    return True if the GODAG has a path from a to b.
    """
    try:
        if nx.has_path(go_dag, a, b):
            return True
    except Exception as e:
        pass
        # log.exception(e)

    return False


class GODAG(object):
    net = obonet.read_obo(os.path.join(DATADIR, 'go.obo'))
    obsolete = {}
    if len(net) == 2:
        net, obsolete = net
        obsolete = {item['id']: item.get('replaced_by', '') for item in obsolete}

    isagraph = nx.DiGraph(filterbyEdgeType(['is_a', 'part_of'], net)).reverse()
    vfunc = np.vectorize(partial(has_path, go_dag=isagraph), otypes=[np.bool_])
    alt_id = {val: key for key, vals in nx.get_node_attributes(net, "alt_id").items() for val in vals}
    idmap = None
    GOIDS = None

    @staticmethod
    def initialize_idmap(idlist, root):
        allnodes = set(GODAG.isagraph.nodes())
        if root:
            root = FUNC_DICT[root]
            allnodes = set(nx.descendants(GODAG.isagraph, root))

        if idlist is not None :
            log.info('loading go funcs of len-{}'.format(len(idlist)))
            # and remove the obsolete go terms from this list
            GODAG.GOIDS = list(set([GODAG.get(id) for id in idlist]).difference(GODAG.obsolete.keys()))
            updatedIdlist = (GODAG.GOIDS).copy()
            GODAG.idmap = {id: index for index, id in enumerate(GODAG.GOIDS)}
            goset = allnodes - GODAG.idmap.keys()
        else:
            with open(os.path.join(DATADIR, 'funcCounts.json')) as inpf:
                funcweights = json.load(inpf)
                obsolete = set(funcweights.keys()) - allnodes
                for i in obsolete:
                    val = funcweights.pop(i)
                    if i in GODAG.alt_id:
                        altid = GODAG.get(i)
                        funcweights[altid] = funcweights.get(altid, 0) + val

            tmp = sorted(funcweights.items(), reverse=True, key=lambda x: x[1])
            # index 0 is reserved for STOPGO
            GODAG.idmap = {GODAG.get(item[0]): index for index, item in enumerate(tmp)}
            goset = allnodes - funcweights.keys()
            GODAG.GOIDS = [item[0] for item in tmp]
            updatedIdlist = (GODAG.GOIDS).copy()
            log.info('len here is {}'.format(len(GODAG.GOIDS)))

        GODAG.GOIDS += list(goset)
        for id in goset:
            GODAG.idmap[GODAG.get(id)] = len(GODAG.idmap)

        log.info('GO data loaded. Total nodes -{}'.format(len(GODAG.idmap)))
        return updatedIdlist

    @staticmethod
    def get_leafnodes(labels):
        """
        function to get only unique paths in GO DAG (children and its ancestors must not be in the list at the same time)
        """
        ulabels = []
        unravel = False
        if isinstance(labels[0], str):
            labels = [labels]
            unravel = True

        for seq in labels:
            uniqset = set(seq)
            for i in seq:
                ## remove any ancestors present in label set
                if i not in GODAG.idmap:
                    ## ignore if node is not in DAG. could be obsolete function
                    try:
                        uniqset.remove(i)
                    except:
                         pass
                        #log.info('leaf node error-{}, {}'.format(str(labels), str(uniqset)))
                    continue

                uniqset.difference_update(nx.ancestors(GODAG.isagraph, i))

            ulabels.append(list(uniqset))

        if unravel is True:
            ulabels = ulabels[0]

        return ulabels

    @staticmethod
    def get_ancestors(node):
        if node == 'STOPGO':
            return []

        return nx.ancestors(GODAG.isagraph, GODAG.get(node))

    @staticmethod
    def get_children(node):
        if node == 'STOPGO':
            return []
        return nx.descendants(GODAG.isagraph, GODAG.get(node))

    @staticmethod
    def get_id(node):
        if node == 'STOPGO':
            return -1

        try:
            return GODAG.idmap[GODAG.get(node)]
        except KeyError as e:
            log.info('unable to find key -{}'.format(str(e)))

        return -1

    @staticmethod
    def id2node(id):
        if id == -1:
            return 'STOPGO'

        return GODAG.GOIDS[id]

    @staticmethod
    def get(node):
        if node == 'STOPGO':
            return node

        return GODAG.alt_id.get(node, node)

    @staticmethod
    def to_npy(funcs):
        mat = np.zeros(len(GODAG.idmap), dtype=np.int32)
        indices = [GODAG.get_id(node) for node in funcs]
        mat[[i for i in indices if i!= -1]] = 1
        return mat

    @staticmethod
    def get_fullmat(funcids):
        funclabels = [GODAG.id2node(i) for i in funcids]
        return np.any(GODAG.vfunc(np.array(GODAG.GOIDS)[:, np.newaxis], funclabels), axis=0)



def get_fullhierarchy(node, godag=None):
    mat = np.zeros(len(godag.GOIDS), dtype=np.bool_)
    mat[[GODAG.get_id(parent) for parent in GODAG.get_ancestors(GODAG.id2node(node))]] = True
    return mat

vectorized_getlabelmat = np.vectorize(partial(get_fullhierarchy, godag=GODAG), otypes=[np.bool_], signature='()->(m)')


class FeatureExtractor():
    ngrammap = dict()
    aminoacidmap = dict()

    @staticmethod
    def to_ngrams(seq, ngram=3):
        return [FeatureExtractor._ngram2id(seq[i: i + ngram]) for i in range(len(seq) -ngram + 1)]

    @staticmethod
    def _ngram2id(ngram):
        if ngram not in FeatureExtractor.ngrammap:
            log.info('unable to find {} in known ngrams'.format(ngram))
            return FeatureExtractor.ngrammap['<unk>']
            #FeatureExtractor.ngrammap[ngram] = len(FeatureExtractor.ngrammap) + 1

        return FeatureExtractor.ngrammap[ngram]

    @staticmethod
    def to_onehot(seq):
        return [FeatureExtractor._aminoacid2id(aacid) for aacid in seq]

    @staticmethod
    def _aminoacid2id(aacid):
        if aacid not in FeatureExtractor.aminoacidmap:
            #FeatureExtractor.aminoacidmap[aacid] = len(FeatureExtractor.aminoacidmap) + 1
            log.info('unable to find {} in known aminoacids'.format(aacid))
            aacid = '<unk>'

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
            FeatureExtractor.aminoacidmap['<unk>'] = len(FeatureExtractor.aminoacidmap)
            log.info('loaded amino acid map of size-{}'.format(len(FeatureExtractor.aminoacidmap)))

        with open(os.path.join(DATADIR, 'ngrams.txt'), 'r') as inpf:
            ngrams = json.load(inpf)
            FeatureExtractor.ngrammap = {key: (index + 1) for index, key in enumerate(ngrams)}
            FeatureExtractor.ngrammap['<unk>'] = len(FeatureExtractor.ngrammap)
            log.info('loaded ngram map of size-{}'.format(len(FeatureExtractor.ngrammap)))


class DataLoader(object):
    def __init__(self, filename='/groups/fungcat/datasets/current/fasta/AllSeqsWithGO_expanded.tar'):
    # def __init__(self, filename='/home/sathap1/workspace/bioFunctionPrediction/AllSeqsWithGO_expanded.tar'):
        self.dir = os.path.isdir(filename)
        if self.dir:
            self.dataobj = filename
            self.members = glob.glob(os.path.join(filename, '*'))
        else:
            if filename.endswith('.tar') or filename.endswith('.tar.gz'):
                self.dataobj = tarfile.open(filename)
                self.members = self.dataobj.getmembers()
            else:
                self.dataobj = None #gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename)
                self.members = [filename]
                self.dir = True

        self.openfiles = set()

    def getmember(self, member):
        """
        return the gzip file obj of member
        """
        membername = member.name if self.dir is False else member
        log.info('opening file - {}'.format(membername))
        if membername in self.openfiles:
            # randint includes both end-points
            return self.getmember(self.members[random.randint(0, len(self.members) - 1)])

        self.openfiles.add(membername)
        if self.dir:
            fobj = gzip.open(member, 'rt') if membername.endswith('.gz') else open(member)
        else:
            fobj = gzip.open(self.dataobj.extractfile(member),
                              mode='rt')

        return (membername, fobj)

    def getrandom(self):
        if len(self.openfiles) == len(self.members):
            return (None, None)

        return self.getmember(self.members[random.randint(0, len(self.members))])

    def close(self):
        if not self.dir:
            self.dataobj.close()


class DataIterator(object):
    def __init__(self, batchsize=1,
                 functype='', size=100, seqlen=2000,
                 featuretype='onehot', dataloader=None,
                 numfiles=1, ngramsize=3, all_labels=True,
                 numfuncs=0, onlyLeafNodes=False, autoreset=False,
                 test=False,
                 **kwargs):
        self.fobj = []
        self.fnames = []
        self.current = 0
        self.numfiles = numfiles
        self.batchsize = batchsize
        self.functype = functype
        self.maxdatasize = size
        self.maxseqlen = seqlen
        self.itersize = 0
        self.featuretype = featuretype
        self.loader = dataloader
        self.ngramsize = ngramsize
        self.all_labels = all_labels
        self.numfuncs = numfuncs
        self.stopiter = False
        self.onlyLeafNodes = onlyLeafNodes
        self.autoreset = autoreset
        self.test = test
        log.info('only leaf nodes will be used as labels - {}'.format(onlyLeafNodes))
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

        if (self.itersize >= self.maxdatasize) or self.stopiter is True:
            if self.autoreset is True:
                self.reset()
            else:
                raise StopIteration

        inputs, labels = [], []
        for fastaObj in SeqIO.parse(self.fobj[self.current], 'fasta'):
            seq = str(fastaObj.seq)
            if len(seq) > self.maxseqlen:
                continue

            inputs.append(self.featureExt(seq))
            if self.test is True:
                labels.append(fastaObj)
            else:
                try:
                    desc = json.loads(fastaObj.description.split(' ', 1)[1])
                    funcs = [GODAG.get(func['go_id']) for func in desc['go_ids']
                             if((func.get('qualifier', '') != 'NOT') and
                             ((self.functype == '') or
                             (self.functype[-1].lower() == func['aspect'].lower()))
                             )]

                    if self.onlyLeafNodes is True:
                        funcs = GODAG.get_leafnodes(funcs)
                        #if self.limit is not None:
                            # only predict limit number of functions
                            #funcs = funcs[:self.limit]
                        ids = [GODAG.get_id(fn) for fn in funcs]
                        labels.append([i for i  in ids  if i != -1])
                    else:
                        labels.append(GODAG.to_npy(funcs))

                except Exception as e:
                    # ipdb.set_trace()
                    log.exception('error in loader - {}'.format(str(e)))
                    continue

            if len(inputs) >= self.batchsize:
                self.itersize += 1
                inputs, labels = self._format(inputs, labels)
                # log.info('sending batch, with labels size-{}'.format(str(labels.shape)))
                return inputs, labels

        if self.itersize >= self.maxdatasize:
            if self.autoreset is True:
                self.reset()
            else:
                raise StopIteration

        self.current = (self.current + 1) % (self.numfiles)
        # if self.numfiles > 1:
        if self.current > len(self.fnames):
            self.loadfile()
        else:
            # Since all files are done processing stop iteration
            # so set stopiter flag as all lines of a file have been read
            self.stopiter = True

        if inputs:
            self.itersize += 1
            inputs, labels = self._format(inputs, labels)
        elif not self.stopiter:
            inputs, labels = self.__next__()
        else:
            raise StopIteration

        return inputs, labels

    def _format(self, inputs, labels):
        inputs = pd.DataFrame(inputs, dtype=np.int32).fillna(0)
        if self.test is False and isinstance(labels[0], list):
            labels = pd.DataFrame(labels, dtype=np.int32).fillna(0, downcast='infer').as_matrix()
            if labels.shape[1] < self.numfuncs:
                diff = self.numfuncs - labels.shape[1]
                labels = np.hstack([labels, np.zeros((labels.shape[0], diff), dtype=np.int32)])
            #log.info('percentage of zeros in labelspace-{}/160'.format((labels==0).sum()))
        else:
            labels = np.vstack(labels) if self.test is False else labels

        # log.info('{}'.format(str(inputs.shape)))
        if inputs.shape[1] < self.expectedshape:
            inputs = np.concatenate([inputs.as_matrix(),
                                      np.zeros((inputs.shape[0],
                                                self.expectedshape - inputs.shape[1]))], axis=1)
        # log.info('batch shape is {}-{}'.format(inputs.shape, labels.shape))
        # log.info('max id is {}'.format(np.max(inputs)))
        # if labels.shape[1] < 10:
        # ipdb.set_trace()

        if self.test is False and self.all_labels is False:
            labels = labels[:, :self.numfuncs]

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

        self.itersize = 0
        self.current = 0
        self.stopiter = False

    def close(self):
        for fno in range(len(self.fobj)):
            self.fobj[fno].close()
