#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
    *.py: Description of what * does.
    Last Modified:
"""

__author__ = "Sathappan Muthiah"
__email__ = "sathap1@vt.edu"
__version__ = "0.0.1"
from utils.dataloader import GODAG
from Bio import SeqIO
import gzip
import numpy as np
from utils import numpy_calc_performance_metrics
from utils.dataloader import FUNC_DICT
import pandas as pd
import os
import json
import ipdb

def read_batches(infile, functype, funcs=None, batchsize=32, order=[]):
    batchtrue, batchpred = [], []
    with gzip.open(infile, 'rt') as inf:
        for ln in SeqIO.parse(inf, 'fasta'):
            #print("reading")
            msg = json.loads(ln.description.split(' ', 1)[1])
            truelabels = [i['go_id'] for i in msg['go_ids']
                          if i['aspect'].lower() == functype[-1].lower()]

            truelabels = GODAG.to_npy(truelabels)
            predictions = np.array(msg['prediction'])[order] #GODAG.to_npy(msg['prediction'])
            #print('read labels')
            batchtrue.append(truelabels)
            batchpred.append(predictions)
            if len(batchtrue) == batchsize:
                #print("here")
                if funcs is None:
                    preds = np.concatenate([np.vstack(batchpred), np.zeros((len(batchpred),
                                            batchtrue[0].shape[0] - len(batchpred[0])))], axis=1)
                else:
                    batchtrue = np.vstack(batchtrue)[:, :len(funcs)]
                    mask = batchtrue.any(axis=1)
                    preds = np.vstack(batchpred)[mask, :]
                    batchtrue = batchtrue[mask, :]
                    #ipdb.set_trace()
                yield (np.vstack(batchtrue), preds)
                batchpred, batchtrue = [], []

    if batchtrue:
        preds = np.concatenate([np.vstack(batchpred), np.zeros((len(batchpred),
                                batchtrue[0].shape[0] - len(batchpred[0])))], axis=1)
        return (np.vstack(batchtrue), preds)

THRESHOLD_RANGE=np.arange(0.05, 0.5, 0.05)

def printnodes(ids):
    print([GODAG.id2node(i) for i  in ids])
    return

def main(FLAGS):
    origfuncs = pd.read_pickle(os.path.join(FLAGS.resources, '{}.pkl'.format(FLAGS.function)))['functions'].values
    revmap = dict(zip([GODAG.get(node) for node in origfuncs], origfuncs))
    funcs = GODAG.initialize_idmap(origfuncs, FLAGS.function)
    print(len(funcs))
    #ipdb.set_trace()
    new_order = [np.where(origfuncs == revmap.get(node, node))[0][0] for node in funcs]
    #ipdb.set_trace()
    avgprec, avgrecall, avgf1, step = np.zeros_like(THRESHOLD_RANGE), np.zeros_like(THRESHOLD_RANGE), np.zeros_like(THRESHOLD_RANGE), 0
    for x, y in read_batches(FLAGS.infile, FLAGS.function, funcs=funcs, batchsize=128, order=new_order):
        prec, recall, f1 = [], [], []
        for thres in THRESHOLD_RANGE:
            p, r, f = numpy_calc_performance_metrics(x, y, thres)
            #ipdb.set_trace()
            prec.append(p)
            recall.append(r)
            f1.append(f)

        avgprec += prec
        avgrecall += recall
        avgf1 += f1
        step += 1.0
        if step % 1000 == 0:
            print('recall-{}, prec-{}, f1-{}'.format(np.round(avgprec/step, 2),
                                                     np.round(avgrecall/step, 2),
                                                     np.round(avgf1/step, 2)))
            #break

    print(THRESHOLD_RANGE)
    print('recall-{}, prec-{}, f1-{}'.format(np.round(avgprec/step, 2),
                                             np.round(avgrecall/step, 2),
                                             np.round(avgf1/step, 2)))



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, help='inputfile')
    ap.add_argument("--function", type=str)
    ap.add_argument("--resources", type=str)
    args = ap.parse_args()
    main(args)
