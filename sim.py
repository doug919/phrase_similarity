
import sys
import os
import argparse
import numpy as np
import math
import logging

from ppdb_avgcnn_hinge_model import ppdb_avgcnn_hinge_model

embedding_file = 'phrase_models/paragram-phrase-XXL.txt'
model_file = 'phrase_models/model_avgcnn.pkl'

def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Measure similarity scores between phrases')
    
    parser.add_argument('input_file', metavar='INPUT_FILE',
                            help='input phrases.')
    parser.add_argument('model_type', metavar='MODEL_TYPE', choices=['CNN', 'WAVG'],
                            help='CNN or WAVG (word average)')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                            help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                            help='show debug messages')

    args = parser.parse_args(argv)
    return args

def get_word_map(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    dim = None
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        if dim is None:
            dim = len(i) - 1
        else:
            assert dim == len(i) - 1

        while j < len(i):
            v.append(float(i[j]))
            j += 1
        # v = np.random.normal(0, 0.01, len(i)-1)
        words[i[0]]=n
        We.append(v)
    # add unknown words if it's necessary
    n_words = len(We)
    if 'UUUNKKK' not in words:
        words['UUUNKKK'] = n_words
        n_words += 1
        We.append(np.random.uniform(low=-0.5/dim, high=0.5/dim, size=dim).tolist())
    return (words, np.array(We, dtype=np.float32))

def phrase_embedding(sp, words, embeddings):
    dim = embeddings.shape[1]
    emb = np.zeros(dim, dtype=np.float32)
    for w in sp:
        if w in words:
            idx = words[w]
        else:
            idx = words['UUUNKKK']
        emb += embeddings[idx]
    return emb

def cosine_similarity(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def word_average_score(s1, s2, words, embeddings):
    s1sp = s1.split(' ')
    s2sp = s2.split(' ')
    s1emb = phrase_embedding(s1sp, words, embeddings)
    s2emb = phrase_embedding(s2sp, words, embeddings)
    return cosine_similarity(s1emb, s2emb)

def prepare_inputs(sent, words):
    toks = sent.split(' ')
    seqs = []
    for tok in toks:
        if tok in words:
            seqs.append(words[tok])
        else:
            seqs.append(words['UUUNKKK'])

    x = np.zeros((1, len(seqs))).astype('int32')
    x_mask = np.zeros((1, len(seqs))).astype(np.float32)
    x[0] = seqs
    for j in range(len(seqs)):
        x_mask[0, j] = 1.0
    return x, x_mask

def cnn_score(s1, s2, words, model):
    s1x, s1mask = prepare_inputs(s1, words)
    s2x, s2mask = prepare_inputs(s2, words)
    s1emb = model.feedforward_function(s1x, s1mask)[0]
    s2emb = model.feedforward_function(s2x, s2mask)[0]
    return cosine_similarity(s1emb, s2emb)

if __name__ == '__main__':
    # get arguments
    args = get_arguments(sys.argv[1:])

    # set debug level
    if args.debug:
        loglevel = logging.DEBUG
    elif args.verbose:
        loglevel = logging.INFO
    else:
        loglevel = logging.WARNING
    logging.basicConfig(format='[%(levelname)s][%(name)s] %(message)s', level=loglevel)

    assert os.path.isfile(args.input_file)

    (words, embeddings) = get_word_map(embedding_file)
    if args.model_type == 'WAVG':
        pass
    elif args.model_type == 'CNN':
        model = ppdb_avgcnn_hinge_model(embeddings, regfile=model_file)
    else:
        raise ValueError('error model file.')

    with open(args.input_file, 'r') as fr:
        ln = -1
        for line in fr:
            ln += 1
            line = line.rstrip('\n')
            sp = line.split('\t')
            if len(sp) != 2:
                logging.warning('skip line {}; potential format error: {}'.format(ln, line))
                continue
            s1, s2 = sp[0], sp[1]
            if args.model_type == 'WAVG':
                print(word_average_score(s1, s2, words, embeddings))
            elif args.model_type == 'CNN':
                print(cnn_score(s1, s2, words, model))
            else:
                raise ValueError('unsupported model type.')



