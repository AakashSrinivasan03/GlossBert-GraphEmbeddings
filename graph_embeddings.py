import numpy as np
import os
from tqdm import tqdm

class Graph(object):
    """docstring for Graph"""

    def __init__(self, source):
        super(Graph, self).__init__()
        self.source = source

    def build(self):
        '''builds adjacency matrix'''
        pass

    def load(self, location):
        embeddings_index = {}
        f = open(location, 'r', encoding='utf-8')
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('found %s word vectors' % len(embeddings_index))
        return embeddings_index

    def get_embeddings(self, location = None):
        if self.source == 'load':
            embeddings = self.load(location)
        else:
            embeddings = self.build()
        return embeddings

    def embedding_to_tuple(self, dic):
        '''converts embedding dictionary to matrix and returns matrix and indexing'''
        name_to_index = {}
        index_to_name = {}
        mat = np.zeros((len(dic),300))
        i=0
        for key, value in dic.items():
            mat[i] = value
            name_to_index[key] = i
            index_to_name[i] = key
            i+=1

        output = {}
        output['embeddings'] = mat
        output['node_2_idx'] = name_to_index
        output['idx_2_node'] = index_to_name
        return output

