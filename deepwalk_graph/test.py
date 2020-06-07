from nltk.corpus import wordnet as wn
import networkx as nx
from itertools import islice
import pickle
import subprocess
from create_wordnet_graph_adjlist import fun
from convert_embedding import con

list1 = ['hypernym']
list2 = [1,2,3,4]

for i in list1:
    fun(i)
    print("######adj list created for {} ##########".format(i))
    for j in list2:
        str = "deepwalk --input wordnet_{}.adjlist --output wn_{}_{}.embeddings --representation-size 300 --window-size {}".format(i,i,j,j)
        subprocess.run(str,shell=True)
        print("deepwalk done for {} with window size {}".format(i,j))
        inp = "wn_{}_{}.embeddings".format(i,j)
        out = "wn_{}_{}_final.vec".format(i,j)
        con(inp, out)
        print("embeddings created for {} with window size {}".format(i,j))


'''
subprocess.run("deepwalk --input wordnet.adjlist --output wn.embeddings --representation-size 300 --window-size 2", shell=True)
python create_wordnet_graph_adjlist.py
deepwalk --input wordnet.adjlist --output wn.embeddings --representation-size 300 --window-size 2
python convert_embedding.py wn.embeddings wn_final.vec
'''
