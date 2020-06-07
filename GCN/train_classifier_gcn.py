from __future__ import absolute_import, division, print_function

import argparse
from collections import OrderedDict
import csv
import logging
import os
import random
import sys
import pandas as pd

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from nltk.corpus import wordnet as wn

from torch.nn import CrossEntropyLoss, MSELoss
from graph_embeddings import Graph
from evaluation_script import evaluate_results

import networkx as nx

import torch.sparse as ts

import scipy.sparse as ss


def get_graph():
    G=nx.Graph()

    vertices = []
    edges = []
    dictionary_name_to_id = {}
    dictionary_id_to_name = {}
    noun_synsets = list(wn.all_synsets())
    vertices = [synset for synset in noun_synsets]

    G.add_nodes_from(range(0, len(vertices)))

    for node in G.nodes():
        G.nodes[node]['name'] = vertices[node].name()
        dictionary_name_to_id[vertices[node].name()] = node
        dictionary_id_to_name[node] = vertices[node]

    for node in G.nodes():
        current_vertex_id = node
        current_vertex_name = G.nodes[node]['name']
        word = vertices[node].lemmas()[0].name()
        current_synset = wn.synsets(word)
        for syn in current_synset:
            # if ".n." in synset.name()
            # if synset.name() >= current_vertice_name:
            synset_id = dictionary_name_to_id[syn.name()]
            G.add_edge(current_vertex_id, synset_id)

    # G.add_edges_from(edges)
    nV = len(G.nodes())
    G.add_node(nV) # dummy for unknown synsets
    G.add_edge(nV, nV)
    # nx.write_adjlist(G, "wordnet.adjlist")

    return G


import torch.nn as nn
import torch.nn.functional as F
# from pygcn.layers import GraphConvolution


class GCN_FCNet(nn.Module):

    def __init__(self, input_dim, G, embedding_dim=300, num_labels=2, emb_scale=1.0, w_scale=1e-2, dropout=0.1, activation=F.sigmoid):
        super(GCN_FCNet, self).__init__()
        self.input_dim = input_dim+embedding_dim
        # self.linear_final = nn.Linear(self.input_dim + embedding_dim, num_labels)
        self.linear_final = nn.Sequential(
                nn.Linear(self.input_dim, 500),
                nn.Sigmoid(),
                # nn.Linear(500, 500),
                # nn.Sigmoid(),
                nn.Linear(500, 500),
                nn.Sigmoid(),
                nn.Linear(500, num_labels)
            )
        # self.linear_final = nn.Linear(self.input_dim, num_labels)

        self.G = G
        self.nV = len(G.nodes())
        edgelist = np.array([[u, v] for u,v,c in nx.to_edgelist(G)]).T
        self.nE = edgelist.shape[1]
        A = ss.csr_matrix((np.ones(self.nE), edgelist), shape=(self.nV, self.nV))
       
        # A = ss.coo_matrix(A)
        # A_ix = torch.LongTensor(np.vstack((A.row, A.col)))
        # A_val = torch.FloatTensor(A.data)
        # A = ts.FloatTensor(A_ix, A_val, torch.Size(A.shape))
        # A.requires_grad = False

        D_half_val = np.array(A.sum(axis=1)).flatten()**-0.5
        D_half_ix = np.arange(self.nV).reshape(1,-1).repeat(2, axis=0)
        D_half = ss.csr_matrix((D_half_val, D_half_ix), shape=(self.nV, self.nV))
        LM = ss.coo_matrix(D_half.dot(A).dot(D_half))

        LM_ix = torch.LongTensor(np.vstack((LM.row, LM.col)))
        LM_val = torch.FloatTensor(LM.data)

        self.LM = ts.FloatTensor(LM_ix, LM_val, torch.Size(LM.shape))
        self.LM.requires_grad = False
        
        self.H0 = torch.randn((self.nV, embedding_dim))*emb_scale
        self.W1 = torch.randn((embedding_dim, embedding_dim))*w_scale
        self.W2 = torch.randn((embedding_dim, embedding_dim))*w_scale

        self.H0 = nn.Parameter(self.H0, requires_grad=True)
        self.W1 = nn.Parameter(self.W1, requires_grad=True)
        self.W2 = nn.Parameter(self.W2, requires_grad=True)

        self.cached = False

        # def backward_hook(m, grad_input, grad_output):
        #     m.cached = False

        # self.register_backward_hook(backward_hook)

        self.dropout = dropout
        self.activation = activation

        # self.gcn = GCN(embedding_dim, embedding_dim, embedding_dim, 0.1, A)


    def forward(self, input_tensor, node):

        if not self.cached:
            # H1 = F.dropout(self.activation(torch.spmm(self.LM, torch.mm(self.H0, self.W1)) ), self.dropout, training=self.training)
            # self.H2 = F.dropout(self.activation(torch.spmm(self.LM, torch.mm(H1, self.W2)) ), self.dropout, training=self.training)
            H1 = self.activation(torch.spmm(self.LM, torch.mm(self.H0, self.W1)) )
            self.H2 = self.activation(torch.spmm(self.LM, torch.mm(H1, self.W2)) )
            # self.H2 = ts.spmm(self.LM, torch.mm(H1, self.W1))
            self.cached = True

        # print(gcn_out.shape)
        # print(input_tensor.shape)

        gcn_out = self.H2[node]
        fc_input = torch.cat((gcn_out, input_tensor), 1)
        scores = self.linear_final(fc_input)
        
        # scores = self.linear_final(input_tensor)

        return scores


def load_dataset(path,train):
    train_data = np.load(path, allow_pickle=True)
    if not train:
        train_data = train_data[()]
    embeddings = train_data['embeddings']
    labels = train_data['labels']
    sense_keys = train_data['synsets']
    synsets = [sc2ss(sensekey) for sensekey in sense_keys]
    print('loaded BERT embeddings')
    return embeddings, labels, synsets

# def attach_graph(graph_dict, sense_keys, embeddings):

#   counter = 0
#   concatenated_representation = []
#   #print('report.n.04' in ix_G_lookup)
#   #print('back.n.03' in ix_G_lookup)
#   for i in range(len(sense_keys)):
#       sensekey = sense_keys[i]
#       synset = sc2ss(sensekey)

#       if(synset in ix_G_lookup):
#           index = ix_G_lookup[synset]
#           vector = graph_dict['embeddings'][index]
#       else:
#           #sensekey not in graph list
#           counter += 1

#           vector = np.zeros_like(graph_dict['embeddings'][0])
#       if(i%1e5==0):
#           print(i,"done")
#   #attach graph vector
#       concatenated_representation.append(np.concatenate([embeddings[i],vector],axis=0))
#   print("shape",np.array(concatenated_representation).shape,counter)
#   return np.array(concatenated_representation)

# def get_graph_embeddings(graph_dict,synset_ids):

#   embeddings = []
#   for synset_id in synset_ids:
#       embeddings.append(graph_dict['embeddings'][synset_id])
#   return np.array(embeddings)

def write_results(path,dataset,probs):

    pred = np.argmax(probs,axis=1)

    with open(os.path.join(path,dataset+'_results.txt'),'w') as f:
        for i,j,k in zip(pred,probs[:,0],probs[:,1]):
            f.write(str(i)+' '+str(j)+' '+str(k)+'\n')
    return os.path.join(path,dataset+'_results.txt')

def sc2ss(sensekey):
    '''Look up a synset given the information from SemCor'''
    ### Assuming it is the same WN version (e.g. 3.0)
    # TO DO: Need a better way of extracting string
    synset = str(wn.lemma_from_key(sensekey).synset())[8:-2]
    #print(synset)
    return synset

# def main():
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--embeddings_data_dir",default=None,type=str,help="The input embedding file")
    parser.add_argument("--dataset",default=None,type=str,help="Dataset name")
    parser.add_argument("--do_train",action='store_true',help="Whether to run training.")
    parser.add_argument("--do_eval",action='store_true',help="Whether to run evaluation")
    parser.add_argument("--out_results_dir",default=None,type=str,help="Output result path")
    parser.add_argument("--load_model_path",default=None,type=str,help="Eval - model to be loaded")
    parser.add_argument("--batch_size",default=32,type=int,help="Total batch size for training.")
    parser.add_argument("--num_epochs",default=5,type=int,help="Number of epochs.")
    # parser.add_argument("--graph_embeddings_loc",default=None,type=str,help="The graph embedding file")

    args = parser.parse_args()

    os.makedirs(args.out_results_dir, exist_ok=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    n_gpu = torch.cuda.device_count()
    print("device",device,"n_gpu",n_gpu)
    # graph_location = args.graph_embeddings_loc
    # graph = Graph('load')
    # g_vector = graph.get_embeddings(location =graph_location)
    # graph_dict = graph.embedding_to_tuple(g_vector)

    G = get_graph()
    # model = GCN_FCNet(300, G)
    # model.to(device)
    # model(1, 2)
    nV = len(G.nodes())
    ix_G_lookup = {name:i for i, name in enumerate(G.nodes())}

    embeddings, labels,synsets = load_dataset(args.embeddings_data_dir,args.do_train)

    # assert list(G.nodes()) == ix_G_lookup.keys()

    all_labels = labels
    synset_mapping = torch.tensor([ix_G_lookup[synset] if synset in ix_G_lookup else -1 for synset in synsets]).long()
    
    # graph_embeddings = torch.tensor(np.concatenate([graph_dict['embeddings'],np.mean(graph_dict['embeddings'],axis=0).reshape(1,-1)],axis=0))
    ###embeddings = attach_graph(graph_dict,sense_keys,embeddings)

    embeddings = torch.tensor(embeddings)
    labels = torch.tensor(labels).long()

    data= TensorDataset(embeddings, labels, synset_mapping)
    shuffle_bool = not args.do_eval
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle_bool)
    num_labels = 2



    ####Semeval07 dev set

    dev_embeddings, dev_labels,dev_synsets = load_dataset("embeddings/semeval2007.npy",False)
    dev_embeddings = torch.tensor(dev_embeddings, device=device)
    all_dev_labels = dev_labels
    dev_synset_mapping = torch.tensor([ix_G_lookup.get(synset, nV-1) for synset in dev_synsets]).long()
    # dev_graph_embeddings = torch.tensor(np.concatenate([graph_dict['embeddings'],np.mean(graph_dict['embeddings'],axis=0).reshape(1,-1)],axis=0))
    # dev_embeddings = torch.tensor(dev_embeddings)

    # dev_concatenated_embeddings = torch.cat((dev_embeddings,dev_graph_embeddings[dev_synset_mapping]),axis=1)







    ##########






    best_accuracy = 0
    if args.do_train:

        output_model_file = os.path.join(args.out_results_dir,"model_save")
        model = GCN_FCNet(embeddings.shape[1], G)
        model.to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        gamma = 0.995
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

        epoch = 0
        loss_fct = CrossEntropyLoss()
        for epoch_no in trange(int(args.num_epochs), desc="Epoch"):

            epoch += 1
            tr_loss = 0

            for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):

                batch = tuple(t.to(device) for t in batch)
                bert_embeddings, labels, synsets = batch
                # graph_embedding_lookup = graph_embeddings[synsets.to('cpu')]
                # inputs = torch.cat((bert_embeddings,graph_embedding_lookup.to(device)),1)
                inputs = bert_embeddings
                logits = model(inputs.float(), synsets)

                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
                tr_loss += loss
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()



            print("Epoch",epoch_no,"Loss",tr_loss)
            dev_logits = model(dev_embeddings.float(), dev_synset_mapping)
            dev_prob_values = nn.Softmax(dim=-1)(dev_logits).cpu().detach().numpy()
            result_path = write_results(".",'semeval2007',dev_prob_values)
            accuracy = evaluate_results('semeval2007',result_path)




            best_accuracy = max(best_accuracy,accuracy)
            # if(best_accuracy==accuracy):
            #     print("saving model..")
            #     torch.save(model.state_dict(), output_model_file)


    if(args.do_eval):



        model = GCN_FCNet(embeddings.shape[1], G)
        model.to(device)
        model.load_state_dict(torch.load(args.load_model_path))

        model.eval()

        probs = np.zeros((embeddings.shape[0],num_labels))

        l = 0
        h = 0

        eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):

            batch = tuple(t.to(device) for t in batch)
            bert_embeddings, labels, synsets = batch
            # graph_embedding_lookup = graph_embeddings[synsets.to('cpu')]
            # inputs = torch.cat((bert_embeddings,graph_embedding_lookup.to(device)),1)
            inputs = bert_embeddings
            logits = model(inputs.float(), synsets)
            prob_values = nn.Softmax(dim=-1)(logits).cpu().detach().numpy()
            h = l + prob_values.shape[0]
            probs[l:h] = prob_values
            l = h


        pred = (probs[:,1]>=0.5).astype(int)
        truth = all_labels.astype(int)
        print(truth.shape)
        print("accuracy",np.sum(pred==truth)*1.0/pred.shape[0])


result_path = write_results(args.out_results_dir,args.dataset,probs)
evaluate_results(args.dataset,result_path)









# if __name__ == "__main__":
#     main()
