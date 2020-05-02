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
from nltk.corpus import wordnet as ewn

from torch.nn import CrossEntropyLoss, MSELoss
from graph_embeddings import Graph

class FCNet(nn.Module):

    def __init__(self, input_dim,num_labels=2):
        super(FCNet, self).__init__()
        self.input_dim = input_dim
        self.linear_final = nn.Linear(self.input_dim, num_labels)

    def forward(self, input_tensor):

        scores = self.linear_final(input_tensor)

        return scores


def load_dataset(path):
	train_data = np.load(path, allow_pickle=True)
	train_data = train_data[()]
	embeddings = train_data['embeddings']
	labels = train_data['labels']
	sense_keys = train_data['synsets']
	return embeddings, labels, sense_keys

def attach_graph(graph_dict, sense_keys, embeddings):

    for i in range(len(sense_keys)):
        sensekey = sense_keys[i]
        synset = sc2ss(sensekey)
        try:
            index = graph_dict['node_2_idx'][synset]
            vector = graph_dict['embeddings'][index]
        except:
            #sensekey not in graph list
            print("oh no! kill me please")
            vector = np.zeros_like(graph_dict['embeddings'][0])
        #attach graph vector
        embeddings[i] = np.hstack(embeddings[i],vector)
    return embeddings

def write_results(path,dataset,probs):

	pred = np.argmax(probs,axis=1)

	with open(os.path.join(path,dataset+'_results.txt'),'w') as f:
		for i,j,k in zip(pred,probs[:,0],probs[:,1]):
			f.write(str(i)+' '+str(j)+' '+str(k)+'\n')

def sc2ss(sensekey):
    '''Look up a synset given the information from SemCor'''
    ### Assuming it is the same WN version (e.g. 3.0)
    return ewn.lemma_from_key(sensekey).synset()

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--embeddings_data_dir",default=None,type=str,help="The input embedding file")
	parser.add_argument("--dataset",default=None,type=str,help="Dataset name")
	parser.add_argument("--do_train",action='store_true',help="Whether to run training.")
	parser.add_argument("--do_eval",action='store_true',help="Whether to run evaluation")
	parser.add_argument("--out_results_dir",default=None,type=str,help="Output result path")
	parser.add_argument("--load_model_path",default=None,type=str,help="Eval - model to be loaded")
	parser.add_argument("--batch_size",default=32,type=int,help="Total batch size for training.")
	parser.add_argument("--num_epochs",default=5,type=int,help="Number of epochs.")
    parser.add_argument("--graph_embeddings_loc",default=None,type=str,help="The graph embedding file")

	args = parser.parse_args()

	os.makedirs(args.out_results_dir, exist_ok=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	n_gpu = torch.cuda.device_count()

    graph_location = args.graph_embeddings_loc
    graph = Graph('load')
    g_vector = graph.get_embeddings(location =graph_location)
    graph_dict = graph.embedding_to_tuple(g_vector)

	embeddings, labels,sense_keys = load_dataset(args.embeddings_data_dir)
	all_labels = labels
    embeddings = attach_graph(graph_dict,sense_keys,embeddings)
    
	embeddings = torch.tensor(embeddings).float()
	labels = torch.tensor(labels).long()

	data= TensorDataset(embeddings, labels)
	shuffle_bool = not args.do_eval
	dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle_bool)
	num_labels = 2

	best_tr_loss = float('inf')
	if args.do_train:

		output_model_file = os.path.join(args.out_results_dir,"model_save")
		model = FCNet(embeddings.shape[1])
		model.train()
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
		gamma = 0.99
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)

		epoch = 0
		loss_fct = CrossEntropyLoss()
		for epoch_no in trange(int(args.num_epochs), desc="Epoch"):

			epoch += 1
			tr_loss = 0

			for step, batch in enumerate(tqdm(dataloader, desc="Iteration")):

				batch = tuple(t.to(device) for t in batch)
				inputs, labels = batch

				logits = model(inputs)

				loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
				tr_loss += loss
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
			print("Epoch",epoch_no,"Loss",tr_loss)
			best_tr_loss = min(best_tr_loss,tr_loss)
			if(best_tr_loss==tr_loss):
				torch.save(model.state_dict(), output_model_file)
		probs = nn.Softmax(dim=-1)(model(embeddings)).cpu().detach().numpy()
		pred = (probs[:,1]>=0.5).astype(int)
		truth = all_labels.astype(int)
		print(truth.shape)
		print("accuracy",np.sum(pred==truth)*1.0/pred.shape[0])
		write_results(args.out_results_dir,args.dataset,probs)
	if(args.do_eval):
		model = FCNet(embeddings.shape[1])
		model.load_state_dict(torch.load(args.load_model_path))

		model.eval()
		logits = model(embeddings)

		probs = nn.Softmax(dim=-1)(logits).cpu().detach().numpy()
		pred = (probs[:,1]>=0.5).astype(int)
		truth = all_labels.astype(int)

		print("accuracy",np.sum(pred==truth)*1.0/pred.shape[0])
		write_results(args.out_results_dir,args.dataset,probs)

if __name__ == "__main__":
    main()
