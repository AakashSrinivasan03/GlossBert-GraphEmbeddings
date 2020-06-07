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
	synsets = [sc2ss(sensekey) for sensekey in sense_keys]
	print('loaded BERT embeddings')
	return embeddings, labels, synsets

def attach_graph(graph_dict, sense_keys, embeddings):

	counter = 0
	concatenated_representation = []
	#print('report.n.04' in graph_dict['node_2_idx'])
	#print('back.n.03' in graph_dict['node_2_idx'])
	for i in range(len(sense_keys)):
		sensekey = sense_keys[i]
		synset = sc2ss(sensekey)

		if(synset in graph_dict['node_2_idx']):
			index = graph_dict['node_2_idx'][synset]
			vector = graph_dict['embeddings'][index]
		else:
			#sensekey not in graph list
			counter += 1

			vector = np.zeros_like(graph_dict['embeddings'][0])
		if(i%1e5==0):
			print(i,"done")
	#attach graph vector
		concatenated_representation.append(np.concatenate([embeddings[i],vector],axis=0))
	print("shape",np.array(concatenated_representation).shape,counter)
	return np.array(concatenated_representation)

def get_graph_embeddings(graph_dict,synset_ids):

	embeddings = []
	for synset_id in synset_ids:
		embeddings.append(graph_dict['embeddings'][synset_id])
	return np.array(embeddings)

def write_results(path,dataset,probs):

	pred = np.argmax(probs,axis=1)

	with open(os.path.join(path,dataset+'_results.txt'),'w') as f:
		for i,j,k in zip(pred,probs[:,0],probs[:,1]):
			f.write(str(i)+' '+str(j)+' '+str(k)+'\n')

def sc2ss(sensekey):
    '''Look up a synset given the information from SemCor'''
    ### Assuming it is the same WN version (e.g. 3.0)
    # TO DO: Need a better way of extracting string
    synset = str(ewn.lemma_from_key(sensekey).synset())[8:-2]
    #print(synset)
    return synset

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
	print("device",device,"n_gpu",n_gpu)
	graph_location = args.graph_embeddings_loc
	graph = Graph('load')
	g_vector = graph.get_embeddings(location =graph_location)
	graph_dict = graph.embedding_to_tuple(g_vector)

	embeddings, labels,synsets = load_dataset(args.embeddings_data_dir)

	all_labels = labels
	synset_mapping = torch.tensor([graph_dict['node_2_idx'][synset] if synset in graph_dict['node_2_idx'] else -1 for synset in synsets]).long()
	
	graph_embeddings = torch.tensor(np.concatenate([graph_dict['embeddings'],np.mean(graph_dict['embeddings'],axis=0).reshape(1,-1)],axis=0))
	###embeddings = attach_graph(graph_dict,sense_keys,embeddings)

	embeddings = torch.tensor(embeddings)
	labels = torch.tensor(labels).long()

	data= TensorDataset(embeddings, labels, synset_mapping)
	shuffle_bool = not args.do_eval
	dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle_bool)
	num_labels = 2

	best_tr_loss = float('inf')
	if args.do_train:

		output_model_file = os.path.join(args.out_results_dir,"model_save")
		model = FCNet(embeddings.shape[1]+graph_embeddings.shape[1])
		model.to(device)
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
				bert_embeddings, labels, synsets = batch
				graph_embedding_lookup = graph_embeddings[synsets.to('cpu')]
				inputs = torch.cat((bert_embeddings,graph_embedding_lookup.to(device)),1)

				logits = model(inputs.float())

				loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
				tr_loss += loss
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
			print("Epoch",epoch_no,"Loss",tr_loss)
			best_tr_loss = min(best_tr_loss,tr_loss)
			if(best_tr_loss==tr_loss):
				torch.save(model.state_dict(), output_model_file)


		model = FCNet(embeddings.shape[1]+graph_embeddings.shape[1])
		model.to(device)
		model.load_state_dict(torch.load(output_model_file))
		model.eval()
		probs = np.zeros((embeddings.shape[0],num_labels))

		l = 0
		h = 0

		eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
		for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):

			batch = tuple(t.to(device) for t in batch)
			bert_embeddings, labels, synsets = batch
			graph_embedding_lookup = graph_embeddings[synsets.to('cpu')]
			inputs = torch.cat((bert_embeddings,graph_embedding_lookup.to(device)),1)

			logits = model(inputs.float())
			prob_values = nn.Softmax(dim=-1)(logits).cpu().detach().numpy()
			h = l + prob_values.shape[0]
			probs[l:h] = prob_values
			l = h


		pred = (probs[:,1]>=0.5).astype(int)
		truth = all_labels.astype(int)
		print(truth.shape)
		print("accuracy",np.sum(pred==truth)*1.0/pred.shape[0])
		write_results(args.out_results_dir,args.dataset,probs)
	if(args.do_eval):
		model = FCNet(embeddings.shape[1]+graph_embeddings.shape[1])
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
			graph_embedding_lookup = graph_embeddings[synsets.to('cpu')]
			inputs = torch.cat((bert_embeddings,graph_embedding_lookup.to(device)),1)

			logits = model(inputs.float())
			prob_values = nn.Softmax(dim=-1)(logits).cpu().detach().numpy()
			h = l + prob_values.shape[0]
			probs[l:h] = prob_values
			l = h


		pred = (probs[:,1]>=0.5).astype(int)
		truth = all_labels.astype(int)
		print(truth.shape)
		print("accuracy",np.sum(pred==truth)*1.0/pred.shape[0])
		write_results(args.out_results_dir,args.dataset,probs)

if __name__ == "__main__":
    main()

