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
from evaluation_script import evaluate_results

class FCNet(nn.Module):

	def __init__(self, input_dim1,input_dim2=None,num_labels=2):
		super(FCNet, self).__init__()
		self.input_dim = input_dim1
		if(input_dim2!=None):
			self.input_dim += input_dim2
		self.linear_final = nn.Linear(self.input_dim, num_labels)

	def forward(self, input_tensor1,input_tensor2=None):

		if(input_tensor2!=None):
			scores = self.linear_final(torch.cat((input_tensor1,input_tensor2),1))
		else:
			scores = self.linear_final(input_tensor1)

		return scores

class Projection(nn.Module):

	def __init__(self, input_dim1,input_dim2,projected_dim=300,num_labels=2,method='concat'):
		super(Projection, self).__init__()

		self.proj1 = nn.Linear(input_dim1, projected_dim)
		self.proj2 = nn.Linear(input_dim2, projected_dim)
		self.method = method
		if(self.method=='concat'):
			self.linear_final = nn.Linear(2*projected_dim, num_labels)
		elif(self.method=='hadamard'):
			self.linear_final = nn.Linear(projected_dim, num_labels)
	def forward(self, input_tensor1,input_tensor2):

		proj1 = nn.ReLU()(self.proj1(input_tensor1))
		proj2 = nn.ReLU()(self.proj2(input_tensor2))
		if(self.method=='concat'):
			features = torch.cat((proj1,proj2),1)
		else:
			features = proj1*proj2
		scores = self.linear_final(features)

		return scores


def load_dataset(path):
	train_data = np.load(path, allow_pickle=True)
	########if(not train):
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
	return os.path.join(path,dataset+'_results.txt')

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
	parser.add_argument("--out_results_dir",default=None,type=str,help="Output result path")
	parser.add_argument("--load_model_path1",default=None,type=str,help="Eval - model 1 to be loaded")
	parser.add_argument("--load_model_path2",default=None,type=str,help="Eval - model 2 to be loaded")
	parser.add_argument("--batch_size",default=512,type=int,help="Total batch size for training.")
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
	
	shuffle_bool = False
	dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle_bool)
	num_labels = 2


	
	model1 = FCNet(embeddings.shape[1])
	model2 = Projection(embeddings.shape[1],graph_embeddings.shape[1])
	
	model1.to(device)
	model1.load_state_dict(torch.load(args.load_model_path1))
	model2.load_state_dict(torch.load(args.load_model_path2))
	model1.eval()
	model2.eval()

	probs1 = np.zeros((embeddings.shape[0],num_labels))
	probs2 = np.zeros((embeddings.shape[0],num_labels))
	l = 0
	h = 0

	eval_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)


	for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):

		batch = tuple(t.to(device) for t in batch)
		

		bert_embeddings, labels, synsets = batch
		graph_embedding_lookup = graph_embeddings[synsets.to('cpu')]
		logits1 = model1(bert_embeddings.float())
		logits2 = model2(bert_embeddings.float(),graph_embedding_lookup.to(device).float())
		
		prob_values1 = nn.Softmax(dim=-1)(logits1).cpu().detach().numpy()
		h = l + prob_values1.shape[0]
		probs1[l:h] = prob_values1
		prob_values2 = nn.Softmax(dim=-1)(logits2).cpu().detach().numpy()
		h = l + prob_values2.shape[0]
		probs2[l:h] = prob_values2
		l = h


	#pred1 = (probs1[:,1]>=0.5).astype(int)
	#pred2 = (probs2[:,1]>=0.5).astype(int)
	
	idx = list(np.where(np.abs(probs1[:,1]-probs2[:,1])>0.25)[0])
	
	test_file_name = './Evaluation_Datasets/'+args.dataset+'/'+args.dataset+'_test_sent_cls.csv'
	test_data = pd.read_csv(test_file_name,sep="\t",na_filter=False)
	print(test_data['sentence'])
	sentence = list(np.array(test_data['sentence'])[idx])
	gloss = list(np.array(test_data['gloss'])[idx])
	label = list(np.array(test_data['label'])[idx])

	f = open("dump_text.csv",'w')
	d = {'sentence:':sentence,'gloss':gloss,'label':label,'prob1':probs1[:,1][idx],'prob2':probs2[:,1][idx]}

	df = pd.DataFrame.from_dict(d)
	df.to_csv(f)
	#ct = 0
	#for s,g,l in zip(sentence,gloss,label):
	#	print(s,g,l,probs1[ct],probs2[ct])
	#	ct += 1





	#truth = all_labels.astype(int)
	#print(truth.shape)
	#print("accuracy",np.sum(pred1==truth)*1.0/pred1.shape[0])
	#print("accuracy",np.sum(pred2==truth)*1.0/pred2.shape[0])

	#result_path = write_results(args.out_results_dir,args.dataset,probs)
	#evaluate_results(args.dataset,result_path)

if __name__ == "__main__":
    main()
