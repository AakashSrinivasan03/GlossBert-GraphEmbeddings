import pickle
import pandas as pd
import nltk
import re
from nltk.corpus import wordnet as ewn
import numpy as np

def load_dataset(path,train):
	train_data = np.load(path, allow_pickle=True)
	########if(not train):
	#train_data = train_data[()]
	embeddings = train_data['embeddings']
	labels = train_data['labels']
	sense_keys = train_data['synsets']
	synsets = [sc2ss(sensekey) for sensekey in sense_keys]
	print('loaded BERT embeddings')
	return embeddings, labels, synsets

def sc2ss(sensekey):
	'''Look up a synset given the information from SemCor'''
	### Assuming it is the same WN version (e.g. 3.0)
	# TO DO: Need a better way of extracting string
	synset = str(ewn.lemma_from_key(sensekey).synset())[8:-2]
	#print(synset)
	return synset

count = 0
def get_neg_sampling(data_loc,loc,save_loc):
    print(data_loc)
    print(loc)
    embeddings, labels, synsets = load_dataset(data_loc,True)

    df = pd.read_csv(loc,sep='\t')

    def get_key(sent):
        return sent.split()[0]

    df['key'] = df['gloss'].apply(get_key)
    print('keys done')
    def sc2ss(sensekey):
        '''Look up a synset given the information from SemCor'''
        ### Assuming it is the same WN version (e.g. 3.0)
        # TO DO: Need a better way of extracting string
        synset = str(ewn.lemma_from_key(sensekey).synset())[8:-2]
        #print(synset)
        return synset

    def get_wordnet_pos(treebank_tag):

        if treebank_tag.startswith('J'):
            return 's'
        elif treebank_tag.startswith('V'):
            return 'v'
        elif treebank_tag.startswith('N'):
            return 'n'
        elif treebank_tag.startswith('R'):
            return 'r'
        else:
            return None

    def sensekey_2_syn(x):
        syn = sc2ss(x).split('.')[1]
        return syn

    df['syn'] = df['sense_key'].apply(sensekey_2_syn)
    print('got syn')
    def get_tag(x):
        sent = x['sentence']
        #key = x['gloss'].split()[0]
        key = x['key']
        #sense = x['sense_key']
        global count
        count+=1
        if(count%2000==0):
            print('We are at line ',count)
        #syn = sc2ss(sense).split('.')[1]
        syn = x['syn']
        #sent is a single sentence
        tokens = nltk.word_tokenize(sent)
        tokens = [t for t in tokens if not re.search(r'[^\w\d\s]',t)]
        tags = nltk.pos_tag(tokens)
        for i in range(len(tokens)):
            if tokens[i]==key:
                val = get_wordnet_pos(tags[i][1])
                if val==syn:
                    return 1
                else:
                    return 0
        return 0
    print('done')
    df['pos'] = df.apply(get_tag,axis=1)
    out = df['pos'].to_numpy()
    #print(df['pos'].head())
    #print(df['pos'].sum())
    #np.save('mask_train_pos.npy',out)
    embeddings = embeddings[out==1]
    labels = labels[out==1]
    synsets = np.array(synsets)[out==1]
    dataset = {}
    dataset['embeddings'] = embeddings
    dataset['labels'] = labels
    dataset['synsets'] = synsets
    with open(save_loc, 'wb') as handle:
    	pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset
	
import argparse
if __name__ =='__main__':
   		
   parser = argparse.ArgumentParser()
   parser.add_argument("--embeddings_loc",default=None,type=str,help="Location to embeddings of numpy")
   parser.add_argument("--csv_loc",default=None,type=str,help="Location to the csv")
   parser.add_argument("--save_location",default=None,type=str,help="Location for the final dataset")
   args = parser.parse_args()
   d =  get_neg_sampling(data_loc=args.embeddings_loc,loc=args.csv_loc,save_loc = args.save_location)
  # d =  get_neg_sampling(data_loc='combined.npy',loc= '/home/pratyushgarg11/data/bert-n-graph-embeddings/GlossBert-GraphEmbeddings/Training_Corpora/SemCor/semcor_train_sent_cls_ws.csv')

'''
count= 0
def count_zeros(word):
    global count
    if not word:
        count+=1
    return 0
_ = words.apply(count_zeros)
print(count)
print(words.head())
'''
