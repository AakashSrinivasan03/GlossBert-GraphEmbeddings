from nltk.corpus import wordnet as wn
import networkx as nx
from itertools import islice
import pickle

def fun(set):

    G=nx.Graph()
    vertices = []
    edges = []
    dictionary_name_to_id = {}
    dictionary_id_to_name = {}

    #get all noun synsets
    noun_synsets = list(wn.all_synsets())
    vertices = [synset.name() for synset in noun_synsets]

    G.add_nodes_from(range(0, len(vertices)))
    print("done")
    for node in G.nodes():
        G.nodes[node]['name'] = vertices[node]
        dictionary_name_to_id[vertices[node]] = node
        dictionary_id_to_name[str(node)] = vertices[node]

    for node in G.nodes():
        current_vertice_id = node
        current_vertice_name = G.nodes[node]['name']
        current_synset = wn.synset(current_vertice_name)
        if set=="hypernym":
            for hypernym in current_synset.hypernyms():
                hypernym_name = hypernym.name()
                hypernym_id = dictionary_name_to_id[hypernym_name]
                edges.append((current_vertice_id, hypernym_id))
        elif set =="synonym":
            count=0
            for synonym in current_synset.lemma_names():
                try:
                    synonym_id = dictionary_name_to_id[synonym]
                    count+=1
                except:
                    continue
                edges.append((current_vertice_id, synonym_id))
            print(count)
        else:
            for hypernym in current_synset.hypernyms():
                hypernym_name = hypernym.name()
                hypernym_id = dictionary_name_to_id[hypernym_name]
                edges.append((current_vertice_id, hypernym_id))
            for synonym in current_synset.synonyms():
                synonym_name = synonym.name()
                synonym_id = dictionary_name_to_id[synonym_name]
                edges.append((current_vertice_id, synonym_id))

    G.add_edges_from(edges)
    nx.write_adjlist(G, "wordnet_{}.adjlist".format(set))

    with open('nodes.pkl', 'wb') as f:
        pickle.dump(dictionary_id_to_name, f, pickle.HIGHEST_PROTOCOL)

    return True
