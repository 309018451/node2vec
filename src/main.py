
# coding: utf-8

# In[15]:

import argparse
import numpy as np
import networkx as nx
import node2vec
from collections import defaultdict
from time import perf_counter
from datetime import timedelta
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from multiprocessing import cpu_count


def read_graph(input_edgelist, directed=False):
    
    G = nx.read_edgelist(input_edgelist, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    probs = defaultdict(dict)
    for node in G.nodes():
        probs[node]['probabilities'] = dict()
            
    if not directed:
        G = G.to_undirected()  
    
    return G, probs


@node2vec.timer('Generating embeddings')
def generate_embeddings(corpus, dimensions, window_size, num_workers, output_file):
    
    model = Word2Vec(corpus, size=dimensions, window=window_size, min_count=0, sg=1, workers=num_workers)
    #model.wv.most_similar('1')
    w2v_emb = model.wv
    w2v_emb.save_word2vec_format(output_file)

    return model, w2v_emb


def process(args):
    
    Graph, init_probabilities = read_graph(args.input, args.directed)
    G = node2vec.Graph(Graph, init_probabilities, args.p, args.q, args.walks, args.length, args.workers)
    G.compute_probabilities()
    walks = G.generate_random_walks()
    model, embeddings = generate_embeddings(walks, args.d, args.window, args.workers, args.output) 
    

    return    


def main():

    parser = argparse.ArgumentParser(description = "node2vec implementation")

    parser.add_argument('--input', default='graph/karate.edgelist', help = 'Path for input edgelist')

    parser.add_argument('--output', default='embeddings/karate_embeddings.txt', help = 'Path for saving output embeddings')

    parser.add_argument('--p', default='1', type=float, help = 'Return parameter')

    parser.add_argument('--q', default='1', type=float, help = 'In-out parameter')

    parser.add_argument('--walks', default=10, type=int, help = 'Walks per node')

    parser.add_argument('--length', default=80, type=int, help = 'Length of each walk')

    parser.add_argument('--d', default=128, type=int, help = 'Dimension of output embeddings')

    parser.add_argument('--window', default=10, type=int, help = 'Window size for word2vec')

    parser.add_argument('--workers', default=cpu_count(), type=int, help = 'Number of workers to assign for random walk and word2vec')

    parser.add_argument('--directed', dest='directed', action ='store_true', help = 'Specify if graph is directed. Default is undirected')
    parser.set_defaults(directed=False)

    args = parser.parse_args()
    process(args)

    return


if __name__ == '__main__':
    main()
    


