#!/bin/bash

python classifier_with_no_graph.py  --embeddings_data_dir combined.npy --dataset semcor --do_train --out_results_dir results/baseline --no_graph_embedding
