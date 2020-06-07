#!/bin/bash
for i in 1 2 3 4
do
  python train_classifier.py --embeddings_data_dir combined.npy --dataset SemCor --do_train --out_results_dir results/"$i" --graph_embeddings_loc ./deepwalk_graph/wn_hypernym_"$i"_final.vec
  echo "$i is done"
done
