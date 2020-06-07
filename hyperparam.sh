#!/bin/bash

LR=(0.0001 0.001 0.01)
model_type=(Projection FC-1layer)
graph_embedding=(1 2 3 4)

for lr in "${LR[@]}"
do
	for model in "${model_type[@]}"
	do
		for i in "${graph_embedding[@]}"
		do
			echo $model, $lr, $i
			python classifier_hyperparam.py --embeddings_data_dir combined.npy --dataset semcor --do_train --out_results_dir results/new_try --model_type $model --graph_embeddings_loc ./deepwalk_graph/wn_hypernym_"$i"_final.vec  --LR $lr > out_hyperparam ;
		done
	done
done &

