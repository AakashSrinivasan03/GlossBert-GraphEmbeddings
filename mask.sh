#!/bin/bash

LR=(0.0001 0.001 0.01)
model_type=(Hadamard Projection FC-1layer)
graph_embedding=(1 2 3 4)

for lr in "${LR[@]}"
do
	for model in "${model_type[@]}"
	do
		for i in "${graph_embedding[@]}"
		do
			echo $model, $lr, $i
			python train_mask.py --embeddings_data_dir combined.npy --dataset semcor --do_train --out_results_dir results/tmp_mask --model_type $model --graph_embeddings_loc ./deepwalk_graph/wn_hypernym_"$i"_final.vec  --LR $lr --use_mask --mask_path  dataset_with_sampling/semcor_mask.npy
		done
	done
done

