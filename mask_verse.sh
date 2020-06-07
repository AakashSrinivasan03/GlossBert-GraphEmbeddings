#!/bin/bash

LR=(0.01)
model_type=(Hadamard Projection FC-1layer)
graph_embedding=(verse)

for lr in "${LR[@]}"
do
	for model in "${model_type[@]}"
	do
		for i in "${graph_embedding[@]}"
		do
			echo $model, $lr, $i
			python train_mask.py --embeddings_data_dir combined.npy --dataset semcor --do_train --out_results_dir results/tmp_mask_verse --model_type $model --graph_embeddings_loc ./VERSE/${i}.vec  --LR $lr --use_mask --mask_path  dataset_with_sampling/semcor_mask.npy
		done
	done
done 

