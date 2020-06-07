#!/bin/bash
declare -a file_array
file_array=(semeval2007 semeval2013 semeval2015 senseval2 senseval3)
for j in 0
do
for i in 1 2 3 4
do
echo ${file_array[j]}_"$i"
  python train_classifier.py --embeddings_data_dir embeddings/${file_array[j]}.npy --dataset ${file_array[j]} --do_eval --out_results_dir new_eval_results/${file_array[j]}/"$i" --load_model_path results/"$i"/model_save --graph_embeddings_loc ./deepwalk_graph/wn_hypernym_"$i"_final.vec
done
done
