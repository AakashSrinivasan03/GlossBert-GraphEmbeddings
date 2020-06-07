#!/bin/bash
for i in semeval2007 semeval2013 semeval2015 senseval2 senseval3
 do
  python projection.py --embeddings_data_dir embeddings/${i}.npy --dataset $i --do_eval --out_results_dir results/projection --load_model_path results/projection/model_save --model_type Projection --graph_embeddings_loc ./deepwalk_graph/wn_hypernym_3_final.vec
done
