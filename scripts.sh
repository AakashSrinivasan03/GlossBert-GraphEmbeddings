seed=1314

train_gloss(){
	CUDA_VISIBLE_DEVICES=0,1,2,3;
	python main.py \
		--task_name WSD \
		--train_data_dir ../src/bert_outputs/train.csv \
		--eval_data_dir ../src/bert_outputs/eval.csv \
		--output_dir results/$seed \
		--config config.json \
		--do_train \
		--do_eval \
		--train_batch_size 64 \
		--eval_batch_size 128 \
		--learning_rate 2e-5 \
		--num_train_epochs 6 \
		--seed $seed ;
}

test_gloss(){
	CUDA_VISIBLE_DEVICES=0,1,2,3;
	python main.py \
		--task_name WSD \
		--eval_data_dir ../src/bert_outputs/test.csv \
		--output_dir results/sent_cls_ws/$seed \
		--config config.json \
		--do_test \
		--train_batch_size 64 \
		--eval_batch_size 128 \
		--learning_rate 2e-5 \
		--num_train_epochs 6.0 \
		--seed $seed ;
}


train_fc_dev(){
	python train_classifier.py \
		--embeddings_data_dir  embeddings/semeval2007.npy \
		--dataset semcor \
		--do_train \
 		--out_results_dir results/graph_bert_model  \
 		--batch_size 128 \
 		--num_epochs 10  \
 		--graph_embeddings_loc embeddings/deepwalk_emb_3.vec 
}


train_fc_dev(){
	python train_classifier.py \
		--embeddings_data_dir  combined.npy \
		--dataset semcor \
		--do_train \
 		--out_results_dir results/graph_bert_model  \
 		--batch_size 128 \
 		--num_epochs 10  \
 		--graph_embeddings_loc embeddings/deepwalk_emb_3.vec 
}

clean(){
	rm -r results/$seed
}


if [ $# -lt 1 ]
then
	echo "Not enough arguments"
else
	if [ $1 = "train_gloss" ]
	then
		train_gloss
	elif [ $1 = "test_gloss" ]
	then
		test_gloss
	elif [ $1 = "train_fc" ]
	then
		train_fc
	elif [ $1 = "clean" ]
	then
		clean
	else
		echo "Invalid argument"
	fi
fi

