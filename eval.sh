#!/bin/bash
i = 0
declare -a file_array
file_array=(semeval2007 semeval2013 semeval2015 senseval2 senseval3)
for train_data_dir in ./Evaluation_Datasets/semeval2007/semeval2007_test_sent_cls_ws.csv ./Evaluation_Datasets/semeval2013/semeval2013_test_sent_cls_ws.csv ./Evaluation_Datasets/semeval2015/semeval2015_test_sent_cls_ws.csv ./Evaluation_Datasets/senseval2/senseval2_test_sent_cls_ws.csv ./Evaluation_Datasets/senseval3/senseval3_test_sent_cls_ws.csv

do
	
	echo $train_data_dir
	python wsd_sent_embeddings.py --task_name WSD --output_dir embeddings --train_data_dir $train_data_dir --bert_model bert-model/pretrained_model/ --file_name ${file_array[i]}
	i=$((i + 1))

done