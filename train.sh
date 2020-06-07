#!/bin/bash
python wsd_sent_embeddings.py --task_name WSD --output_dir embeddings --train_data_dir ./Training_Corpora/SemCor/semcor_train_sent_cls_ws.csv --bert_model bert-model/pretrained_model/ --file_name semcor --train_batch_size 64
