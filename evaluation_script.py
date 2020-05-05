import pandas as pd
import numpy as np
import argparse
import os

def evaluate_results(dataset,input_file_name):
    
    train_file_name = './Evaluation_Datasets/'+dataset+'/'+dataset+'.csv'
    train_data = pd.read_csv(train_file_name,sep="\t",na_filter=False).values
    words_train = []
    for i in range(len(train_data)):
        words_train.append(train_data[i][4]) # get lemmas

    test_file_name = './Evaluation_Datasets/'+dataset+'/'+dataset+'_test_sent_cls.csv'
    test_data = pd.read_csv(test_file_name,sep="\t",na_filter=False).values

    seg = [0]
    for i in range(1,len(test_data)):
        if test_data[i][0] != test_data[i-1][0]:
            seg.append(i)
            

    results=[]
    num=0
    with open(input_file_name, "r", encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            q=float(s.split()[-1])
            results.append((q,test_data[num][-1]))
            num+=1
            s = f.readline().strip()


    predicted_sense_keys_dict = {}
    
    for i in range(len(seg)):
        if i!=len(seg)-1:
            result=results[seg[i]:seg[i+1]]
        else:
            result=results[seg[i]:-1]
        result.sort(key=lambda x:x[0],reverse=True)
        predicted_sense_keys_dict[test_data[seg[i]][0]] = [result[0][1]]

    gold_dataset_path = './Evaluation_Datasets/'+dataset+'/'+dataset+'.gold.key.txt'
    gold_sense_keys_dict = {}

    with open(gold_dataset_path, "r", encoding="utf-8") as f:
        s=f.readline().strip()
        while s:
            l = s.split()
            key=l[0]
            gold_sense_keys_dict[key] = l[1:]
            s = f.readline().strip()


    ok = 0
    not_ok = 0
    for key in predicted_sense_keys_dict:
        
        if(key not in gold_sense_keys_dict):
            continue

        local_ok = len(set(predicted_sense_keys_dict[key]).intersection(gold_sense_keys_dict[key]))/len(set(predicted_sense_keys_dict[key]))
        local_not_ok = len(set(predicted_sense_keys_dict[key]).difference(gold_sense_keys_dict[key]))/len(set(predicted_sense_keys_dict[key]))
        ok += local_ok
        not_ok += local_not_ok
    P = 100*(ok/(ok+not_ok))
    R = 100*(ok/len(gold_sense_keys_dict))
    
    F = (2*P*R)/(P+R)
    print("Precision",P)
    print("Recall",R)
    print("F1-score",F)
    return F

