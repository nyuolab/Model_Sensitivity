#!/usr/bin/env python
# coding: utf-8

from audioop import mul
from tkinter import WORD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.calibration import CalibratedClassifierCV
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import random
import sys
sys.path.insert(0, "/home/gy654/Documents/preprocess/classifier_sensitivity")
from change_words import *

'''
generate the filtered and perturbed datasets for the metric
'''






def get_max_frequency_word(m, filtered_series, word): 
    cv = CountVectorizer()
    try:
        corpus = [''.join(filtered_series["extracted_text"].tolist())]
    except:
        corpus = [''.join(filtered_series["filtered"].tolist())]
    cv_fit = cv.fit_transform(corpus).toarray()
    count_list = cv_fit.sum(axis=0)
    word_list = cv.get_feature_names_out()
    word_count_dict = dict(zip(word_list,count_list))
    sorted_vocab_dict = {k: word_count_dict[k] for k in sorted(word_count_dict, key=word_count_dict.get, reverse=True)}
    top_30_most_frequent_list = list(sorted_vocab_dict.keys())[:30]
    print(top_30_most_frequent_list)
    stop_words = stopwords.words('english')
    filtered_w_list = [w for w in top_30_most_frequent_list if not w.lower() in stop_words]
    filtered_w_list = [w for w in filtered_w_list if w != word]
    filtered_w_list = [w for w in filtered_w_list if w.isalpha()]
    top_m_most_frequent_list = filtered_w_list[:m]
    return top_m_most_frequent_list

def get_uniform_substitutes(e,  word):
    m = e.M
    print(f'what is m:{m}')
    random.seed(100)
    
    num_list = []
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    word_num = tokenizer.encode(word)[1]
    count = 0
    while count < m:
        ran = random.randint(1117, 28125)  # start, end index in the vocab file
        if ran != word_num:
            num_list.append(ran)
            count +=1
    substitute_list = []
    for num in num_list:
        substitute = tokenizer.decode(num)
        substitute_list.append(substitute)
    print(f"What are the {m} tokens we selected in the uniform task? {substitute_list}")
    return substitute_list

"""
mask out the words in word_list in the filtered_df, which must contain the column "filtered"
word_list: the list of words to mask out
filtered_df: filtered df based on the word to test sensitivity, containing the "filtered" column
masked_text: a list of text, with the word to test masked out
"""
def mask(word_list, filtered_df, mode): 
    _dict = dict_constructor(word_list, "[MASK]")
    masked_text = switch_columns(["filtered"], filtered_df, _dict, mode)
    return masked_text


"""
check is the pmode_word.csv exist:
    if so, return it, and None for filtered_df
    else return a filtered_df
returns a filtered ds anyways, pmode ds could be None or an actual ds
"""
def check(e, word, pmode, BASE_PATH):
    print(f"check : {word}")
    filtered_df = None
    exist = False
    if isinstance(word, str): 
        word_list = [word]
    elif isinstance(word, list): 
        word_list = word
        word = word_list[0]

    pmode_filename = f"{pmode}_{word}.csv"
    pmode_path = BASE_PATH+pmode_filename
    filtered_path = e.mid_dir+'filtered/'+f"fil_{word}.csv"
    if os.path.exists(pmode_path):
        print(f"[WORD {word}]: {pmode_filename} exists")
        exist = True
    else: 
        print(f"[WORD {word}]: {pmode_filename} does NOT exist")
        if os.path.exists(filtered_path):
            print(f"[WORD {word}]: filtered file exists")
            filtered_df = pd.read_csv(filtered_path)
        else: 
            print(f"[WORD {word}]: filtered_file does not exist: filtering...")
            filtered_df = filter(e, word_list) 
            print(f"[WORD {word}]: filtered file saved: {os.path.exists(filtered_path)}")
  
    return word, word_list, filtered_df,  exist



"""
save (return) a dataframe of six columns (replaces w with 5 different w' under p_mode = uniform)
1.check if the {pmode}_{word} dataset exist, if so, return
2.if pmode ds doesn't exist, check if the filtered ds exit
    if not so, make one filtered ds, then from that, make a pmode_word ds
"""
def uniform_1gram_w_prime(e, word, pmode, mode, BASE_PATH):
    M = e.M
    if e.multiple_swap:
        swap = 'multiple'
    else:
        swap = 'onetime' 

    word, word_list, filtered_df, exist = check(e, word, pmode, BASE_PATH)
    if exist:
        return
    else: 
        if pmode == "uniform": 
            substitutes = get_uniform_substitutes(e, word) # get the 5 words substitution
        elif pmode == "1gram": 
            substitutes = get_max_frequency_word(M,filtered_df,word)

    df_new = filtered_df.copy()
    for substitute in substitutes: 
        _dict = dict_constructor(word_list, substitute) # substitute a list of words with one word picked according to the scheme
        new_column = switch_columns(["filtered"], filtered_df, _dict, mode)
        new_column_name = pmode+"_"+substitute
        df_new[new_column_name] = new_column
        print(f"[WORD {word} TASK {pmode} REPLACEw/ {substitute}]: finished!")
    
    pmode_filename  = f'{swap}/{pmode}_{word}.csv'
    pmode_path = BASE_PATH+pmode_filename
    df_new.to_csv(pmode_path)
    print(f"[WORD {word}]: {pmode_filename} created? {os.path.exists(pmode_path)}")



def uniform_w_prime(e, word,mode, BASE_PATH):
    uniform_1gram_w_prime(e, word, "uniform", mode, BASE_PATH)

# save (return) a dataframe of six columns (replaces w with 5 different w' under p_mode = 1gram)
def max_w_prime(e, word, mode, BASE_PATH):
    uniform_1gram_w_prime(e, word, "1gram", mode, BASE_PATH)
 

'''
@param: 
word: can be a list (for now, only len(list)=1 works) or str. 
This function check if the pmode ds for context mlm has been created,
if not so, create one and save
'''
# save (return) a dataframe of six columns (replaces w with 5 different w' under p_mode = context)
def context_w_prime(e, word, mode, BASE_PATH):
    M = e.M
    if e.multiple_swap:
        swap = 'multiple'
    else:
        swap = 'onetime' 

    print(f"WORD: {word}")
    word, word_list, filtered_df, exist = check(e, word,"context", BASE_PATH)

    if exist:
        return
    else:
        CLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"
        masked_column = mask(word_list, filtered_df, mode)
        df_new = filtered_df.copy()
         
        new_columns = replace_all_mask(masked_column, CLINICAL_BERT,M)
        df_perturbed = pd.DataFrame(new_columns, columns = [f"context_{i}" for i in range(M)])
        #print(f'df_perturbed: {df_perturbed.head(2)}')
        df_new = pd.concat([df_new, df_perturbed], axis =1)
        print(f"[WORD {word} TASK mlm]: finished!")
    

    file_path = BASE_PATH + f'{swap}/context_{word}.csv'
    df_new.to_csv(file_path, index = False)
    print(f"[WORD {word}]: context_{word}.csv created? {os.path.exists(file_path)}\n\n\n")



def make_dataset(e):
    MULTIPLE_SWAP = e.multiple_swap
    BASE_PATH = e.perturbed_dir
    WORD_LIST = e.wl
    if isinstance(WORD_LIST, str):
        WORD_LIST = [WORD_LIST]
    for word in WORD_LIST:
        uniform_w_prime(e, word, MULTIPLE_SWAP, BASE_PATH)
        max_w_prime(e, word, MULTIPLE_SWAP, BASE_PATH)
        context_w_prime(e, word, MULTIPLE_SWAP, BASE_PATH)




#xmake_dataset(WORD_LIST, multiple_swap)

