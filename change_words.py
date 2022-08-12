#!/usr/bin/env python
# coding: utf-8

"""
potential error: 
sequence length > 512: check the encode, tokenizer, model parameters, add the max_length param
"""

from curses.ascii import isalpha
from pickle import GLOBAL
import sys
from unittest.util import _MAX_LENGTH

import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, LongformerForMaskedLM,pipeline
from transformers import BertForMaskedLM, BertConfig

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import FloatTensor
from sklearn.metrics import roc_curve
from transformers import set_seed
import re
import random
import csv
from transformers.pipelines import PipelineException
sys.path.insert(0, '/home/gy654/Documents/text_classification/experiments/sensitivity_analysis')


#FILTER_BASE = e.mid_dir+'filtered/'

def to1_list(text):
    new = []
    for line_index in range(len(text)):
        line = text[line_index]
        line = re.split("\s|\.|,|:", line) # split into words
        new.extend(line)
    return new

def isBlank (myString):
    if myString and myString.strip():
        #myString is not None AND myString is not empty or blank
        return False
    #myString is None OR myString is empty or blank
    return True

def max_frequency_word(df): # df is the filtered df
    oex = df["filtered"]
    all_words = to1_list(oex)
    all_words_s = pd.Series(all_words).value_counts()
    max_frequency_word  = list(all_words_s.index)[0]
    #print(list(all_words_s.index))
    #print("word:", max_frequency_word)
    counter = 1
    while isBlank(max_frequency_word) or not max_frequency_word.isalpha(): 
        max_frequency_word = list(all_words_s.index)[counter]
        #print(counter, max_frequency_word )
        counter +=1
    
    return max_frequency_word
    

def uniform_substitute():
    random.seed(100)
    num = random.randint(1117, 28125) # start, end index in the vocab file
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    token = tokenizer.decode(num)
    print(f"What is the token we selected in the uniform task? {token}")
    return token


'''
if mode == True, means MULTIPLE SWAP == TRUE, we swap all occurences
'''
# _dict can contain many key value pairs, the function will replace all keys wit pairs
def switch_columns(columns, df, _dict, mode):
    print(f"mode: {mode}")
    total_swaps =0
    # modify each column: extracted_text, text
    for column in columns:
        changed_column_name = "switched_" + column
        text = list(df[column])
        swapped_text = [] # could be swapped text or masked text
        print("swapping...")
        for line_index in range(len(text)):
            #=================== swap word starts ==========================
            line = text[line_index]

            # we wanna "normalize" the special tokens [MASK] contained in the original data, that before our own masking
            line = encode_mask_in_original_text(line, "mask") # swap the original [MASK] with mask

            swap = line # before our own preprocess, the line should contain no [MASK]
            words_to_change = list(_dict.keys())
            for word_to_change in words_to_change:
                search_scheme = generate_search_scheme(word_to_change, "discrete")
                ind = re.search(search_scheme, swap)
                
                """
                if ind != None:
                    st = ind.start()
                    e = ind.end()
                    try: 
                        print(f"In swap: did we find it? line: {line[st-10:e+10]}\nword to change:{word_to_change}")
                    except:
                        print("Wrong index!")
                        continue
                """
                
                if mode: # if multiple times
                    while ind != None:  # if this key exist in the sentence, keep swapping
                        swap = swap[: ind.start()] +f" {_dict[word_to_change]} " + swap[ind.end()-1:]
                        total_swaps+=1
                        ind = re.search(search_scheme, swap)
                else: # if one_time
                    try:
                        swap = swap[: ind.start()] +f" {_dict[word_to_change]} " + swap[ind.end()-1:]
                        total_swaps+=1
                    except AttributeError as e:
                        print(swap)
                        print(f'one swap: Attribute Error: {e}, what is {ind}')
                        sys.exit()

            swapped_text.append(swap)

    print(f"Total swaps from {word_to_change} to {_dict[word_to_change]}: {total_swaps}")

    return swapped_text


    


# if the length of encoding is > 512, we truncate the string to 512 tokens
def adjust_length(tokenizer, text):
    encoded = tokenizer.encode(text)
    if len(encoded) > 512: 
        text = tokenizer.decode(encoded[:512])
    return text


'''
returns a list of M objects of type obj
'''
def M_obj_generator(M, obj):
    container = []
    for i in range(M):
        container.append(obj)
    
    return container


'''
@param: sentence - str
@return: a list of new_sentences with top k possible prediction
Replace all the [MASK] in a string
'''
def mlm(model_mlm, tokenizer, unmasker, sentence, k):
    occurance = sentence.count("[MASK]")
    new_sentences = []
    new_sentence = sentence
    if occurance == 0:
        for i in range(k):
            new_sentences.append(new_sentence)
    elif occurance == 1:
        if k>5: 
            print("Error: k>5!")
            exit()
        if "[MASK]" in sentence:
            try: 
                result = unmasker(sentence, truncation = True)
                masks = [i["token_str"] for i in result]
                for i in range(k): 
                    new_sentence  = result[i]["sequence"]
                    new_sentences.append(new_sentence)
                

                """ # only use the alphabetic predictions
                #print(f"what is the mask? {masks}")
                not_all_hashtag = any(mask.isalpha() for mask in masks) # true if there exist one prediction to be alpha
                #print(result)
                regular_mask = [] # alpha numeric 
                for i in masks: 
                    if i.isalpha():
                        regular_mask.append(i)
                if not_all_hashtag == True:
                    for _dict in result:
                        if new_sentence != sentence: 
                            continue
                        if not _dict['token_str'].isalnum():
                            continue
                        new_sentence = _dict["sequence"]

                else:
                    new_sentence  = result[k]["sequence"]
                """
            except PipelineException: 
                 # if we cannot mask when there is a [MASK], but is trucated so the mln fails
                 # we use the original sentence
                print("special token [MASK] is after 512, truncated!")
                for i in range(k):
                    new_sentences.append(new_sentence)
    elif occurance > 1:
        new_sentences = multiple_mask_mlm(sentence, model_mlm, tokenizer, 5)

    #print(f"length of new_sentences: {len(new_sentences)}")
    if len(new_sentences)!=k:
        print(f"Error: length of new_sentences is not k!\n\nSentence: {sentence}")
        sys.exit()
    
    return new_sentences

# replace masked column with mlm result
def replace_all_mask(masked_column, model_path, M):
    # model setup
    model_mlm = BertForMaskedLM.from_pretrained(model_path)
    tokenizer =  AutoTokenizer.from_pretrained(model_path, model_max_length=512)
    unmasker = pipeline('fill-mask', model=model_mlm, tokenizer = tokenizer)

    new_columns = []
    for line in masked_column: # line-each line # masked_column : number of rows * 1
        container = re.split("\.", line)  # container: (number of sentences in a line * 1)
        #print(f"container: {container}")

        masked_container = [] # per row/line
        for sentence in container: # each sentence
            new_sentences = mlm(model_mlm, tokenizer, unmasker, sentence,M) # a list of M variant /predictions
            #print(f"replace_all_mask: new_sentences: {new_sentences}")
            masked_container.append(new_sentences) # [[sen1 has 5 variant], 
                                                   #  [sen2 has 5 variant]]
            #print(f"shape is {M}? : len: {len(new_sentences)}")

        #print(masked_container)
        new_line = [] # should contain M variations for each line, len() == M
        for col_ind in range(M):
            col = ""
            for row_ind in range(len(masked_container)):
                #print(f"masked_container_shape:{masked_container.shape}")
                
                col+= masked_container[row_ind][col_ind] + ". "
            new_line.append(col)
        # new_line : (5*1)
        
        if len(new_line) != M:
            print(f"Error: Length of {new_line} != {M}")
            exit()



        new_columns.append(new_line)

    return new_columns


# replace a sentence with >1 masks with mlm results
# returns new_s as the replaced sentence, and best_guess and topk_guess as token id lists
def multiple_mask_mlm(sentence, model, tokenizer, topk):
    token_ids = tokenizer.encode(sentence, max_length = 512, truncation = True, return_tensors='pt')
    token_ids_tk = tokenizer.tokenize(sentence, return_tensors='pt')
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position ]
    with torch.no_grad():
        output = model(token_ids) # do we need trucatin == True here?

    last_hidden_state = output[0].squeeze()
    list_of_list =[]
    for mask_index in masked_pos:
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=100, dim=0)[1]
        words_id = [i.item() for i in idx]
        list_of_list.append(words_id)

    top_k_guess = [i[:topk] for i in list_of_list]
    #print(f"top k guess: {top_k_guess}")
    #print(f"masked_position: {masked_position}")
    new_sentences=[]
    for t in range(topk): 
        variation = token_ids[0]
        for ind in range(len(masked_position)): # which masked position (1st, 2nd..?)
            token_pos = masked_position[ind][0] # masked position specific position id
            #print(f"ind: {ind}, t: {t}, token_pos: {token_pos}")
            variation[token_pos] = top_k_guess[ind][t] 
        new_sentence = tokenizer.decode(variation)
        new_sentences.append(new_sentence)
        #print(f"mul: {new_sentences}")

    return new_sentences #, best_guess, top_k_guess
    
    """
    #print(f"topk {top_k_guess}")
    best_guess = [] # a list of token ids 
    for mask_list in list_of_list:
        #print("mask list: ", mask_list)
        words_mask_list = [tokenizer.decode(i).strip() for i in mask_list]
        not_all_hashtag = any(prediction.isalpha() for prediction in words_mask_list)
        if not_all_hashtag == True:
            #print("not_all_hashtag == True")
            for prediction in mask_list:
                if tokenizer.decode(prediction).strip().isalnum():
                    best_prediction = prediction
                    break
        else:
            best_prediction = mask_list[0]


        best_guess.append(best_prediction)

    count = 0
    token_ids = token_ids[0]
    for token_pos in masked_position: 
        token_ids[token_pos] = best_guess[count]
        count+=1

    new_s = tokenizer.decode(token_ids)
    """

    

def dict_constructor(word_list, value):
    d = {}
    for word in word_list: 
        d[word] = value
    return d

# modify the dataset based on all words in word_list
# df is the filtered df based on the word_list
def change_1_word_list(word_list, schemes, df, mode):
    df_new = df.copy()

    for scheme in schemes:
        if scheme == "context":
            # mask out the word
            mask = dict_constructor(word_list, "[MASK]")
            #mask = {word1: "[MASK]", word2: "[MASK]"}
            masked_column = switch_columns(["filtered"], df, mask, mode)
            print(f"[TASK {scheme}]: masking finished!")
            # replace the word
            CLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"
            new_column = replace_all_mask(masked_column, CLINICAL_BERT)
            df_new[scheme] = new_column
            print(f"[TASK {scheme}]: swapping finished!")

        else:
            if scheme == "1_gram":
                substitute = max_frequency_word(df) # top k
            elif scheme == "uniform":
                substitute = uniform_substitute()

            _dict = dict_constructor(word_list, substitute) # substitute a list of words with one word picked according to the scheme
            new_column = switch_columns(["filtered"], df, _dict, mode)
            new_column_name = scheme+"_"+substitute
            df_new[new_column_name] = new_column
            print(f"[TASK {scheme}]: swapping finished!")
    return df_new


def change_many_words(e, words_list_of_list, odf, save_path):
    file_path_list = []
    schemes = ["context", "1_gram", "uniform"]
    filter_nothing_count = 0
    records = None
    
    for word_list in words_list_of_list: 
        print(f"This is the word_list {word_list}")
        # take in original df, select relevant df for the word
        word_relevant_notes = filter(e, odf, 'extracted_text', word_list) 
        if isinstance(word_relevant_notes, type(None)): 
            print("filtered nothing!")
            filter_nothing_count +=1
            continue
        if len(word_relevant_notes) <= 10:
            print("too few records! SKIP this word")
            continue
        df_word = change_1_word_list(word_list, schemes, word_relevant_notes) 
        df_name = f"swap_{word_list[0]}.csv"
        df_word.to_csv(save_path+df_name) # save the modified csv specific to that word list
        file_path_list.append([save_path+df_name])
        print(f"\n\n\n\n\nThe current word list is {word_list}:")
        print(df_word["context"].head(3))
        print("\n\n\n\n\n")
    print(f"filter nothing count {filter_nothing_count}")
    return file_path_list


# rp: word to replace
def generate_search_scheme(rp, mode):
    if mode == "discrete":  # eg. "/[MASK]", " [MASK] "
        p1 = f"[^a-zA-Z\d]{rp}[^a-zA-Z\d]" 
        p2 = f"^{rp}[^a-zA-Z\d]" 
        p3 = f"[^a-zA-Z\d]{rp}$" 
        p4 = f"^{rp}$"
        match_scheme = f"{p1}|{p2}|{p3}|{p4}"
    elif mode == "connected":  #eg. "for[MASK]"
        p1 = f"[a-zA-Z\d]+{rp}[a-zA-Z\d]+" 
        p2 = f"{rp}[a-zA-Z\d]+" 
        p3 = f"[a-zA-Z\d]+{rp}" 
        match_scheme = f"{p1}|{p2}|{p3}"
    #print(f"{mode}: {match_scheme}")
    return match_scheme

def encode_mask_in_original_text(sentence, encoding): 
    match_scheme = "\[MASK\]"
    k = re.search(match_scheme, sentence)
    if k == None: # there is no [MASK], return the original sentence
        return sentence
    start = k.start()
    end = k.end()
    encoded_s = sentence
    encoded_s = sentence[: start]+ encoding + sentence[end:]
    print(f"PREENCODE: {sentence}\nPOSTENCODE: {encoded_s}\n")
    return encoded_s



# filter per word
# @param wordlist can be a single word or a list of words we wanna filter; not the same as the wordlist provided to Experiment
# define a function that filters out a subset of notes that contain the words that we are interested in replacing.
# the function returns a series of notes that contains the words and save the filtered notes to the assigned file
def filter(e, wordlist):
    df = pd.read_csv(e.input_dir + 'test.csv')
    hit = 0
    substring = ""
    for ind in range(len(wordlist)): 
        word = wordlist[ind]
        nw = generate_search_scheme(word, "discrete")
        if ind != len(wordlist)-1: 
            substring+=nw + "|"
        else: 
            substring+=nw
    print(f"The substring we search for is:{substring}")
    
    #substring =  ' | '.join(wordlist)
    #print(substring)
    all_notes = df['text']
    #print(all_notes)
    relevant_notes = []
    for i in range(len(all_notes)):
        note_i = list(all_notes)[i].lower()
        #print(note_i)
        match_exist = re.search(substring,note_i)
        if match_exist is not None:
            matches = re.finditer(substring,note_i)
            relevant_notes.append(note_i)
            for i in matches:
                st = i.start()
                end = i.end()
                hit+=1

                try:
                    continue
                    #print(f"hit: {note_i[st-10:e+10]}")
                except:
                    #print("Wring index")
                    continue
                
            

    print(f"In filter: Total hits: {hit}")
    if hit == 0: 
        return None
    base_path = e.mid_dir+'filtered/'
    file_name = f"fil_{wordlist[0]}.csv"
    saved_path = base_path+file_name
    relevant_notes = pd.Series(relevant_notes).to_frame('filtered')
    relevant_notes.to_csv(saved_path, index= False)
    print(f"Did we save the filtered ds? {os.path.exists(saved_path)}, path = {saved_path}")
    #print(relevant_notes)
    return relevant_notes


def get_words(mode, num_of_words):
    if mode == "sensitive": 
        PATH = '/home/gy654/Documents/preprocess/classifier_sensitivity/top200.csv'
    elif mode == "nonsensitive": 
        PATH = '/home/gy654/Documents/preprocess/classifier_sensitivity/low200.csv'

    words_list_of_list = [] 
    words_table = pd.read_csv(PATH)
    i = 0
    while len(words_list_of_list) < num_of_words: 
        admit = words_table['admit'][i]
        non_admit = words_table['non-admit'][i]
        if admit not in words_list_of_list: 
            words_list_of_list.append(admit)
        if non_admit not in words_list_of_list: 
            words_list_of_list.append(non_admit)
        i+=1

    words_list_of_list_1 = [[i] for i in words_list_of_list]

    print(f"{mode} words: {words_list_of_list_1}")
    return words_list_of_list_1, words_list_of_list


def log_preprocessed_words(list_of_list): 
    with open("/home/gy654/Documents/preprocess/classifier_sensitivity/preprocessed_words.csv", "a") as f:
        writer = csv.writer(f)
        for new_row in list_of_list: 
            writer.writerow(new_row)
        print(f"\tcsv Records file appended.")


