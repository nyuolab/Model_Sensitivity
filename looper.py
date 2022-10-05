#!/usr/bin/env python
# coding: utf-8

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
from netcal.scaling import TemperatureScaling
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import random
import seaborn as sns
import itertools
from looper_data import *
import sys
sys.path.insert(0, "/home/gy654/Documents/preprocess/classifier_sensitivity") 
from change_words import *
from ratio_freq import *
import nltk
nltk.download('stopwords')

print("Looper starts!")



#WORD_LIST = ['residual', 'thirty', 'increase', 'atrial', 'prescribed', 'nameis', 'week', 'medication', 'units', 'extended', 'aspirin', 'facility', 'date', 'number', 'major', 'chief', 'procedure', 'namepattern1', 'patient', 'family', 'care', 'follow', 'stitle', 'dr', 'md', 'pain', 'telephone', 'instructions', 'release', 'capsule', 'times', 'disp', 'refills', 'discharge', 'day', 'daily', 'po', 'sig', 'mg', 'tablet']
#WORD_LIST = ['mother', 'father','blood','patient','fall','vaccination', 'labor', 'ulcer', 'arthritis', 'prematurity', 'hypoglycemia']
WORD_LIST = ['cancer', 'mg', 'colon', 'expired', 'deceased', 'heparin', 'died', 'father', 'mother', 'mouthwash', 'regimen', 'congenital', 'thinner']


# fundtion used by count_fre in the class Experiment
def read_corpus(corpus_path):
    mode = "csv"
    # Read the whole text.
    if mode == "csv":
        fn = corpus_path
        df = pd.read_csv(fn)
        text = df[['text', 'labels']]
        corpus = [''.join(text[text['labels']==label]['text'].tolist()) for label in [0,1]]

    else:
        raise Exception(f"mode {mode} not implemented!")
    return corpus



def calibrate_model(e):
    model_path = e.model_path
    val_path = e.input_dir + 'val.csv'
    model = joblib.load(model_path)
    complete_df = pd.read_csv(val_path)
    X_calib = complete_df['text']
    y_calib = complete_df['labels']
    calibrated_model = CalibratedClassifierCV(base_estimator=model, cv='prefit')
    calibrated_model.fit(X_calib, y_calib)
    return calibrated_model


def generate_pred_tfidf(e, model, complete_df):
    m = e.M
    feature_names = complete_df.columns
    feature_dict = {}
    pred_dict = {}
    try:
        for i in range(1, m+2):
            feature_dict['feature'+str(i)]= feature_names[i]
        for i in range(1, m+2):
            pred_dict['pred'+str(i)] = model.predict_proba(list(complete_df[feature_dict['feature'+str(i)]]))[:,0]
    except: 
        for i in range(1, m+2):
            feature_dict['feature'+str(i)]= feature_names[i-1]
        for i in range(1, m+2):
            pred_dict['pred'+str(i)] = list(model.predict_proba(list(complete_df[feature_dict['feature'+str(i)]]))[:,0])
    return pred_dict



def agreement_rate_tfidf(pred_dict):
    joined = [list(v) for k, v in pred_dict.items()]
    l1_matrix = (1/len(pred_dict['pred1'])) * manhattan_distances(joined, joined)
    first_line = l1_matrix[0]
    return first_line


def generate_pred_finetune(e, checkpoint_path, complete_df):
    m = e.M
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path) 
    tokenizer = AutoTokenizer.from_pretrained(e.pretrain_path, model_max_length=512,truncation = True, pad_to_max_length=True)
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device =0)
    feature_names = complete_df.columns
    feature_dict = {}
    try:
        for i in range(1, m+2):
            feature_dict['feature'+str(i)]= feature_names[i]

    except:
        for i in range(0, m+1):
            feature_dict['feature'+str(i+1)]= feature_names[i]
 
    pipe_dataloader_dict = {}
    for i in range(1, m+2):
        pipe_dataloader_dict['pipe_dataloader'+str(i)] = torch.utils.data.DataLoader(list(complete_df[feature_dict['feature'+str(i)]]), batch_size = 32)

    result_dict = {}
    for i in range(1, m+2):
        result_dict['result'+str(i)] = []
    
    for i in range(1, m+2):
        for j, batch in enumerate(pipe_dataloader_dict['pipe_dataloader'+str(i)]):
            outputs = pipe(batch, truncation = True)
            result_dict['result'+str(i)].append(outputs)
    pred_dict = {}
    for i in range(1, m+2):
        pred_dict['pred'+str(i)] = list(chain(*result_dict['result'+str(i)]))
        pred_dict['pred'+str(i)] = [[1-pred_dict['pred'+str(i)][j]['score'] if pred_dict['pred'+str(i)][j]['label']=='LABEL_0' else pred_dict['pred'+str(i)][j]['score']  for j in range(len(pred_dict['pred'+str(i)]))]]
    
    return pred_dict
    
def generate_val_pred(e):
    VAL_PATH = e.input_dir + 'val.csv'
    model_path = e.model_path
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(e.pretrain_path, model_max_length=512,truncation = True, pad_to_max_length=True)
    pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device =0)
    complete_df = pd.read_csv(VAL_PATH)
    val_labels = complete_df['labels']
    raw_dataset1 = list(complete_df['text'])
    pipe_dataloader1 = torch.utils.data.DataLoader(raw_dataset1, batch_size = 32)
    result = []
    for i, batch in enumerate(pipe_dataloader1):
        outputs = pipe(batch, truncation = True)
        result.append(outputs)
    val_pred = list(chain(*result))
    val_pred = [[val_pred[i]['score'] for i in range(len(val_pred))]]
    return val_pred, val_labels

def calibrate_test_score(val_pred, val_labels, pred_dict, m):
    temperature = TemperatureScaling()
    temperature.fit(np.array(val_pred[0]), np.array(val_labels))
    calibrated_dict = {}
    for i in range(1, m+2):
        calibrated_dict['pred'+str(i)] = temperature.transform(np.array(pred_dict['pred'+str(i)][0]))
    return calibrated_dict


def agreement_rate_finetune(calibrated_dict):
    joined = [list(v) for k, v in calibrated_dict.items()]
    l1_matrix = (1/len(calibrated_dict['pred1'])) * manhattan_distances(joined, joined)
    first_line = l1_matrix[0]
    return first_line




# p_mode selected from ['uniform', '1gram', 'context']
# when called, append one row(l1 of using p_mode on word) to word_df
def add_l1_of_p_mode(e, word, word_df, p_mode):
    BASE_PATH = e.mid_dir
    M = e.M
    IF_CALIBRATE = e.calibrate
    MODEL_PATH = e.model_path
    model_type = e.model_type
    if e.multiple_swap:
        swap = 'multiple'
    else:
        swap = 'onetime'


    file_path = BASE_PATH + 'perturbed/'+f'{swap}/{p_mode}_{word}.csv'
    pred_dict = {}
    if not os.path.exists(file_path):
        print(f'what is file path?:{file_path}')
        print('The file does not exist')
        return word_df, pred_dict
    
    perturbed_df = pd.read_csv(file_path)
    if model_type == 'tfidf':
        if IF_CALIBRATE:
            calibrated_model = calibrate_model(e)
            pred_dict = generate_pred_tfidf(e, calibrated_model, perturbed_df)
            #print(f'what is pred_dict:{pred_dict}')
        else:
            model = joblib.load(MODEL_PATH)
            pred_dict = generate_pred_tfidf(e, model, perturbed_df) 
        first_line =  agreement_rate_tfidf(pred_dict)
        word_df.loc[p_mode] = first_line
    else:
        val_pred, val_labels =  generate_val_pred(e)
        pred_dict = generate_pred_finetune(e, MODEL_PATH, perturbed_df)
        if IF_CALIBRATE:
            pred_dict = calibrate_test_score(val_pred, val_labels, pred_dict, M)
        first_line = agreement_rate_finetune(pred_dict)
        word_df.loc[p_mode] = first_line
    return word_df


# for each word, generate a dataframe that records its array of l1s under different p_mode
def gen_df_for_word(e, word):
    M = e.M
    modes = ['uniform', '1gram', 'context']
    columns = ['filtered']
    for i in range(1, M+1):
        columns.append('w'+str(i))
    word_df = pd.DataFrame(columns = columns)
    for mode in modes:
        word_df = add_l1_of_p_mode(e, word, word_df, mode)
    #print(word_df)
    return word_df


def expectation(word_df): 
    print("expectation word_df: {word_df}")
    word_df.drop(columns= word_df.columns[0], axis=1, inplace=True)
    print(f"{word_df}")
    all_l1 = word_df.to_numpy().flatten()
    weighted_avg = all_l1.mean() 
    return weighted_avg

def search_fre_for_word(w):
    fre_df = pd.read_csv('/home/gy654/Documents/preprocess/3_corpus_ratio_fre.csv', index_col=[0])
    return fre_df.loc[w]['train_fre']


def look_up_fre(e, w):
    combined = e.fre_ref
    if combined.index.name != 'word':
        combined.set_index('word', inplace = True)
    return combined.loc[w]['train_fre'], combined.loc[w]['test_fre']




def islist(a):
    if a.isinstance(list):
        return True
    elif a.isinstance(str):
        return False
    else:
        print("Error: not list nor str")
        sys.exit()
        

# decide if we need to calculate the word's l1 or look it up. Seperate the wordlist at the beginning
def calculate_or_look_up_words_l1(e):
    if e.multiple_swap:
        swap = 'multiple'
    else:
        swap = 'onetime'
        # reference file name ex: tfidf_onetime_reference.csv
    file_path = f'{e.model_type}_{swap}_reference.csv'
    REFERENCE_PATH = e.reference_dir + file_path
    # make a list if the input to Experiment is not a list
    wl = e.wl
    try:
        reference = pd.read_csv(REFERENCE_PATH, index_col =[0])
        print(f'what is reference: {reference}')
    except:
        reference = pd.DataFrame()
    look_up_wl = []
    calculate_wl = []
    for w in wl:
        if w in list(reference.index):
            look_up_wl.append(w)
        else:
            calculate_wl.append(w)
    print(f'look up what: {look_up_wl}, calculate: {calculate_wl}')
    return look_up_wl, calculate_wl, reference

def calculate_df_summary(e, calculate_wl):
    if calculate_wl == []:
        df_summary = pd.DataFrame(columns = ['train_fre', 'test_fre', 'sensitivity_score'])
    else:
        mc_l1 = []
        train_fre_l = []
        test_fre_l = [] 
        for word in calculate_wl:
            print(f'what is word: {word}')
            word_df = gen_df_for_word(e, word) 
            print(f'word_df: {word_df}')
            word_l1_avg = expectation(word_df) 
            mc_l1.append(word_l1_avg) # append the dissimilarity score
            train_fre, test_fre = look_up_fre(e, word)
            train_fre_l.append(train_fre)
            test_fre_l.append(test_fre)
        df_summary = pd.DataFrame({'train_fre':train_fre_l ,'test_fre':test_fre_l,'sensitivity_score':mc_l1}, index = calculate_wl)
    return df_summary


def add_look_up_words_to_df_summary(reference, look_up_wl, df_summary):
    if look_up_wl == []:
        return df_summary
    else:
        print(f"WHAT IS reference index: {reference.index}")
        look_up_data = []
        look_up_index = []
        for word in look_up_wl:
            train_fre = reference.loc[word][0]
            test_fre = reference.loc[word][1]
            mc_l1 = reference.loc[word][2]
            look_up_data.append([train_fre,test_fre, mc_l1])
            look_up_index.append(word)
        df2 = pd.DataFrame(look_up_data, columns=['train_fre','test_fre','sensitivity_score'], index= look_up_index)
        df_summary = df_summary.append(df2)
        return df_summary # all k words's dissimilarity score and frequency, including the ones already calculated


# returns: return the entire df_summary incuding all the other k words, and their ['index', 'train_fre', 'test_fre','sensitivity_score', 'rank']
def rank_the_combined_df_summary(df_summary):
    df_summary.reset_index(inplace=True)
    df_summary.columns = ['index', 'train_fre', 'test_fre','sensitivity_score']
    #df_summary['train_fre'] = df_summary['index'].apply(search_fre_for_word)
    #df_summary['train_ratio'] = df_summary['index'].apply(search_ratio_for_word)
    df_summary.set_index('index', drop=True, inplace=True)
    print(f'what is df_summary{df_summary}')
    df_summary['rank'] = df_summary['sensitivity_score'].rank(ascending=False).astype(int)
    print(f'df_summary : {df_summary}')
    return df_summary


def rank_reference(reference_path):
    df_reference = pd.read_csv(reference_path)
    df_reference.columns = ['word', 'train_fre', 'test_fre','sensitivity_score', 'rank']
    df_reference['rank'] = df_reference['sensitivity_score'].rank(ascending=False).astype(int)
    df_reference.sort_values(by = 'rank', inplace = True)
    return df_reference


def get_calculated_word_l1_and_ranking(calculate_wl, df_summary):
    l1_rank_dict = {}
    for word in calculate_wl:
        word_l1 = df_summary.loc[word]['sensitivity_score']
        word_rank = df_summary.loc[word]['rank'] # rank not based on the existing words in the record file, but the k neightbors of this word
        l1_rank_dict[word] = [word_l1, word_rank]
    return l1_rank_dict


def add_info_to_file(e, l1_rank_dict):
    if e.multiple_swap:
        swap = 'multiple'
    else:
        swap = 'onetime'
    REFERENCE_PATH = e.reference_dir + f'{e.model_type}_{swap}_reference.csv'
    if l1_rank_dict == {}:
        return
    else:
        if not os.path.exists(REFERENCE_PATH):
            header=['word','train_fre','test_fre','sensitivity_score','rank']
            pd.DataFrame([header]).to_csv(REFERENCE_PATH, header= None, index = False)

        with open(REFERENCE_PATH) as r:
            text = r.read()
        added_str = ''
        for word, l1_rank_list in l1_rank_dict.items():
            # word, frequency, l1, relative rank
            train_fre, test_fre = look_up_fre(e, word)
            mc_l1 = l1_rank_list[0]
            relative_rank = l1_rank_list[1]
            added_str = added_str + f'{word}, {train_fre}, {test_fre}, {mc_l1} ,{relative_rank}\n'
        with open(REFERENCE_PATH, 'a') as f:
            if not text.endswith('\n'):
                f.write('\n')
            f.write(added_str)
            print(f"line added to {REFERENCE_PATH}")


def order_file_by_fre(e):
    REFERENCE_PATH = e.reference_dir
    if e.multiple_swap:
        swap = 'multiple'
    else:
        swap = 'onetime'
        
    REFERENCE_PATH = REFERENCE_PATH+f'{e.model_type}_{swap}_reference.csv'
    reference = pd.read_csv(REFERENCE_PATH, index_col = [0])
    reference.sort_values(by = ['test_fre'], inplace = True)
    reference.to_csv(REFERENCE_PATH)


# the rerank_reference is independent of the experiment, for future use.
def rerank_reference(reference_path): 
    reference= pd.read_csv(reference_path)
    print(reference)
    reference['rank'] = reference['sensitivity_score'].rank(ascending=False).astype(int)
    reference = reference.sort_values(by = 'rank', ascending= True)
    print(reference)
    reference = reference.reset_index()
    reference = reference.drop(['index'], axis = 1)
    big_more_sensitive_list = []
    print(reference.columns)
    wl = list(reference['Unnamed: 0'])
    train_fre = list(reference['train_fre'])
    for i in range(len(wl)):
        fre = train_fre[i]
        more_sensitive_list = []
        for j in range(i+1, len(wl)):
            if train_fre[j]> fre:
                more_sensitive_list.append(wl[j])
        big_more_sensitive_list.append(more_sensitive_list)
    reference['more_sensitive_wl'] = big_more_sensitive_list 
    print(reference)
    return reference


# new_main() when given a word, it returns the relative rank of the model's sensitivity to this word compared with k other 
# words with the closest frequency.
def new_main(e, combined):
    # we have tested "expired", "vaccination", "mg", "death", 
    #wl = ['ulcer','diabetes','collapse', 'necrosis', 'wheeze', 'malnutrition', "expired", "vaccination", "mg", "death"]
    wl = ['expired', 'mg', 'diabetes']
    for w in wl:
        k = 10
        modes = ['uniform', '1gram', 'context']
        word_list = find_neighboring_words(combined, w, k)
        print(f'wordlist: {word_list}')
        make_dataset(e)
        look_up_wl, calculate_wl, reference = calculate_or_look_up_words_l1(word_list)
        #print(f'look up and calculate: {look_up_wl}, {calculate_wl}')
        df_summary = calculate_df_summary(e, calculate_wl)
        df_summary = add_look_up_words_to_df_summary(reference, look_up_wl, df_summary)
        df_summary = rank_the_combined_df_summary(df_summary)
        l1_rank_dict = get_calculated_word_l1_and_ranking(calculate_wl, df_summary)
        add_info_to_file(e, combined, l1_rank_dict)
        order_file_by_fre(e)



def sensitivity_on_wl(e):
    word_list = e.wl
    print(f'wordlist: {word_list}')
    make_dataset(e) # check if the datasets are made, if not, make them
    look_up_wl, calculate_wl, reference = calculate_or_look_up_words_l1(e)
    df_summary = calculate_df_summary(e, calculate_wl)
    df_summary = add_look_up_words_to_df_summary(reference, look_up_wl, df_summary)
    df_summary = rank_the_combined_df_summary(df_summary)
    #print(f'summary data frame for all words we select:{df_summary}\n\n\n')
    l1_rank_dict = get_calculated_word_l1_and_ranking(calculate_wl, df_summary)
    add_info_to_file(e, l1_rank_dict)
    order_file_by_fre(e)
    return df_summary



