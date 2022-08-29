#!/usr/bin/env python
# coding: utf-8



from looper import * 
from looper_data import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text as sk_text
import shutil
from omegaconf import OmegaConf




class Experiment:
    def __init__( self, wl, input_dir,  model_type, model_path , pretrain_path, mid_dir, output_dir , m , mul, cal):
        self.model_type = model_type
        self.input_dir = input_dir

        self.mid_dir = mid_dir
        if mid_dir == 'default':
            self.mid_dir = '/'.join(input_dir.split('/')[:-2])+'/mid_dir/'
            if not os.path.exists(self.mid_dir):
                os.mkdir(self.mid_dir)
        
        self.out_dir = output_dir
        if output_dir == 'default':
            self.out_dir = '/'.join(input_dir.split('/')[:-2])+'/out_dir/'
            if not os.path.exists(self.out_dir):
                os.mkdir(self.out_dir)

        self.calibrate = cal
        if cal == 'default':
            self.calibrate = True
        
        self.multiple_swap = mul
        if mul == 'default':
            self.multiple_swap = False
        
        self.model_path = model_path

        self.M = m
        if m == 'default':
            self.M = 5
        
        self.wl = self.make_sure_list(wl) # list, len>=1

        self.filter_dir = self.mid_dir + "filtered/"
        if not os.path.exists(self.filter_dir):
            os.mkdir(self.filter_dir)


        self.perturbed_dir = self.mid_dir + "perturbed/"
        if not os.path.exists(self.perturbed_dir):
            os.mkdir(self.perturbed_dir)
        if not os.path.exists(self.perturbed_dir+'multiple/'):
            os.mkdir(self.perturbed_dir+'multiple/')
        if not os.path.exists(self.perturbed_dir+'onetime/'):
            os.mkdir(self.perturbed_dir+'onetime/')
        self.reference_dir = self.mid_dir + "reference/"
        if not os.path.exists(self.reference_dir):
            os.mkdir(self.reference_dir)
        self.pretrain_path = pretrain_path 
        self.fre_ref = None
        self.output = None
        # ...
        pass

    def make_sure_list(self, wl):
        if not isinstance(wl, list):
            wl = [wl]
        return wl
    
    # the main function to check sensitivity score and relative rankings of a wordlist. return a df that contains all information
    def check_sensitivity_for_wl(self):
        self.set_fre_ref() 
        # sensitivity_on_wl is from looper.py
        output = sensitivity_on_wl(self)
        output_path = self.out_dir + 'out.csv'
        output.to_csv(output_path)
        ranked_reference = self.read_reference()
        print(f'Result for this experiment:\n{output}')
        return output, ranked_reference

    
    # the function to gain insights from the reference saved from past experiment. If no past experience, the function will return null
    # the function is bug-free for now



    
    def read_reference(self):
        if self.multiple_swap:
            swap = 'multiple'
        else:
            swap = 'onetime'
        # reference file name ex: tfidf_onetime_reference.csv
        reference_path = f'{self.reference_dir}{self.model_type}_{swap}_reference.csv'
        try:
            # rank_reference is from looper.py, delete sd in later versions
            ranked_reference = rank_reference(reference_path)
            print(f'Experiment Setting: {self.model_type}+{swap}')
            print(f'See Result of All Previous Experiment:\n {ranked_reference}')
            return ranked_reference
        except (FileNotFoundError, IOError) as exc:
            print('No previous experiment or reference file not found')
            return None


    # provided the corpus, the method returns a reference_df that contains the occurences of each word in the training and test corpus. bug_free now
    def set_fre_ref(self):
        classes = ["negative", "positive"]
        SAVE_FRE_PATH = self.mid_dir+'/corpus_fre.csv'

        if os.path.exists(SAVE_FRE_PATH):
            combined = pd.read_csv(SAVE_FRE_PATH, index_col = [0])
        else:
            custom_words = []#["patient","pt","date","discharge","admission","history","hospital", "md", "right", "left","mg","normal","cm"]
            my_stop_words = sk_text.ENGLISH_STOP_WORDS.union(custom_words)
            vec = CountVectorizer(stop_words=my_stop_words,
                                preprocessor=lambda x: re.sub(r'(\d[\d\.])+', '', x.lower()),
                                ngram_range=(1,1))

            train_path = self.input_dir+'train.csv'
            test_path = self.input_dir+'test.csv'
            train_corpus = read_corpus(train_path)
            test_corpus = read_corpus(test_path)

            X_train = vec.fit_transform(train_corpus)
            X_train = X_train.toarray()
            bow_train = pd.DataFrame(X_train,columns = vec.get_feature_names_out())
            bow_train.index = classes
            train_fre = X_train[0] + X_train[1]


            train_df = pd.DataFrame(list(train_fre),index=vec.get_feature_names_out())
            train_df.reset_index(inplace = True)
            train_df.rename(columns = {'index':'word', 0: 'train_fre'}, inplace= True)

            X_test = vec.fit_transform(test_corpus)
            X_test = X_test.toarray()
            bow_test = pd.DataFrame(X_test,columns = vec.get_feature_names_out())
            bow_test.index = classes
            test_fre = X_test[0] + X_test[1]

            test_df = pd.DataFrame(list(test_fre),index=vec.get_feature_names_out())
            test_df.reset_index(inplace = True)
            test_df.rename(columns = {'index':'word', 0: 'test_fre'}, inplace= True)

            combined = pd.merge(train_df, test_df, on='word',how='outer')
            combined.to_csv(SAVE_FRE_PATH)
        self.fre_ref = combined
        return combined

def check_input_error(conf):
    if conf.model_type not in ['tfidf', 'finetune']:
        print('Unsupported model type!')
        sys.exit()
    if len(conf.wl.split(','))==1:
        print('Warning: recommend at least two words to get relative rankings.')
    try:
        files_in_input_dir = os.listdir(conf.input_dir[:-1])
    except:
        print('Input directory cannot be located')
        sys.exit()
    if 'train.csv' not in files_in_input_dir or 'val.csv' not in files_in_input_dir or 'test.csv' not in files_in_input_dir:
        print('The input directory should contain three files: train.csv, val.csv, test.csv')
        sys.exit()
    if conf.mid_dir != 'default':
        if not os.path.exists(conf.mid_dir):
            print('Mid-dir cannot be located.')
            sys.exit()
    if conf.output_dir != 'default':
        if not os.path.exists(conf.output_dir):
            print('Output directory cannot be located.')
            sys.exit()
    if not os.path.exists(conf.model_path):
        print('The model cannot be located.')
        sys.exit()
    if conf.model_type == 'finetune':
        if not os.path.exists(conf.pretrain_path):
            print('The pretrain model cannot be located.')
            sys.exit()
    print('Parameters condition satisfied!')


# optional: all files used during the calculation is deleted after the experiment if called. 
def clear_cache(self):
    mid_dir = self.mid_dir[:-1]
    out_dir = self.out_dir[:-1]
    shutil.rmtree(mid_dir)
    shutil.rmtree(out_dir)
    print('Cache is cleared from the server.')


# the highest-level function up till now
# @param: the config_path
# @returns: the sensitivity scores of the words in the word_list provided; the sensitivity score of all words calculated hitherto
def Sensitivity(config_path):
    try:     
        conf = OmegaConf.load(config_path)
    except FileNotFoundError:
        print("Config file cannot be located.")
        sys.exit()
    
    check_input_error(conf)
    e = Experiment(model_type = conf.model_type,
                    wl =  conf.wl.split(','),
                    input_dir = conf.input_dir,
                    mid_dir = conf.mid_dir,
                    output_dir = conf.output_dir,
                    model_path = conf.model_path,
                    pretrain_path = conf.pretrain_path, 
                    m = conf.M, 
                    mul = conf.multiple_swap, 
                    cal = conf.calibrate)

    e.check_sensitivity_for_wl()



CONFIG_PATH =  '/home/gy654/Documents/text_classification/experiments/sensitivity_analysis/api_config.yaml'
Sensitivity(CONFIG_PATH)





