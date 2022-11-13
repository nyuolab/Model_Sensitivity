# Contacting the Authors
Grace Yang: gy654@nyu.edu
Ming Cao: mc7787@nyu.edu


# Model_Sensitivity

Sensitivity: a library for evaluating a nlp classification model's sensitivity score to a word.

The nlp models are usually considered as black boxes making classifications that is challenging to interpret.
To further enhance the model’s explainability and detect bias if exists, we present a sensitivity analysis method.
In our recent paper, we show how the metric is designed and implemented.

This repo contains a library called Sensitivity that computes the sensitivity scores and their relative rankings for a list of word given by the user.

Arguments:

```
wl(list): a string of words separated by comma. The function will calculate the sensitivity score and relative rankings of those words.

input_dir(string): the classifier will read dataset from the input directory path. The directory should include three csv files ('train.csv', 'val.csv', 'test.csv'), which should be aligned with the dataset the user used to train the classifier
                    Each csv file has two column, the first one is named 'text', the second one is named 'labels'.

model_type('tfidf' or 'finetune') = the classifier's model type (only accept tfidf or language model for now)

model_path(string): the path where the classifier can be loaded

pretrain_path(string): needed when model_type =='finetune', the huggingface path to the pretrained Bert model

optional parameters:

mid_dir(string, optional): the directory path to store all datasets needed during computation. Default to build another directory in the parent directory of the input_dir

out_dir(string, optional): when computation is done, the output (sensitivity result dataframe) will be saved in the directory indicated by out_dir, default to build another directory in the parent directory of the input_dir

M(int, optional): the number of substitutions under one perturbation scheme. M accepts integer smaller than 5 for now. Default set to 5

calibrate(boolean, optional): A boolean value of whether the model's prediction score should be calibrated. Default set to False

multiple_swap(boolean, optional): A boolean value of substitution mode. if multiple_swap set to True, all occurences of a word in a note will be substituted. If set to False, only one occurences of a word in a note will be substituted. Defalut set to False

```

Features:
This library is designed for BERT-like model and the TFIDF+XGboost model.

- Bring your own dataset: while we do provide you with a toy dataset originating from MIMIC_III dataset. It is required that you provide the dataset on which your model is trained on.
- Bring your own model: while we do provide you with 2 models: ClinicalBERT finetuned on our readmission task, and tfidf+XGBoost model, it is required that you test with your own model.
- Runtime: when running the metric for the first time on a set of words, it might take some time to generate the perturbations. The runtime decreases significantly in later calls.
- Relativity and interpretability: This metric depicts the relative sensitivity across different words. Since each model may have different levels of sensitivity, this metric is not independent of the model in use. Instead, to make sense of the magnitude of the model sensitivity, please compare the scores across different words using one model consistently.
- Frequency adjustment: The difference between perturbations tend to be larger when the word is more frequent (since the number of changes done is greater). If the target word list has a large standard deviation in terms of word frequency, it is recommend to set “swap” as “onetime” in the config file.  If there are multiple occurrences of the target word in one row of text, the “onetime” option will only perturb the first occurrence.

Example:

example configuration file

```yaml
wl: "cancer,expired,thinner,hypoglycemia,vaccination"

model_type:  'tfidf'

input_dir: '/home/gy654/Documents/API_test2/input_dir/'

model_path: '/home/gy654/Documents/sensitivity_score/checkpoint-11535'

pretrain_path: "emilyalsentzer/Bio_ClinicalBERT"

mid_dir: 'default'

output_dir: 'default'

M: 'default'

multiple_swap: 'default'

calibrate: 'default'
```

Installation:

1. `git clone git@github.com:nyuolab/Model_Sensitivity.git`
2. Go install the [Dependencies](https://github.com/willwhitney/reprieve/blob/master/readme.md#dependencies). Since installations of Pytorch and JAX are both highly context-dependent, I won't include an install script.
4. `pip install -e .`

Dependencies:

```yaml

The standard Python data kit, including numpy, pandas, sklearn, re, nltk, matplotlib, os, sys.
Pytorch:
	torch==1.9.0+cu111
OmegaConf:
	omegaconf==2.2.2
transformers:
	transformers==4.5.0
joblib:
	joblib=1.1.0=pyhd3eb1b0_0

```

Algorithms:

see paper …

# Full API Documentation:

Class Experiment

```python
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
```

Create a Experiment. 

Arguments:

1. wl: the word or word list to test model sensitivity
2. input_dir: the directory that contains train.csv, val.csv, and test.csv used for the model in test
3. model_type: surported model types are: “tfidf” and “finetune”
4. model_path: custom path of the user saved model
5. pretrain_path: If the model type is finetune, meaning BERT, the pretrained_path is the huggingface model path. eg. "emilyalsentzer/Bio_ClinicalBERT”
6. mid_dir(optional): if not specified, default is parallel to the input_dir. This directory saves the intermediate data files used to compute the sensitivity score, including the filtered datasets, perturbed datasets, and previous outputs. 
7. output_dir(optional): if not specified, default is parallel to the input_dir. The output directory contains the outputs of the words provided in wl. 
8. m(optional): number of random perturbations under each scheme. Larger m would increase runtime significantly, it is recommended to choose a value ≤5. If not specified, the default value is 5. 
9. mul (optional): If True, every occurrence of the target word or word list in the corpus is perturbed. If False, only the first occurrence of each target word in each row of the corpus is swapped. If the words in the word list has a large variance in frequency, it is recommend to set mul as False. If not specified, the default is False. 
10. cal(optional): whether to calibrate the model or not. If not specified, The default is True. 

Effects: this class specifies all the necessary parameters for the metric to work. 

### **Functions**

```python
read_reference(e)
```

display all previous experiment results under the same setting. 

Arguments:

- e: (Experiment object) the object e belongs to the class Experiment. e has several attributes: multiple_swap, reference_dir and model_type.

The reference file will be updated after each experiment. The reference df will be stored at different paths depending on the aforementioned attributes of e. The function read_reference(e) will check previous experiment results with the same setting and display them to the user. If the experiment is executed for the first time, the function will print a warning “No previous experiment or reference file not found.” 

```python
set_fre_ref(self)
```

set the fre_ref attribute of object e and returns the fre_ref data frame

Arguments:

- e: (Experiment object) the object e belongs to the class Experiment. e has several attributes: multiple_swap, reference_dir and model_type.

Returns: the fre_ref data frame

the function set_fre_ref(self) counts the occurrences of each word in the training and test corpus respectively and generates a data frame for reference only . 

```python
plot_output(self)
```

plot the output of the experiment and save the plot to the output directory

Arguments:

- e: (Experiment object) the object e belongs to the class Experiment. e has several attributes: model_type, multiple_swap, reference_dir and out_dir.

Returns: a sensitivity_frequency plot

the plot generated by the function offer reader insights about the sensitivity score and the relative ranking of the words they select as compared with their frequency in the corpus. The x-axis of the plot is word frequency and the y-axis of the plot is the word’s sensitivity score. Each scatter point on the plot is labeled with the word it represent. The function also saves a png file in the e.out_dir. 

```python
clear_cache(self):
```

All files used during the calculation is deleted after the experiment if called.

Arguments:

- e: (Experiment object) the object e belongs to the class Experiment. e has several attributes: multiple_swap, reference_dir and model_type.

Lots of files are saved on the user’s server when computing the sensitivity score for words due to the design of the sensitivity metric. Readers has the option to delete all files after obtaining the experiment result. Avoid calling this function if the user intends to test the same wordlist but with different experiment settings to accelerate the computation process. 

```python
check_sensitivity_for_wl(self)
```

Arguments:

- e: (Experiment object) the object e belongs to the class Experiment. e has several attributes: e. wl.

Returns:

1. an output data frame pertaining to this experiment only
2. a reference data frame containing all data of previous experiments

the function calculates the sensitivity scores of a list of words specified in e.wl and rank them according to their sensitivity score.

```python
Sensitivity(config_path)
```

Create and run the experiment based on the config file. Automatically calculate the sensitivity score of all words provided in the config file. 

Arguments:

- config_path: the file path of the config file that specifies all the necessary parameters to calculate the sensitivity scores. The target word or wordlist should be specified in the config file.

Returns: the experiment object created. 

Outputs: this method outputs two data frames and : 1. generate a scatter plot of the result, the y-axis is model sensitivity and x-axis is word frequency; 2. the sensitivity scores of the target words in the config file; 3. the sensitivity scores of all the words tested hitherto.
