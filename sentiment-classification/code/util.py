import torch
import os
import sys
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
import time
from datetime import datetime
import logging
from pathlib import Path
import spacy
import random
import vader_negation_util


nlp = spacy.load("en_core_web_md")

def filter_amazon(plot_data): 
    amazon_names = ['Pet Supplies', 'Luxury Beauty', 'Automotive', 'Cellphones', 'Sports']
    amazon_names = [val.strip().replace(" ", "_").lower() for val in amazon_names]   
    plot_data_amz = []
    plot_data_non_amz = []
    for d in plot_data:
        if d['name'].lower() in amazon_names:
            plot_data_amz.append(d)
        else:
            plot_data_non_amz.append(d)

    return plot_data_amz, plot_data_non_amz


def get_filename(time: int, util_name:str =""):   
    filename = str(time.strftime('%b-%d-%Y_%H-%M-%S'))
    if util_name != "":
        filename = util_name+"_"+filename
    return filename


# If there's a GPU available...
def get_device(device_no: int):
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda:"+str(device_no))

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    
    return device

def read_samples(filename0: str, filename1: str, seed_val:int, n_samples:int = None, sentence_flag:bool=False):
    yelp_reviews_0 = []
    yelp_reviews_1 = []
    with open(filename0, "r") as f:
        yelp_reviews_0 = f.read().splitlines()        

    with open(filename1, "r") as f:
        yelp_reviews_1 = f.readlines()

    seed_val = 23
    np.random.seed(seed_val)

    reviews = []
    labels = []
    if n_samples != None:
        if min(len(yelp_reviews_0), len(yelp_reviews_1)) < n_samples:
            n_samples = min(len(yelp_reviews_0), len(yelp_reviews_1))

        indices = np.random.choice(np.arange(len(yelp_reviews_0)), size=min(len(yelp_reviews_0), n_samples), replace=False)
        yelp_reviews_0_new = [yelp_reviews_0[idx] for idx in indices]

        indices = np.random.choice(np.arange(len(yelp_reviews_1)), size=min(len(yelp_reviews_1), n_samples), replace=False)
        yelp_reviews_1_new = [yelp_reviews_1[idx] for idx in indices]
        
        reviews = yelp_reviews_0_new+yelp_reviews_1_new
        labels = [0]*len(yelp_reviews_0_new) + [1]*len(yelp_reviews_1_new)
    else:
        reviews = yelp_reviews_0+yelp_reviews_1
        labels = [0]*len(yelp_reviews_0) + [1]*len(yelp_reviews_1)
    reviews = [rev.lower() for rev in reviews]
    return reviews, labels

def read_samples_new_util(filename, n_samples, sentence_flag):
    data = []
    with open(filename, "r") as f:
        reviews = f.read().splitlines()
        if n_samples == None:
            n_samples = sys.maxsize
        random.shuffle(reviews)
        for rev in reviews:            
            if sentence_flag:
                doc = nlp(rev)
                for sent in doc.sents:
                    data.append(sent.string)
                    if len(data) >= n_samples:
                        break
                if len(data) >= n_samples:
                    break
            else:
                data.append(rev)                
            if len(data) >= n_samples:
                break
        return data

def read_samples_new(filename0: str, filename1: str, seed_val:int, n_samples:int = None, sentence_flag:bool=False):
    seed_val = 23
    random.seed(seed_val)
    reviews_0 = read_samples_new_util(filename0, n_samples, sentence_flag)
    reviews_1 = read_samples_new_util(filename1, n_samples, sentence_flag)
    labels = [0]*len(reviews_0) + [1]*len(reviews_1)
    reviews = reviews_0+reviews_1
    reviews = [rev.lower() for rev in reviews]
    return reviews, labels


def has_negation(text):
    words = text.strip("\n").split()
    neg_count = vader_negation_util.negated(words)
    return neg_count
        
def read_file(filename):
    reviews = []    
    with open(filename, "r") as fin:    
        for line in fin:
            line = line.strip("\n")
            reviews.append(line)
        return reviews