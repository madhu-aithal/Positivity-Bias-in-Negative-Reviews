import os
import spacy
import numpy as np
import random
import torch
from scipy import stats

nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def filter_plot_data(plot_data): 
    amazon_names = ['Pet Supplies', 'Luxury Beauty', 'Automotive', 'Cellphones', 'Sports']
    amazon_names = [val.lower() for val in amazon_names]   
    plot_data_amz = []
    plot_data_non_amz = []
    for d in plot_data:
        if d['name'].lower() in amazon_names:
            plot_data_amz.append(d)
        else:
            plot_data_non_amz.append(d)

    return plot_data_amz, plot_data_non_amz

def get_samples(data, n_samples, seed_val=23):
    np.random.seed(seed_val)
    if n_samples == None:
        return data
        
    indices = np.random.choice(np.arange(len(data)), size=min(len(data),n_samples), replace=False)
    sampled_data = [data[idx] for idx in indices]
    return sampled_data

def filter_samples(data, max_token_length):
    selected_data = []
    for text in data:
        if len(text.split()) <= max_token_length:
            selected_data.append(text)
    return selected_data

def write_file(sents, out_file):
    with open(out_file, "w") as fout:
        for s in sents:
            fout.write(s.strip("\n")+"\n")

def read_file(filename):
    reviews = []    
    with open(filename, "r") as fin:    
        for line in fin:
            line = line.strip("\n")
            reviews.append(line)
        return reviews

def get_sents(reviews, min_no_of_tokens: int = 5):
    all_sents = []    
    for review in reviews:
        doc = nlp(review)
        for sent in doc.sents:
            tokens = tokenizer(sent.string.strip())
            if len(tokens) >= min_no_of_tokens:
                all_sents.append(sent.string.strip().strip("\n"))

    return all_sents


# Get SEM value based on analysis type. 
# If 'sent_level', then compute sem using per-sentence counts
def get_sem_value(a_type, sent_counts, token_count_a_type):
    if a_type == 'sent_level':
        sem_value = stats.sem(sent_counts)
    else:
        sem_value = stats.sem(token_count_a_type)
    return sem_value

def get_ttest_value(a_type, pos_sent_counts, neg_sent_counts, 
    pos_token_count_a_type, neg_token_count_a_type):

    if a_type == 'sent_level':
        ttest_value = stats.ttest_ind(pos_sent_counts, neg_sent_counts, equal_var = False)
    else:
        ttest_value = stats.ttest_ind(pos_token_count_a_type, 
            neg_token_count_a_type, equal_var = False)
            
    return ttest_value

def get_mean_value(a_type, sent_counts, token_count_a_type):
    if a_type == 'sent_level':
        sem_value = np.mean(sent_counts)
    else:
        sem_value = np.mean(token_count_a_type)
    return sem_value