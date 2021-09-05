import os
import spacy
import numpy as np
import random

nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def get_samples(data, n_samples):
    indices = np.random.choice(np.arange(len(data)), size=n_samples)
    sampled_data = [data[idx] for idx in indices]
    return sampled_data

def write_data_to_file(data, filename):
    with open(filename, "w") as fout:
        for d in data:
            fout.write(d.strip("\n")+"\n")

def get_clean_review(review: str):
    review = review.replace("\n", " ")
    review = re.sub(' +', ' ', review)
    review = review.strip("\n")
    return review

def write_file(sents, out_file):
    with open(out_file, "w") as fout:
        for s in sents:
            fout.write(s.strip("\n")+"\n")

def read_file(filename, empty_line=False):
    reviews = []    
    with open(filename, "r") as fin:    
        for line in fin:
            if not empty_line:
                if line.strip():
                    line = line.strip("\n")
            else:
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