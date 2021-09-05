import numpy as np
import spacy
import json
import gzip
from pathlib import Path
import os
import random
import pprint
from os import listdir
from os.path import isfile, join
from amazon_read_reviews import get_clean_review
import utils
import argparse

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

def read_file(filename):
    pos_reviews = []
    neg_reviews = []
    with open(filename, "r") as f:    
        json_content = json.load(f)        
        for data in json_content["Reviews"]:
            if data["Content"]:
                ratings = float(data["Ratings"]["Overall"])
                if ratings >= 4:
                    pos_reviews.append(get_clean_review(data["Content"]))
                elif ratings <= 2:
                    neg_reviews.append(get_clean_review(data["Content"]))

        return pos_reviews, neg_reviews


def read_json_gz(args: dict()):
    dataset_dir = args["dataset_dir"]
    output_dir = args["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fout = open(os.path.join(output_dir, "reviews"), "w")    

    dataset_files = [join(dataset_dir, f) for f in listdir(dataset_dir) if isfile(join(dataset_dir, f))]    

    all_pos_reviews = []
    all_neg_reviews = []
    
    for filename in dataset_files:
        pos_reviews, neg_reviews = read_file(filename)
        for rev in pos_reviews:
            fout.write(rev+"\n")
        for rev in neg_reviews:
            fout.write(rev+"\n")
        all_pos_reviews.extend(pos_reviews)
        all_neg_reviews.extend(neg_reviews)

    return all_pos_reviews, all_neg_reviews
   
if __name__ == "__main__": 
    data = [
        {
            "dataset_dir": "/data/madhu/tripadvisor/json_data",
            "name": "Tripadvisor reviews",
            "output_dir": "/data/madhu/tripadvisor/processed_data"                        
        }
    ]    
    n_samples = int(2*1e4)

    for d in data:
        myprint(d)
        pos_reviews, neg_reviews = read_json_gz(d)
        random.shuffle(pos_reviews)
        random.shuffle(neg_reviews)

        train_size_pos = int(len(pos_reviews)*3/4)
        train_size_neg = int(len(neg_reviews)*3/4)

        train_pos_reviews = pos_reviews[:train_size_pos]
        train_neg_reviews = neg_reviews[:train_size_neg]
        utils.write_data_to_file(train_neg_reviews, d["output_dir"]+"/neg_reviews_train")
        utils.write_data_to_file(train_pos_reviews, d["output_dir"]+"/pos_reviews_train")


        
        test_pos_reviews = pos_reviews[train_size_pos+1:]
        test_neg_reviews = neg_reviews[train_size_neg+1:]
        utils.write_data_to_file(test_neg_reviews, d["output_dir"]+"/neg_reviews_test")
        utils.write_data_to_file(test_pos_reviews, d["output_dir"]+"/pos_reviews_test")

        test_pos_reviews_samples = utils.get_samples(test_pos_reviews, n_samples)
        test_neg_reviews_samples = utils.get_samples(test_neg_reviews, n_samples)

        test_pos_sents = utils.get_sents(test_pos_reviews_samples)
        test_neg_sents = utils.get_sents(test_neg_reviews_samples)

        utils.write_data_to_file(test_pos_sents, d["output_dir"]+"/pos_sents_test")
        utils.write_data_to_file(test_neg_sents, d["output_dir"]+"/neg_sents_test")


        
        train_pos_reviews_samples = utils.get_samples(train_pos_reviews, n_samples)
        train_neg_reviews_samples = utils.get_samples(train_neg_reviews, n_samples)

        utils.write_data_to_file(train_pos_reviews_samples, d["output_dir"]+"/pos_reviews_train_"+str(n_samples))
        utils.write_data_to_file(train_neg_reviews_samples, d["output_dir"]+"/neg_reviews_train_"+str(n_samples))

        train_pos_sents = utils.get_sents(train_pos_reviews_samples)
        train_neg_sents = utils.get_sents(train_neg_reviews_samples)

        utils.write_data_to_file(train_pos_sents, d["output_dir"]+"/pos_sents_train")
        utils.write_data_to_file(train_neg_sents, d["output_dir"]+"/neg_sents_train")
