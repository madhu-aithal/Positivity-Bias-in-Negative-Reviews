import numpy as np
import spacy
import json
import gzip
from pathlib import Path
import os
import random
import pprint
import re
import utils

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

def read_json_gz(args: dict()):
    filename = args["data_filepath"]
    output_dir = args["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fout = open(os.path.join(output_dir, "reviews"), "w")    
    
    pos_reviews = []
    neg_reviews = []

    with gzip.open(filename, "rb") as f:
        jl_data = f.read().decode('utf-8')
        jl_data = list(jl_data.split("\n"))
        for line in jl_data:
            if line:
                json_content = json.loads(line)
                if "reviewText" in json_content:
                    if json_content["reviewText"].strip() != "":
                        rev = utils.get_clean_review(json_content["reviewText"])                        
                        rating = float(json_content["overall"])
                        fout.write(rev+"\n")
                        if rating >= 4:                    
                            pos_reviews.append(rev)
                        elif rating <= 2:                    
                            neg_reviews.append(rev)
        return pos_reviews, neg_reviews

if __name__ == "__main__":
    seed_val = 23
    np.random.seed(seed_val)

    data = [
        {
            "data_filepath": "/data/madhu/amazon-reviews-2018/Cell_Phones_and_Accessories_5.json.gz",
            "name": "Amazon Cellphones reviews",
            "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/cellphones_and_accessories/"                        
        },       
        {
            "data_filepath": "/data/madhu/amazon-reviews-2018/Automotive_5.json.gz",
            "name": "Automotive reviews",
            "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/automotive/"                        
        },
        {
            "data_filepath": "/data/madhu/amazon-reviews-2018/Luxury_Beauty_5.json.gz",
            "name": "Luxury Beauty reviews",
            "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/luxury_beauty/"                        
        },
        {
            "data_filepath": "/data/madhu/amazon-reviews-2018/Pet_Supplies_5.json.gz",
            "name": "Pet supplies reviews",
            "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/pet_supplies/"                        
        },
        {
            "data_filepath": "/data/madhu/amazon-reviews-2018/Sports_and_Outdoors_5.json.gz",
            "name": "Sports and Outdoors reviews",
            "output_dir": "/data/madhu/amazon-reviews-2018/processed_data/sports_and_outdoors/"
        }
    ]    
    NO_OF_SAMPLES = int(2*1e4)

    for d in data:        
        pos_reviews, neg_reviews = read_json_gz(d)
        random.shuffle(pos_reviews)
        random.shuffle(neg_reviews)

        train_size_pos = int(len(pos_reviews)*3/4)
        train_size_neg = int(len(neg_reviews)*3/4)

        # Train data - reviews and sentences        
        train_pos_reviews = pos_reviews[:train_size_pos]
        train_neg_reviews = neg_reviews[:train_size_neg]
        utils.write_data_to_file(train_pos_reviews, d["output_dir"]+"/pos_reviews_train")
        utils.write_data_to_file(train_neg_reviews, d["output_dir"]+"/neg_reviews_train")

        train_pos_reviews_samples = utils.get_samples(train_pos_reviews, NO_OF_SAMPLES)
        train_neg_reviews_samples = utils.get_samples(train_neg_reviews, NO_OF_SAMPLES)

        utils.write_data_to_file(train_pos_reviews_samples, d["output_dir"]+"/pos_reviews_train_"+str(NO_OF_SAMPLES))
        utils.write_data_to_file(train_neg_reviews_samples, d["output_dir"]+"/neg_reviews_train_"+str(NO_OF_SAMPLES))
        
        train_pos_sents = utils.get_sents(train_pos_reviews_samples)
        train_neg_sents = utils.get_sents(train_neg_reviews_samples)

        utils.write_data_to_file(train_pos_sents, d["output_dir"]+"/pos_sents_train")
        utils.write_data_to_file(train_neg_sents, d["output_dir"]+"/neg_sents_train")


        # Test data - reviews and sentences (sentences in 20K sampled reviews)
        test_pos_reviews = pos_reviews[train_size_pos+1:]
        test_neg_reviews = neg_reviews[train_size_neg+1:]
        utils.write_data_to_file(test_pos_reviews, d["output_dir"]+"/pos_reviews_test")
        utils.write_data_to_file(test_neg_reviews, d["output_dir"]+"/neg_reviews_test")

        test_pos_reviews_samples = utils.get_samples(test_pos_reviews, NO_OF_SAMPLES)
        test_neg_reviews_samples = utils.get_samples(test_neg_reviews, NO_OF_SAMPLES)
        
        test_pos_sents = utils.get_sents(test_pos_reviews_samples)
        test_neg_sents = utils.get_sents(test_neg_reviews_samples)

        utils.write_data_to_file(test_pos_sents, d["output_dir"]+"/pos_sents_test")
        utils.write_data_to_file(test_neg_sents, d["output_dir"]+"/neg_sents_test")
