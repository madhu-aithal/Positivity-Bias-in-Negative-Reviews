import pickle
import gzip
import json
import pprint
import argparse
import os
import utils
import random

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

def sample_data(args, pos_filepath, neg_filepath, neutral_filepath):
    pos_reviews = utils.read_file(pos_filepath)
    neg_reviews = utils.read_file(neg_filepath)
    neutral_reviews = utils.read_file(neutral_filepath)

    random.shuffle(pos_reviews)
    random.shuffle(neg_reviews)
    random.shuffle(neutral_reviews)
    
    train_size_pos = int(len(pos_reviews)*3/4)
    train_size_neg = int(len(neg_reviews)*3/4)
    train_size_neutral = int(len(neutral_reviews)*3/4)

    train_pos_reviews = pos_reviews[:train_size_pos]
    train_neg_reviews = neg_reviews[:train_size_neg]
    train_neutral_reviews = neutral_reviews[:train_size_neutral]

    utils.write_file(train_pos_reviews, args.out_dir+"/pos_reviews_train")
    utils.write_file(train_neg_reviews, args.out_dir+"/neg_reviews_train")
    utils.write_file(train_neutral_reviews, args.out_dir+"/neutral_reviews_train")

    test_pos_reviews = pos_reviews[train_size_pos+1:]
    test_neg_reviews = neg_reviews[train_size_neg+1:]
    test_neutral_reviews = neutral_reviews[train_size_neutral+1:]

    utils.write_file(test_pos_reviews, args.out_dir+"/pos_reviews_test")
    utils.write_file(test_neg_reviews, args.out_dir+"/neg_reviews_test")
    utils.write_file(test_neutral_reviews, args.out_dir+"/neutral_reviews_test")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--yelp_review_filepath",
                        default=None,
                        type=str,
                        required=True,
                        help="")
    parser.add_argument("--yelp_business_filepath",
                        default=None,
                        type=str,
                        required=True,
                        help="")    
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="")    
    
    args = parser.parse_args() 
    myprint(f"args: {args}")   

    neg_review_filepath = os.path.join(args.output_dir, "pos_reviews")
    pos_review_filepath = os.path.join(args.output_dir, "neg_reviews")
    neutral_review_filepath = os.path.join(args.output_dir, "neutral_reviews")

    restaurant_business_ids = set()
    with gzip.open(args.yelp_business_filepath, "rb") as f:    
        jl_data = f.read().decode('utf-8') 
        jl_data = list(jl_data.split("\n"))
        for line in jl_data:  
            if line:    
                json_content = json.loads(line)
                if json_content["categories"] != None:                        
                    categories = [val.lower().strip() for val in json_content["categories"].split(",")]
                    if "restaurants" in categories:
                        restaurant_business_ids.add(json_content["business_id"])
    
    print("Finished reading the business.json file")
    neg_review_file = open(neg_review_filepath,"w")
    pos_review_file = open(pos_review_filepath,"w")    
    neutral_review_file = open(neutral_review_filepath,"w")    
    
    with gzip.open(args.yelp_review_filepath, "rb") as f:    
        jl_data = f.read().decode('utf-8') 
        jl_data = list(jl_data.split("\n"))
        for line in jl_data:  
            if line:    
                json_content = json.loads(line)
                if json_content["business_id"] in restaurant_business_ids:
                    if float(json_content["stars"]) >= 3:
                        pos_review_file.write(json_content["text"].replace("\n", " ")+"\n")
                    elif float(json_content["stars"]) <= 2:
                        neg_review_file.write(json_content["text"].replace("\n", " ")+"\n")
                    else:
                        neutral_review_file.write(json_content["text"].replace("\n", " ")+"\n")
                        
    print("Finished reading the review.json file")

    sample_data(args, pos_review_filepath, neg_review_filepath, neutral_review_filepath)