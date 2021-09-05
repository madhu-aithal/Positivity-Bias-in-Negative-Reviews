import pickle
import numpy as np
import spacy
import sys
from scipy import stats
import pprint
import json
import os
from pathlib import Path
import util
import argparse
import random

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint
nlp = spacy.load("en_core_web_md")

def compute_dataset_length(data):    
    data_filepath = data["data_filepath"]
    all_reviews = []
    
    with open(data_filepath, "r") as fin:
        for line in fin:
            all_reviews.append(line.strip("\n"))

        n_samples = None

        if "n_samples" in data:
            n_samples = int(data["n_samples"])
        else:
            n_samples = len(all_reviews)

        if n_samples == None:
            n_samples = len(all_reviews)

        random.shuffle(all_reviews)
        selected_reviews = all_reviews[:min(len(all_reviews),n_samples)]
        all_reviews_data = []
        
        for review in selected_reviews:
            token_counts_per_sentence = []
            doc = nlp(review)
            no_of_tokens = len(doc)
            no_of_sents = 0
            
            for sent in doc.sents:                 
                no_of_sents += 1
                token_counts_per_sentence.append(len(sent))

            all_reviews_data.append({
                "no_of_tokens": no_of_tokens,
                "no_of_sents": no_of_sents,                
                "token_counts_per_sentence": token_counts_per_sentence
            })

        token_counts_review_level = list(map(lambda x:x['no_of_tokens'], all_reviews_data))
        token_counts_sent_level = list(map(lambda x:1.0*x['no_of_tokens']/x["no_of_sents"], all_reviews_data))        
        token_counts_per_sentence = [val for x in all_reviews_data for val in x['token_counts_per_sentence']]

        return token_counts_review_level, token_counts_sent_level, token_counts_per_sentence

if __name__ == "__main__": 
    '''
    Command to run this file:
    python code/dataset_length_dist.py --datasets_info_json="./input.json"
    '''
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--datasets_info_json",
                        default='./input.json',
                        type=str,
                        required=False,
                        help="JSON file with all datasets information")
    parser.add_argument("--saves_dir_name",
                        default="./saves",
                        type=str,
                        help="Main saves directory name")
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="")
    
    args = parser.parse_args()
    myprint(f"args: {args}")

    random.seed(args.seed_val)
    
    PICKLE_SAVE_PREFIX = "dataset_length_dist"

    saves_dir = os.path.join(args.saves_dir_name, "dataset_length")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)
    plot_data={
        "sent_level": [],
        "review_level": []
    }
    analysis_types = list(plot_data.keys())
    
    datasets = json.loads(open(args.datasets_info_json, "r").read())

    for data in datasets:        
        # Analysis of positive reviews
        token_counts = {}        
        token_counts["review_level"], token_counts["sent_level"], token_counts_per_sentence = compute_dataset_length(data["positive"])
        for analysis_type in analysis_types:            
            plot_data[analysis_type].append({
                "category": "positive reviews",
                "name": data["name"],
                "value": np.mean(token_counts[analysis_type]),
                "sem_value": util.get_sem_value(analysis_type, token_counts_per_sentence, token_counts[analysis_type]),
                "all_samples_data": token_counts[analysis_type]
            })

        # Analysis of negative reviews
        token_counts = {}        
        token_counts["review_level"], token_counts["sent_level"], token_counts_per_sentence = compute_dataset_length(data["negative"])
        for analysis_type in analysis_types:                        
            plot_data[analysis_type].append({
                "category": "negative reviews",
                "name": data["name"],
                "value": np.mean(token_counts[analysis_type]),
                "sem_value": util.get_sem_value(analysis_type, token_counts_per_sentence, token_counts[analysis_type]),
                "all_samples_data": token_counts[analysis_type]
            })
        
        # Analysis of neutral reviews
        token_counts = {}        
        token_counts["review_level"], token_counts["sent_level"], token_counts_per_sentence = compute_dataset_length(data["neutral"])
        for analysis_type in analysis_types:            
            plot_data[analysis_type].append({
                "category": "neutral",
                "name": data["name"],
                "value": np.mean(token_counts[analysis_type]),
                "sem_value": util.get_sem_value(analysis_type, token_counts_per_sentence, token_counts[analysis_type]),
                "all_samples_data": token_counts[analysis_type]
            })

    pickle.dump({
        "plot_data": plot_data
    }, open(os.path.join(saves_dir, PICKLE_SAVE_PREFIX+".pickle"), "wb"))