import pickle
import spacy
import vader_negation_util
import numpy as np
import pprint
import json
import os
from pathlib import Path
import util
import argparse

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint
nlp = spacy.load("en_core_web_md")

def compute_negation_words_only(args: dict, seed_val: int): 
    np.random.seed(seed_val)
    dataset_file = args["data_filepath"]
    n_samples = None

    with open(dataset_file, "r") as fin:        
        all_reviews = fin.readlines()

        if "n_samples" in args:
            n_samples = int(args["n_samples"])
        else:
            n_samples = len(all_reviews)

        indices = np.random.choice(np.arange(len(all_reviews)), size=min(len(all_reviews),n_samples), replace=False)       

        sampled_reviews = [all_reviews[idx] for idx in indices]

        neg_words_count = []
        rev_count = 0
        
        for rev in sampled_reviews:   
            rev = rev.lower()
            per_sentence_negation_count = []
            rev=rev.strip("\n")
            neg_count = 0
            sents_count = 0
            
            doc = nlp(rev)
            tokens_count = len(doc)
            for sent in doc.sents:
                sents_count += 1
                words = sent.text.strip("\n").split()
                temp_count = vader_negation_util.negated(words)
                per_sentence_negation_count.append(temp_count)
                neg_count += temp_count

            neg_words_count.append({
                "review": rev,
                "negative_words_count": neg_count,
                "sents_count": sents_count,
                "tokens_count": tokens_count,
                "per_sentence_negation_count": per_sentence_negation_count
            })
            rev_count += 1

        return neg_words_count

def compute_negation_util(data_file, name, seed_val, plot_data, category, analysis_types):
    neg_words_count = compute_negation_words_only(data_file, seed_val)
    per_sentence_negation_count = []
    for analysis in analysis_types:
        if analysis == "word_level":   
            neg_words_normalized = list(map(lambda x: x["negative_words_count"]*1.0/x["tokens_count"], neg_words_count))
        elif analysis == "sent_level":   
            neg_words_normalized = list(map(lambda x: x["negative_words_count"]*1.0/x["sents_count"], neg_words_count))
            per_sentence_negation_count = list(map(lambda x: x["per_sentence_negation_count"], neg_words_count))
            per_sentence_negation_count = [val for x in per_sentence_negation_count for val in x]
        else:
            neg_words_normalized = list(map(lambda x: x["negative_words_count"], neg_words_count))
    
        plot_data[analysis].append({
            "name": name,
            "category": category,
            "value": np.mean(neg_words_normalized),
            "sem_value": util.get_sem_value(analysis, per_sentence_negation_count, neg_words_normalized),
            "value_per_sentence_counts": util.get_mean_value(analysis, per_sentence_negation_count, neg_words_normalized),
            "all_samples_data": neg_words_normalized
        })

if __name__ == "__main__":
    '''
    Command to run this file:
    python code/vader_negation_only_dist.py --datasets_info_json="./input.json"
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets_info_json",
                        default="./input.json",
                        type=str,
                        required=False,
                        help="JSON file with all datasets information")
    parser.add_argument("--saves_dir_name",
                        default="saves",
                        type=str,                        
                        help="Main saves directory name")    
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="Seed value for random function")
   
    args = parser.parse_args()    
    myprint(f"args: {args}")   
    np.random.seed(args.seed_val)    

    saves_dir = os.path.join(args.saves_dir_name, "vader_negation_only")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)   
    plot_save_prefix = "vader_negation_only_dist"
        
    plot_data = {
        "word_level": [],
        "sent_level": [],
        "review_level": []
    }
    analysis_types = list(plot_data.keys())

    dataset_files = json.loads(open(args.datasets_info_json).read())

    for data_file in dataset_files:            
        myprint(f"Dataset: {data_file}")            
        compute_negation_util(data_file["positive"], data_file["name"], args.seed_val, plot_data, 
            "positive", analysis_types)
        compute_negation_util(data_file["negative"], data_file["name"], args.seed_val, plot_data, 
            "negative", analysis_types)    
        compute_negation_util(data_file["neutral"], data_file["name"], args.seed_val, plot_data, 
            "neutral", analysis_types)        
        print()
        print()
        
    pickle.dump(plot_data, open(os.path.join(saves_dir, plot_save_prefix+".pickle"), "wb"))        