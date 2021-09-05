import liwc_util
import re
import pickle
import numpy as np
import pandas as pd
import spacy
import sys
import json
from pathlib import Path
import os
import csv
import util
import pprint
import argparse

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint
nlp = spacy.load("en_core_web_md")

def read_vader_sentiment_dict(filepath):
    vader_sentiment_scores = {}
    with open(filepath, "r") as fin:
        for line in fin:
            values = line.split("\t")
            vader_sentiment_scores[values[0]] = float(values[1])
    return vader_sentiment_scores

def compute_vadersentiment(data, dataset_name, vader_sentiment_scores):
    data_filepath = data["data_filepath"]
    
    n_samples = None
    if "n_samples" in data:
        n_samples = data["n_samples"]
    
    review_data = []    
    with open(data_filepath, "r") as f:
        all_reviews = []
        for rev in f.readlines():
            rev = rev.strip("\n")
            all_reviews.append(rev)

        if n_samples == None:
            n_samples = len(all_reviews)

        indices = np.random.choice(np.arange(len(all_reviews)), size=min(len(all_reviews),n_samples), replace=False)
        selected_reviews = [all_reviews[idx] for idx in indices]
        count = 0
        for rev in selected_reviews:
            rev = rev.lower()
            per_sentence_pos_counts = []
            per_sentence_neg_counts = []
            count += 1
            doc = nlp(rev)            
            token_count = len(doc)
            review_level_count = {}
            sent_count = 0
            pos_words = []
            neg_words = []
            matched_tokens_count = 0

            for sent in doc.sents:  
                sent_count += 1
                temp_pos = 0
                temp_neg = 0
                for idx,token in enumerate(sent):
                    if token.text in vader_sentiment_scores:
                        matched_tokens_count+=1
                        sent_score = vader_sentiment_scores[token.text]
                        if sent_score>=1:
                            pos_words.append(token.text)
                            temp_pos += 1
                        elif sent_score<=-1:
                            neg_words.append(token.text)
                            temp_neg += 1
                per_sentence_pos_counts.append(temp_pos)
                per_sentence_neg_counts.append(temp_neg)

            review_level_count["review_text"] = rev
            review_level_count["total_no_of_tokens"] = token_count
            review_level_count["total_no_of_sents"] = sent_count                        
            review_level_count["neg_words"] = neg_words
            review_level_count["pos_words"] = pos_words
            review_level_count["neg_words_count"] = len(neg_words)
            review_level_count["pos_words_count"] = len(pos_words)
            review_level_count["matched_tokens_count"] = matched_tokens_count
            review_level_count["per_sentence_pos_counts"] = per_sentence_pos_counts
            review_level_count["per_sentence_neg_counts"] = per_sentence_neg_counts
            
            review_data.append(review_level_count)

        return review_data

def compute_vadersentiment_util(data, name, vader_sentiment_scores, category,
    plot_data, analysis_types):

    analysis_data = compute_vadersentiment(data, name, vader_sentiment_scores)
    per_sentence_neg_counts = []
    per_sentence_pos_counts = []

    for analysis in analysis_types:
        if analysis == "review_level":   
            pos_count_normalized = list(map(lambda x:x['pos_words_count'], analysis_data))        
        elif analysis == "sent_level":  
            pos_count_normalized = list(map(lambda x:1.0*x['pos_words_count']/x["total_no_of_sents"], analysis_data))
            per_sentence_pos_counts = list(map(lambda x:x['per_sentence_pos_counts'], analysis_data))
            per_sentence_pos_counts = [val for x in per_sentence_pos_counts for val in x]
        else:
            pos_count_normalized = list(map(lambda x:1.0*x['pos_words_count']/x["total_no_of_tokens"], analysis_data))        

        if analysis == "review_level":   
            neg_count_normalized = list(map(lambda x:x['neg_words_count'], analysis_data))        
        elif analysis == "sent_level":              
            neg_count_normalized = list(map(lambda x:1.0*x['neg_words_count']/x["total_no_of_sents"], analysis_data))       
            per_sentence_neg_counts = list(map(lambda x:x['per_sentence_neg_counts'], analysis_data))
            per_sentence_neg_counts = [val for x in per_sentence_neg_counts for val in x]

        else:
            neg_count_normalized = list(map(lambda x:1.0*x['neg_words_count']/x["total_no_of_tokens"], analysis_data))
           
        plot_data[analysis].append({
            "category": "positive - "+category+" review ",
            "review category": category,
            "text sentiment": "positive",
            "name": name,
            "value": np.mean(pos_count_normalized),
            "sem_value": util.get_sem_value(analysis, per_sentence_pos_counts, pos_count_normalized),
            "value_per_sentence_counts": util.get_mean_value(analysis, per_sentence_pos_counts, pos_count_normalized),
            "all_samples_data": pos_count_normalized
        })
        plot_data[analysis].append({
            "category": "negative - "+category+" review ",
            "review category": category,
            "text sentiment": "negative",
            "name": name,
            "value": np.mean(neg_count_normalized),
            "sem_value": util.get_sem_value(analysis, per_sentence_neg_counts, neg_count_normalized),
            "value_per_sentence_counts": util.get_mean_value(analysis, per_sentence_neg_counts, neg_count_normalized),
            "all_samples_data": neg_count_normalized
        })


if __name__ == "__main__":
    '''
    Command:
    python code/vader_pos_neg_dist.py --datasets_info_json="./input.json" --vader_lexicon_path=<path_to_vader_lexicon_file>
    '''

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datasets_info_json",
                        default="./input.json",
                        type=str,
                        required=False,
                        help="")
    parser.add_argument("--saves_dir_name",
                        default="saves",
                        type=str,                        
                        help="")   
    parser.add_argument("--vader_lexicon_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to the file containing vader lexicons i.e. vaderSentiment/vaderSentiment/vader_lexicon.txt")
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="Seed value for random function") 
    
    args = parser.parse_args()    
    np.random.seed(args.seed_val)
    myprint(f"args: {args}")

    plot_data = {
        "word_level": [],
        "sent_level": [],
        "review_level": []
    }

    saves_dir = os.path.join(args.saves_dir_name, "vader_pos_neg_dist")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)
    plot_save_prefix = "vader_pos_neg_dist"

    analysis_types = list(plot_data.keys())
    
    vader_sentiment_scores = read_vader_sentiment_dict(args.vader_lexicon_path)
    datasets = json.loads(open(args.datasets_info_json).read())
    
    for data in datasets:
        myprint(data)                
        compute_vadersentiment_util(data["positive"], data["name"], vader_sentiment_scores, "positive", 
            plot_data, analysis_types)
        
        compute_vadersentiment_util(data["negative"], data["name"], vader_sentiment_scores, "negative", 
            plot_data, analysis_types)

        compute_vadersentiment_util(data["neutral"], data["name"], vader_sentiment_scores, "neutral", 
            plot_data, analysis_types)
        print()
        print()

    pickle.dump(plot_data, open(os.path.join(saves_dir, plot_save_prefix+".pickle"), "wb"))