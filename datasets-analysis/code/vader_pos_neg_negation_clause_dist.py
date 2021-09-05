import re
import pickle
import numpy as np
import spacy
import sys
import vader_negation_util
import pprint
import json
from pathlib import Path
import os
import util
import argparse
import re
import string

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

def check_sentiment_in_clause(clause, vader_sentiment_scores):
    pos = False
    neg = False
    for word in clause.split():
        sent_score = vader_sentiment_scores[word]
        if sent_score >= 1:
            pos = True
        if sent_score <= -1:
            neg = True    
    return pos, neg

def compute_vadersentiment(data, dataset_name, vader_sentiment_scores, saves_dir):   
    data_filepath = data["data_filepath"]    
    Path(saves_dir).mkdir(parents=True, exist_ok=True)   
    save_pickle_path = os.path.join(saves_dir, dataset_name+"_vader_pos_neg_negation_dist.pickle")

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

        for rev in selected_reviews:            
            rev = rev.lower()
            per_sentence_negation_pos_count = []
            per_sentence_negation_neg_count = []
            doc = nlp(rev)            
            token_count = len(doc)
            review_level_count = {}
            sent_count = 0
            negation_pos_count = 0
            negation_neg_count = 0
            posemo_words = list()
            negemo_words = list()
            matched_tokens_count = 0
            clause_count = 0
            
            for sent in doc.sents:
                sent_count += 1
                temp_negation_pos_count = 0
                temp_negation_neg_count = 0

                clauses = re.split(r'['+string.punctuation+']', sent.string)
                for clause in clauses:
                    clause_count += 1
                    negation_flag = False
                    pos_count = 0
                    neg_count = 0
                    for word in clause.split():
                        if word in vader_negation_util.NEGATE or "n't" in word:
                            negation_flag = True

                        if word in vader_sentiment_scores:
                            matched_tokens_count += 1
                            sent_score = vader_sentiment_scores[word]
                            if sent_score >= 1:
                                pos_count += 1
                            if sent_score <= -1:
                                neg_count += 1
                    if negation_flag:                        
                        if pos_count>0:
                            negation_pos_count += 1
                            temp_negation_pos_count += 1
                        if neg_count>0:
                            negation_neg_count += 1
                            temp_negation_neg_count += 1

                per_sentence_negation_pos_count.append(temp_negation_pos_count)
                per_sentence_negation_neg_count.append(temp_negation_neg_count)

            review_level_count["total_no_of_tokens"] = token_count
            review_level_count["total_no_of_sents"] = sent_count
            review_level_count["total_no_of_clauses"] = clause_count            
            review_level_count["negation_posemo"] = negation_pos_count
            review_level_count["negation_negemo"] = negation_neg_count
            review_level_count["per_sentence_negation_pos_count"] = per_sentence_negation_pos_count
            review_level_count["per_sentence_negation_neg_count"] = per_sentence_negation_neg_count            
            review_data.append(review_level_count)

        pickle.dump({
            "negation_analysis_data": review_data
        }, open(save_pickle_path, "wb"))
        
        return review_data

def compute_vadersentiment_util(data, name, vader_sentiment_scores, category, 
    plot_data, analysis_types, saves_dir):
    analysis_data = compute_vadersentiment(data, name, vader_sentiment_scores, saves_dir)    
    per_sentence_negation_neg_counts = []
    per_sentence_negation_pos_counts = []
    for analysis in analysis_types:
        if analysis == "review_level":   
            negation_pos_count_normalized = list(map(lambda x:x['negation_posemo'], analysis_data))        
        elif analysis == "sent_level":  
            negation_pos_count_normalized = list(map(lambda x:1.0*x['negation_posemo']/x["total_no_of_sents"], analysis_data))       
            per_sentence_negation_pos_counts = list(map(lambda x:x['per_sentence_negation_pos_count'], analysis_data))
            per_sentence_negation_pos_counts = [val for x in per_sentence_negation_pos_counts for val in x]
        else:
            negation_pos_count_normalized = list(map(lambda x:1.0*x['negation_posemo']/x["total_no_of_tokens"], analysis_data))        

        if analysis == "review_level":   
            negation_neg_count_normalized = list(map(lambda x:x['negation_negemo'], analysis_data))        
        elif analysis == "sent_level":  
            negation_neg_count_normalized = list(map(lambda x:1.0*x['negation_negemo']/x["total_no_of_sents"], analysis_data))       
            per_sentence_negation_neg_counts = list(map(lambda x:x['per_sentence_negation_neg_count'], analysis_data))
            per_sentence_negation_neg_counts = [val for x in per_sentence_negation_neg_counts for val in x]
        else:
            negation_neg_count_normalized = list(map(lambda x:1.0*x['negation_negemo']/x["total_no_of_tokens"], analysis_data))
          
           
        plot_data[analysis].append({
            "category": "positive - "+category+" review ",
            "review category": category,
            "text sentiment": "positive",
            "name": name,
            "value": np.mean(negation_pos_count_normalized),
            "sem_value": util.get_sem_value(analysis, per_sentence_negation_pos_counts, negation_pos_count_normalized),
            "value_per_sentence_counts": util.get_mean_value(analysis, per_sentence_negation_pos_counts, negation_pos_count_normalized),
            "all_samples_data": negation_pos_count_normalized
        })
        plot_data[analysis].append({
            "category": "negative - "+category+" review ",
            "review category": category,
            "text sentiment": "negative",
            "name": name,
            "value": np.mean(negation_neg_count_normalized),
            "sem_value": util.get_sem_value(analysis, per_sentence_negation_neg_counts, negation_neg_count_normalized),
            "value_per_sentence_counts": util.get_mean_value(analysis, per_sentence_negation_neg_counts, negation_neg_count_normalized),
            "all_samples_data": negation_neg_count_normalized
        })

if __name__ == "__main__": 
    '''
    Command:
    python code/vader_pos_neg_negation_clause_dist.py --datasets_info_json="./input.json" 
        --vader_lexicon_path=<path_to_vader_lexicon_file>
    '''

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--datasets_info_json",
                        default="./input.json",
                        type=str,
                        required=False,
                        help="")
    parser.add_argument("--saves_dir_name",
                        default="saves",
                        type=str,
                        # required=True,
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
    myprint(f"args: {args}")   
    np.random.seed(args.seed_val)
    
    amazon_names = ['Pet Supplies', 'Luxury Beauty', 'Automotive', 'Cellphones', 'Sports']

    plot_data = {
        "word_level": [],
        "sent_level": [],
        "review_level": []
    }
    analysis_types = list(plot_data.keys())

    saves_dir = os.path.join(args.saves_dir_name, "vader_pos_neg_negation_clause_dist")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)
    plot_save_prefix = "vader_pos_neg_negation_clause_dist"

    vader_sentiment_scores = read_vader_sentiment_dict(args.vader_lexicon_path)
    datasets = json.loads(open(args.datasets_info_json, "r").read())

    for data in datasets:
        myprint(data)        
        compute_vadersentiment_util(data["positive"], data["name"], vader_sentiment_scores, "positive", 
            plot_data, analysis_types, saves_dir)

        compute_vadersentiment_util(data["negative"], data["name"], vader_sentiment_scores, "negative", 
            plot_data, analysis_types, saves_dir)

        compute_vadersentiment_util(data["neutral"], data["name"], vader_sentiment_scores, "neutral", 
            plot_data, analysis_types, saves_dir)

        print()
        print()

    pickle.dump(plot_data, open(os.path.join(saves_dir, plot_save_prefix+".pickle"), "wb"))
