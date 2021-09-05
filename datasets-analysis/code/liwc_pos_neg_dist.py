import liwc_util
import re
import pickle
import numpy as np
import spacy
import sys
import json
import os
from pathlib import Path
import pprint
import util
import argparse

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

def compute_liwc(plot_data, data, dataset_name, review_category, 
    required_categories, result, class_id, cluster_result, 
    categories, category_reverse, analysis_types):
    
    data_filepath = data["data_filepath"]

    n_samples = None
    if "n_samples" in data:
        n_samples = int(data["n_samples"])
    
    reviews = util.read_file(data_filepath)
    selected_reviews = util.get_samples(reviews, n_samples)
    all_reviews_data = []
    for rev in selected_reviews:
        doc = nlp(rev)            
        token_count = len(doc)
        review_data = {}
        sent_count = 0
        
        negation_pos = 0
        negation_neg = 0
        for category in required_categories:
            review_data[category] = 0

        # Storing counts (LIWC pos/neg) for each sentence. 
        # Eg: If there are 3 sentences in a review, then the corrsponding 'sentence_counts' list length would be 3
        per_sentence_liwc_counts = []

        for sent in doc.sents:  
            sent_count += 1   
            temp_token_counts = {}
            for category in required_categories:
                temp_token_counts[category] = 0
                
            for idx,token in enumerate(sent):
                for category in required_categories:
                    for pattern in cluster_result[category_reverse[category]]:
                        if (pattern.endswith("*") and token.text.startswith(pattern[:-1])) or (pattern==token.text):
                            review_data[category] = review_data.get(category, 0) + 1  
                            temp_token_counts[category] = temp_token_counts.get(category, 0) + 1 
            per_sentence_liwc_counts.append(temp_token_counts)

        review_data["total_no_of_tokens"] = token_count
        review_data["total_no_of_sents"] = sent_count
        review_data["per_sentence_liwc_counts"] = per_sentence_liwc_counts
    
        all_reviews_data.append(review_data)

    category_counts = {}
    for category in required_categories:       
        category_counts["word_level"] = list(map(lambda x:1.0*x[category]/x["total_no_of_tokens"], all_reviews_data))
        category_counts["sent_level"] = list(map(lambda x:1.0*x[category]/x["total_no_of_sents"], all_reviews_data))
        category_counts["review_level"] = list(map(lambda x:x[category], all_reviews_data))
        
        sentence_counts_category = []
        for review_data in all_reviews_data:
            for val in review_data['per_sentence_liwc_counts']:
                if category in val:
                    sentence_counts_category.append(val[category])
        
        for a_type in analysis_types:
            plot_data[a_type].append({                
                "review_category": review_category,
                "liwc_category": category,
                "name": dataset_name,
                "value": np.mean(category_counts[a_type]),
                "sem_value": util.get_sem_value(a_type, sentence_counts_category, category_counts[a_type]),
                "per_sentence_counts": util.get_mean_value(a_type, sentence_counts_category, category_counts[a_type]),
                "all_samples_data": category_counts[a_type]
            })               

    return plot_data

if __name__ == "__main__": 
    '''
    Command:
    python code/liwc_count.py --datasets_info_json="./input.json" --liwc_filepath=<path_to_liwc_dictonary_file>
    '''

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--datasets_info_json",
                        default='./input.json',
                        type=str,
                        required=False,
                        help="JSON file with all datasets information")
    parser.add_argument("--saves_dir_name",
                        default="saves",
                        type=str,
                        help="Main saves directory name")
    parser.add_argument("--liwc_filepath",
                        default='/data/LIWC2007/Dictionaries/LIWC2007_English100131.dic',
                        type=str,
                        required=False,
                        help="Path to the LIWC dictionary file")
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="Seed value for random function")
    
    args = parser.parse_args() 
    myprint(f"args: {args}")       
    np.random.seed(args.seed_val)
    nlp = spacy.load("en_core_web_md")

    saves_dir = os.path.join(args.saves_dir_name, "liwc_dist")
    Path(saves_dir).mkdir(parents=True, exist_ok=True)
    plot_save_prefix = "liwc"

    analysis_types = [
        "sent_level", 
        "review_level", 
        "word_level"
    ]
    required_categories = [
        "posemo", 
        "negemo", 
        "anger", 
        "sad", 
    ]

    amazon_names = ['Pet Supplies', 'Luxury Beauty', 'Automotive', 'Cellphones', 'Sports']
    
    result, class_id, cluster_result, categories, category_reverse = liwc_util.load_liwc(args.liwc_filepath)

    datasets = json.loads(open(args.datasets_info_json, "r").read())

    plot_data = {}    
    for a_type in analysis_types:
        plot_data[a_type] = []

    for data in datasets:
        myprint(data)
        plot_data = compute_liwc(plot_data, data["positive"], data["name"], "positive", 
            required_categories, result, class_id, cluster_result, categories, category_reverse, analysis_types)
        plot_data = compute_liwc(plot_data, data["negative"], data["name"], "negative", 
            required_categories, result, class_id, cluster_result, categories, category_reverse, analysis_types)
        plot_data = compute_liwc(plot_data, data["neutral"], data["name"], "neutral", 
            required_categories, result, class_id, cluster_result, categories, category_reverse, analysis_types)
        
    pickle_save_dir = os.path.join(saves_dir, "all")
    Path(pickle_save_dir).mkdir(parents=True, exist_ok=True)
    pickle.dump(plot_data, open(os.path.join(pickle_save_dir, "liwc_dist_data.pickle"), "wb"))    