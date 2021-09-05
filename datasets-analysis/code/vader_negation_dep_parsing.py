import spacy
import argparse
import numpy as np
import util
import pickle
import os
from pathlib import Path
import vader_negation_util
import json
import pprint
import argparse

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

nlp = spacy.load('en_core_web_md')

def count_negation(text, vader_sentiment_scores):
    doc = nlp(text)
    negation_count = 0
    dep_dict = {}
    sent_count = 0
    pos_negation_count = 0
    neg_negation_count = 0
    per_sentence_negation_count = []
    per_sentence_negation_pos_count = []
    per_sentence_negation_neg_count = []

    for sent in doc.sents:
        sent_count += 1
        temp_overall_negation_count = 0
        temp_negation_pos_count = 0
        temp_negation_neg_count = 0

        doc_sent = nlp(sent.text)
        for token in doc_sent: 
            dep_dict[token.text] = [(child.text, child.dep_) for child in token.children]  
            if token.dep_ == "neg":
                negation_count += 1
                temp_overall_negation_count += 1
                if token.head.pos_ == "AUX":
                    if token.head.dep_ in ["acomp", "advmod"] and token.head.head.text in vader_sentiment_scores:
                        sent_score = vader_sentiment_scores[token.head.head.text]
                        if sent_score >= 1:
                            pos_negation_count += 1
                            temp_negation_pos_count += 1
                        elif sent_score <= -1:
                            neg_negation_count += 1
                            temp_negation_neg_count += 1
                elif token.head.pos_ in ["NOUN", "VERB", "ADJ", "ADV"] and token.head.text in vader_sentiment_scores:
                    sent_score = vader_sentiment_scores[token.head.text]
                    if sent_score >= 1:
                        pos_negation_count += 1
                        temp_negation_pos_count += 1
                    elif sent_score <= -1:
                        neg_negation_count += 1
                        temp_negation_neg_count += 1
        
        per_sentence_negation_count.append(temp_overall_negation_count)
        per_sentence_negation_neg_count.append(temp_negation_neg_count)
        per_sentence_negation_pos_count.append(temp_negation_pos_count) 
    
    negation_count_dict = {
        "word_level": 1.0*negation_count/len(doc),
        "sent_level": 1.0*negation_count/sent_count,
        "review_level": negation_count,
        "per_sentence_count": per_sentence_negation_count
    }
    pos_negation_count_dict = {
        "word_level": 1.0*pos_negation_count/len(doc),
        "sent_level": 1.0*pos_negation_count/sent_count,
        "review_level": pos_negation_count,
        "per_sentence_count": per_sentence_negation_pos_count
    }
    neg_negation_count_dict = {
        "word_level": 1.0*neg_negation_count/len(doc),
        "sent_level": 1.0*neg_negation_count/sent_count,
        "review_level": neg_negation_count,
        "per_sentence_count": per_sentence_negation_neg_count
    }
    return negation_count_dict, pos_negation_count_dict, neg_negation_count_dict, dep_dict


if __name__ == "__main__":
    '''
    Command to run this file:
    python code/vader_negation_dep_parsing.py --datasets_info_json="./input.json 
        --analysis_type=<analysis_type> --vader_lexicon_path=<path_to_vader_lexicons_file>"
    '''
    
    pos_neg_save_prefix = "pos_neg_negation_depparsing"
    overall_save_prefix = "overall_negation_depparsing"

    parser = argparse.ArgumentParser()

    parser.add_argument("--analysis_type",
                        default='sent_level',
                        type=str,
                        required=False,
                        help="Type of analysis - word_level, sent_level, review_level")   
    parser.add_argument("--datasets_info_json",
                        default="./input.json",
                        type=str,
                        required=False,
                        help="JSON file with all datasets information")    
    parser.add_argument("--saves_dir_name",
                        default="saves",
                        type=str,
                        help="Main saves directory name")   
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

    saves_dir = os.path.join(args.saves_dir_name, "negation_dep_parsing")
    Path(saves_dir).mkdir(parents=True, exist_ok=True) 
    Path(os.path.join(saves_dir, "overall")).mkdir(parents=True, exist_ok=True) 
    Path(os.path.join(saves_dir, "pos_neg_negation")).mkdir(parents=True, exist_ok=True) 
    datasets = json.loads(open(args.datasets_info_json).read())

    np.random.seed(args.seed_val)

    vader_sentiment_scores = vader_negation_util.read_vader_sentiment_dict(args.vader_lexicon_path)

    negation_count_data = {}
    pos_negation_count_data = {}
    neg_negation_count_data = {}

    selected_samples = {}
    for data in datasets:
        myprint(data)
        selected_samples[data["name"]] = {}
        for category in ["positive", "negative", "neutral"]:
            texts = util.read_file(data[category]["data_filepath"])
            n_samples = None
            if "n_samples" in data[category]:
                n_samples = int(data[category]["n_samples"])
            selected_texts = util.get_samples(texts, n_samples)
            selected_samples[data["name"]][category] = selected_texts

    plot_data = []
    plot_data_overall_negation = []

    for data in datasets:
        dep_data = {}        
        for category in ["positive", "negative", "neutral"]:            
            dep_data[category] = []
            negation_count_data = []
            pos_negation_count_data = []
            neg_negation_count_data = []
            per_sentence_negation_counts = []
            per_sentence_negation_neg_counts = []
            per_sentence_negation_pos_counts = []

            selected_texts = selected_samples[data["name"]][category]
            for text in selected_texts:
                negation_count_dict, pos_negation_count_dict, neg_negation_count_dict, dep_dict = count_negation(text, 
                    vader_sentiment_scores) 
                                
                per_sentence_negation_counts.append(negation_count_dict['per_sentence_count'])
                per_sentence_negation_pos_counts.append(pos_negation_count_dict['per_sentence_count'])
                per_sentence_negation_neg_counts.append(neg_negation_count_dict['per_sentence_count'])

                negation_count_data.append(negation_count_dict[args.analysis_type])
                pos_negation_count_data.append(pos_negation_count_dict[args.analysis_type])
                neg_negation_count_data.append(neg_negation_count_dict[args.analysis_type])                    

            per_sentence_negation_counts = [val for x in per_sentence_negation_counts for val in x]
            per_sentence_negation_pos_counts = [val for x in per_sentence_negation_pos_counts for val in x]
            per_sentence_negation_neg_counts = [val for x in per_sentence_negation_neg_counts for val in x]

            dep_data[category].append({
                "text": text,
                "dep_info": dep_dict                        
            })

            plot_data.append({
                "category": "negative - "+category+" review ",
                "review category": category,
                "text sentiment": "negative",
                "name": data["name"],
                "value": np.mean(neg_negation_count_data),                
                "sem_value": util.get_sem_value(args.analysis_type, per_sentence_negation_neg_counts, neg_negation_count_data),
                "value_per_sentence_count": util.get_mean_value(args.analysis_type, per_sentence_negation_neg_counts, neg_negation_count_data),
                "all_samples_data": neg_negation_count_data
            })
            plot_data.append({
                "category": "positive - "+category+" review ",
                "review category": category,
                "text sentiment": "positive",
                "name": data["name"],
                "value": np.mean(pos_negation_count_data),
                "sem_value": util.get_sem_value(args.analysis_type, per_sentence_negation_pos_counts, pos_negation_count_data),
                "value_per_sentence_count": util.get_mean_value(args.analysis_type, per_sentence_negation_neg_counts, neg_negation_count_data),
                "all_samples_data": pos_negation_count_data
            })
            plot_data_overall_negation.append({
                "category": category,
                "name": data["name"],
                "value": np.mean(negation_count_data),                
                "sem_value": util.get_sem_value(args.analysis_type, per_sentence_negation_counts, negation_count_data),
                "value_per_sentence_count": util.get_mean_value(args.analysis_type, per_sentence_negation_counts, negation_count_data),
                "all_samples_data": negation_count_data
            })
            
        pickle.dump(dep_data, 
            open(os.path.join(saves_dir, data["name"]+"_depparsing_data.pickle"), "wb")
        )
                
    pickle.dump(plot_data, 
        open(os.path.join(saves_dir, "pos_neg_negation", args.analysis_type+"_"+pos_neg_save_prefix+".pickle"), "wb")
    )
    pickle.dump(plot_data_overall_negation, 
        open(os.path.join(saves_dir, "overall", args.analysis_type+"_"+overall_save_prefix+".pickle"), "wb")
    )