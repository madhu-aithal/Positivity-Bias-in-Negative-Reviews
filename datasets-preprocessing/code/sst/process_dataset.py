import pickle
from utils import process_invalid_sents
from pathlib import Path
import os
import argparse
import pprint

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

def read_dictionary(dictionary_file, index_sentiment_map):
    phrase_sentiment_map = {}
    with open(dictionary_file, "r", encoding='utf-8') as f:
        for row in f.readlines():
            phrase = row.split("|")[0].strip()
            index = row.split("|")[1].strip()
            phrase_sentiment_map[phrase.lower()] = index_sentiment_map[index]            
    return phrase_sentiment_map

def read_sentiment_labels(sentiment_label_file):
    index_sentiment_map = {}
    with open(sentiment_label_file, "r", encoding='utf-8') as f:
        for row in f.readlines():
            index = row.split("|")[0].strip()
            sent_score = row.split("|")[1].strip()
            index_sentiment_map[index] = sent_score

    return index_sentiment_map

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--rt_snippets_filepath",
                        default=None,
                        type=str,
                        required=True,
                        help="File path of original_rt_snippets.txt file in the SST dataset directory")    
    parser.add_argument("--dataset_sentences_filepath",
                        default=None,
                        type=str,
                        required=True,
                        help="File path of datasetSentences.txt file in the SST dataset directory")
    parser.add_argument("--output_dir",
                        default='./valid_data',
                        type=str,
                        required=True,
                        help="Output directory to store the processed reviews as pos_reviews and neg_reviews")
    
    args = parser.parse_args() 
    myprint(f"args: {args}") 

    rt_snippets_file = args.rt_snippets_filepath
    datasetsentences_file = args.dataset_sentences_filepath
    output_dir = args.output_dir

    index_sentiment_map = read_sentiment_labels("stanfordSentimentTreebank/sentiment_labels.txt")
    phrase_sentiment_map = read_dictionary("stanfordSentimentTreebank/dictionary.txt", index_sentiment_map)

    total_review_count = 0
    total_sents_count = 0
    
    pos_reviews = {
        "pos_sents": 0,
        "neg_sents": 0,
        "neutral_sents": 0,
        "total_count": 0,
        "total_sents": 0,
        "reviews": [],
    }

    neg_reviews = {
        "pos_sents": 0,
        "neg_sents": 0,
        "neutral_sents": 0,
        "total_count": 0,
        "total_sents": 0,
        "reviews": [],
    }

    neutral_reviews = {
        "pos_sents": 0,
        "neg_sents": 0,
        "neutral_sents": 0,
        "total_count": 0,
        "total_sents": 0,
        "reviews": [],
    }

    reviews = []
    sents = []
    invalid_sents = []
    invalid_reviews = []

    from pycorenlp import StanfordCoreNLP
    nlp = StanfordCoreNLP('http://localhost:9000')
    
    with open(rt_snippets_file, "r") as f:
        for row in f.readlines():
            row = row.strip("\n")
            output = nlp.annotate(row, properties={
                'annotators': 'ssplit',
                'outputFormat': 'json'
            })
            review = ""
            pos_sents = []
            neg_sents = []
            neutral_sents = []
            sents_count = 0
            for sent_tokens in output['sentences']:
                sent = ""
                for token in sent_tokens["tokens"]:
                    if token["originalText"] =="(":
                        sent += "("+" "
                    elif token["originalText"] == ")":
                        sent += ")"+" "
                    else:
                        sent += token["word"]+" "
                sent = sent.strip()
                sents_count += 1
                total_sents_count += 1
                if sent in phrase_sentiment_map:
                    if float(phrase_sentiment_map[sent]) >= 0.6:                        
                        pos_sents.append(sent)
                    elif float(phrase_sentiment_map[sent]) <= 0.4:                        
                        neg_sents.append(sent)
                    else:                        
                        neutral_sents.append(sent)
                else:
                    invalid_sents.append(sent)
                review += sent + " "

            total_review_count += 1
            if total_review_count%1000 == 0:
                print("Total review count: ", total_review_count)

            review = review.strip()
            if review in phrase_sentiment_map:        
                if float(phrase_sentiment_map[review]) >= 0.6:
                    pos_reviews["total_count"] += 1
                    pos_reviews["pos_sents"] += len(pos_sents)
                    pos_reviews["neg_sents"] += len(neg_sents)
                    pos_reviews["neutral_sents"] += len(neutral_sents)
                    pos_reviews["total_sents"] += sents_count
                    pos_reviews["reviews"] += [review]
                elif float(phrase_sentiment_map[review]) <= 0.4:
                    neg_reviews["total_count"] += 1
                    neg_reviews["pos_sents"] += len(pos_sents)
                    neg_reviews["neg_sents"] += len(neg_sents)  
                    neg_reviews["neutral_sents"] += len(neutral_sents)         
                    neg_reviews["total_sents"] += sents_count
                    neg_reviews["reviews"] += [review]
                else:
                    neutral_reviews["total_count"] += 1
                    neutral_reviews["pos_sents"] += len(pos_sents)
                    neutral_reviews["neg_sents"] += len(neg_sents)  
                    neutral_reviews["neutral_sents"] += len(neutral_sents)         
                    neutral_reviews["total_sents"] += sents_count
                    neutral_reviews["reviews"] += [review]
            else:
                invalid_reviews.append(review)

    neutral_reviews, pos_review, neg_review = process_invalid_sents(invalid_sents, invalid_reviews, 
                                            neutral_reviews, pos_reviews, neg_reviews)

    print(f"total_review_count: {total_review_count}")
    print(f"total_sent_count: {total_sents_count}")
    print("neg_reviews: ", neg_reviews)
    print("pos_reviews: ", pos_reviews)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open("invalid_reviews.txt", "w") as f:
        for review in invalid_reviews:
            f.write(review+"\n")

    with open("invalid_sents.txt", "w") as f:
        for sent in invalid_sents:
            f.write(sent+"\n")

    with open(output_dir+"/neutral_reviews", "w") as fout:
        for review in neutral_reviews["reviews"]:
            fout.write(review.strip("\n")+"\n")
    
    with open(output_dir+"/pos_reviews", "w") as fout:
        for review in pos_reviews["reviews"]:
            fout.write(review.strip("\n")+"\n")

    with open(output_dir+"/neg_reviews", "w") as fout:
        for review in neg_reviews["reviews"]:
            fout.write(review.strip("\n")+"\n")
