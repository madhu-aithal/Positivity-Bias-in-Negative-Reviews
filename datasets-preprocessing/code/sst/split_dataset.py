# Code to read the reviews of the SST dataset and split them into train and test set
import os
import spacy
import random
import utils
import argparse
import pprint

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

SST_DATA_DIR = "/data/madhu/stanford-sentiment-treebank/matched_data/"

def split_data(reviews, size, filename_train, filename_test):
    random.shuffle(reviews)
    size = int(size)
    train_reviews = reviews[:size]
    test_reviews = reviews[size:]

    with open(filename_train, "w") as fout:
        for rev in train_reviews:
            fout.write(rev+"\n")

    with open(filename_test, "w") as fout:
        for rev in test_reviews:
            fout.write(rev+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sst_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Directory containing processed reviews of the SST dataset i.e. output directory of the process_dataset.py file")
    
    args = parser.parse_args() 
    myprint(f"args: {args}")  

    seed_val = 23
    random.seed(seed_val)

    pos_reviews = utils.read_file(os.path.join(SST_DATA_DIR, "pos_reviews"))
    neg_reviews = utils.read_file(os.path.join(SST_DATA_DIR, "neg_reviews"))    

    split_data(pos_reviews, len(pos_reviews)*3.0/4, os.path.join(SST_DATA_DIR, "pos_reviews_train"), os.path.join(SST_DATA_DIR, "pos_reviews_test"))
    split_data(neg_reviews, len(neg_reviews)*3.0/4, os.path.join(SST_DATA_DIR, "neg_reviews_train"), os.path.join(SST_DATA_DIR, "neg_reviews_test"))

    print("execution finished")  
    