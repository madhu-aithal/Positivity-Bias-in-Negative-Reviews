# Code to read the reviews of PeerRead dataset.
import os
import spacy
import numpy as np
import random
import json
import utils
from pathlib import Path
import argparse
import pprint

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def read_files(dataset_dir):   
    onlyfiles = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]    
    pos_reviews = []
    neg_reviews = []
    for file in onlyfiles:
        print(file)
        file_path = os.path.join(dataset_dir, file)
        with open(file_path, "r") as f:
            line = f.readline().strip("\n")
            json_content = json.loads(line)
            reviews = json_content["reviews"]
            for rev in reviews:
                score = float(rev["RECOMMENDATION"])
                if score <=3:
                    neg_reviews.append(rev["comments"].replace("\n", " ").strip("\n"))
                elif score >= 4:
                    pos_reviews.append(rev["comments"].replace("\n", " ").strip("\n"))
    return pos_reviews, neg_reviews   

if __name__ == "__main__":
    seed_val = 23
    np.random.seed(seed_val)
    random.seed(seed_val)

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="For example - /data/madhu/PeerRead/data/acl_2017/train/reviews")
    
    args = parser.parse_args() 
    myprint(f"args: {args}")  

    pos_reviews, neg_reviews = read_files(args.dataset_dir)

    out_dir = os.path.join(args.dataset_dir, "processed_data")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    utils.write_file(pos_reviews, os.path.join(out_dir, 'pos_reviews'))
    utils.write_file(neg_reviews, os.path.join(out_dir, 'neg_reviews'))
    
    print("Execution finished")