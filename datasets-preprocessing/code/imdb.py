import os
import spacy
import numpy as np
import random
import argparse
import utils

nlp = spacy.load("en_core_web_md")
tokenizer = nlp.Defaults.create_tokenizer(nlp)

def read_files(catgeory):
    files_dir = os.path.join(IMDB_DATASET_DIR, data_category, catgeory)
    
    onlyfiles = [f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f))]
    out_dir = "/data/madhu/imdb_dataset/processed_data"
    f_out = open(os.path.join(out_dir, catgeory+"_reviews_"+data_category), "w")
    samples = []
    for file in onlyfiles:
        file_path = os.path.join(files_dir, file)
        with open(file_path, "r") as f:
            line = f.readline().strip("\n").replace("<br />", ". ")
            samples.append(line)
            f_out.write(line+"\n")
    return samples

if __name__ == "__main__":
    IMDB_DATASET_DIR = "/data/madhu/imdb_dataset/aclImdb/"
    data_category = "train"

    seed_val = 23
    np.random.seed(seed_val)
    random.seed(seed_val)

    pos_samples = read_files("pos")
    neg_samples = read_files("neg")

    pos_sents = utils.get_sents(pos_samples)
    neg_sents = utils.get_sents(neg_samples)

    n_samples = int(1e5)

    pos_indices = np.random.choice(np.arange(len(pos_sents)), size=n_samples)
    neg_indices = np.random.choice(np.arange(len(neg_sents)), size=n_samples)

    pos_selected_sents = [pos_sents[idx] for idx in pos_indices]
    neg_selected_sents = [neg_sents[idx] for idx in neg_indices]

    out_dir = "/data/madhu/imdb_dataset/processed_data"

    if data_category == "train":
        utils.write_sents(pos_selected_sents[:n_samples-5000], os.path.join(out_dir,"pos_"+data_category+"_reviews_"+str(n_samples-5000)+"sents"))
        utils.write_sents(neg_selected_sents[:n_samples-5000], os.path.join(out_dir,"neg_"+data_category+"_reviews_"+str(n_samples-5000)+"sents"))

        utils.write_sents(pos_selected_sents[n_samples-5000:], os.path.join(out_dir,"pos_dev_reviews_"+str(5000)+"sents"))
        utils.write_sents(neg_selected_sents[n_samples-5000:], os.path.join(out_dir,"neg_dev_reviews_"+str(5000)+"sents"))
    else:
        utils.write_sents(pos_selected_sents, os.path.join(out_dir,"pos_"+data_category+"_reviews_"+str(n_samples)+"sents"))
        utils.write_sents(neg_selected_sents, os.path.join(out_dir,"neg_"+data_category+"_reviews_"+str(n_samples)+"sents"))
    n_samples = int(1e4)

    pos_indices = np.random.choice(np.arange(len(pos_samples)), size=n_samples)
    neg_indices = np.random.choice(np.arange(len(neg_samples)), size=n_samples)

    selected_pos_samples = [pos_samples[idx] for idx in pos_indices]
    selected_neg_samples = [neg_samples[idx] for idx in neg_indices]

    pos_sents_same_review = utils.get_sents(selected_pos_samples)
    neg_sents_same_review = utilsget_sents(selected_neg_samples)

    utils.write_sents(pos_sents_same_review, os.path.join(out_dir,"pos_"+data_category+"_reviews_"+str(n_samples)+"sents_same_review"))
    utils.write_sents(neg_sents_same_review, os.path.join(out_dir,"neg_"+data_category+"_reviews_"+str(n_samples)+"sents_same_review"))

    print("Execution finished")

