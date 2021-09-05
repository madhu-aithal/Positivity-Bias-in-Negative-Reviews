import torch
import os
import sys
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import random
from pathlib import Path
import util
import datetime
import logging
import pprint
import pandas as pd
import argparse
import pickle
from pathlib import Path
import vader_negation_util
import csv
import spacy
nlp = spacy.load("en_core_web_md")

pp = pprint.PrettyPrinter(indent=4)
iprint = pp.pprint

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def test(args: dict(), save_flag: bool, seed_val):    
    device = util.get_device(device_no=args.device_no)   
    model = torch.load(args.model_path, map_location=device)

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    testfile = args.input_file
    true_label = args.label
    truncation = args.truncation
    n_samples = None
    if "n_samples" in args:
        n_samples = args.n_samples
   
    # Load the BERT tokenizer.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_len = 0
    reviews = []
    labels = []
    with open(testfile, "r") as fin:
        reviews = fin.readlines()
    
    reviews = [rev.lower() for rev in reviews]
    
    if n_samples == None:
        n_samples = len(reviews)

    indices = np.random.choice(np.arange(len(reviews)), size=min(len(reviews), n_samples), replace=False)
    selected_reviews = [reviews[idx] for idx in indices]

    labels = [0 if true_label == "negative" else 1]*len(selected_reviews)

    input_ids = []
    attention_masks = []
    
    for rev in selected_reviews:        
        input_id = tokenizer.encode(rev, add_special_tokens=True)
        if len(input_id) > 512:                        
            if truncation == "tail-only":
                # tail-only truncation
                input_id = [tokenizer.cls_token_id]+input_id[-511:]      
            elif truncation == "head-and-tail":
                # head-and-tail truncation       
                input_id = [tokenizer.cls_token_id]+input_id[1:129]+input_id[-382:]+[tokenizer.sep_token_id]
            else:
                # head-only truncation
                input_id = input_id[:511]+[tokenizer.sep_token_id]
                
            input_ids.append(torch.tensor(input_id).view(1,-1))
            attention_masks.append(torch.ones([1,len(input_id)], dtype=torch.long))
        else:
            encoded_dict = tokenizer.encode_plus(
                                rev,                      
                                add_special_tokens = True,
                                max_length = 512,         
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'pt',    
                        )                            
            input_ids.append(encoded_dict['input_ids'])            
            attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    # Set the batch size.  
    batch_size = 8  

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    
    print('DONE.')
    return predictions, true_labels, selected_reviews

if __name__=="__main__":    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to the input file containing the inputs to test i.e. test set file")
    parser.add_argument("--saves_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="Directory to save the testing predictions results and stats")
    parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Directory containing the model and its configuration")
    parser.add_argument("--truncation",
                        default="head-and-tail",
                        type=str,
                        required=True,
                        help="Truncation technique to use. `head-and-tail' or `head-only` or `tail-only`")
    parser.add_argument("--label",
                        default=None,
                        type=str,
                        required=True,
                        help="Expected label for the input sentences. Values can be `positive` or `negative`")
    parser.add_argument("--name",
                        default=None,
                        type=str,
                        required=True,
                        help="Dataset name")
    parser.add_argument("--n_samples",
                        default=None,
                        type=int,
                        help="Number of samples to use for testing")
    parser.add_argument("--device_no",
                        default=2,
                        type=int,
                        help="GPU number, in case of multiple GPUs")
    parser.add_argument("--seed_val",
                        default=23,
                        type=int,
                        help="Seed value to use with random function")
    
    args = parser.parse_args()
    

    with open(args.saves_dir+"/"+args.name+"_"+args.label+'.csv', "w") as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)        
        accuracies_df = pd.DataFrame(columns=['dataset', 'seed_val', 'accuracy', 'score'])
        vader_sentiment_scores = vader_negation_util.read_vader_sentiment_dict()
        iprint(f"Args: {args}")
        
        preds, true_labels, reviews = test(args, False, args.seed_val)
    
        negation_count_values = []
        pos_count_values = []
        neg_count_values = []
        for rev in reviews:
            negation_count_values.append(util.has_negation(rev))
            doc = nlp(rev)
            pos = 0
            neg = 0
            for token in doc:
                if token.text in vader_sentiment_scores and vader_sentiment_scores[token.text.lower()] >= 1:
                    pos += 1
                if token.text in vader_sentiment_scores and vader_sentiment_scores[token.text.lower()] >= 1:
                    neg += 1
            pos_count_values.append(pos)
            neg_count_values.append(neg)
        
        flat_predictions = np.concatenate(preds, axis=0)

        # For each sample, pick the label (0 or 1) with the higher score.
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        # Combine the correct labels for each batch into a single list.
        flat_true_labels = np.concatenate(true_labels, axis=0)

        accuracy = 1.0*(np.sum(flat_predictions == flat_true_labels))/flat_predictions.shape[0]
        
        score = 0
        correct_count = np.sum(flat_predictions == flat_true_labels)
        total_count = flat_predictions.shape[0]
        
        if args.label == "positive":
            TPR = 1.0*correct_count/total_count
            FNR = 1.0*(total_count-correct_count)/total_count
            score = accuracy*TPR+(1-accuracy)*FNR
            true_rate = TPR
            false_rate = FNR

        elif args.label == "negative":            
            TNR = 1.0*correct_count/total_count
            FPR = 1.0*(total_count-correct_count)/total_count
            score = accuracy*TNR+(1-accuracy)*FPR
            true_rate = TNR
            false_rate = FPR

        correct_indices = np.argwhere(flat_predictions == flat_true_labels).flatten()
        incorrect_indices = np.argwhere(flat_predictions != flat_true_labels).flatten()
        correct_negation_count = 0
        incorrect_negation_count = 0
        
        has_pos_count = 0
        for val in pos_count_values:
            if val>0:
                has_pos_count += 1
        no_pos_count = len(pos_count_values)-has_pos_count

        has_negation_count = 0
        for val in negation_count_values:
            if val>0:
                has_negation_count += 1
        no_negation_count = len(negation_count_values)-has_negation_count

        correct_pos_count = 0
        incorrect_pos_count = 0

        for idx in correct_indices:
            if negation_count_values[idx] > 0:
                correct_negation_count+=1
        for idx in incorrect_indices:
            if negation_count_values[idx] > 0:
                incorrect_negation_count+=1

        for idx in correct_indices:
            if pos_count_values[idx] > 0:
                correct_pos_count+=1
        for idx in incorrect_indices:
            if pos_count_values[idx] > 0:
                incorrect_pos_count+=1
        
        accuracies_df = accuracies_df.append({                
            "dataset": args.input_file,
            "name": args.name,
            "seed_val": args.seed_val,
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "true_rate": true_rate,
            "false_rate": false_rate,
            "total_negation_count": has_negation_count,
            "correct_count_with_negation": correct_negation_count,
            "incorrect_count_with_negation": incorrect_negation_count,
            "total_count_with_pos_words": has_pos_count,
            "correct_count_with_pos_words": correct_pos_count,
            "incorrect_count_with_pos_words": incorrect_pos_count
        }, ignore_index=True)
        
        csv_writer.writerow([
            args.input_file,
            args.name,
            args.seed_val,
            accuracy,
            correct_count,
            total_count,
            has_negation_count,
            correct_negation_count,
            incorrect_negation_count,
            has_pos_count,
            correct_pos_count,
            incorrect_pos_count
        ])
        
        iprint(f"Accuracy: {accuracy}")            

        save_pickle_path = os.path.join(args.saves_dir,             
            args.name,
            os.path.basename(args.input_file))
        Path(os.path.dirname(save_pickle_path)).mkdir(parents=True, exist_ok=True)
        
        pickle.dump(accuracies_df, open(save_pickle_path, "wb"))
            
        print("accuracies_df: ", accuracies_df)

    
