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
from torch import nn
import argparse
import pprint

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train_model(args: dict, hparams:dict):       
    pos_file = args.pos_file
    neg_file = args.neg_file
    truncation = args.truncation
    n_samples = args.n_samples
    seed_val = hparams["seed_val"]
    device = util.get_device(device_no=args.device_no)
    saves_dir = "saves/"

    Path(saves_dir).mkdir(parents=True, exist_ok=True)   
    time = datetime.datetime.now()

    saves_path = os.path.join(saves_dir, util.get_filename(time))
    Path(saves_path).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(saves_path, "training.log")

    logging.basicConfig(filename=log_path, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info("Pos file: "+str(pos_file))
    logger.info("Neg file: "+str(neg_file))
    logger.info("Parameters: "+str(args))
    logger.info("Truncation: "+truncation)    
    logger.info('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_len = 0

    reviews, labels = util.read_samples_new(filename0=neg_file, filename1=pos_file, 
        seed_val=seed_val, n_samples=n_samples, sentence_flag=True)
    print(len(reviews), len(labels))
    
    input_ids = []
    attention_masks = []
    
    for rev in reviews:        
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

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info('{:>5,} training samples'.format(train_size))
    logger.info('{:>5,} validation samples'.format(val_size))    
    batch_size = hparams["batch_size"]

    train_dataloader = DataLoader(
                train_dataset,  
                sampler = RandomSampler(train_dataset), 
                batch_size = batch_size 
            )

    validation_dataloader = DataLoader(
                val_dataset, 
                sampler = SequentialSampler(val_dataset), 
                batch_size = batch_size
            )
    model = BertForSequenceClassification.from_pretrained(        
        "bert-base-uncased", 
        num_labels = 2,                         
        output_attentions = False,
        output_hidden_states = False,
    )
        
    model = model.to(device=device)    

    optimizer = AdamW(model.parameters(),
                    lr = hparams["learning_rate"], 
                    eps = hparams["adam_epsilon"]
                    )

    epochs = 4
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    training_stats = []

    # For each epoch...
    for epoch_i in range(0, epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.
        logger.info("")
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logger.info('Training...')

        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:               
                # Report progress.
                logger.info('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)           

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.detach().cpu().numpy()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            

        logger.info("")
        logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))

        logger.info("")
        logger.info("Running Validation...")

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        
                (loss, logits) = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
            # Accumulate the validation loss.
            total_eval_loss += loss.detach().cpu().numpy()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        logger.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
               
        logger.info("  Validation Loss: {0:.2f}".format(avg_val_loss))        

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,                
            }
        )

        model_save_path = os.path.join(saves_path, "model_"+str(epoch_i+1)+"epochs")
        torch.save(model, model_save_path)

    logger.info("")
    logger.info("Training complete!")
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

if __name__=="__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_file",
                        default=None,
                        type=str,
                        required=True,
                        help="File containing positive reviews of the dataset")
    parser.add_argument("--neg_file",
                        default=None,
                        type=str,
                        required=True,
                        help="File containing negative reviews of the dataset")
    parser.add_argument("--truncation",
                        default=None,
                        type=str,
                        required=True,
                        help="Truncation technique to use for longer reviews")
    parser.add_argument("--n_samples",
                        default=None,
                        type=int,
                        help="Number of samples to use for training (train set and validation set)")
    parser.add_argument("--device_no",
                        default=0,
                        type=int,
                        help="GPU number, in case of multiple GPUs")
    
    args = parser.parse_args()

    hyperparams = [       
        {
            "learning_rate": 2e-5,
            "batch_size": 8,
            "seed_val": 23,
            "adam_epsilon": 1e-8
        }     
    ]

    for hparams in hyperparams:         
        myprint(f"args: {args}")    
        myprint(f"hparams: {hparams}")
        train_model(args, hparams)