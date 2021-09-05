# Sentiment Classification
This directory contains code related to the sentence level sentiment classification (section 4.3).

# How to run
Please run `python <file> --help` to read about the required parameters for training and testing. You can look at the below sample commands as a reference.
```
# Training
python code/training.py --pos_file=/data/yelp/processed_data/pos_reviews_train --neg_file=/data/yelp/processed_data/neg_reviews_train --truncation=head-and-tail --n_samples=25000

# Testing with test set reviews to find the performance of the finetuned classifier
python code/testing.py --input_file=/data/yelp/processed_data/neg_reviews_test --label="negative" --n_samples=5000 --name="yelp" --model_path=./saves/yelp/model_2epochs --device_no=0 --truncation='head-and-tail' --saves_dir=./review_testing_outputs/

# Testing with sentences of test set reviews to estimate the amount of positive/negative sentences in negative reviews
python code/testing.py --input_file=/data/madhu/yelp/processed_data/neg_sents_test --name="yelp" --n_samples=5000 --label="negative" --model_path=./saves/yelp/model_2epochs --device_no=0 --truncation='head-and-tail' --saves_dir=./sentence_testing_outputs/


```