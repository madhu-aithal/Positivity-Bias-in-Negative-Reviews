# Datasets Analysis
This directory contains code related to our analysis mentioned in the paper. Please note that this directory doesn't contain code related to our sentiment prediction experiment done using BERT. You can find it [here](../sentiment-classification).

# How to run
Please use the below sample commands to run. For more info, you can run `python <file> --help`
```
python code/dataset_length_dist.py --datasets_info_json="./input.json"

python code/liwc_count.py --datasets_info_json="./input.json" --liwc_filepath=<path_to_liwc_dictonary_file>

python code/vader_negation_dep_parsing.py --datasets_info_json="./input.json --analysis_type=<analysis_type> --vader_lexicon_path=<path_to_vader_lexicons_file>"
    
python code/vader_negation_only_dist.py --datasets_info_json="./input.json"

python code/vader_pos_neg_dist.py --datasets_info_json="./input.json" --vader_lexicon_path=<path_to_vader_lexicon_file>

python code/vader_pos_neg_negation_clause_dist.py --datasets_info_json="./input.json" 
        --vader_lexicon_path=<path_to_vader_lexicon_file>

python code/vader_pos_neg_negation_dist.py --datasets_info_json="./input.json" --vader_lexicon_path=<path_to_vader_lexicon_file>
```

To Do
* Add requirements.txt