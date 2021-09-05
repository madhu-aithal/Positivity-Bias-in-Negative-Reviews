# Datasets Preprocessing

## Datasets
1. Yelp restaurant reviews
2. IMDb movie reviews
3. SST
4. PeerRead
5. Amazon reviews

## How to run
1. Yelp
    ```
    python code/yelp.py --yelp_review_filepath="<path_to_yelp_academic_dataset_review.json.gz>" --yelp_business_filepath="<path_to_yelp_academic_dataset_business.json.gz>" --output_dir="<output_dir_to_store_processed_files>"
    ```
2. IMDb
    ```
    python code/imdb.py
    ```
3. SST - Check the README of `code/sst/` directory
4. PeerRead
    ```
    python code/peerread.py --dataset_dir=<peerread_dataset_dir>
    ```
5. Amazon reviews - Update the `data_filepath` and `output_dir` values of the `data` variable in `amazon_reviews.py` with the correct values, and run `python code/amazon_reviews.py`

## To DO
1. Simplify the logic of reading and processing reviews of SST dataset.
2. Improve the README