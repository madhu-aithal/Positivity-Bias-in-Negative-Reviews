First run `process_dataset.py` file, and then run `split_dataset.py` file as shown below.

```
cd code/sst/

python process_dataset.py --rt_snippets_filepath=<path_to_rt_snippets.txt> --dataset_sentences_filepath=<path_to_datasetSentences.txt> --output_dir=<output_dir>

python split_dataset.py --sst_data_dir=<output_dir>
```
