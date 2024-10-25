# Datasets
This is were all features are extract from the raw data and stored for the **gilani-2017**, **varol-icwsm** datasets.

## To Run Regular Dataset Preperation
If the raw data is available here is how you run the code for the gilani-2017 and varol-icwsm datasets:
```bash
source ../venv/bin/activate
python3 paths.py
python3 user_extracted_features_clean.py
python3 timeline_extracted_features_clean.py
python3 description_tweet_embedding.py
python3 combine.py
```

If the raw data is available here is how you run the code for the **twibot** datasets:
```bash
source ../venv/bin/activate
python3 paths_twibot.py
python3 user_extracted_features_clean_twibot.py
python3 timeline_extracted_features_clean_twibot.py
python3 description_tweet_embedding_twibot.py
python3 combine_twibot.py
```

## To Run Adversary Testing (Tweet Modification)
You can only run this testing for Gilani-2017 as of now, upon changing the date in line 420 in `timeline_extracted_features_clean_bot_sample_modifed_single_tweets.py` to 2023-03-05, it can be run for varol.

```bash
DATASET=gilani-2017
source ../venv/bin/activate
python timeline_extracted_features_clean_bot_sample_modifed_single_tweets.py $DATASET
python description_tweet_embedding_sample_modified_single_tweets.py $DATASET
python combine_sample_modified_single_tweets.py $DATASET
```
