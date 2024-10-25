# BotSSCL
This self-supervised contrastive learning framework implementation is adapted from [pytorch-scarf](https://github.com/clabrugere/pytorch-scarf).

## Pre-Installation of Pytorch-Scarf
Please ensure you have Python 3.7.4 installed, once this requirement is fulfilled, then run the following script.

```bash
python3 -m venv venv
source venv/bin/activate
pip install git+https://github.com/clabrugere/pytorch-scarf.git
python3 check_pyscarf.py
pip install -r requirements.txt
```

## Data Preparation
From `_1_Data_Collection_and_Feature_Extraction/DATASETs`, you will need to copy over the `*clean.csv` data files containing the table of users per DATASET.

```bash
cp '../_1_Data_Collection_and_Feature_Extraction/DATASETs/*clean.csv' .
```

Then depending on the augmentation method, there will be different steps to take.

### Augmentation 1
```bash
source venv/bin/activate
python3 twiiter_user_representation.py
```

### Augmentation 2
```bash
source venv/bin/activate
python3 run_imputer.py
python3 combined_corrupted.py
```

### Augmention 3
No additional data processing.

## Running Experiments

You can currently run experiments either on `varol-icwsm` and `gilani-2017`.

### Augmentation 1

#### Baseline
```bash
source venv/bin/activate
python3 botsscl_augmentation_1.py $DATASET
```

#### Batch Sizes
```bash
source venv/bin/activate
python botsscl_augmentation_1_batch_size.py $DATASET
```

#### Corruption Rates
```bash
source venv/bin/activate
python botsscl_augmentation_1_corr_rate.py varol-icwsm
```

#### Supervision
```bash
source venv/bin/activate
python botsscl_augmentation_1_supervised.py $DATASET 64 64
```

#### Supervision v2
```bash
source venv/bin/activate
python botsscl_augmentation_1_supervised_v2.py $DATASET 64 64
```

### Adversarial Testing (Sample Modified Single Tweets)
You will need to do additional data preperation by running the following scripts. There were rate limits on the ChatGPT API so there were sets were the number of tweets to be paraphrased (re-written as part of Tweet manipulation). There were a 20 sets total. Different sets will have to be run seperately or together depending on the current limits.

```bash
DATASET=gilani-2017
source venv/bin/activate
python3 submit_prompts_chat.py $DATASET 1
python3 submit_prompts_chat.py $DATASET 2
python3 submit_prompts_chat.py $DATASET 3
python3 submit_prompts_chat.py $DATASET 4
python3 submit_prompts_chat.py $DATASET 5
python3 submit_prompts_chat.py $DATASET 6
python3 submit_prompts_chat.py $DATASET 7
python3 submit_prompts_chat.py $DATASET 8
python3 submit_prompts_chat.py $DATASET 9
python3 submit_prompts_chat.py $DATASET 10
python3 submit_prompts_chat.py $DATASET 11
python3 submit_prompts_chat.py $DATASET 12
python3 submit_prompts_chat.py $DATASET 13
python3 submit_prompts_chat.py $DATASET 14
python3 submit_prompts_chat.py $DATASET 15
python3 submit_prompts_chat.py $DATASET 16
python3 submit_prompts_chat.py $DATASET 17
python3 submit_prompts_chat.py $DATASET 18
python3 submit_prompts_chat.py $DATASET 19
python3 submit_prompts_chat.py $DATASET 20
```

Then each set has to be combined using the following script.
```bash
source venv/bin/activate
python combine_submit_prompts_chat.py
```

Then copy the output file of this script then follow the instructions found there to prepare the main data file.
```bash
$DATASET=gilani-2017
cp ${DATASET}-sample-modified-tweets.csv ../_1_Data_Collection_and_Feature_Extraction/DATASETs
```

Copy the data file `{$DATASET}-clean-sample-modified-single-tweets.csv` to the Data folder in this directory. Then run the following scripts.


```bash
source venv/bin/activate
DATASET=gilani-2017
python twitter_user_representation_sample_modified_single_tweets.py $DATASET
```

Once the representations are produced, the following script can be the run and the labels are outputted.
```bash
source venv/bin/activate
python botsscl_augmentation_1_sample_modified_single_tweets_no_graphs.py $DATASET 64 64
```

### Augmentation 2
```bash
source venv/bin/activate
python3 botsscl_augmentation_2.py $DATASET
```

### Augmentation 3
```bash
source venv/bin/activate
python3 botsscl_augmentation_3.py $DATASET
```
