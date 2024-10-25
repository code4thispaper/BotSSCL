import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import torch
import pickle
from glob import glob

def linear_layer(in_features, out_features):
    linear = torch.nn.Linear(in_features, out_features)
    return linear
    
def run(out_features):
    # Step 1: Extract Datasets
    mode = "clean"
    datasets = glob(f"Data/*{mode}.csv")
    datasets_original = [pd.read_csv(k) for k in datasets]
    datasets = [k for k in datasets_original]

    # Step 2: Running checks for unique columns and removing some additional information columns
    for i in range(len(datasets)):
        datasets[i] = datasets[i].drop(columns=['id','label', 'clean_description'])
        datasets[i]['mean_positive_and_negative_score_ratio_per_tweet'] = datasets[i]['mean_positive_and_negative_score_ratio_per_tweet'].fillna(0)
        datasets[i]['mean_positive_and_negative_score_ratio_per_tweet'] = datasets[i]['mean_positive_and_negative_score_ratio_per_tweet'].replace([-np.inf, np.inf], 0)

    ## Drop all unique columns
    for i in range(len(datasets)):
        # constant_cols = ['profile_image_url', 'hashtag_in_username', 'emojis_in_username', 'mean_user_mentions_per_tweet', 'unique_mention_rate_per_tweet'] for both datasets
        constant_cols = [c for c in datasets[i].columns if datasets[i][c].nunique() == 1]
        datasets[i].drop(columns=constant_cols, inplace=True)

    # Step 3: Feature Split (For just understanding)
    ## User Based/Metadata Feature
    user_metadata_columns = list(datasets[0].columns)[:33]
    print(f'There are {len(user_metadata_columns)} User Based/Metadata Features')
    print()
    print(user_metadata_columns)
    print()
    # count = 31

    ## Tweet Metadata (Content + Language) Feature
    content_language_columns = list(datasets[0].columns)[33:39] + [c for c in list(datasets[0].columns)[39:39+19] if 'emoticons' in c and not 'entropy' in c]
    # count = 9

    ## Tweet Metadata (Sentiment) Feature
    sentiment_columns = [c for c in list(datasets[0].columns)[39:39+19] if not 'emoticons_per_tweet' in c]
    # count = 16

    ## Tweet Metadata Total (Sentiment) Feature
    tweet_metadata_columns = content_language_columns + sentiment_columns + ['mean_favourites_per_tweet', 'mean_retweets_per_tweet', 'no_retweet_tweets', 'retweet_as_tweet_rate']
    print(f'There are {len(tweet_metadata_columns)} Tweet Metadata Features')
    print()
    print(tweet_metadata_columns)
    print()

    ## Tweet Temporal Feature
    temporal_columns = list(datasets[0].columns)[39+19:]
    temporal_columns.remove('mean_favourites_per_tweet')
    temporal_columns.remove('mean_retweets_per_tweet')
    temporal_columns.remove('no_retweet_tweets')
    temporal_columns.remove('retweet_as_tweet_rate')

    print(f'There are {len(temporal_columns)} Tweet Temporal Feature')
    print()
    print(temporal_columns)
    print()
    ## count = 16

    total_features = user_metadata_columns + tweet_metadata_columns + temporal_columns
    diff = set(datasets[0].columns).difference(set(total_features))
    print(diff)

    print(f'There are a total of {len(datasets[0].columns)} features')
    ## Total Features 69

    print("Hello")

    # Step 4: Impute NaN values using MICE
    for i in range(len(datasets)):
        imputer = IterativeImputer(random_state=100, max_iter=10)
        # fit on the dataset
        imputer.fit(datasets[i])
        df_imputed = imputer.transform(datasets[i])
        # column_headers = list(datasets[i].columns.values)
        datasets[i].iloc[:, :] = df_imputed
        del imputer

    # Step 5: Spliting Features
    user_metadatas = [df[user_metadata_columns] for df in datasets]
    tweet_metadatas = [df[tweet_metadata_columns] for df in datasets]
    temporal_datas = [df[temporal_columns] for df in datasets]

    # Step 6: Getting Tweet+Description Embeddings features
    with open('Data/description_tweet_embeddings.pickle', 'rb') as f:
        data_tweet = pickle.load(f)

    ## Merge on each dataset
    df_embed = pd.DataFrame().from_dict(data_tweet)
    tweet_datas = []
    for i in range(len(datasets)):
        datasets_original[i] = datasets_original[i].merge(df_embed, on='id')
        tweet_datas.append(datasets_original[i][['clean_combined_embedding']])
    
    # Step 7: Normalise Features
    # Normalizing other tabular features
    scaler = StandardScaler()
    for i in range(len(datasets)):
        user_metadatas[i]= scaler.fit_transform(user_metadatas[i])
        tweet_metadatas[i] = scaler.fit_transform(tweet_metadatas[i])
        temporal_datas[i] = scaler.fit_transform(temporal_datas[i])

    # Step 8: Getting user metadata from linear layer
    torch.manual_seed(43)
    in_features = len(user_metadata_columns)
    user_linear = linear_layer(in_features, out_features)
    r_ums = [[] for _ in range(len(datasets))]
    for k in range(len(datasets)):
        for i in range(0, datasets[k].shape[0]):
            user_metadata = torch.tensor(user_metadatas[k].astype(np.float32))
            r = user_linear(user_metadata[i])
            r = r.detach().numpy()
            r = r.reshape(1, out_features)
            r_ums[k].append(r)
    
    # Step 9: Add user tweets
    user_tweets = []
    for k in range(len(datasets)):
        tweet_data_embd = []
        for i in range(0,tweet_datas[k].shape[0]):
            tweet_data_embd.append(tweet_datas[k].iloc[i][0])
        tweet_data_embd = np.array(tweet_data_embd, dtype=np.float32)
        user_tweets.append(torch.tensor(tweet_data_embd))
                
    # Step 9: Get Tweet Datas as 2D Array
    for i in range(len(datasets)):
        tweet_datas[i]['clean_combined_embedding'] =  tweet_datas[i]['clean_combined_embedding'].apply(lambda x: x[0])
        tweet_datas[i] = np.array(list(tweet_datas[i]['clean_combined_embedding']), dtype="float32")

    ## Getting user tweet from linear layer
    torch.manual_seed(43)
    user_tweet_linear = linear_layer(768, out_features)
    r_uts = [[] for _ in range(len(datasets))]
    for k in range(len(datasets)):
        for i in range(0, datasets[k].shape[0]):
            user_tweet = torch.tensor(tweet_datas[k])
            r = user_tweet_linear(user_tweet[i])
            r = r.detach().numpy()
            r = r.reshape(1, out_features)
            r_uts[k].append(r)

    # Step 10: Getting tweet metadata from linear layer
    torch.manual_seed(43)
    in_features = len(tweet_metadata_columns)
    tweet_metadata_linear = linear_layer(in_features, out_features)
    r_tms = [[] for _ in range(len(datasets))]
    for k in range(len(datasets)):
        for i in range(0, datasets[k].shape[0]):
            tweet_metadata = torch.tensor(tweet_metadatas[k].astype(np.float32))
            r = tweet_metadata_linear(tweet_metadata[i])
            r = r.detach().numpy()
            r = r.reshape(1, out_features)
            r_tms[k].append(r)
    
    # Getting tweet temporal from linear layer
    torch.manual_seed(43)
    in_features = len(temporal_columns)
    tweet_temporal_linear = linear_layer(in_features, out_features)
    r_tts = [[] for _ in range(len(datasets))]
    for k in range(len(datasets)):
        for i in range(0, datasets[k].shape[0]):
            temporal_data = torch.tensor(temporal_datas[k].astype(np.float32))
            r = tweet_temporal_linear(temporal_data[i])
            r = r.detach().numpy()
            r = r.reshape(1, out_features)
            r_tts[k].append(r)

    # Concatenation
    varol_reps = np.concatenate((r_ums[0], r_uts[0], r_tms[0], r_tts[0]),axis=2)
    varol_reps_shape = (varol_reps.shape[0], varol_reps.shape[-1])
    gilani_reps = np.concatenate((r_ums[-1], r_uts[-1], r_tms[-1],r_tts[-1]),axis=2)
    gilani_reps_shape = (gilani_reps.shape[0], gilani_reps.shape[-1])
    # Conversion
    varol_reps = varol_reps.tolist()
    gilani_reps = gilani_reps.tolist()
    # Reshape
    varol_reps = np.reshape(varol_reps,  varol_reps_shape)
    gilani_reps = np.reshape(gilani_reps, gilani_reps_shape)

    # Save this to use for SCARF, this is your embeddings for all three features (96) dimensions
    with open(f'Data/varol_test_{out_features}_with_tweets.npy', 'wb') as f:
        np.save(f, varol_reps)
    with open(f'Data/gilani_test_{out_features}_with_tweets.npy', 'wb') as f:
        np.save(f, gilani_reps)


if __name__ == "__main__":

    no_dims = [16, 32, 64, 128]
    for d in no_dims:
        run(d)
