import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import torch
import pickle
from time import time
import sys

def linear_layer(in_features, out_features):
    linear = torch.nn.Linear(in_features, out_features)
    return linear
    
def run(out_features, d_name):
    # Step 1: Extract Datasets
    # varol_data = pd.read_csv(f'Data/varol-icwsm-{mode}.csv')
    gilani_data = pd.read_csv(f'Data/{d_name}-clean-sample-modified-single-tweets.csv')
    datasets = [gilani_data]

    # Step 2: Running checks for unique columns and removing some additional information columns
    for i in range(len(datasets)):
        datasets[i]['mean_positive_and_negative_score_ratio_per_tweet'] = datasets[i]['mean_positive_and_negative_score_ratio_per_tweet'].fillna(0)
        datasets[i]['mean_positive_and_negative_score_ratio_per_tweet'] = datasets[i]['mean_positive_and_negative_score_ratio_per_tweet'].replace([-np.inf, np.inf], 0)

    ## Drop all unique columns
    for i in range(len(datasets)):
        constant_cols = ['profile_image_url', 'hashtag_in_username', 'emojis_in_username', 'mean_user_mentions_per_tweet', 'unique_mention_rate_per_tweet']
        datasets[i].drop(columns=constant_cols, inplace=True)

    # Step 3: Feature Split (For just understanding)
    ## User Based/Metadata Feature
    user_metadata_columns = ['followers_count', 'friends_count', 'listed_count', 'verified', 'user_age', 'follower_growth_rate', 'friends_growth_rate', 'listed_growth_rate', 'followers_friend_ratio', 'name_length', 'username_length', 'description_length', 'num_digits_in_name', 'num_digits_in_username', 'names_ratio', 'name_freq', 'name_entropy', 'username_entropy', 'description_entropy', 'description_sentiment', 'names_sim', 'url_in_description', 'bot_in_names', 'hashtag_in_description', 'hashtag_in_name', 'numbers_in_description', 'numbers_in_name', 'numbers_in_username', 'emojis_in_description', 'emojis_in_name', 'favourites_count', 'status_count', 'default_profile']

    ## Tweet Metadata Total (Sentiment) Feature
    tweet_metadata_columns = ['mean_no_emoticons', 'mean_no_urls_per_tweet', 'mean_no_media_per_tweet', 'mean_no_words', 'no_languages', 'mean_no_hashtags', 'mean_number_of_positive_emoticons_per_tweet', 'mean_number_of_negative_emoticons_per_tweet', 'mean_number_of_neutral_emoticons_per_tweet', 'mean_tweet_sentiment', 'mean_positive_valence_score_per_tweet', 'mean_negative_valence_score_per_tweet', 'mean_neutral_valence_score_per_tweet', 'positive_valence_score_of_aggregated_tweets', 'negative_valence_score_of_aggregated_tweets', 'neutral_valence_score_of_aggregated_tweets', 'mean_positive_and_negative_score_ratio_per_tweet', 'mean_emoticons_entropy_per_tweet', 'mean_emoticons_entropy_of_aggregated_tweets', 'mean_negative_emoticons_entropy_of_aggregated_tweets', 'mean_positive_emoticons_entropy_of_aggregated_tweets', 'mean_neutral_emoticons_entropy_of_aggregated_tweets', 'mean_positive_emoticons_entropy_per_tweet', 'mean_negative_emoticons_entropy_per_tweet', 'mean_neutral_emoticons_entropy_per_tweet', 'mean_favourites_per_tweet', 'mean_retweets_per_tweet', 'no_retweet_tweets', 'retweet_as_tweet_rate']

    ## Tweet Temporal Feature
    temporal_columns = ['time_between_tweets', 'tweet_frequency', 'min_tweets_per_hour', 'min_tweets_per_day', 'max_tweets_per_hour', 'max_tweets_per_day', 'max_occurence_of_same_gap']

    # Step 4: Impute NaN values using MICE
    # No imputation needed since there is no NAN values
    '''
    for i in range(len(datasets)):
        imputer = IterativeImputer(random_state=100, max_iter=10)
        # fit on the dataset
        imputer.fit(datasets[i])
        df_imputed = imputer.transform(datasets[i])
        # column_headers = list(datasets[i].columns.values)
        datasets[i].iloc[:, :] = df_imputed
        del imputer
    '''

    # Step 6: Getting Tweet+Description Embeddings features
    with open('Data/description_sample_modified_single_tweets_embeddings.pickle', 'rb') as f:
        data_tweet = pickle.load(f)
    
    ## Merge each dataset by creating unique keys
    df_embed = pd.DataFrame().from_dict(data_tweet)
    df_embed['merge_id'] = df_embed.apply(lambda x: f"{x['id']}_{x['tweet_index']}", axis=1)
    for i in range(len(datasets)):
        datasets[i]['merge_id'] = datasets[i].apply(lambda row: f"{row['id']}_{row['tweet_index']}", axis=1)
        datasets[i] =  datasets[i].merge(df_embed, on='merge_id')
        # Sort datasets (same ids even if seper)
        datasets[i] = datasets[i].sort_values(by=['id_x', 'tweet_index_x'])
        datasets[i].reset_index(drop=True, inplace=True)                                              

    for i in range(len(datasets)):
        
        df = datasets[i]

        for user in df['id_x'].unique():

            user_df = df[df['id_x'] == user]

            for index, row in user_df.iterrows():
                
                df_excluded_selected_user = df[(df['id_x'] != user) & (df['tweet_index_x'] == 0)].copy()
                df_included_selected_user = df_excluded_selected_user
                df_included_selected_user.loc[df_included_selected_user.shape[0]] = row

                # User metadata is the same
                user_metadata = df_included_selected_user[user_metadata_columns]
                # Temporal metadata is assumed to be same
                temporal_data = df_included_selected_user[temporal_columns]
                # Tweet Metadata
                tweet_metadata = df_included_selected_user[tweet_metadata_columns]
                # Get embeddings
                tweet_datas = df_included_selected_user['clean_combined_embedding']

                # Step 7: Normalising Features
                scaler = StandardScaler()
                user_metadatas = scaler.fit_transform(user_metadata)
                temporal_datas = scaler.fit_transform(temporal_data) 
                tweet_metadatas = scaler.fit_transform(tweet_metadata)
            
                # Step 8: Getting user metadata from linear layer
                torch.manual_seed(43)
                in_features = len(user_metadata_columns)
                user_linear = linear_layer(in_features, out_features)
                r_um = []
                for i in range(0, df_included_selected_user.shape[0]):
                    user_metadata = torch.tensor(user_metadatas.astype(np.float32))
                    r = user_linear(user_metadata[i])
                    r = r.detach().numpy()
                    r = r.reshape(1, out_features)
                    r_um.append(r)
        
                # Step 9: Get Tweet Datas as 2D Array
                tweet_datas =  tweet_datas.apply(lambda x: x[0])
                tweet_datas = np.array(tweet_datas.tolist(), dtype="float32")
                torch.manual_seed(43)
                user_tweet_linear = linear_layer(768, out_features)
                r_ut = []
                for i in range(0, df_included_selected_user.shape[0]):
                    user_tweet = torch.tensor(tweet_datas)
                    r = user_tweet_linear(user_tweet[i])
                    r = r.detach().numpy()
                    r = r.reshape(1, out_features)
                    r_ut.append(r)
    
                # Step 10: Getting tweet metadata from linear layer
                torch.manual_seed(43)
                in_features = len(tweet_metadata_columns)
                tweet_metadata_linear = linear_layer(in_features, out_features)
                r_tm = []
                for i in range(0, df_included_selected_user.shape[0]):
                    tweet_metadata = torch.tensor(tweet_metadatas.astype(np.float32))
                    r = tweet_metadata_linear(tweet_metadata[i])
                    r = r.detach().numpy()
                    r = r.reshape(1, out_features)
                    r_tm.append(r)

                # Getting tweet temporal from linear layer
                torch.manual_seed(43)
                in_features = len(temporal_columns)
                tweet_temporal_linear = linear_layer(in_features, out_features)
                r_tt = []
                for i in range(0, df_included_selected_user.shape[0]):
                    temporal_data = torch.tensor(temporal_datas.astype(np.float32))
                    r = tweet_temporal_linear(temporal_data[i])
                    r = r.detach().numpy()
                    r = r.reshape(1, out_features)
                    r_tt.append(r)

                # Concatenation
                reps = np.concatenate((r_um, r_ut, r_tm, r_tt),axis=2)
                reps_shape = (reps.shape[0], reps.shape[-1])
                list_reps = reps.tolist()
                final_reps = np.reshape(list_reps, reps_shape)

                # Save this to use for SCARF, this is your embeddings for all three features (96) dimensions
                # with open(f'Data/varol_test_{out_features}_with_tweets.npy', 'wb') as f:
                #    np.save(f, varol_reps)
                with open(f'Data/{d_name}_test_{user}_{row["tweet_index_x"]}_{out_features}_with_sample_modified_tweets.npy', 'wb') as f:
                    np.save(f, final_reps)
        
            print(f"User {user} is finished")


if __name__ == "__main__":

    a = time()
    run(64, "gilani-2017")
    b = time()
    print(f"Ttime taken to create representation: {b - a}")
    # run(16, "varol")


