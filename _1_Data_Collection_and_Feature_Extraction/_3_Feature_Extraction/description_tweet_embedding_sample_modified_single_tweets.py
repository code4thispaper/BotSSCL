from sentence_transformers import SentenceTransformer
import numpy as np
import ujson
import pandas as pd
import os
import pickle
import emoji
import re
import sys
from time import time

def cleanString(s):
    clean_s = emoji.replace_emoji(s, '')
    clean_s = re.sub(r'http\S+', '', clean_s)
    clean_s = re.sub('#', '', clean_s)
    clean_s = re.sub('/\s\s+/g', '', clean_s)
    clean_s = clean_s.rstrip()
    clean_s = clean_s.lstrip()
    return clean_s


if __name__ == "__main__":

    '''
    # Text file of user folder paths (they contain the paths to all of the user folders)
    paths = "paths.txt"
    with open(paths, "r") as f:
        users = f.read().splitlines()
    '''

    dataset = sys.argv[1]
    df = pd.read_csv(f"{dataset}-sample-modified-tweets.csv")

    # Bert model
    model = SentenceTransformer('bert-base-nli-mean-tokens', device='cuda')

    ids = []
    embeddings = []
    tweet_index = []

    a = time()

    for user in df['id'].unique()[:20]:

        # Open files
        with open(f"{dataset}/users/{user}/profile.json", 'r') as f:
            data_profile = ujson.load(f)

        # Append description embedding
        clean_desc = [cleanString(data_profile['description'])]
        desc_embedding = model.encode(clean_desc)

        # Append tweet embeddings
        df_tweets = df[df['id'] == user]
        for i in range(df_tweets.shape[0]):
            
            tweet_embeddings = []

            for tweet in df_tweets['rewritten_tweet'].iloc[:i]:
                tweet_embedding = model.encode([cleanString([tweet])])
                tweet_embeddings.append(tweet_embedding.tolist()[0])
                del tweet_embedding
            for tweet in df_tweets['original_tweet'].iloc[i:]:
                tweet_embedding = model.encode([cleanString([tweet])])
                tweet_embeddings.append(tweet_embedding.tolist()[0])
                del tweet_embedding
        
            # Average it out
            if len(tweet_embeddings) == 0:
                embeddings.append(desc_embedding)
            else:
                tweet_embeddings.append(desc_embedding[0])
                total_embedding = np.array(tweet_embeddings)
                embedding = total_embedding.mean(axis=0).tolist()
                embeddings.append([embedding])
                        
        
        ids += [user] * df_tweets.shape[0]
        # Just to differentiate tweets
        tweet_index += [k for k in range(df_tweets.shape[0])]
        print(f"{user} is finished")
        del clean_desc
        del data_profile
    
    b = time()

    print(f"Time took for BERT: {b - a}")

    df = pd.DataFrame()
    df['id'] = ids
    df['clean_combined_embedding'] = embeddings
    df['tweet_index'] = tweet_index

    print(len(embeddings))
    data = df.to_dict()
    with open('description_sample_modified_single_tweets_embeddings.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
