from sentence_transformers import SentenceTransformer
import numpy as np
import ujson
import pandas as pd
import os
import pickle
import emoji
import re


def cleanString(s):
    clean_s = emoji.replace_emoji(s, '')
    clean_s = re.sub(r'http\S+', '', clean_s)
    clean_s = re.sub('#', '', clean_s)
    clean_s = re.sub('/\s\s+/g', '', clean_s)
    clean_s = clean_s.rstrip()
    clean_s = clean_s.lstrip()
    return clean_s


if __name__ == "__main__":

    # Text file of user folder paths (they contain the paths to all of the user folders)
    paths = "paths_twibot.txt"
    # Bert model
    model = SentenceTransformer('bert-base-nli-mean-tokens', device='cuda')

    with open(paths, "r") as f:
        users = f.read().splitlines()

    ids = []
    embeddings = []

    for user in users:

        if not os.path.exists(user + "/timeline.json") or not os.path.exists(user + "/profile.json"):
            continue

        # Open files
        with open(user + "/timeline.json", 'r') as f:
            data_timeline = ujson.load(f)
        with open(user + "/profile.json", 'r') as f:
            data_profile = ujson.load(f)

        # Append description embedding
        clean_desc = [cleanString(data_profile['description'])]
        desc_embedding = model.encode(clean_desc)
        tweet_embeddings = []

        # Append tweet embeddings
        tweet_embeddings = []
        for tweet in data_timeline:
            tweet_embedding = model.encode([cleanString([tweet['text']])])
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
        
        
        ids.append(int(user.split('/')[-1]))

        del data_timeline
        del data_profile
        del clean_desc

    df = pd.DataFrame()
    df['clean_combined_embedding'] = embeddings
    df['id'] = ids

    print(len(embeddings))
    data = df.to_dict()
    with open('description_tweet_embeddings_twibot.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
