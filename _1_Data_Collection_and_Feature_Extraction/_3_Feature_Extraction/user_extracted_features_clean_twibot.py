from copy import deepcopy
import emoji
import numpy as np
import os
import sys
import re
import pandas as pd
import ujson
from datetime import datetime
from collections import Counter
from math import log2
from collections import Counter
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import emoji
nltk.download('wordnet')
nltk.download('stopwords')


PROFILE_INFO = {
    "followers_count": None,
    "status_count": None,
    "tweet_frequency": None,
    "friends_count": None,
    "listed_count": None,
    "default_profile": None,
    "verified": None,
    "user_age": None,
    "follower_growth_rate": None,
    "friends_growth_rate": None,
    "listed_growth_rate": None,
    "followers_friend_ratio": None,
    "name_length": None,
    "username_length": None,
    "description_length": None,
    "num_digits_in_name": None,
    "num_digits_in_username": None,
    "names_ratio": None,
    "name_freq": None,
    "name_entropy": None,
    "username_entropy": None,
    'clean_description': None,
    "description_entropy": None,
    "description_sentiment": None,
    "names_sim": None,
    "profile_image_url": None,
    'url_in_description': None,
    'bot_in_names': None,
    'hashtag_in_description': None,
    'hashtag_in_name': None,
    'hashtag_in_username': None,
    'numbers_in_description': None,
    "numbers_in_name": None,
    "numbers_in_username": None,
    'emojis_in_description': None,
    "emojis_in_name": None,
    "emojis_in_username": None,
    'id': None,
    'dataset': None
}


def cleanString(s):
    clean_s = emoji.replace_emoji(s, '')
    clean_s = re.sub(r'http\S+', '', clean_s)
    clean_s = re.sub('#', '', clean_s)
    clean_s = re.sub('/\s\s+/g', '', clean_s)
    clean_s = clean_s.rstrip()
    clean_s = clean_s.lstrip()
    return clean_s


def emojisInString(s):
    return int(emoji.emoji_count(s) > 0)


def numbersInString(s):
    return int(re.search(r'\d+', s) != None)


def hasHashtag(s):
    return int(re.search("#[^ ]+", s) != None)


def botInString(s1, s2):
    m1 = re.search("bot", s1.lower()) != None
    m2 = re.search("bot", s2.lower()) != None
    return int(m1 or m2)


def checkURL(s):
    # from https://www.studytonight.com/python-programs/python-program-to-check-for-url-in-a-string
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.search(regex, s)
    return int(url != None)


def text_similarity(t1, t2):
    # Tokenize and lemmatize the texts
    text1 = ''.join(filter(lambda x: x in string.printable, t1))
    text2 = ''.join(filter(lambda x: x in string.printable, t2))

    if not text1 or not text2:
        return 0

    try:
        tokens1 = word_tokenize(text1)
        tokens2 = word_tokenize(text2)
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

        # Remove stopwords
        stop_words = stopwords.words('english')
        tokens1 = [token for token in tokens1 if token not in stop_words]
        tokens2 = [token for token in tokens2 if token not in stop_words]

        # Create the TF-IDF vectors
        vectorizer = TfidfVectorizer()
        vector1 = vectorizer.fit_transform(tokens1)
        vector2 = vectorizer.transform(tokens2)

        # Calculate the cosine similarity
        similarity = cosine_similarity(vector1, vector2)
        np_sim = np.array(similarity)
        sim_avg = np_sim.sum()/np_sim.size

        return sim_avg

    except:
        return 0


def calculate_entropy(iter):
    # Shannon entropy
    word_length = len(iter)
    counter = Counter(iter)
    entropy = 0.0
    for _, count in counter.items():
        probability = count / word_length
        entropy += -probability * log2(probability)
    return entropy


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S+00:00")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


def user_extracted_features(profile_file, date):

    profile_info = deepcopy(PROFILE_INFO)
    profile_info['id'] = profile_file.split('/')[-2]
    profile_info['dataset'] = profile_file.split('/')[-4]

    with open(profile_file, "r") as f:
        data = ujson.load(f)
        f.close()

    # Age
    profile_info['user_age'] = days_between(data['created_at'], date)

    # Bigram Info
    try:
        bigram_count = Counter(
            map(''.join, zip(data['name'], data['name'][1:])))
        profile_info['name_freq'] = sum(
            bigram_count.values())/len(bigram_count.values())
    except:
        profile_info['name_freq'] = 0

    # Entropy
    clean_desc = cleanString(data['description'])
    profile_info['clean_description'] = clean_desc
    profile_info['name_entropy'] = calculate_entropy(data['name'])
    profile_info['username_entropy'] = calculate_entropy(data['username'])
    profile_info['description_entropy'] = calculate_entropy(clean_desc)
    profile_info['description_sentiment'] = SentimentIntensityAnalyzer(
    ).polarity_scores(clean_desc)['compound']
    profile_info['names_sim'] = text_similarity(data['name'], data['username'])

    # Follower friend ratio
    try:
        profile_info['followers_friend_ratio'] = data['public_metrics']['followers_count'] / \
            data['public_metrics']['following_count']
    except:
        profile_info['followers_friend_ratio'] = 0
    
    # Default profile
    profile_info['default_profile'] = 1 if re.match("default", data["profile_image_url"]) else 0

    # Temporal Features
    try:
        profile_info['status_count'] = profile_info['public_metrics']['tweet_count']
    except:
        print(profile_info)
        sys.exit(1)
    profile_info['tweet_frequency'] = profile_info['status_count'] / \
        days_between(data[0]['user']['created_at'], date)

    # Easily extractable information
    profile_info['followers_count'] = data['public_metrics']['followers_count']
    profile_info['friends_count'] = data['public_metrics']['following_count']
    profile_info['listed_count'] = data['public_metrics']['listed_count']
    profile_info['verified'] = int(data['verified'])
    profile_info['follower_growth_rate'] = data['public_metrics']['followers_count'] / \
        profile_info['user_age']
    profile_info['friends_growth_rate'] = data['public_metrics']['following_count'] / \
        profile_info['user_age']
    profile_info['listed_growth_rate'] = data['public_metrics']['listed_count'] / \
        profile_info['user_age']
    profile_info['name_length'] = len(data['name'])
    profile_info['username_length'] = len(data['username'])
    profile_info['description_length'] = len(data['description'])
    profile_info['num_digits_in_name'] = sum(
        c.isdigit() for c in data['name'])
    profile_info["num_digits_in_username"] = sum(
        c.isdigit() for c in data['username'])
    profile_info['names_ratio'] = len(data['name'])/len(data['username'])
    profile_info['profile_image_url'] = 1 if 'profile_image_url' in data else 0
    profile_info['url_in_description'] = checkURL(data['description'])
    profile_info['bot_in_names'] = botInString(data['name'], data['username'])
    profile_info['hashtag_in_description'] = hasHashtag(data['description'])
    profile_info['hashtag_in_username'] = hasHashtag(data['username'])
    profile_info['hashtag_in_name'] = hasHashtag(data['name'])
    profile_info['numbers_in_description'] = numbersInString(
        data['description'])
    profile_info['numbers_in_name'] = numbersInString(data['name'])
    profile_info['numbers_in_username'] = numbersInString(data['username'])
    profile_info['emojis_in_description'] = emojisInString(data['description'])
    profile_info['emojis_in_name'] = emojisInString(data['name'])
    profile_info['emojis_in_username'] = emojisInString(data['username'])

    return profile_info


if __name__ == "__main__":

    # Text file of user folder paths (they contain the paths to all of the user folders)
    paths = "paths_twibot.txt"

    with open(paths, "r") as f:
        users = f.read().splitlines()

    df = pd.DataFrame()
    for user in users:
        profile_file = user + "/profile.json"
        if not os.path.exists(profile_file):
            continue

        if 'varol' in user:
            date_scraped = '2023-03-05'
        else:
            date_scraped = '2023-03-07'

        feature_per_user = user_extracted_features(
            profile_file, date_scraped)
        print(f"{user.split('/')[-1]} Extracted")
        df_dictionary = pd.DataFrame([feature_per_user])
        df = pd.concat([df, df_dictionary], ignore_index=True)

    df.to_csv(f"profile_clean_twibot.csv", index=False)
