import string
import os
import sys
import pandas as pd
import ujson
from datetime import datetime, timedelta
from collections import Counter
from math import log2
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import emoji
import re
from scipy import stats
from nltk import download
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from copy import deepcopy
# Install nltk lexicons
download('punkt')
download('wordnet')
download('stopwords')
download('vader_lexicon')

TIMELINE_INFO = {
    # User Features
    'favourites_count': None,
    'status_count': None,
    'default_profile_timeline': None,
    # Content and Langauge Features
    'mean_no_emoticons': None,
    'mean_no_urls_per_tweet': None,
    'mean_no_media_per_tweet': None,
    'mean_no_words': None,
    'no_languages': None,
    'mean_no_hashtags': None,
    # Sentiment Analysis
    'mean_tweet_sentiment': None,
    'mean_positive_valence_score_per_tweet': None,
    'mean_negative_valence_score_per_tweet': None,
    'mean_neutral_valence_score_per_tweet': None,
    'positive_valence_score_of_aggregated_tweets': None,
    'negative_valence_score_of_aggregated_tweets': None,
    'neutral_valence_score_of_aggregated_tweets': None,
    'mean_positive_and_negative_score_ratio_per_tweet': None,
    'mean_emoticons_entropy_per_tweet': None,
    'mean_emoticons_entropy_of_aggregated_tweets': None,
    'mean_negative_emoticons_entropy_of_aggregated_tweets': None,
    'mean_positive_emoticons_entropy_of_aggregated_tweets': None,
    'mean_neutral_emoticons_entropy_of_aggregated_tweets': None,
    'mean_positive_emoticons_entropy_per_tweet': None,
    'mean_negative_emoticons_entropy_per_tweet': None,
    'mean_neutral_emoticons_entropy_per_tweet': None,
    'mean_number_of_positive_emoticons_per_tweet': None,
    'mean_number_of_negative_emoticons_per_tweet': None,
    'mean_number_of_neutral_emoticons_per_tweet': None,
    # Temporal Features
    'time_between_tweets': None,
    'tweet_frequency': None,
    'mean_favourites_per_tweet': None,
    'mean_retweets_per_tweet': None,
    'no_retweet_tweets': None,
    'retweet_as_tweet_rate': None,
    'min_tweets_per_hour': None,
    'min_tweets_per_day': None,
    'max_tweets_per_hour': None,
    'max_tweets_per_day': None,
    'max_occurence_of_same_gap': None,
    'mean_user_mentions_per_tweet': None,
    'unique_mention_rate_per_tweet': None,
    # Add user info
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


def get_consecutive_days(dates):
    l = dates[0]
    dayList = []
    numOfDays = (dates[-1] - dates[0]).days
    for n in range(numOfDays + 1):
        dayList.append(l + timedelta(n))
    return dayList


def get_consecutive_hours(dates):
    l = dates[0]
    dateList = []
    numOfHours = int((dates[-1] - dates[0]).total_seconds()/3600)
    for n in range(numOfHours + 1):
        dateList.append(l + (timedelta(hours=n)))
    return dateList


def get_max_min_tweets_per_day(timestamps):
    # https://github.com/idimitriadis/bot_detection_WDM/blob/master/features/temporal_features.py#L130
    # print('get max min tweets per day')
    dates = []
    for t in timestamps:
        tweet_date = datetime.fromtimestamp(t).date()
        dates.append(tweet_date)
    date_list = get_consecutive_days(dates)
    # print (date_list)
    c = Counter(dates)
    if len(date_list) > 0:
        for d in date_list:
            if d not in c:
                c[d] = 0
        # print (c.values())
        return min(c.values()), max(c.values())
    else:
        # print('get_max_min_tweets_per_day')
        return 0, 0


def get_max_min_tweets_per_hour(timestamps):
    # https://github.com/idimitriadis/bot_detection_WDM/blob/master/features/temporal_features.py#L130
    # print('get max min tweets per day')
    dates = []
    for t in timestamps:
        tweet_date = datetime.fromtimestamp(t)
        tweet_date = tweet_date.replace(minute=0, second=0)
        dates.append(tweet_date)
    date_list = get_consecutive_hours(dates)
    # print (date_list)
    c = Counter(dates)
    # print (c)
    if len(date_list) > 0:
        for d in date_list:
            if d not in c:
                c[d] = 0
        return min(c.values()), max(c.values())
    else:
        # print ('error get_max_min_tweets_per_hour')
        return 0, 0


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
    try:
        word_length = len(iter)
        counter = Counter(iter)
        entropy = 0.0
        for _, count in counter.items():
            probability = count / word_length
            entropy += -probability * log2(probability)
        return entropy
    except:
        return 0.0


def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%a %b %d %H:%M:%S +0000 %Y")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


def find_tweet_valence(tweet, analyser):
    d = analyser.polarity_scores(tweet)
    return d


def extract_emojis(s):
    return [k['emoji'] for k in emoji.emoji_list(s)]


def get_positive_emojis(emojis, analyser):
    es = []
    for e in emojis:
        metric = find_tweet_valence(e, analyser)['compound']
        if metric > 0.05:
            es.append(e)
    return es


def get_negative_emojis(emojis, analyser):
    es = []
    for e in emojis:
        metric = find_tweet_valence(e, analyser)['compound']
        if metric < -0.05:
            es.append(e)
    return es


def get_neutral_emojis(emojis, analyser):
    es = []
    for e in emojis:
        metric = find_tweet_valence(e, analyser)['compound']
        if metric > -0.05 and metric < 0.05:
            es.append(e)
    return es


def generate_df(data):
    created_at = []
    languages = []
    tweets = []
    no_urls = []
    no_media = []
    no_hashtags = []
    no_favourites = []
    no_retweets = []
    no_mentions = []
    for tweet in data:
        created_at_time = datetime.strptime(
            tweet['created_at'], "%a %b %d %H:%M:%S +0000 %Y").timestamp()
        created_at.append(created_at_time)
        languages.append(tweet['lang'])
        tweets.append(tweet['text'])
        no_favourites.append(tweet['favorite_count'])
        no_retweets.append(tweet['retweet_count'])
        # Count
        if 'media' in tweet['entities']:
            no_media.append(len(tweet['entities']['media']))
        else:
            no_media.append(0)
        if 'urls' in tweet['entities']:
            no_urls.append(len(tweet['entities']['urls']))
        else:
            no_urls.append(0)
        if 'hashtags' in tweet['entities']:
            no_hashtags.append(len(tweet['entities']['hashtags']))
        else:
            no_hashtags.append(0)
        if 'mentions' in tweet['entities']:
            no_mentions.append(len(tweet['entities']['mentions']))
        else:
            no_mentions.append(0)

    # Add raw data to df
    df = pd.DataFrame()
    df['urls'] = no_urls
    df['hashtags'] = no_hashtags
    df['media'] = no_media
    df['created_at'] = created_at
    df['languages'] = languages
    df['tweet'] = tweets
    df['retweet_count'] = no_retweets
    df['favourites_count'] = no_favourites
    df['mentions_count'] = no_mentions
    return df


def tweet_extracted_features(timeline_file, date_scraped):

    timeline_info = deepcopy(TIMELINE_INFO)
    timeline_info['id'] = timeline_file.split('/')[-2]
    timeline_info['dataset'] = timeline_file.split('/')[-4]

    with open(timeline_file, "r") as f:
        data = ujson.load(f)
        f.close()
    df = generate_df(data)

    if len(df['tweet']) == 0:
        return timeline_info

    # User Features
    timeline_info['favourites_count'] = data[0]['user']['favourites_count']
    timeline_info['status_count'] = data[0]['user']['statuses_count']
    timeline_info['default_profile_timeline'] = int(
        data[0]['user']['default_profile'])

    # Content and Langauge Features
    timeline_info['mean_no_emoticons'] = df['tweet'].apply(
        lambda x: emoji.emoji_count(x)).mean()
    timeline_info['mean_no_urls_per_tweet'] = df['urls'].mean()
    timeline_info['mean_no_media_per_tweet'] = df['media'].mean()
    timeline_info['mean_no_words'] = df['tweet'].apply(
        lambda x: len(x.split())).mean()
    timeline_info['no_languages'] = len(df['languages'].unique())
    timeline_info['mean_no_hashtags'] = df['hashtags'].mean()

    # Sentiment Analysis
    df['clean_tweet'] = df['tweet'].apply(lambda x: cleanString(x))
    analyser = SentimentIntensityAnalyzer()
    df['sentiment'] = df['clean_tweet'].apply(
        lambda x: find_tweet_valence(x, analyser))
    timeline_info['mean_tweet_sentiment'] = df['sentiment'].apply(
        lambda x: x['compound']).mean()
    timeline_info['mean_positive_valence_score_per_tweet'] = df['sentiment'].apply(
        lambda x: x['pos']).mean()
    timeline_info['mean_negative_valence_score_per_tweet'] = df['sentiment'].apply(
        lambda x: x['neg']).mean()
    timeline_info['mean_neutral_valence_score_per_tweet'] = df['sentiment'].apply(
        lambda x: x['neu']).mean()
    try:
        timeline_info['mean_positive_and_negative_score_ratio_per_tweet'] = timeline_info['mean_positive_valence_score_per_tweet'] / \
            timeline_info['mean_negative_valence_score_per_tweet']
    except:
        timeline_info['mean_positive_and_negative_score_ratio_per_tweet'] = 0
    agg_tweets_clean = "\n".join(df['clean_tweet'])
    agg_tweets_raw = "\n".join(df['tweet'])
    agg_valence_scores = find_tweet_valence(agg_tweets_clean, analyser)
    timeline_info['positive_valence_score_of_aggregated_tweets'] = agg_valence_scores['pos']
    timeline_info['negative_valence_score_of_aggregated_tweets'] = agg_valence_scores['neg']
    timeline_info['neutral_valence_score_of_aggregated_tweets'] = agg_valence_scores['neu']
    df['emoticons'] = df['tweet'].apply(lambda x: extract_emojis(x))
    timeline_info['mean_emoticons_entropy_per_tweet'] = df['emoticons'].apply(
        lambda x: calculate_entropy(x)).mean()
    timeline_info['mean_emoticons_entropy_of_aggregated_tweets'] = calculate_entropy(
        extract_emojis(agg_tweets_raw))
    timeline_info['mean_negative_emoticons_entropy_of_aggregated_tweets'] = calculate_entropy(
        get_negative_emojis(extract_emojis(agg_tweets_raw), analyser))
    timeline_info['mean_positive_emoticons_entropy_of_aggregated_tweets'] = calculate_entropy(
        get_positive_emojis(extract_emojis(agg_tweets_raw), analyser))
    timeline_info['mean_neutral_emoticons_entropy_of_aggregated_tweets'] = calculate_entropy(
        get_neutral_emojis(extract_emojis(agg_tweets_raw), analyser))
    df['pos_emoticons'] = df['emoticons'].apply(
        lambda x: get_positive_emojis(x, analyser))
    timeline_info['mean_positive_emoticons_entropy_per_tweet'] = df['pos_emoticons'].apply(
        lambda x: calculate_entropy(x)).mean()
    timeline_info['mean_number_of_positive_emoticons_per_tweet'] = df['pos_emoticons'].apply(
        lambda x: len(x)).mean()
    df['neg_emoticons'] = df['emoticons'].apply(
        lambda x: get_negative_emojis(x, analyser))
    timeline_info['mean_negative_emoticons_entropy_per_tweet'] = df['neg_emoticons'].apply(
        lambda x: calculate_entropy(x)).mean()
    timeline_info['mean_number_of_negative_emoticons_per_tweet'] = df['neg_emoticons'].apply(
        lambda x: len(x)).mean()
    df['neu_emoticons'] = df['emoticons'].apply(
        lambda x: get_neutral_emojis(x, analyser))
    timeline_info['mean_neutral_emoticons_entropy_per_tweet'] = df['neu_emoticons'].apply(
        lambda x: calculate_entropy(x)).mean()
    timeline_info['mean_number_of_neutral_emoticons_per_tweet'] = df['neu_emoticons'].apply(
        lambda x: len(x)).mean()

    # Temporal Features
    df.sort_values(by=['created_at'], inplace=True)
    df['delta'] = (df['created_at'] - df['created_at'].shift()).fillna(0)
    timeline_info['time_between_tweets'] = df['delta'].mean()
    timeline_info['tweet_frequency'] = timeline_info['status_count'] / \
        days_between(data[0]['user']['created_at'], date_scraped)
    timeline_info['mean_favourites_per_tweet'] = df['favourites_count'].mean()
    timeline_info['mean_retweets_per_tweet'] = df['retweet_count'].mean()
    df['retweet'] = df['tweet'].apply(
        lambda x: int(re.search("^RT\s", x) != None))
    timeline_info['no_retweet_tweets'] = df['retweet'].sum()
    timeline_info['retweet_as_tweet_rate'] = df['retweet'].mean()
    max_min_hour = get_max_min_tweets_per_hour(df['created_at'])
    max_min_day = get_max_min_tweets_per_day(df['created_at'])
    timeline_info['min_tweets_per_hour'] = max_min_hour[0]
    timeline_info['min_tweets_per_day'] = max_min_day[0]
    timeline_info['max_tweets_per_hour'] = max_min_hour[1]
    timeline_info['max_tweets_per_day'] = max_min_day[1]
    timeline_info['max_occurence_of_same_gap'] = Counter(
        df['delta']).most_common(1)[0][1]
    timeline_info['mean_user_mentions_per_tweet'] = df['mentions_count'].mean()
    timeline_info['unique_mention_rate_per_tweet'] = df['mentions_count'].apply(
        lambda x: x == 1).mean()

    return timeline_info


if __name__ == "__main__":

    # Text file of user folder paths (they contain the paths to all of the user folders)
    paths = "paths.txt"

    with open(paths, "r") as f:
        users = f.read().splitlines()

    df = pd.DataFrame()
    for user in users:
        timeline_file = user + "/timeline.json"
        if not os.path.exists(timeline_file):
            continue

        if 'varol' in user:
            date_scraped = '2023-03-05'
        else:
            date_scraped = '2023-03-07'

        feature_per_user = tweet_extracted_features(timeline_file, date_scraped)
        print(f"{user.split('/')[-1]} Extracted")
        df_dictionary = pd.DataFrame([feature_per_user])
        df = pd.concat([df, df_dictionary], ignore_index=True)

    df.to_csv("timeline_clean.csv", index=False)
