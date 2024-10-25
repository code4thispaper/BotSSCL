import requests
import json
import sys
import os
from data_collection import ratelimit_manager

####################
# Helper functions #
####################

class twitterv2:

    def __init__(self, config_file: str):
        self.limiter = ratelimit_manager.RateLimiter()
        with open(f'data_collection/{config_file}', 'r') as f:
            jsondata = json.loads(f.read())
        self.bearer_token = jsondata['BEARER_TOKEN']

    def mkdir(name):
        """
        Creates a directory, but if one already exists, returns the size of the existing directory
        """

        try:
            os.mkdir(name)
        except:
            # Directory must exist - check size vs expected size
            return len(os.listdir(name))
        return 0

    def bearer_oauth(self, r):
        """
        Method required by bearer token authentication.
        """

        r.headers["Authorization"] = f"Bearer {self.bearer_token}"
        r.headers["User-Agent"] = "v2R"
        return r


    def connect_to_endpoint(self, url: str, params={}) -> dict:
        """
        Connecting to endpoint with parameters and url given. Will raise error if return code is not 200.
        """
        while True:
            response = requests.get(url, auth=self.bearer_oauth, params=params)
            if response.status_code != 200:
                if response.status_code == 429:
                    self.limiter.too_many_requests(url)
                    continue
                if response.status_code == 500:
                    self.limiter.internal_error()
                    continue
                raise Exception(response.status_code, response.text)
            return response.json()

    ##################
    # MAIN FUNCTIONS #
    ##################


    def get_users(self, ids: list | str) -> list | dict:
        """
        Returns a list of user objects based on the amount of user_ids given.

        Parameters
        ----------
        * ids: list of twitter user_id's

        Returns
        -------
        List of user objects

        """
        users = []
        url = "https://api.twitter.com/2/users"
        tweet_fields = "attachments,author_id,context_annotations,conversation_id,created_at,entities,geo,id,in_reply_to_user_id,lang,public_metrics,possibly_sensitive,referenced_tweets,reply_settings,source,text,withheld"
        user_fields = "created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified,withheld"

        params = {
            "tweet.fields": tweet_fields,
            "user.fields": user_fields
        }

        while (len(ids) > 100):
            params['ids'] = ",".join(ids[:100])
            self.limiter.check("users")
            response = self.connect_to_endpoint(url, params)
            try:
                data = response['data']
                users.extend(data)
            except:
                pass
            ids = ids[100:]
        params['ids'] = ",".join(ids)
        self.limiter.check("users")
        response = self.connect_to_endpoint(url, params)
        try:
            data = response['data']
            users.extend(data)
        except:
            pass

        return users


    def get_tweets(self, ids: list | str) -> list | dict:
        """
        Returns a list of tweet objects based on the amount of tweet_ids given.

        Parameters
        ----------
        * ids: list of twitter tweet_id's

        Returns
        -------
        List of tweet objects

        """
        tweets = []
        url = "https://api.twitter.com/2/tweets"
        expansions = "author_id"
        place_fields = "contained_within,country,country_code,full_name,geo,id,name,place_type"
        tweet_fields = "attachments,author_id,context_annotations,conversation_id,created_at,edit_controls,entities,geo,id,in_reply_to_user_id,lang,public_metrics,possibly_sensitive,referenced_tweets,reply_settings,source,text"
        user_fields = "created_at,description,entities,id,location,name,pinned_tweet_id,profile_image_url,protected,public_metrics,url,username,verified,withheld"

        params = {
            "ids": ids,
            "expansions": expansions,
            "place.fields": place_fields,
            "tweet.fields": tweet_fields,
            "user.fields": user_fields
        }

        while (len(ids) > 100):
            print(f'collecting {100}/{len(ids)}')
            params['ids'] = ",".join(ids[:100])
            self.limiter.check("tweets")
            response = self.connect_to_endpoint(url, params)
            try:
                data = response['data']
                tweets.extend(response['data'])
            except:
                pass
            ids = ids[100:]
        params['ids'] = ",".join(ids)
        self.limiter.check("tweets")
        response = self.connect_to_endpoint(url, params)
        try:
            data = response['data']
            tweets.extend(data)
        except:
            pass

        return tweets
    
    def get_following(self, id: str) -> list | dict:
        """
        Returns a list of tweet objects based on the amount of tweet_ids given.

        Parameters
        ----------
        * ids: list of twitter tweet_id's

        Returns
        -------
        List of tweet objects

        """
        following = []

        print(f'collecting following for {id}')
        params = {
            "max_results": 200
        }
        self.limiter.check("following")
        response = self.connect_to_endpoint(f'https://api.twitter.com/2/users/{id}/following', params)
        try:
            data = response['data']
            following.append(data)
        except Exception as e:
            print(e) 
            
        return following
    

