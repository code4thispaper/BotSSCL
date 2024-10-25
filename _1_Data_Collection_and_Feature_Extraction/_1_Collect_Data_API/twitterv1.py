import requests
from requests_oauthlib import OAuth1
import json
import os
from auxillary import ratelimit_manager

########################
### twitterv1 Class ###
########################
class twitterv1:

    def __init__(self, config_file: str):
        self.limiter = ratelimit_manager.RateLimiter()
        self.url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
        # Get auth data
        with open(f'data_collection/{config_file}', 'r') as f:
            jsondata = json.loads(f.read())
        self.auth = OAuth1(jsondata['API_KEY'], jsondata['API_SECRET'],
                    jsondata['ACCESS_TOKEN'], jsondata['ACCESS_TOKEN_SECRET'])

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

    def connect_to_endpoint(self, url: str, params={}) -> dict:
        """
        Connecting to endpoint with parameters and url given. Will raise error if return code is not 200.
        """
        while True:
            response = requests.get(url, auth=self.auth, params=params)
            if response.status_code != 200:
                if response.status_code == 429:
                    self.limiter.too_many_requests(url)
                    continue
                if response.status_code == 401:
                    break
                if response.status_code == 500:
                    self.limiter.internal_error()
                    continue
                raise Exception(response.status_code, response.text)
            return response.json()

    #####################
    ### GET FUNCTIONS ###
    #####################
    def get_followers(self, user_id: str):
        params = {
            "user_id": user_id,
            "count": 200
        }
        followers = []
        self.limiter.check("followers")
        temp_url = "https://api.twitter.com/1.1/followers/ids.json"
        response = self.connect_to_endpoint(temp_url, params)
        try:
            followers.append(response)
        except Exception as e:
            print(e)
        return followers
    
    
    def get_user_timeline(self,
                        user_id: str = None,
                        screen_name: str = None,
                        since_id: int = None,
                        count: int = 200,
                        max_id: int = None,
                        exclude_replies: bool = False,
                        include_rts: bool = True) -> list | dict:
        """
        Returns a collection of the most recent Tweets posted by the user indicated by the screen_name or user_id parameters.

        Parameters
        ----------
        * user_id: (optional) Twitter User ID
        * screen_name: (optional) Twitter username
        * since_id: (optional) Specified first tweet to collect from
        * count: (optional) Number of tweets (default 200). If > 200, function loops
        * max_id: (optional) Specified last tweet to collect from
        * exclude_replies: (optional) Default to True, excludes reply tweet objects
        * include_rts: (optional) Default to False, includes retweets

        Returns
        -------
        List of timeline objects based on count % 200 [timeline object]

        """
        if (user_id == None and screen_name == None) or (user_id != None and screen_name != None):
            raise Exception(
                "get_user_timeline must have either user_id or screen_name")
        if (count < 0):
            raise Exception("Count needs to be a positive integer")
        params = {
            'count': count,
            'exclude_replies': exclude_replies,
            'include_rts': include_rts
        }
        if (user_id):
            params['user_id'] = user_id
        if (screen_name):
            params['user_id'] = user_id
        if (since_id):
            params['since_id'] = since_id
        if (max_id):
            params['max_id'] = max_id

        timeline = []
        while (count > 200):
            self.limiter.check('user_timeline')
            params['count'] = 200
            self.limiter.check("user_timeline")
            response = self.connect_to_endpoint(self.url, params)
            if (response != None):
                timeline.extend(response)
            try:
                params['since_id'] = response[-1]['id']
            except:
                return timeline
            count -= 200

        params['count'] = count
        self.limiter.check("user_timeline")
        response = self.connect_to_endpoint(self.url, params)
        if (response != None):
            timeline.extend(response)
        return timeline
