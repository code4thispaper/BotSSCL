import botometer
import json
import glob
import re
import os

if __name__ == "__main__":

    # Botometer API Key
    rapidapi_key = ""

    # Open either config to download data from Twitter API
    with open("config.json", "r") as f:
        config = json.load(f)
        f.close()

    twitter_app_auth = {
        'consumer_key': config['API_KEY'],
        'consumer_secret': config['API_SECRET'],
        'access_token': config['ACCESS_TOKEN'],
        'access_token_secret': config['ACCESS_TOKEN_SECRET'],
    }

    bom = botometer.Botometer(wait_on_ratelimit=True,
                            rapidapi_key=rapidapi_key,
                            **twitter_app_auth)        
    
    if not os.path.isdir("collected_data"):
        os.mkdir("collected_data")
    
    datasets = glob.glob("active_users/*.txt")
    for file in datasets:
        # Make folders
        tempStr = file.split('/')[-1]
        tempStr = re.sub(".txt", "", tempStr)
        folder = f"data/{tempStr}"
        # Check if folders exist
        if not os.path.isdir(folder):
            os.mkdir(folder)
        # Get list of users
        with open(file, "r") as f:
            ids = [line.rstrip() for line in f.readlines()]
            f.close()
        # Generate files
        for id, result in bom.check_accounts_in(ids):
            with open(f"{folder}/{id}.json", "w") as of:
                json.dump(result, of)
                of.close()
