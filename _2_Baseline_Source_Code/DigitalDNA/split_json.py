import json
import pandas as pd
from glob import glob

if __name__ == "__main__":
    
    files = glob("*.json")
    
    for f in files:
        with open(f, "r") as g:
            data = json.load(g)
        
        df = pd.read_csv(f"../TwiBot-22/datasets/{f.split('_')[0]}/split.csv")
        df = df[df["split"] == "test"]
        df["id"] = df["id"].apply(lambda x: x[1:])
        tweets = []
        for tweet in data:
            if tweet["user"]["id"] in list(df["id"]):
                tweets.append(tweet) 
        with open(f, "w") as z:
            json.dump(tweets, z, indent=4)





