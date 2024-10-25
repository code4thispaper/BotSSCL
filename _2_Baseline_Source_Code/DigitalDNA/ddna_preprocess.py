import json
from glob import glob

if __name__ == "__main__":
    
    folders = ["../../Data Collection/Datasets/gilani-2017", "../../Data Collection/Datasets/varol-icwsm"]
    
    for folder in folders:
        user_folders = glob(f"{folder}/users/*/timeline.json")
        timelines = []
        for user in user_folders:

            with open(user, "r") as f:
                data = json.load(f)
                f.close()
            if not data:
                continue
            for tweet in data:
                to_append = {
                    "user": {
                        "id": tweet['user']['id_str']
                    },
                    "in_reply_to_user_id": tweet['in_reply_to_user_id'],
                    "retweeted_status": {
                        "id": tweet['retweeted_status']['id_str'] if 'retweeted_status' in tweet else None 
                    },
                    "entities": tweet["entities"]
                }
                timelines.append(to_append)
                print(f"{tweet['user']['id']} added to timelines")
        
        with open(f"{folder.split('/')[-1]}_timelines.json", "w") as t:
            json.dump(timelines,t,indent=4)
            t.close()
        
        print(f"{folder.split('/')[-1]}")
            

