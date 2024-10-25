import ujson
from glob import glob
import pandas as pd

def process_timeline(filename, df: pd.DataFrame):
    with open(filename, "r") as f:
        data = ujson.load(f)
    if len(data) == 0:
        return
    for row in data:
        df.loc[len(df)] = ["post", 'u' + row['user']['id_str'], 't' + row['id_str']]

def process_followers(filename, users, df):
    user = filename.split('/')[-2]
    with open(filename, "r") as f:
        data = ujson.load(f)
    if None in data or len(data) == 0:
        return
    for id in data[0]['ids']:
        if str(id) in users:
            df.loc[len(df)] = ["follower", 'u' + user, 'u' + str(id)]

def process_following(filename, users, df):
    user = filename.split('/')[-2]
    with open(filename, "r") as f:
        data = ujson.load(f)
    if None in data or len(data) == 0:
        return
    for id in data[0]:
        if id['id'] in users:
            df.loc[len(df)] = ["follows", 'u' + user, 'u' + id['id']]


if __name__ == "__main__":

    folders = [
        "../../../Data Collection/Datasets/gilani-2017/users",
        "../../../Data Collection/Datasets/varol-icwsm/users",
    ]

    for folder in folders:

        dataset = folder.split('/')[-2]
        
        users = glob(folder + "/*")
        
        print(f"{folder} finished globbing users")

        df = pd.DataFrame({
            "relation": [],
            "source_id": [],
            'target_id': []
        })
        users_ids = [k.split('/')[-1] for k in users]

        for user in users:

            print(user)

            no_files = len(glob(user + "/*"))

            if no_files == 0:
                continue

            process_timeline(user + "/timeline.json", df)

            if no_files != 4:
                continue

            process_followers(user + '/followers.json', users, df)
            process_following(user + '/following.json', users, df)

        df.to_csv(f"{folder.split('/')[-2]}/edge.csv", index=False)

        print(f"{folder} finished generating edge")

