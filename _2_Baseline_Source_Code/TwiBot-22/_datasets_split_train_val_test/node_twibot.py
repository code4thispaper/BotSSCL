import ujson
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":

    # Generate nodes
    folders = [
        "../../../Data Collection/Datasets/twibot-s1",
        "../../../Data Collection/Datasets/twibot-s2"
    ]

    # Add u
    for folder in folders:
        profiles = []
        users = glob(f"{folder}/users/*/profile.json")
        print(f"{folder} {len(users)}")
        print("Started")
        for user in users:
            with open(user, "r") as f:
                data = ujson.load(f)
                if not 'location' in data:
                    data['location'] = ""
                profiles.append(data)
        new_folder = folder.split('/')[-1]
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        with open(new_folder + "/node.json", "w") as g:
            ujson.dump(profiles, g, indent = 4)
            g.close()
        print(f"Finished {new_folder}")