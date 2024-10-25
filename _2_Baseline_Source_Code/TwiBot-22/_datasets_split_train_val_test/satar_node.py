import ujson
from glob import glob
import os

def run():

    # Generate nodes
    folders = [
        "../../Data Collection/Datasets/gilani-2017",
        "../../Data Collection/Datasets/varol-icwsm",
        "../../Data Collection/Datasets/twibot-s1",
        "../../Data Collection/Datasets/twibot-s2"
    ] 

    for data_folder in folders:
        user_folder = data_folder + "/users/*"
        user_folders = glob(user_folder)
        profiles = []
        for user in user_folders:
            print(user)
            if len(glob(user + "/*")) == 0:
                continue
            with open(user + "/profile.json", "r") as f:
                data = ujson.load(f)
                data['id'] = 'u' + data['id']
                if not 'location' in data:
                    data['location'] = ""
                profiles.append(data)
            with open(user + "/timeline.json", "r") as f:
                data = ujson.load(f)
                for t in data:
                    t['id_str'] = 't' + t['id_str']
                    t['id'] = t['id_str']
                    profiles.append(t)
        new_folder = data_folder.split('/')[-1]
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        with open(new_folder + f"/node_satar.json", "w") as g:
            ujson.dump(profiles, g, indent = 4)
            g.close()
        print(f"Finished {data_folder}")

if __name__ == "__main__":

    run()