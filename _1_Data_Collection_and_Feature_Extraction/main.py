from _1_Collect_Data_API import twitterv1, twitterv2
import os
import sys
import json
import pandas as pd

def collect_users(
        config_file: str,
        workload: str,
        folder_title: str,
        user_filename: str,
        datasets_folder: str,
    ):
        
    df = pd.read_csv(f"{datasets_folder}/active_users/{dataset_file_name}")
    user_ids = list(df['id'])

    # Splitting workload into two - with argument of 0, will do first half, argument of 1 will do second half.
    mylen = len(user_ids)
    if (workload == '0'):
        user_ids = user_ids[:int(mylen/2)]
    elif (workload == '1'):
        user_ids = user_ids[int(mylen/2):]
    else:
        print("Proceeding with full amount")
    
    # Check if Datasets folders exist
    if not os.path.exists("Datasets"):
        try:
            os.mkdir(f"Datasets/")
        except Exception as e:
            print(e)
            sys.exit(1)
    if not os.path.exists(f"Datasets/{folder_title}/users"):
        try:
            os.mkdir(f"Datasets/{folder_title}")
            os.mkdir(f"Datasets/{folder_title}/users")
        except Exception as e:
            print(e)
            sys.exit(1)
                
    for i in range(len(user_ids)):
        print(f"{i + 1}/{len(user_ids)} users")
        if not os.path.exists(f"Datasets/{folder_title}/users/{user_ids[i]}"):
            try:
                os.mkdir(f"Datasets/{folder_title}/users/{user_ids[i]}")
            except Exception as e:
                print(e)
                sys.exit(1)
        twitter1 = twitterv1.twitterv1(config_file)
        twitter2 = twitterv2.twitterv2(config_file)
        try:
            print(user_ids[i], "profile")
            userjson = twitter2.get_users([user_ids[i]])[0]
            print(user_ids[i], "time")
            timelinejson = twitter1.get_user_timeline(
                user_id=user_ids[i], count=200)
            print(user_ids[i], "following")
            followingjson = twitter2.get_following(id=user_ids[i])
            print(user_ids[i], "followers")
            followersjson = twitter1.get_followers(user_id=user_ids[i])

        except:
            continue

        with open(f"Datasets/{folder_title}/users/{user_ids[i]}/profile.json", 'w') as f:
            f.write(json.dumps(userjson, indent=4))
        with open(f"Datasets/{folder_title}/users/{user_ids[i]}/timeline.json", 'w') as f:
            f.write(json.dumps(timelinejson, indent=4))
        with open(f"Datasets/{folder_title}/users/{user_ids[i]}/following.json", 'w') as f:
            f.write(json.dumps(followingjson, indent=4))
        with open(f"Datasets/{folder_title}/users/{user_ids[i]}/followers.json", 'w') as f:
            f.write(json.dumps(followersjson, indent=4))
 
if __name__ == "__main__":

    """
    Parameters
    ----------
    * folder_title: The name of the folder you want to create (will be sub-folder of Datasets)
    * user_filename: The name of the file in datasets that you'd like to read user_ids from.

    * argv[1]: config.json file (credentials for usage of API)
    * argv[2]: <0 | 1 | full>, which partition of data must be collected

    Returns
    -------
    Writes into Datasets/{folder_title}/users all the timelines and appropriate data needed to analyse. 
    """

    if (len(sys.argv) < 5):
        print("Usage: python3 main.py <configX.json> <0 | 1 | full> dataset_folder_name dataset_file_name")
        sys.exit(1)

    config_file = sys.argv[1]
    workload = sys.argv[2]
    dataset_folder_name = sys.argv[3]
    dataset_file_name = sys.argv[4]
    labels_folder = "Raw Labels"
    # Check if files exist
    if os.path.exists(f"{labels_folder}/active_users/{dataset_file_name}"):
        collect_users(config_file,
                      workload,
                      dataset_folder_name,
                      dataset_file_name,
                      labels_folder
                    )
    else:
        print("Check if dataset files exist")
